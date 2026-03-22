"""Scene-aware multi-modal dialogue extraction pipeline (v2).

Four-stage pipeline:
  Stage 0: Shot boundary detection (av1an)
  Stage 1: Scene grouping (shots → scenes → chunks)
  Stage 2: Dialogue detection (SAM-Audio text_only, scene-aligned chunks)
  Stage 3: Face scan + character clustering + SAM3 body tracking + visual separation
           Phase A: keyframe face detection on dialogue chunks → cluster → profiles
           Phase B: SAM3 multi-object body tracking + SAM-Audio visual separation
  Stage 4: Reconciliation + timeline assembly
"""

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import time
from pathlib import Path

# Reduce CUDA memory fragmentation from SAM3↔SAM-Audio model swapping
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchaudio
from tqdm import tqdm

logger = logging.getLogger(__name__)


# --- Audio helpers ---


def extract_audio(
    video_path: str,
    sample_rate: int = 48000,
    stream_index: int | None = None,
) -> torch.Tensor:
    """Extract mono audio from video file at target sample rate.

    Args:
        video_path: Path to the video file.
        sample_rate: Target sample rate.
        stream_index: Optional audio stream index for ffmpeg (e.g., 6 for 0:a:4).
    """
    if stream_index is not None:
        # Use ffmpeg to extract specific audio stream
        import subprocess
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-map",
            f"0:{stream_index}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            tmp.name,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        wav, sr = torchaudio.load(tmp.name)
        Path(tmp.name).unlink()
        return wav[:1]  # mono
    else:
        wav, sr = torchaudio.load(video_path)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        return wav.mean(0, keepdim=True)


def rms_db(audio: torch.Tensor) -> float:
    rms = audio.float().pow(2).mean().sqrt()
    if rms < 1e-10:
        return -100.0
    return 20 * math.log10(rms.item())


def analyze_chunk(
    target: torch.Tensor,
    residual: torch.Tensor,
    start_sec: float,
    end_sec: float,
    chunk_index: int,
    sample_rate: int,
    rms_threshold_db: float,
) -> dict:
    """Analyze a separated chunk into 1-second segments."""
    seg_samples = sample_rate
    total_samples = target.shape[-1]

    segments = []
    has_any_dialogue = False
    for seg_start in range(0, total_samples, seg_samples):
        seg_end = min(seg_start + seg_samples, total_samples)
        t_seg = target[..., seg_start:seg_end].flatten()
        r_seg = residual[..., seg_start:seg_end].flatten()

        t_rms = rms_db(t_seg)
        r_rms = rms_db(r_seg)
        t_to_r = t_rms - r_rms

        has_dialogue = t_rms > rms_threshold_db and t_to_r > 0
        if has_dialogue:
            has_any_dialogue = True

        seg_start_time = start_sec + seg_start / sample_rate
        seg_end_time = start_sec + seg_end / sample_rate
        segments.append(
            {
                "start_time": round(seg_start_time, 3),
                "end_time": round(seg_end_time, 3),
                "has_dialogue": has_dialogue,
                "target_rms_db": round(t_rms, 1),
                "target_to_residual_db": round(t_to_r, 1),
            }
        )

    return {
        "chunk_index": chunk_index,
        "start_time": round(start_sec, 3),
        "end_time": round(end_sec, 3),
        "has_any_dialogue": has_any_dialogue,
        "segments": segments,
    }


def extract_chunk_frames(
    video_decoder,
    start_sec: float,
    end_sec: float,
    fps: float,
    max_frames: int = 150,
    max_height: int = 480,
) -> torch.Tensor:
    """Extract video frames for a time range, downscaled for VRAM efficiency.

    The SAMAudioProcessor resamples frames to the audio token count (25/sec),
    so a 90s chunk produces 2250 frames. At 1080p that's ~53GB — must downscale.
    """
    duration = end_sec - start_sec
    num_frames = min(int(duration * fps), max_frames)
    if num_frames <= 0:
        num_frames = 1
    timestamps = [start_sec + i * duration / num_frames for i in range(num_frames)]
    batch = video_decoder.get_frames_played_at(timestamps)
    frames = batch.data  # [N, C, H, W]

    # Downscale if needed to avoid OOM in vision encoder
    _, _, h, w = frames.shape
    if h > max_height:
        scale = max_height / h
        new_h = max_height
        new_w = int(w * scale)
        frames = torch.nn.functional.interpolate(
            frames.float(), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).to(frames.dtype)

    return frames


# --- Resume support ---


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {
        "stage": 0,
        "pass2_completed": [],
        "pass3_completed": [],
        "chunk_results": [],
    }


def save_progress(progress_path: Path, progress: dict):
    with open(progress_path, "w") as f:
        json.dump(progress, f)


# --- Timeline building ---


def build_timeline(chunk_results: list[dict], min_gap_seconds: float) -> dict:
    """Merge chunk segments into a continuous dialogue timeline."""
    all_segments = {}
    for chunk in chunk_results:
        for seg in chunk["segments"]:
            key = seg["start_time"]
            all_segments[key] = seg

    sorted_segs = sorted(all_segments.values(), key=lambda s: s["start_time"])

    # Build per-character speaker map from character_separation data
    char_sep_map = {}
    for chunk in chunk_results:
        if "character_separation" not in chunk:
            continue
        for seg in chunk["segments"]:
            key = seg["start_time"]
            speakers = []
            for char_sep in chunk["character_separation"]:
                for cseg in char_sep["segments"]:
                    if (
                        abs(cseg["start_time"] - seg["start_time"]) < 0.01
                        and cseg["has_dialogue"]
                    ):
                        speakers.append(char_sep["character_id"])
            if speakers:
                char_sep_map[key] = speakers

    # Merge consecutive dialogue segments
    dialogue_segments = []
    current = None
    for seg in sorted_segs:
        speakers = char_sep_map.get(seg["start_time"])
        if seg["has_dialogue"]:
            if current is None:
                current = {
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "speakers": speakers,
                    "rms_values": [seg["target_rms_db"]],
                }
            elif (
                seg["start_time"] - current["end_time"] < 0.01
                and current["speakers"] == speakers
            ):
                current["end_time"] = seg["end_time"]
                current["rms_values"].append(seg["target_rms_db"])
            else:
                current["duration"] = round(
                    current["end_time"] - current["start_time"], 3
                )
                current["avg_target_rms_db"] = round(
                    sum(current["rms_values"]) / len(current["rms_values"]), 1
                )
                del current["rms_values"]
                dialogue_segments.append(current)
                current = {
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "speakers": speakers,
                    "rms_values": [seg["target_rms_db"]],
                }
        else:
            if current is not None:
                current["duration"] = round(
                    current["end_time"] - current["start_time"], 3
                )
                current["avg_target_rms_db"] = round(
                    sum(current["rms_values"]) / len(current["rms_values"]), 1
                )
                del current["rms_values"]
                dialogue_segments.append(current)
                current = None

    if current is not None:
        current["duration"] = round(current["end_time"] - current["start_time"], 3)
        current["avg_target_rms_db"] = round(
            sum(current["rms_values"]) / len(current["rms_values"]), 1
        )
        del current["rms_values"]
        dialogue_segments.append(current)

    # Build gaps
    gaps = []
    if sorted_segs:
        movie_start = sorted_segs[0]["start_time"]
        movie_end = sorted_segs[-1]["end_time"]
        prev_end = movie_start
        for ds in dialogue_segments:
            gap_dur = ds["start_time"] - prev_end
            if gap_dur >= min_gap_seconds:
                gaps.append(
                    {
                        "start_time": round(prev_end, 3),
                        "end_time": round(ds["start_time"], 3),
                        "duration": round(gap_dur, 3),
                    }
                )
            prev_end = ds["end_time"]
        if movie_end - prev_end >= min_gap_seconds:
            gaps.append(
                {
                    "start_time": round(prev_end, 3),
                    "end_time": round(movie_end, 3),
                    "duration": round(movie_end - prev_end, 3),
                }
            )

    return {"dialogue_segments": dialogue_segments, "gaps": gaps}


# --- Main pipeline ---


def _source_tag(input_path: str) -> str:
    """Short hash from source filename to namespace workspace artifacts."""
    name = Path(input_path).stem
    h = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{name}.{h}"


def process_movie(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # Namespace all artifacts by source file to prevent cross-video overwrites
    tag = _source_tag(args.input)
    progress_path = workspace / f"pipeline.{tag}.progress.json"
    progress = (
        load_progress(progress_path)
        if args.resume
        else {
            "stage": 0,
            "pass2_completed": [],
            "pass3_completed": [],
            "chunk_results": [],
        }
    )

    t_start = time.time()
    shots_scenes_path = workspace / f"shots_scenes.{tag}.json"

    # ================================================================
    # STAGE 0: Shot Boundary Detection
    # ================================================================
    from scene_detector import (
        detect_shots,
        generate_scene_chunks,
        group_shots_into_scenes,
        save_shots_and_scenes,
    )

    logger.info("=== Stage 0: Shot Boundary Detection ===")
    t_stage0_start = time.time()

    shots_json = workspace / f"shots.{tag}.json"
    if args.shots_json and Path(args.shots_json).exists():
        from scene_detector import load_shots_from_json

        shots = load_shots_from_json(args.shots_json, args.input)
        logger.info(f"Loaded {len(shots)} shots from {args.shots_json}")
    elif shots_json.exists():
        from scene_detector import load_shots_from_json

        shots = load_shots_from_json(str(shots_json), args.input)
        logger.info(f"Loaded {len(shots)} cached shots from {shots_json}")
    else:
        shots = detect_shots(args.input, str(shots_json))

    t_stage0 = time.time() - t_stage0_start
    logger.info(f"Stage 0 complete: {len(shots)} shots in {t_stage0:.1f}s")

    # ================================================================
    # STAGE 1: Scene Grouping
    # ================================================================
    logger.info("=== Stage 1: Scene Grouping ===")
    t_stage1_start = time.time()

    scenes = group_shots_into_scenes(
        shots,
        max_gap_sec=args.max_scene_gap,
        max_scene_duration=args.max_scene_duration,
    )
    save_shots_and_scenes(shots, scenes, str(shots_scenes_path))

    # Generate scene-aligned chunks (no character info yet — that comes in Stage 3)
    scene_chunks = generate_scene_chunks(scenes, max_chunk_seconds=args.window_seconds)

    t_stage1 = time.time() - t_stage1_start
    logger.info(
        f"Stage 1 complete: {len(scenes)} scenes, "
        f"{len(scene_chunks)} chunks in {t_stage1:.1f}s"
    )

    # ================================================================
    # STAGE 2: Dialogue Detection (audio-only, no video decode)
    # ================================================================
    from sam_audio import SAMAudio, SAMAudioProcessor

    logger.info("=== Stage 2: Dialogue Detection ===")
    t_stage2_start = time.time()

    model = SAMAudio.from_pretrained(args.checkpoint).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(args.checkpoint)

    # Extract audio
    # TODO: This loads the entire movie audio into RAM (~1.7GB for a 2.5hr film).
    # We only need ~90s chunks at a time (~17MB). Could instead extract to a temp
    # WAV file once, then use torchaudio.load(offset=, num_frames=) for random
    # access per chunk — WAV supports seeking, video containers don't.
    logger.info("Extracting audio from video")
    full_audio = extract_audio(
        args.input, sample_rate=48000, stream_index=args.audio_stream
    )
    total_duration = full_audio.shape[-1] / 48000

    chunk_results = []
    completed_indices = set(progress["pass2_completed"])
    existing_results = {r["chunk_index"]: r for r in progress.get("chunk_results", [])}

    for chunk_info in tqdm(scene_chunks, desc="Stage 2: dialogue detection"):
        idx = chunk_info["chunk_index"]
        if idx in completed_indices:
            chunk_results.append(existing_results[idx])
            continue

        start_sample = int(chunk_info["start_sec"] * 48000)
        end_sample = int(chunk_info["end_sec"] * 48000)
        chunk_audio = full_audio[:, start_sample:end_sample]

        if chunk_audio.shape[-1] < 48000:  # skip chunks < 1s
            continue

        batch = processor(descriptions=["speech"], audios=[chunk_audio]).to(device)
        result = model.separate(batch, max_chunk_tokens=args.max_chunk_tokens)

        chunk_meta = analyze_chunk(
            result.target[0].cpu(),
            result.residual[0].cpu(),
            chunk_info["start_sec"],
            chunk_info["end_sec"],
            idx,
            48000,
            args.rms_threshold_db,
        )

        chunk_meta["scene_id"] = chunk_info["scene_id"]

        chunk_results.append(chunk_meta)
        progress["pass2_completed"].append(idx)
        progress["chunk_results"] = chunk_results
        save_progress(progress_path, progress)

        del batch, result
        torch.cuda.empty_cache()

    # Mark scenes with dialogue
    scene_has_dialogue = {}
    for cr in chunk_results:
        sid = cr.get("scene_id", -1)
        if cr["has_any_dialogue"]:
            scene_has_dialogue[sid] = True
    for scene in scenes:
        scene.has_dialogue = scene_has_dialogue.get(scene.scene_id, False)

    t_stage2 = time.time() - t_stage2_start
    logger.info(f"Stage 2 complete: {len(chunk_results)} chunks in {t_stage2:.1f}s")

    # ================================================================
    # STAGE 3: Face Scan + Clustering + SAM3 Tracking + Visual Separation
    #
    # Phase A: Load InsightFace on GPU, detect faces on 1 keyframe per
    #          dialogue chunk (at 480p — same resolution as SAM-Audio).
    #          Then cluster → character profiles. Unload InsightFace.
    # Phase B: Load SAM3 on GPU. For multi-speaker chunks, SAM3 body
    #          tracking + SAM-Audio visual separation.
    # ================================================================
    from face_tracker import FaceTracker
    from torchcodec.decoders import VideoDecoder

    logger.info("=== Stage 3: Character Detection + Visual Separation ===")
    t_stage3_start = time.time()

    video_decoder = VideoDecoder(args.input, dimension_order="NCHW")
    fps = video_decoder.metadata.average_fps_from_header

    # --- Phase A: face detection + clustering (InsightFace on GPU) ---
    stage3a_cache = workspace / f"stage3a.{tag}.pkl"

    dialogue_chunks = [c for c in chunk_results if c["has_any_dialogue"]]

    if stage3a_cache.exists() and args.resume:
        with open(stage3a_cache, "rb") as f:
            cache = pickle.load(f)
        chunk_keyframe_dets = cache["chunk_keyframe_dets"]
        profiles = cache["profiles"]
        total_face_dets = sum(len(d) for d in chunk_keyframe_dets.values())
        logger.info(
            f"Phase A: loaded {total_face_dets} cached face detections, "
            f"{len(profiles)} characters from {stage3a_cache.name}"
        )
    else:
        face_tracker = FaceTracker(
            sam3_predictor=None,
            det_threshold=args.face_det_threshold,
            cluster_threshold=args.cluster_threshold,
        )

        logger.info(
            f"Phase A: scanning {len(dialogue_chunks)} dialogue chunks for faces"
        )

        # Detect faces at FULL resolution for high-quality embeddings.
        # InsightFace needs faces ≥50px for reliable 512-dim ArcFace embeddings;
        # at 480p many faces are <30px and produce garbage embeddings.
        # After detection+embedding, scale bboxes down to 480p for SAM3/SAM-Audio.
        chunk_keyframe_dets: dict[int, list] = {}
        for chunk_meta in tqdm(dialogue_chunks, desc="Stage 3A: face detection"):
            idx = chunk_meta["chunk_index"]
            chunk_duration = chunk_meta["end_time"] - chunk_meta["start_time"]
            keyframe_t = chunk_meta["start_time"] + chunk_duration / 3
            # Decode at full resolution for face detection quality
            keyframe_full = extract_chunk_frames(
                video_decoder,
                keyframe_t,
                keyframe_t + 0.01,
                fps,
                max_frames=1,
                max_height=9999,  # no downscale
            )  # [1, C, H, W]
            _, _, full_h, full_w = keyframe_full.shape

            dets = face_tracker.detect_faces(keyframe_full, frame_indices=[0])
            del keyframe_full

            # Quality filter at full resolution
            min_face_area = 2500
            good_dets = []
            scale = 480 / full_h if full_h > 480 else 1.0
            for det in dets[0]:
                x1, y1, x2, y2 = det.bbox
                w, h = x2 - x1, y2 - y1
                area = w * h
                ar = w / h if h > 0 else 0.0
                if area < min_face_area or det.confidence < 0.5 or ar < 0.4 or ar > 2.5:
                    continue
                # Scale bbox to 480p for SAM3/SAM-Audio
                det.bbox = (
                    int(x1 * scale),
                    int(y1 * scale),
                    int(x2 * scale),
                    int(y2 * scale),
                )
                det.timestamp = keyframe_t
                good_dets.append(det)

            chunk_keyframe_dets[idx] = good_dets

        total_face_dets = sum(len(d) for d in chunk_keyframe_dets.values())
        logger.info(
            f"Phase A: {total_face_dets} face detections "
            f"from {len(dialogue_chunks)} keyframes"
        )

        # Cluster all keyframe face detections → character profiles
        all_keyframe_dets = [list(dets) for dets in chunk_keyframe_dets.values()]
        all_keyframe_dets = [d for d in all_keyframe_dets if d]

        if total_face_dets > 0:
            profiles = face_tracker.cluster_to_profiles(all_keyframe_dets)
            profiles = dict(list(profiles.items())[: args.max_characters])
            logger.info(f"Clustered into {len(profiles)} characters")
        else:
            profiles = {}
            logger.warning("No face detections found, skipping visual separation")

        # Cache Stage 3A results
        with open(stage3a_cache, "wb") as f:
            pickle.dump(
                {
                    "chunk_keyframe_dets": chunk_keyframe_dets,
                    "profiles": profiles,
                },
                f,
            )

        # Unload InsightFace from GPU
        del face_tracker.face_app
        torch.cuda.empty_cache()
        logger.info("Unloaded InsightFace, freeing GPU for SAM3")

    # Build char_id lookup per chunk from keyframe detections
    for idx, dets in chunk_keyframe_dets.items():
        char_ids = sorted({d.character_id for d in dets if d.character_id >= 0})
        for cm in chunk_results:
            if cm["chunk_index"] == idx:
                cm["visible_characters"] = char_ids
                cm["needs_visual_pass"] = len(char_ids) >= 2 and cm["has_any_dialogue"]
                break

    # --- Phase B: SAM3 body tracking + SAM-Audio separation ---
    from sam3.model_builder import build_sam3_video_predictor
    from voice_tracker import VoiceTracker

    sam3_predictor = build_sam3_video_predictor()
    # When resuming from Stage 3A cache, face_tracker was never created
    # (FaceTracker.__init__ loads InsightFace which we want to skip).
    # Create a lightweight stand-in that only has .sam3 for tracking.
    try:
        face_tracker.sam3 = sam3_predictor
    except UnboundLocalError:
        face_tracker = FaceTracker.__new__(FaceTracker)
        face_tracker.sam3 = sam3_predictor
    logger.info("SAM3 video predictor loaded for body tracking")

    voice_tracker = VoiceTracker(device=device)

    multi_speaker_chunks = [c for c in chunk_results if c.get("needs_visual_pass")]
    logger.info(
        f"Phase B: {len(multi_speaker_chunks)} / {len(chunk_results)} "
        f"chunks need visual separation"
    )

    completed_pass3 = set(progress["pass3_completed"])

    for chunk_meta in tqdm(multi_speaker_chunks, desc="Stage 3B: visual separation"):
        idx = chunk_meta["chunk_index"]
        if idx in completed_pass3:
            continue

        start_sample = int(chunk_meta["start_time"] * 48000)
        end_sample = int(chunk_meta["end_time"] * 48000)
        chunk_audio = full_audio[:, start_sample:end_sample]

        eligible_chars = chunk_meta["visible_characters"]

        # Build character_detections from keyframe dets (at 480p)
        char_dets = {}
        for det in chunk_keyframe_dets.get(idx, []):
            cid = det.character_id
            if cid in eligible_chars:
                if cid not in char_dets or det.confidence > char_dets[cid].confidence:
                    char_dets[cid] = det

        if len(char_dets) < 2:
            continue

        # Single decode — shared by SAM3 and SAM-Audio (no temp files)
        chunk_frames = extract_chunk_frames(
            video_decoder,
            chunk_meta["start_time"],
            chunk_meta["end_time"],
            fps,
            max_frames=150,
        )

        # Build prompt frame indices from keyframe timestamps
        chunk_duration = chunk_meta["end_time"] - chunk_meta["start_time"]
        prompt_indices = {}
        for cid, det in char_dets.items():
            det_time = getattr(det, "timestamp", -1.0)
            if det_time >= 0:
                t_frac = (det_time - chunk_meta["start_time"]) / chunk_duration
                local_idx = int(t_frac * chunk_frames.shape[0])
            else:
                local_idx = chunk_frames.shape[0] // 3
            prompt_indices[cid] = max(0, min(local_idx, chunk_frames.shape[0] - 1))

        # SAM3 tracks ALL characters using shared frames (no re-decode)
        try:
            per_char_masks = face_tracker.track_characters_in_chunk(
                chunk_frames=chunk_frames,
                character_detections=char_dets,
                prompt_frame_indices=prompt_indices,
            )
        except Exception as e:
            logger.warning(f"Chunk {idx}: SAM3 tracking failed: {e}")
            continue

        # Offload SAM3 to CPU to free ~30GB VRAM for SAM-Audio separation
        sam3_predictor.model.to("cpu")
        torch.cuda.empty_cache()

        character_segments = []

        # Run SAM-Audio separation per character using SAM3's body masks
        # Masks are already at 480p with same frame count — no resampling
        for char_id, masks in per_char_masks.items():
            masked_video = processor.mask_videos([chunk_frames], [masks])

            batch = processor(
                descriptions=[""],
                audios=[chunk_audio],
                masked_videos=masked_video,
            ).to(device)
            result = model.separate(batch, max_chunk_tokens=args.max_chunk_tokens)

            target_cpu = result.target[0].cpu()
            residual_cpu = result.residual[0].cpu()

            char_analysis = analyze_chunk(
                target_cpu,
                residual_cpu,
                chunk_meta["start_time"],
                chunk_meta["end_time"],
                idx,
                48000,
                args.rms_threshold_db,
            )

            char_seg_entry = {
                "character_id": char_id,
                "segments": char_analysis["segments"],
            }

            # Voice fingerprinting — extract from dialogue segments only
            # (full-chunk RMS is dominated by non-speech, failing the threshold)
            dial_segs = [s for s in char_analysis["segments"] if s["has_dialogue"]]
            if dial_segs and char_id in profiles:
                seg_start = int(
                    (dial_segs[0]["start_time"] - chunk_meta["start_time"]) * 48000
                )
                seg_end = int(
                    (dial_segs[-1]["end_time"] - chunk_meta["start_time"]) * 48000
                )
                voice_emb = voice_tracker.extract_embedding(
                    target_cpu[..., seg_start:seg_end], sample_rate=48000
                )
                if voice_emb is not None:
                    voice_tracker.update_profile(profiles[char_id], voice_emb)
                    char_seg_entry["has_voice_sample"] = True

            character_segments.append(char_seg_entry)

            del batch, result
            torch.cuda.empty_cache()

        del chunk_frames, per_char_masks
        torch.cuda.empty_cache()

        # Restore SAM3 to GPU for next chunk
        sam3_predictor.model.to(device)

        chunk_meta["character_separation"] = character_segments

        # Voice-based discovery
        if args.enable_voice_discovery:
            _voice_discovery_pass(
                chunk_meta, chunk_audio, voice_tracker, profiles, 48000
            )

        progress["pass3_completed"].append(idx)
        progress["chunk_results"] = chunk_results
        save_progress(progress_path, progress)

    # --- Phase C: Identity mapping for remaining dialogue chunks ---
    # Two sub-phases:
    #   C1: 1-face chunks — directly attribute + collect voice samples
    #   C2: 0-face chunks — SAM3 confirms body presence, voice matches identity
    # Both use face + voice signals for character identification.

    single_face_chunks = [
        c
        for c in chunk_results
        if c["has_any_dialogue"]
        and not c.get("needs_visual_pass")
        and len(c.get("visible_characters", [])) == 1
    ]
    no_face_chunks = [
        c
        for c in chunk_results
        if c["has_any_dialogue"]
        and not c.get("needs_visual_pass")
        and len(c.get("visible_characters", [])) == 0
    ]

    # --- C1: Direct attribution for 1-face chunks ---
    # Face is already detected — attribute all dialogue to that character.
    # Also extract voice to enrich the profile.
    sam3_predictor.model.to("cpu")
    torch.cuda.empty_cache()

    c1_mapped = 0
    for chunk_meta in tqdm(single_face_chunks, desc="Stage 3C1: single-face"):
        char_id = chunk_meta["visible_characters"][0]
        if char_id not in profiles:
            continue

        chunk_meta["voice_match"] = {
            "character_id": char_id,
            "voice_similarity": 1.0,
            "source": "face_direct",
        }
        c1_mapped += 1

        # Extract voice sample to enrich profile
        start_sample = int(chunk_meta["start_time"] * 48000)
        end_sample = int(chunk_meta["end_time"] * 48000)
        chunk_audio = full_audio[:, start_sample:end_sample]

        batch = processor(descriptions=["speech"], audios=[chunk_audio]).to(device)
        result = model.separate(batch, max_chunk_tokens=args.max_chunk_tokens)

        target_cpu = result.target[0].cpu()
        residual_cpu = result.residual[0].cpu()
        del batch, result
        torch.cuda.empty_cache()

        dial_segs = [s for s in chunk_meta["segments"] if s["has_dialogue"]]
        if dial_segs:
            seg_start = int(
                (dial_segs[0]["start_time"] - chunk_meta["start_time"]) * 48000
            )
            seg_end = int(
                (dial_segs[-1]["end_time"] - chunk_meta["start_time"]) * 48000
            )
            voice_emb = voice_tracker.extract_embedding(
                target_cpu[..., seg_start:seg_end], sample_rate=48000
            )
            if voice_emb is not None:
                voice_tracker.update_profile(profiles[char_id], voice_emb)

    logger.info(
        f"Phase C1: {c1_mapped}/{len(single_face_chunks)} single-face chunks attributed"
    )

    # --- C2: SAM3 body detection + voice matching for 0-face chunks ---
    # SAM3 confirms a person is on screen, then voice identifies them.
    sam3_predictor.model.to(device)

    c2_voice_embs: dict[int, object] = {}
    c2_has_body: dict[int, bool] = {}

    # For 0-face chunks, SAM3 body detection confirms person presence.
    # Use a lightweight SAM3 check (1 frame) then voice-match.
    from PIL import Image

    sam3_predictor.model.to("cpu")
    torch.cuda.empty_cache()

    for chunk_meta in tqdm(no_face_chunks, desc="Stage 3C2: body+voice"):
        idx = chunk_meta["chunk_index"]

        # Quick SAM3 body check on a single frame
        sam3_predictor.model.to(device)
        keyframe_t = (
            chunk_meta["start_time"]
            + (chunk_meta["end_time"] - chunk_meta["start_time"]) / 3
        )
        keyframe = extract_chunk_frames(
            video_decoder,
            keyframe_t,
            keyframe_t + 0.01,
            fps,
            max_frames=1,
            max_height=480,
        )
        pil_frame = Image.fromarray(keyframe[0].permute(1, 2, 0).cpu().numpy())
        resp = sam3_predictor.handle_request(
            {"type": "start_session", "resource_path": [pil_frame]}
        )
        sid = resp["session_id"]
        resp = sam3_predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": sid,
                "frame_index": 0,
                "text": "person",
            }
        )
        outputs = resp.get("outputs")
        body_count = len(outputs["out_obj_ids"]) if outputs is not None else 0
        sam3_predictor.handle_request({"type": "close_session", "session_id": sid})
        del keyframe, pil_frame

        c2_has_body[idx] = body_count > 0
        if not c2_has_body[idx]:
            continue

        # Person confirmed — extract voice
        sam3_predictor.model.to("cpu")
        torch.cuda.empty_cache()

        start_sample = int(chunk_meta["start_time"] * 48000)
        end_sample = int(chunk_meta["end_time"] * 48000)
        chunk_audio = full_audio[:, start_sample:end_sample]

        batch = processor(descriptions=["speech"], audios=[chunk_audio]).to(device)
        result = model.separate(batch, max_chunk_tokens=args.max_chunk_tokens)

        target_cpu = result.target[0].cpu()
        residual_cpu = result.residual[0].cpu()
        del batch, result
        torch.cuda.empty_cache()

        dial_segs = [s for s in chunk_meta["segments"] if s["has_dialogue"]]
        if dial_segs:
            seg_start = int(
                (dial_segs[0]["start_time"] - chunk_meta["start_time"]) * 48000
            )
            seg_end = int(
                (dial_segs[-1]["end_time"] - chunk_meta["start_time"]) * 48000
            )
            voice_emb = voice_tracker.extract_embedding(
                target_cpu[..., seg_start:seg_end], sample_rate=48000
            )
            if voice_emb is not None:
                c2_voice_embs[idx] = voice_emb

    logger.info(
        f"Phase C2: {sum(c2_has_body.values())} bodies detected, "
        f"{len(c2_voice_embs)} voice embeddings from {len(no_face_chunks)} chunks"
    )

    # Multi-pass voice matching for 0-face chunks
    total_c2 = 0
    for pass_num in range(3):
        mapped_this_pass = 0
        for chunk_meta in no_face_chunks:
            idx = chunk_meta["chunk_index"]
            if chunk_meta.get("voice_match") or idx not in c2_voice_embs:
                continue
            voice_emb = c2_voice_embs[idx]
            match_id, sim = voice_tracker.match_to_profiles(voice_emb, profiles)
            if match_id is not None:
                chunk_meta["voice_match"] = {
                    "character_id": match_id,
                    "voice_similarity": round(sim, 3),
                    "source": "body_voice",
                }
                voice_tracker.update_profile(profiles[match_id], voice_emb)
                mapped_this_pass += 1
        total_c2 += mapped_this_pass
        if mapped_this_pass == 0:
            break
        logger.info(
            f"Phase C2 pass {pass_num + 1}: matched {mapped_this_pass} (total {total_c2})"
        )

    logger.info(
        f"Phase C total: {c1_mapped + total_c2} chunks attributed "
        f"({c1_mapped} face-direct, {total_c2} body+voice)"
    )

    t_stage3 = time.time() - t_stage3_start
    logger.info(f"Stage 3 complete in {t_stage3:.1f}s")

    # ================================================================
    # STAGE 4: Reconciliation + Timeline Assembly
    # ================================================================
    logger.info("=== Stage 4: Reconciliation + Timeline Assembly ===")
    t_stage4_start = time.time()

    timeline = build_timeline(chunk_results, args.min_gap_seconds)

    # Compute per-character stats (visual separation + voice-matched monologues)
    char_stats = []
    for cid, profile in profiles.items():
        total_speaking = 0.0
        for chunk in chunk_results:
            # Visual separation (multi-speaker chunks)
            if "character_separation" in chunk:
                for cs in chunk["character_separation"]:
                    if cs["character_id"] == cid:
                        for seg in cs["segments"]:
                            if seg["has_dialogue"]:
                                total_speaking += seg["end_time"] - seg["start_time"]
            # Voice-matched monologues (single-speaker chunks)
            vm = chunk.get("voice_match")
            if vm and vm["character_id"] == cid:
                for seg in chunk["segments"]:
                    if seg["has_dialogue"]:
                        total_speaking += seg["end_time"] - seg["start_time"]

        face_det_count = sum(
            1
            for dets in chunk_keyframe_dets.values()
            for d in dets
            if d.character_id == cid
        )

        char_stats.append(
            {
                "id": cid,
                "name": None,
                "face_detections": face_det_count,
                "voice_samples": len(profile.voice_embeddings),
                "speaking_sec": round(total_speaking, 1),
                "identity_sources": profile.identity_sources,
            }
        )

    # Build per-character speaking segments
    per_character = {}
    for cid in profiles:
        segments = []
        for chunk in chunk_results:
            # Visual separation
            if "character_separation" in chunk:
                for cs in chunk["character_separation"]:
                    if cs["character_id"] == cid:
                        for seg in cs["segments"]:
                            if seg["has_dialogue"]:
                                segments.append([seg["start_time"], seg["end_time"]])
            # Voice-matched monologues
            vm = chunk.get("voice_match")
            if vm and vm["character_id"] == cid:
                for seg in chunk["segments"]:
                    if seg["has_dialogue"]:
                        segments.append([seg["start_time"], seg["end_time"]])
        merged = _merge_segments(segments)
        per_character[str(cid)] = {"segments": merged}

    # Dialogue timeline (all dialogue regardless of character)
    all_segments = {}
    for chunk in chunk_results:
        for seg in chunk["segments"]:
            all_segments[seg["start_time"]] = seg
    total_dialogue = sum(
        seg["end_time"] - seg["start_time"]
        for seg in all_segments.values()
        if seg["has_dialogue"]
    )

    t_stage4 = time.time() - t_stage4_start
    t_total = time.time() - t_start

    # Build scenes summary
    scenes_summary = [
        {
            "scene_id": sc.scene_id,
            "start": round(sc.start_sec, 1),
            "end": round(sc.end_sec, 1),
            "num_shots": len(sc.shots),
            "has_dialogue": sc.has_dialogue,
        }
        for sc in scenes
    ]

    metadata = {
        "version": "2.1",
        "source": {
            "file": args.input,
            "duration": round(total_duration, 1),
        },
        "scenes": scenes_summary,
        "characters": char_stats,
        "per_character": per_character,
        "dialogue_timeline": [
            [ds["start_time"], ds["end_time"]] for ds in timeline["dialogue_segments"]
        ],
        "timeline": timeline,
        "processing": {
            "model": args.checkpoint,
            "num_shots": len(shots),
            "num_scenes": len(scenes),
            "num_chunks": len(scene_chunks),
            "num_dialogue_chunks": len(dialogue_chunks),
            "num_multi_speaker_chunks": len(multi_speaker_chunks),
            "face_det_threshold": args.face_det_threshold,
            "stage0_time": round(t_stage0, 1),
            "stage1_time": round(t_stage1, 1),
            "stage2_time": round(t_stage2, 1),
            "stage3_time": round(t_stage3, 1),
            "stage4_time": round(t_stage4, 1),
            "total_time": round(t_total, 1),
        },
        "statistics": {
            "dialogue_percentage": round(total_dialogue / total_duration * 100, 1)
            if total_duration > 0
            else 0,
            "total_dialogue_seconds": round(total_dialogue, 1),
            "num_gaps": len(timeline["gaps"]),
        },
        "chunks": chunk_results,
    }

    with open(args.output, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata to {args.output}")

    # Clean up progress file on success
    if progress_path.exists():
        progress_path.unlink()

    logger.info(f"Pipeline complete in {t_total:.1f}s ({t_total / 60:.1f}m)")


def _voice_discovery_pass(
    chunk_meta: dict,
    chunk_audio: torch.Tensor,
    voice_tracker,
    profiles: dict,
    sample_rate: int,
):
    """Try to identify unattributed dialogue segments via voice matching."""
    if not chunk_meta.get("character_separation"):
        return

    # Find 1s segments where dialogue exists but no character claims it
    attributed_times = set()
    for cs in chunk_meta["character_separation"]:
        for seg in cs["segments"]:
            if seg["has_dialogue"]:
                attributed_times.add(seg["start_time"])

    unattributed = []
    for seg in chunk_meta["segments"]:
        if seg["has_dialogue"] and seg["start_time"] not in attributed_times:
            unattributed.append(seg)

    if not unattributed:
        return

    # Extract voice from unattributed segments and try to match
    voice_matches = []
    for seg in unattributed:
        seg_start = int((seg["start_time"] - chunk_meta["start_time"]) * sample_rate)
        seg_end = int((seg["end_time"] - chunk_meta["start_time"]) * sample_rate)
        seg_audio = chunk_audio[:, seg_start:seg_end]

        emb = voice_tracker.extract_embedding(seg_audio, sample_rate)
        if emb is None:
            continue

        match_id, sim = voice_tracker.match_to_profiles(emb, profiles)
        if match_id is not None:
            voice_matches.append(
                {
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "character_id": match_id,
                    "voice_similarity": round(sim, 3),
                    "source": "voice_discovery",
                }
            )

    if voice_matches:
        chunk_meta["voice_discoveries"] = voice_matches
        logger.debug(
            f"Chunk {chunk_meta['chunk_index']}: "
            f"{len(voice_matches)} voice-discovered segments"
        )


def _merge_segments(
    segments: list[list[float]], gap: float = 0.01
) -> list[list[float]]:
    """Merge adjacent/overlapping time segments."""
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda s: s[0])
    merged = [sorted_segs[0][:]]
    for s in sorted_segs[1:]:
        if s[0] - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(s[:])
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Scene-aware multi-modal dialogue extraction pipeline (v2)"
    )
    parser.add_argument("--input", required=True, help="Input video file (mkv/mp4)")
    parser.add_argument(
        "--output", default="dialogue_metadata_v2.json", help="Output JSON path"
    )
    parser.add_argument(
        "--workspace",
        default="./workspace/v2",
        help="Directory for intermediate files",
    )
    parser.add_argument(
        "--checkpoint",
        default="facebook/sam-audio-base-tv",
        help="SAM-Audio model checkpoint",
    )
    parser.add_argument("--device", default=None, help="Device (default: auto)")

    # Shot detection
    parser.add_argument(
        "--shots-json",
        default=None,
        help="Pre-computed shot boundaries JSON (skip av1an)",
    )

    # Face detection
    parser.add_argument(
        "--face-det-threshold",
        type=float,
        default=0.3,
        help="Face detection confidence threshold",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.6,
        help="Face clustering distance threshold",
    )

    # Scene grouping
    parser.add_argument(
        "--max-scene-gap",
        type=float,
        default=2.0,
        help="Max gap between shots to merge into same scene",
    )
    parser.add_argument(
        "--max-scene-duration",
        type=float,
        default=300.0,
        help="Max scene duration in seconds",
    )

    # Dialogue detection
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=90,
        help="Max chunk size in seconds",
    )
    parser.add_argument(
        "--rms-threshold-db",
        type=float,
        default=-50,
        help="RMS threshold for dialogue detection (combined with target>residual ratio)",
    )
    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=3.0,
        help="Minimum gap duration to report",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=500,
        help="Max tokens for chunked decoding",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=8,
        help="Max characters to track",
    )

    # Audio stream
    parser.add_argument(
        "--audio-stream",
        type=int,
        default=None,
        help="Audio stream index in ffmpeg (e.g., 6 for stream 0:a:4)",
    )

    # Voice
    parser.add_argument(
        "--voice-quality-threshold",
        type=float,
        default=5.0,
        help="Min target-to-residual dB for voice extraction",
    )
    parser.add_argument(
        "--enable-voice-discovery",
        action="store_true",
        help="Enable voice-based character discovery for unattributed dialogue",
    )

    # Resume
    parser.add_argument(
        "--resume", action="store_true", help="Resume from progress file"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    process_movie(args)


if __name__ == "__main__":
    main()
