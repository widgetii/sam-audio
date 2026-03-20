"""Scene-aware multi-modal dialogue extraction pipeline (v2).

Five-stage pipeline:
  Stage 0: Shot boundary detection (av1an)
  Stage 1: Dense face detection + clustering (InsightFace, 1080p, 0.5s sampling)
  Stage 2: Scene grouping + character propagation
  Stage 3: Dialogue detection (SAM-Audio text_only, scene-aligned chunks)
  Stage 4: Visual separation + voice fingerprinting (SAM-Audio visual + ECAPA-TDNN)
  Stage 5: Reconciliation + timeline assembly
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
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

        has_dialogue = t_rms > rms_threshold_db
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


# --- Face detection cache ---


def _save_face_cache(
    path: Path,
    all_detections: list[list],
    sample_timestamps: list[float],
    sample_shot_indices: list[int],
    profiles: dict,
):
    """Save face detection results to disk for fast resume."""
    import pickle

    import numpy as np

    data = {
        "sample_timestamps": np.array(sample_timestamps, dtype=np.float64),
        "sample_shot_indices": np.array(sample_shot_indices, dtype=np.int32),
    }

    # Serialize detections: flatten into parallel arrays
    det_frame_ids = []  # which frame each detection belongs to
    det_bboxes = []
    det_embeddings = []
    det_confidences = []
    det_frame_indices = []
    det_timestamps = []
    det_character_ids = []
    det_shot_indices = []

    for frame_i, frame_dets in enumerate(all_detections):
        for det in frame_dets:
            det_frame_ids.append(frame_i)
            det_bboxes.append(list(det.bbox))
            det_embeddings.append(det.embedding)
            det_confidences.append(det.confidence)
            det_frame_indices.append(det.frame_index)
            det_timestamps.append(getattr(det, "timestamp", -1.0))
            det_character_ids.append(det.character_id)
            det_shot_indices.append(getattr(det, "shot_index", -1))

    data["num_frames"] = len(all_detections)
    if det_embeddings:
        data["det_frame_ids"] = np.array(det_frame_ids, dtype=np.int32)
        data["det_bboxes"] = np.array(det_bboxes, dtype=np.int32)
        data["det_embeddings"] = np.stack(det_embeddings)
        data["det_confidences"] = np.array(det_confidences, dtype=np.float32)
        data["det_frame_indices"] = np.array(det_frame_indices, dtype=np.int32)
        data["det_timestamps"] = np.array(det_timestamps, dtype=np.float64)
        data["det_character_ids"] = np.array(det_character_ids, dtype=np.int32)
        data["det_shot_indices"] = np.array(det_shot_indices, dtype=np.int32)

    # Serialize profiles
    data["profiles_pickle"] = np.void(pickle.dumps(profiles))

    np.savez_compressed(path, **data)


def _load_face_cache(path: Path):
    """Load cached face detection results."""
    import pickle

    from character_profile import FaceDetection

    data = dict(np.load(str(path), allow_pickle=True))

    sample_timestamps = data["sample_timestamps"].tolist()
    sample_shot_indices = data["sample_shot_indices"].tolist()

    num_frames = int(data["num_frames"])
    all_detections: list[list[FaceDetection]] = [[] for _ in range(num_frames)]

    if "det_embeddings" in data:
        for i in range(len(data["det_frame_ids"])):
            det = FaceDetection(
                bbox=tuple(data["det_bboxes"][i].tolist()),
                embedding=data["det_embeddings"][i],
                confidence=float(data["det_confidences"][i]),
                frame_index=int(data["det_frame_indices"][i]),
                timestamp=float(data["det_timestamps"][i]),
                character_id=int(data["det_character_ids"][i]),
                shot_index=int(data["det_shot_indices"][i]),
            )
            all_detections[int(data["det_frame_ids"][i])].append(det)

    profiles = pickle.loads(bytes(data["profiles_pickle"]))

    return all_detections, sample_timestamps, sample_shot_indices, profiles


# --- Resume support ---


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {
        "stage": 0,
        "pass3_completed": [],
        "pass4_completed": [],
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


def process_movie(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    progress_path = workspace / "pipeline.progress.json"
    progress = (
        load_progress(progress_path)
        if args.resume
        else {
            "stage": 0,
            "pass3_completed": [],
            "pass4_completed": [],
            "chunk_results": [],
        }
    )

    t_start = time.time()
    shots_scenes_path = workspace / "shots_scenes.json"

    # ================================================================
    # STAGE 0: Shot Boundary Detection
    # ================================================================
    from scene_detector import (
        detect_shots,
        generate_scene_chunks,
        group_shots_into_scenes,
        propagate_characters,
        save_shots_and_scenes,
    )

    logger.info("=== Stage 0: Shot Boundary Detection ===")
    t_stage0_start = time.time()

    shots_json = workspace / "shots_av1an.json"
    if args.shots_json and Path(args.shots_json).exists():
        # User provided pre-computed shot boundaries
        from scene_detector import load_shots_from_json

        shots = load_shots_from_json(args.shots_json, args.input)
        logger.info(f"Loaded {len(shots)} shots from {args.shots_json}")
    elif shots_json.exists() and args.resume:
        from scene_detector import load_shots_from_json

        shots = load_shots_from_json(str(shots_json), args.input)
        logger.info(f"Resumed {len(shots)} shots from {shots_json}")
    else:
        shots = detect_shots(args.input, str(shots_json))

    t_stage0 = time.time() - t_stage0_start
    logger.info(f"Stage 0 complete: {len(shots)} shots in {t_stage0:.1f}s")

    # ================================================================
    # STAGE 1: Dense Face Detection + Clustering
    # ================================================================
    from face_tracker import FaceTracker
    from torchcodec.decoders import VideoDecoder

    logger.info("=== Stage 1: Dense Face Detection + Clustering ===")
    t_stage1_start = time.time()

    face_cache_path = workspace / "face_detections.npz"
    video_decoder = VideoDecoder(args.input, dimension_order="NCHW")
    fps = video_decoder.metadata.average_fps_from_header

    face_tracker = FaceTracker(
        sam3_predictor=None,  # SAM3 loaded later if needed
        det_threshold=args.face_det_threshold,
        cluster_threshold=args.cluster_threshold,
    )

    if face_cache_path.exists() and args.resume:
        logger.info(f"Loading cached face detections from {face_cache_path}")
        all_detections, sample_timestamps, sample_shot_indices, profiles = (
            _load_face_cache(face_cache_path)
        )
        profiles = dict(list(profiles.items())[: args.max_characters])
        logger.info(
            f"Loaded {len(all_detections)} frames, {len(profiles)} characters from cache"
        )
    else:
        # Shot-aware face detection: 0.5s interval, min 2 frames per shot
        all_detections, sample_timestamps, sample_shot_indices = (
            face_tracker.detect_faces_for_shots(
                video_decoder,
                shots,
                sample_interval=args.sample_interval,
                min_frames_per_shot=2,
                batch_size=32,
            )
        )

        # Cluster into character profiles
        logger.info("Clustering faces into characters")
        profiles = face_tracker.cluster_to_profiles(all_detections)
        profiles = dict(list(profiles.items())[: args.max_characters])
        logger.info(f"Found {len(profiles)} characters")

        # Cache to disk
        _save_face_cache(
            face_cache_path,
            all_detections,
            sample_timestamps,
            sample_shot_indices,
            profiles,
        )
        logger.info(f"Cached face detections to {face_cache_path}")

    # Assign detected characters to shots
    for frame_dets, shot_idx in zip(all_detections, sample_shot_indices, strict=False):
        shot = shots[shot_idx] if shot_idx < len(shots) else None
        if shot is None:
            continue
        for det in frame_dets:
            if det.character_id >= 0 and det.character_id in profiles:
                shot.character_ids.add(det.character_id)
                shot.face_detections.append(det)

    t_stage1 = time.time() - t_stage1_start
    logger.info(
        f"Stage 1 complete: {len(all_detections)} frames processed in {t_stage1:.1f}s"
    )

    # ================================================================
    # STAGE 2: Scene Grouping + Character Propagation
    # ================================================================
    logger.info("=== Stage 2: Scene Grouping + Character Propagation ===")
    t_stage2_start = time.time()

    scenes = group_shots_into_scenes(
        shots,
        max_gap_sec=args.max_scene_gap,
        max_scene_duration=args.max_scene_duration,
    )
    propagate_characters(scenes)
    save_shots_and_scenes(shots, scenes, str(shots_scenes_path))

    # Generate scene-aligned chunks
    scene_chunks = generate_scene_chunks(scenes, max_chunk_seconds=args.window_seconds)

    t_stage2 = time.time() - t_stage2_start
    logger.info(
        f"Stage 2 complete: {len(scenes)} scenes, "
        f"{len(scene_chunks)} chunks in {t_stage2:.1f}s"
    )

    # ================================================================
    # STAGE 3: Dialogue Detection (scene-aligned)
    # ================================================================
    from sam_audio import SAMAudio, SAMAudioProcessor

    logger.info("=== Stage 3: Dialogue Detection ===")
    t_stage3_start = time.time()

    model = SAMAudio.from_pretrained(args.checkpoint).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(args.checkpoint)

    # Extract audio
    logger.info("Extracting audio from video")
    full_audio = extract_audio(
        args.input, sample_rate=48000, stream_index=args.audio_stream
    )
    total_duration = full_audio.shape[-1] / 48000

    chunk_results = []
    completed_indices = set(progress["pass3_completed"])
    existing_results = {r["chunk_index"]: r for r in progress.get("chunk_results", [])}

    for chunk_info in tqdm(scene_chunks, desc="Stage 3: dialogue detection"):
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

        # Scene-aware character info (from propagation, not just face detection in chunk)
        chunk_meta["scene_id"] = chunk_info["scene_id"]
        chunk_meta["visible_characters"] = chunk_info["characters"]
        chunk_meta["needs_visual_pass"] = (
            len(chunk_info["characters"]) >= 2 and chunk_meta["has_any_dialogue"]
        )

        chunk_results.append(chunk_meta)
        progress["pass3_completed"].append(idx)
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

    t_stage3 = time.time() - t_stage3_start
    logger.info(f"Stage 3 complete: {len(chunk_results)} chunks in {t_stage3:.1f}s")

    # ================================================================
    # STAGE 4: Visual Separation + Voice Fingerprinting
    # ================================================================
    from voice_tracker import VoiceTracker

    logger.info("=== Stage 4: Visual Separation + Voice Fingerprinting ===")
    t_stage4_start = time.time()

    voice_tracker = VoiceTracker(device=device)

    sam3_predictor = None
    if not args.no_sam3:
        try:
            from sam3.model_builder import build_sam3_video_predictor

            sam3_predictor = build_sam3_video_predictor()
            face_tracker.sam3 = sam3_predictor
        except ImportError:
            logger.warning("SAM3 not available, using bbox masks")

    multi_speaker_chunks = [c for c in chunk_results if c.get("needs_visual_pass")]
    logger.info(
        f"{len(multi_speaker_chunks)} / {len(chunk_results)} chunks need visual pass"
    )

    completed_pass4 = set(progress["pass4_completed"])

    for chunk_meta in tqdm(multi_speaker_chunks, desc="Stage 4: character separation"):
        idx = chunk_meta["chunk_index"]
        if idx in completed_pass4:
            continue

        start_sample = int(chunk_meta["start_time"] * 48000)
        end_sample = int(chunk_meta["end_time"] * 48000)
        chunk_audio = full_audio[:, start_sample:end_sample]

        chunk_frames = extract_chunk_frames(
            video_decoder,
            chunk_meta["start_time"],
            chunk_meta["end_time"],
            fps,
            max_frames=150,
        )

        # Get detections aligned to chunk frames
        sample_interval = args.sample_interval
        chunk_start_frame = int(chunk_meta["start_time"] / sample_interval)
        chunk_end_frame = int(chunk_meta["end_time"] / sample_interval)
        chunk_detections = face_tracker.get_detections_for_range(
            all_detections, chunk_start_frame, chunk_end_frame
        )

        if len(chunk_detections) > 0 and chunk_frames.shape[0] > 0:
            det_indices = (
                torch.linspace(0, len(chunk_detections) - 1, chunk_frames.shape[0])
                .round()
                .long()
                .tolist()
            )
            aligned_detections = [chunk_detections[i] for i in det_indices]
        else:
            aligned_detections = [[] for _ in range(chunk_frames.shape[0])]

        character_segments = []
        eligible_chars = chunk_meta["visible_characters"]
        if len(eligible_chars) < 2:
            continue

        for char_id in eligible_chars:
            masks = face_tracker.generate_masks(
                chunk_frames, aligned_detections, char_id, video_file=args.input
            )
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

            # Voice fingerprinting: extract speaker embedding from clean separations
            voice_emb = voice_tracker.extract_from_separation(
                target_cpu,
                residual_cpu,
                sample_rate=48000,
                min_target_to_residual_db=args.voice_quality_threshold,
            )
            if voice_emb is not None and char_id in profiles:
                voice_tracker.update_profile(profiles[char_id], voice_emb)
                char_seg_entry["has_voice_sample"] = True

            character_segments.append(char_seg_entry)

            del batch, result
            torch.cuda.empty_cache()

        del chunk_frames
        torch.cuda.empty_cache()

        chunk_meta["character_separation"] = character_segments

        # Voice-based discovery: check for unattributed dialogue
        # If chunk has dialogue but some segments don't match any face-detected character,
        # try matching via voice
        if args.enable_voice_discovery:
            _voice_discovery_pass(
                chunk_meta, chunk_audio, voice_tracker, profiles, 48000
            )

        progress["pass4_completed"].append(idx)
        progress["chunk_results"] = chunk_results
        save_progress(progress_path, progress)

    t_stage4 = time.time() - t_stage4_start
    logger.info(f"Stage 4 complete in {t_stage4:.1f}s")

    # ================================================================
    # STAGE 5: Reconciliation + Timeline Assembly
    # ================================================================
    logger.info("=== Stage 5: Reconciliation + Timeline Assembly ===")
    t_stage5_start = time.time()

    timeline = build_timeline(chunk_results, args.min_gap_seconds)

    # Compute per-character stats
    char_stats = []
    for cid, profile in profiles.items():
        total_speaking = 0.0
        for chunk in chunk_results:
            if "character_separation" not in chunk:
                continue
            for cs in chunk["character_separation"]:
                if cs["character_id"] == cid:
                    for seg in cs["segments"]:
                        if seg["has_dialogue"]:
                            total_speaking += seg["end_time"] - seg["start_time"]

        char_stats.append(
            {
                "id": cid,
                "name": None,
                "face_detections": profile.face_detections_count,
                "voice_samples": len(profile.voice_embeddings),
                "screen_time_sec": round(
                    profile.face_detections_count * args.sample_interval, 1
                ),
                "speaking_sec": round(total_speaking, 1),
                "identity_sources": profile.identity_sources,
            }
        )

    # Build per-character speaking segments
    per_character = {}
    for cid in profiles:
        segments = []
        for chunk in chunk_results:
            if "character_separation" not in chunk:
                continue
            for cs in chunk["character_separation"]:
                if cs["character_id"] == cid:
                    for seg in cs["segments"]:
                        if seg["has_dialogue"]:
                            segments.append([seg["start_time"], seg["end_time"]])
        # Merge adjacent
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

    t_stage5 = time.time() - t_stage5_start
    t_total = time.time() - t_start

    # Build scenes summary
    scenes_summary = [
        {
            "scene_id": sc.scene_id,
            "start": round(sc.start_sec, 1),
            "end": round(sc.end_sec, 1),
            "num_shots": len(sc.shots),
            "characters": sorted(sc.confirmed_characters),
            "has_dialogue": sc.has_dialogue,
        }
        for sc in scenes
    ]

    metadata = {
        "version": "2.0",
        "source": {
            "file": args.input,
            "resolution": "1920x1080",
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
            "num_multi_speaker_chunks": len(multi_speaker_chunks),
            "sample_interval": args.sample_interval,
            "face_det_threshold": args.face_det_threshold,
            "stage0_time": round(t_stage0, 1),
            "stage1_time": round(t_stage1, 1),
            "stage2_time": round(t_stage2, 1),
            "stage3_time": round(t_stage3, 1),
            "stage4_time": round(t_stage4, 1),
            "stage5_time": round(t_stage5, 1),
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
        help="Face detection confidence threshold (default: 0.3, lowered from v1's 0.5)",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.6,
        help="Face clustering distance threshold",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
        help="Face sampling interval in seconds (default: 0.5, 4x denser than v1)",
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
        default=-40,
        help="RMS threshold for dialogue detection",
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

    # Visual separation
    parser.add_argument(
        "--no-sam3",
        action="store_true",
        help="Use bbox masks instead of SAM3",
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
