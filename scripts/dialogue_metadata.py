"""Movie dialogue extraction pipeline with character tracking.

Two-pass selective approach:
  Pass 0: Preprocessing — extract audio, detect/cluster faces, generate masks
  Pass 1: Text-only dialogue detection (fast, all chunks)
  Pass 2: Visual per-character separation (selective, multi-speaker chunks only)
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
import torchaudio
from face_tracker import FaceTracker
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.model.judge import SAMAudioJudgeModel
from sam_audio.processor import SAMAudioJudgeProcessor

logger = logging.getLogger(__name__)


# --- Audio helpers ---


def extract_audio(video_path: str, sample_rate: int = 48000) -> torch.Tensor:
    """Extract mono audio from video file at target sample rate."""
    wav, sr = torchaudio.load(video_path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.mean(0, keepdim=True)  # mono [1, samples]


def generate_chunks(
    total_samples: int,
    sample_rate: int,
    window_seconds: float,
    overlap_seconds: float,
) -> list[tuple[int, int, int]]:
    """Generate (start_sample, end_sample, chunk_index) tuples."""
    window = int(window_seconds * sample_rate)
    overlap = int(overlap_seconds * sample_rate)
    stride = window - overlap
    chunks = []
    idx = 0
    start = 0
    while start < total_samples:
        end = min(start + window, total_samples)
        chunks.append((start, end, idx))
        start += stride
        idx += 1
    return chunks


def rms_db(audio: torch.Tensor) -> float:
    """Compute RMS in dB for a 1D audio tensor."""
    rms = audio.float().pow(2).mean().sqrt()
    if rms < 1e-10:
        return -100.0
    return 20 * math.log10(rms.item())


def analyze_chunk(
    target: torch.Tensor,
    residual: torch.Tensor,
    start_sample: int,
    end_sample: int,
    chunk_index: int,
    sample_rate: int,
    rms_threshold_db: float,
) -> dict:
    """Analyze a separated chunk into 1-second segments."""
    seg_samples = sample_rate  # 1 second
    total_samples = target.shape[-1]
    start_time = start_sample / sample_rate
    end_time = end_sample / sample_rate

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

        seg_start_time = start_time + seg_start / sample_rate
        seg_end_time = start_time + seg_end / sample_rate
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
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "has_any_dialogue": has_any_dialogue,
        "segments": segments,
    }


def crossfade_stitch(
    chunks: list[torch.Tensor],
    overlap_samples: int,
    sample_rate: int,
) -> torch.Tensor:
    """Stitch audio chunks with equal-power crossfade in overlap regions."""
    if len(chunks) == 0:
        return torch.zeros(1, 0)
    if len(chunks) == 1:
        return chunks[0]

    result = chunks[0]
    for i in range(1, len(chunks)):
        if overlap_samples > 0 and result.shape[-1] >= overlap_samples:
            fade_out = torch.linspace(1, 0, overlap_samples).pow(0.5)
            fade_in = torch.linspace(0, 1, overlap_samples).pow(0.5)
            overlap_region = (
                result[..., -overlap_samples:] * fade_out
                + chunks[i][..., :overlap_samples] * fade_in
            )
            result = torch.cat(
                [
                    result[..., :-overlap_samples],
                    overlap_region,
                    chunks[i][..., overlap_samples:],
                ],
                dim=-1,
            )
        else:
            result = torch.cat([result, chunks[i]], dim=-1)
    return result


# --- Judge scoring ---


def score_with_judge(
    judge_model: SAMAudioJudgeModel,
    judge_processor: SAMAudioJudgeProcessor,
    input_audio: torch.Tensor,
    target_audio: torch.Tensor,
    description: str,
    device: torch.device,
) -> dict:
    """Score separation quality with Judge model."""
    with torch.inference_mode():
        processed = judge_processor(
            text=[description],
            input_audio=[input_audio.cpu()],
            separated_audio=[target_audio.cpu()],
            sampling_rate=48000,
        ).to(device)
        result = judge_model(**processed)
        return {
            "overall": round(result.overall.squeeze(-1).cpu().item(), 2),
            "precision": round(result.precision.squeeze(-1).cpu().item(), 2),
            "recall": round(result.recall.squeeze(-1).cpu().item(), 2),
            "faithfulness": round(result.faithfulness.squeeze(-1).cpu().item(), 2),
        }


# --- Video frame extraction ---


def extract_chunk_frames(
    video_decoder: VideoDecoder,
    start_sec: float,
    end_sec: float,
    fps: float,
) -> torch.Tensor:
    """Extract video frames for a time range."""
    start_frame = max(0, int(start_sec * fps))
    end_frame = min(len(video_decoder), int(end_sec * fps))
    if start_frame >= end_frame:
        end_frame = start_frame + 1
    return video_decoder.get_frames_in_range(start_frame, end_frame - start_frame).data


# --- Resume support ---


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {"pass1_completed": [], "pass2_completed": [], "results": []}


def save_progress(progress_path: Path, progress: dict):
    with open(progress_path, "w") as f:
        json.dump(progress, f)


# --- Timeline merging ---


def build_timeline(chunk_results: list[dict], min_gap_seconds: float) -> dict:
    """Merge chunk segments into a continuous timeline with dialogue segments and gaps."""
    # Collect all segments across chunks, dedup overlapping regions by keeping latest chunk
    all_segments = {}
    for chunk in chunk_results:
        for seg in chunk["segments"]:
            key = seg["start_time"]
            all_segments[key] = seg

    sorted_segs = sorted(all_segments.values(), key=lambda s: s["start_time"])

    # For segments with character separation, determine speakers
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
        # Trailing gap
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
    progress_path = Path(args.output).with_suffix(".progress.json")
    progress = (
        load_progress(progress_path)
        if args.resume
        else {"pass1_completed": [], "pass2_completed": [], "results": []}
    )

    t_start = time.time()

    # === PASS 0: Preprocessing ===
    logger.info("Pass 0: Loading models and preprocessing")

    model = SAMAudio.from_pretrained(args.checkpoint).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(args.checkpoint)

    judge_model = None
    judge_processor = None
    if not args.no_judge:
        judge_model = SAMAudioJudgeModel.from_pretrained("facebook/sam-audio-judge").to(
            device
        )
        judge_processor = SAMAudioJudgeProcessor.from_pretrained(
            "facebook/sam-audio-judge"
        )

    sam3_predictor = None
    if not args.no_sam3:
        from sam3.model_builder import build_sam3_video_predictor

        sam3_predictor = build_sam3_video_predictor()

    face_tracker = FaceTracker(
        sam3_predictor=sam3_predictor,
        det_threshold=args.face_det_threshold,
        cluster_threshold=args.cluster_threshold,
    )

    # Extract audio
    logger.info("Extracting audio from video")
    full_audio = extract_audio(args.input, sample_rate=48000)
    total_duration = full_audio.shape[-1] / 48000

    # Open video decoder
    video_decoder = VideoDecoder(args.input, dimension_order="NCHW")
    fps = video_decoder.metadata.average_fps_from_header
    total_video_frames = len(video_decoder)

    # Face detection on sampled frames (every 0.5s)
    sample_interval = 0.5
    sample_frame_indices = list(
        range(0, total_video_frames, max(1, int(fps * sample_interval)))
    )
    logger.info(f"Detecting faces on {len(sample_frame_indices)} sampled frames")

    all_detections = []
    batch_size = 32
    for batch_start in tqdm(
        range(0, len(sample_frame_indices), batch_size), desc="Pass 0: face detection"
    ):
        batch_indices = sample_frame_indices[batch_start : batch_start + batch_size]
        frames_batch = video_decoder.get_frames_in_range(
            batch_indices[0], len(batch_indices)
        ).data
        dets = face_tracker.detect_faces(frames_batch, frame_indices=batch_indices)
        all_detections.extend(dets)

    # Cluster into characters
    logger.info("Clustering faces into characters")
    characters = face_tracker.cluster_characters(all_detections)
    # Limit to max_characters most prominent
    characters = dict(list(characters.items())[: args.max_characters])
    logger.info(f"Found {len(characters)} characters")

    t_pass0 = time.time() - t_start

    # === PASS 1: Text-only dialogue detection ===
    logger.info("Pass 1: Text-only dialogue detection")
    t_pass1_start = time.time()

    chunks = generate_chunks(
        full_audio.shape[-1], 48000, args.window_seconds, args.overlap_seconds
    )
    chunk_results = []

    # Restore already-completed results from progress
    completed_indices = set(progress["pass1_completed"])
    existing_results = {r["chunk_index"]: r for r in progress.get("results", [])}

    for start, end, idx in tqdm(chunks, desc="Pass 1: dialogue detection"):
        if idx in completed_indices:
            chunk_results.append(existing_results[idx])
            continue

        chunk_audio = full_audio[:, start:end]
        batch = processor(descriptions=["speech"], audios=[chunk_audio]).to(device)
        result = model.separate(batch, max_chunk_tokens=args.max_chunk_tokens)

        chunk_meta = analyze_chunk(
            result.target[0].cpu(),
            result.residual[0].cpu(),
            start,
            end,
            idx,
            48000,
            args.rms_threshold_db,
        )

        # Judge scoring
        if judge_model is not None:
            chunk_meta["quality"] = score_with_judge(
                judge_model,
                judge_processor,
                chunk_audio.squeeze(0),
                result.target[0].cpu(),
                "speech",
                device,
            )

        # Determine multi-speaker status
        chunk_start_sec = start / 48000
        chunk_end_sec = end / 48000
        visible_chars = face_tracker.characters_in_range(
            all_detections,
            chunk_start_sec,
            chunk_end_sec,
            1.0 / sample_interval,
            sample_interval,
        )
        # Filter to tracked characters only
        visible_char_ids = [
            c.character_id for c in visible_chars if c.character_id in characters
        ]
        chunk_meta["visible_characters"] = visible_char_ids
        chunk_meta["needs_visual_pass"] = (
            len(visible_char_ids) >= 2 and chunk_meta["has_any_dialogue"]
        )

        chunk_results.append(chunk_meta)
        progress["pass1_completed"].append(idx)
        progress["results"] = chunk_results
        save_progress(progress_path, progress)

        del batch, result
        torch.cuda.empty_cache()

    t_pass1 = time.time() - t_pass1_start

    # === PASS 2: Visual per-character separation (selective) ===
    logger.info("Pass 2: Visual per-character separation")
    t_pass2_start = time.time()

    multi_speaker_chunks = [c for c in chunk_results if c["needs_visual_pass"]]
    logger.info(f"{len(multi_speaker_chunks)} / {len(chunks)} chunks need visual pass")

    completed_pass2 = set(progress["pass2_completed"])

    target_chunks_per_char: dict[int, list[tuple[int, torch.Tensor]]] = {
        cid: [] for cid in characters
    }

    for chunk_meta in tqdm(multi_speaker_chunks, desc="Pass 2: character separation"):
        idx = chunk_meta["chunk_index"]
        if idx in completed_pass2:
            continue

        start = int(chunk_meta["start_time"] * 48000)
        end = int(chunk_meta["end_time"] * 48000)
        chunk_audio = full_audio[:, start:end]

        # Extract video frames for this chunk
        chunk_frames = extract_chunk_frames(
            video_decoder, chunk_meta["start_time"], chunk_meta["end_time"], fps
        )

        # Get detections aligned to chunk frames
        chunk_start_frame = int(chunk_meta["start_time"] / sample_interval)
        chunk_end_frame = int(chunk_meta["end_time"] / sample_interval)
        chunk_detections = face_tracker.get_detections_for_range(
            all_detections, chunk_start_frame, chunk_end_frame
        )

        # Subsample detections to match chunk_frames length
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
        for char_id in chunk_meta["visible_characters"]:
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

            char_analysis = analyze_chunk(
                result.target[0].cpu(),
                result.residual[0].cpu(),
                start,
                end,
                idx,
                48000,
                args.rms_threshold_db,
            )
            character_segments.append(
                {
                    "character_id": char_id,
                    "segments": char_analysis["segments"],
                }
            )

            if args.save_audio:
                target_chunks_per_char[char_id].append((idx, result.target[0].cpu()))

            del batch, result
            torch.cuda.empty_cache()

        del chunk_frames
        torch.cuda.empty_cache()

        chunk_meta["character_separation"] = character_segments
        progress["pass2_completed"].append(idx)
        progress["results"] = chunk_results
        save_progress(progress_path, progress)

    t_pass2 = time.time() - t_pass2_start

    # === Save separated audio per character ===
    if args.save_audio:
        audio_dir = Path(args.audio_output_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)

        overlap_samples = int(args.overlap_seconds * 48000)
        for char_id, char_chunks in target_chunks_per_char.items():
            if not char_chunks:
                continue
            sorted_chunks = sorted(char_chunks, key=lambda x: x[0])
            audio_tensors = [c[1] for c in sorted_chunks]
            stitched = crossfade_stitch(audio_tensors, overlap_samples, 48000)
            out_path = audio_dir / f"character_{char_id}_dialogue.wav"
            torchaudio.save(
                str(out_path),
                stitched.unsqueeze(0) if stitched.ndim == 1 else stitched,
                48000,
            )
            logger.info(f"Saved {out_path}")

    # === Build final metadata ===
    t_total = time.time() - t_start

    # Compute per-character stats
    char_stats = []
    for cid, info in characters.items():
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
                "character_id": cid,
                "total_screen_time_seconds": round(
                    info.total_frames * sample_interval, 1
                ),
                "total_speaking_seconds": round(total_speaking, 1),
            }
        )

    timeline = build_timeline(chunk_results, args.min_gap_seconds)

    # Statistics
    total_dialogue = sum(
        seg["end_time"] - seg["start_time"]
        for chunk in chunk_results
        for seg in chunk["segments"]
        if seg["has_dialogue"]
    )
    quality_scores = [
        chunk["quality"]["overall"] for chunk in chunk_results if "quality" in chunk
    ]

    metadata = {
        "version": "1.0",
        "source": {
            "file": args.input,
            "duration_seconds": round(total_duration, 1),
            "sample_rate": 48000,
        },
        "characters": char_stats,
        "processing": {
            "model": args.checkpoint,
            "window_seconds": args.window_seconds,
            "overlap_seconds": args.overlap_seconds,
            "num_chunks": len(chunks),
            "num_multi_speaker_chunks": len(multi_speaker_chunks),
            "pass0_time_seconds": round(t_pass0, 1),
            "pass1_time_seconds": round(t_pass1, 1),
            "pass2_time_seconds": round(t_pass2, 1),
            "total_time_seconds": round(t_total, 1),
        },
        "chunks": chunk_results,
        "timeline": timeline,
        "statistics": {
            "dialogue_percentage": round(total_dialogue / total_duration * 100, 1)
            if total_duration > 0
            else 0,
            "total_dialogue_seconds": round(total_dialogue, 1),
            "num_gaps": len(timeline["gaps"]),
            "avg_quality_overall": round(sum(quality_scores) / len(quality_scores), 2)
            if quality_scores
            else None,
        },
    }

    with open(args.output, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata to {args.output}")

    # Clean up progress file on success
    if progress_path.exists():
        progress_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Movie dialogue extraction with character tracking"
    )
    parser.add_argument("--input", required=True, help="Input video file (mkv/mp4)")
    parser.add_argument(
        "--output", default="dialogue_metadata.json", help="Output JSON path"
    )
    parser.add_argument(
        "--checkpoint",
        default="facebook/sam-audio-base-tv",
        help="SAM-Audio model checkpoint (default: facebook/sam-audio-base-tv)",
    )
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    parser.add_argument(
        "--window-seconds", type=float, default=180, help="Chunk window size in seconds"
    )
    parser.add_argument(
        "--overlap-seconds", type=float, default=10, help="Chunk overlap in seconds"
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
        "--max-characters", type=int, default=8, help="Max characters to track"
    )
    parser.add_argument("--no-judge", action="store_true", help="Skip Judge scoring")
    parser.add_argument(
        "--no-sam3", action="store_true", help="Use bbox masks instead of SAM3"
    )
    parser.add_argument(
        "--face-det-threshold",
        type=float,
        default=0.5,
        help="Face detection confidence threshold",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.6,
        help="Face clustering distance threshold",
    )
    parser.add_argument(
        "--save-audio", action="store_true", help="Save separated audio tracks"
    )
    parser.add_argument(
        "--audio-output-dir",
        default="./separated/",
        help="Directory for separated audio",
    )
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
