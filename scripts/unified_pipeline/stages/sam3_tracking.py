"""Stage 2: SAM3 person tracking at 1fps across all shots.

SAM3 is a video segmentation model with temporal state — it provides
consistent obj_ids across frames without needing ByteTrack.

Each shot is a separate SAM3 session. Text prompt "person" detects all
people on the first frame, then propagates through the shot at 1fps.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from unified_pipeline.db import AnalysisDB, mask_to_rle

log = logging.getLogger(__name__)

STAGE = "stage2"


def _decode_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_dir: str,
    fps: float = 1.0,
) -> list[tuple[float, str]]:
    """Decode frames at given fps from a video segment, return (timestamp, path) pairs."""
    duration = end_sec - start_sec
    if duration < 0.1:
        return []

    pattern = os.path.join(output_dir, "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(start_sec),
        "-t",
        str(duration),
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        "-qscale:v",
        "2",
        pattern,
    ]
    subprocess.run(cmd, check=True)

    dt = 1.0 / fps
    frames = []
    for i, path in enumerate(sorted(Path(output_dir).glob("frame_*.jpg"))):
        t = start_sec + i * dt
        frames.append((t, str(path)))

    return frames


def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Extract bounding box (x1, y1, x2, y2) from a binary mask."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def run_stage2(
    db: AnalysisDB,
    video_path: str,
    audio_stream: int = 0,
    device: str = "cuda",
    store_masks: bool = True,
    tracking_fps: float = 1.0,
):
    """Run SAM3 person tracking for every shot.

    Args:
        db: Analysis database.
        video_path: Path to source video.
        audio_stream: Audio stream index (unused here, passed for consistency).
        device: CUDA device.
        store_masks: Whether to store RLE masks (large but needed for SAM-Audio visual sep).
        tracking_fps: Frames per second to track at (default 1.0).
    """
    shots = db.get_shots()
    if not shots:
        raise RuntimeError("Stage 2: no shots in DB — run stages 0-1 first")

    progress = db.get_progress(STAGE)
    remaining = [s for s in shots if progress.get(str(s["shot_id"])) != "done"]
    if not remaining:
        log.info("Stage 2: all shots already tracked, skipping")
        return

    log.info(
        f"Stage 2: tracking persons in {len(remaining)}/{len(shots)} shots "
        f"at {tracking_fps}fps"
    )

    # Load SAM3
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from sam3.model_builder import build_sam3_video_predictor

    sam3 = build_sam3_video_predictor()
    sam3.model.to(device)

    try:
        _process_shots(db, sam3, video_path, remaining, store_masks, tracking_fps)
    finally:
        sam3.model.to("cpu")
        import torch

        torch.cuda.empty_cache()


# Max frames per SAM3 session to avoid OOM (at 1080p: ~200 frames ≈ 1.2GB)
MAX_FRAMES_PER_CHUNK = 200


def _process_shots(
    db: AnalysisDB,
    sam3,
    video_path: str,
    shots: list[dict],
    store_masks: bool,
    tracking_fps: float = 1.0,
):
    from PIL import Image

    total_tracks = 0

    for shot_idx, shot in enumerate(shots):
        shot_id = shot["shot_id"]

        with tempfile.TemporaryDirectory() as tmpdir:
            frames = _decode_frames(
                video_path, shot["start_sec"], shot["end_sec"], tmpdir, tracking_fps
            )

            if len(frames) == 0:
                db.mark_progress(STAGE, str(shot_id), "done")
                continue

            # Process in chunks to avoid OOM on long shots
            shot_tracks = []
            for chunk_start in range(0, len(frames), MAX_FRAMES_PER_CHUNK):
                chunk_frames = frames[chunk_start : chunk_start + MAX_FRAMES_PER_CHUNK]
                pil_frames = [
                    Image.open(path).convert("RGB") for _, path in chunk_frames
                ]
                timestamps = [t for t, _ in chunk_frames]

                # Offset obj_ids for subsequent chunks to avoid collisions
                obj_id_offset = chunk_start

                tracks = _track_persons_in_shot(
                    sam3, pil_frames, timestamps, shot_id, store_masks, obj_id_offset
                )
                shot_tracks.extend(tracks)

                # Free memory between chunks
                del pil_frames

        if shot_tracks:
            db.insert_person_tracks(shot_tracks)
            total_tracks += len(shot_tracks)

        db.mark_progress(STAGE, str(shot_id), "done")

        if (shot_idx + 1) % 50 == 0 or shot_idx == len(shots) - 1:
            log.info(
                f"Stage 2: {shot_idx + 1}/{len(shots)} shots done, "
                f"{total_tracks} track rows total"
            )


def _track_persons_in_shot(
    sam3,
    pil_frames: list,
    timestamps: list[float],
    shot_id: int,
    store_masks: bool,
    obj_id_offset: int = 0,
) -> list[dict]:
    """Run SAM3 text="person" on a shot's frames, return track rows."""
    if len(pil_frames) == 0:
        return []

    # Start SAM3 session with frames
    response = sam3.handle_request(
        {
            "type": "start_session",
            "resource_path": pil_frames,
        }
    )
    session_id = response["session_id"]

    try:
        # Text prompt "person" on first frame — detects all people
        response = sam3.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": "person",
            }
        )

        # Collect initial detections
        outputs = response.get("outputs")
        if outputs is None or len(outputs.get("out_obj_ids", [])) == 0:
            return []

        # Propagate through all frames
        per_frame: dict[int, dict[int, np.ndarray]] = {}

        # Store frame 0 detections
        for obj_id, mask in zip(
            outputs["out_obj_ids"], outputs["out_binary_masks"], strict=True
        ):
            per_frame.setdefault(0, {})[int(obj_id)] = mask

        # Propagate forward
        for result in sam3.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
            }
        ):
            frame_out = result.get("outputs")
            if frame_out is None:
                continue
            fi = result["frame_index"]
            for obj_id, mask in zip(
                frame_out["out_obj_ids"], frame_out["out_binary_masks"], strict=True
            ):
                per_frame.setdefault(fi, {})[int(obj_id)] = mask

        # Build track rows
        track_rows = []
        for fi in range(len(pil_frames)):
            if fi not in per_frame:
                continue
            t = timestamps[fi]
            for obj_id, mask in per_frame[fi].items():
                bbox = _bbox_from_mask(mask)
                if bbox[2] - bbox[0] < 1 and bbox[3] - bbox[1] < 1:
                    continue  # skip empty masks

                row = {
                    "shot_id": shot_id,
                    "sam3_obj_id": obj_id + obj_id_offset,
                    "frame_sec": t,
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "mask_rle": mask_to_rle(mask.astype(np.uint8))
                    if store_masks
                    else None,
                }
                track_rows.append(row)

        return track_rows

    finally:
        sam3.handle_request({"type": "close_session", "session_id": session_id})
