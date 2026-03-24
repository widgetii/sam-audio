"""Stage 2: SAM3 person tracking across all shots.

SAM3 is a video segmentation model with temporal state — it provides
consistent obj_ids across frames without needing ByteTrack.

Each shot is a separate SAM3 session. Text prompt "person" detects all
people on the first frame, then propagates through the shot.

Frames are decoded via ffmpeg pipe → GPU tensors (no PIL, no disk I/O).
Resize + normalize happen on GPU. Masks are stored as FFV1 label-map
videos (one per shot). head_cx is precomputed from the mask silhouette.
"""

import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from unified_pipeline.db import AnalysisDB

log = logging.getLogger(__name__)

STAGE = "stage2"

# Max frames per SAM3 session to avoid OOM
MAX_FRAMES_PER_CHUNK = 200

# Head region: top fraction of mask silhouette for head_cx computation
HEAD_TOP_FRACTION = 0.10


def _get_video_dimensions(video_path: str) -> tuple[int, int]:
    """Get video width and height via ffprobe."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    w, h = [int(x) for x in probe.stdout.strip().split(",")]
    return w, h


SAM3_IMAGE_SIZE = 1008
SAM3_MEAN = 0.5
SAM3_STD = 0.5


def _decode_frames_to_gpu(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float,
    vid_w: int,
    vid_h: int,
    max_frames: int = 0,
) -> tuple[torch.Tensor | None, list[float]]:
    """Decode frames via ffmpeg pipe → GPU tensors, pre-processed for SAM3.

    Returns (tensor [N,3,1008,1008] float16 on CUDA, list of timestamps).
    Resize and normalize happen on GPU — no PIL involved.
    """
    duration = end_sec - start_sec
    if duration < 0.04:
        return None, []

    cmd = [
        "ffmpeg",
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
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-vsync",
        "cfr",
        "pipe:1",
    ]

    frame_bytes = vid_w * vid_h * 3
    dt = 1.0 / fps

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    tensors = []
    timestamps = []
    try:
        i = 0
        while True:
            if max_frames and i >= max_frames:
                break
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            # Raw bytes → GPU tensor, resize + normalize on GPU
            t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
                vid_h, vid_w, 3
            )
            t = t.permute(2, 0, 1).unsqueeze(0).cuda().half() / 255.0
            t = F.interpolate(
                t,
                size=(SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE),
                mode="bicubic",
                align_corners=False,
            )
            t = (t - SAM3_MEAN) / SAM3_STD
            tensors.append(t.squeeze(0))
            timestamps.append(start_sec + i * dt)
            i += 1
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()

    if not tensors:
        return None, []

    return torch.stack(tensors), timestamps


def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Extract bounding box (x1, y1, x2, y2) from a binary mask."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _head_cx_from_mask(mask: np.ndarray) -> float | None:
    """Compute head horizontal center from the top portion of a mask silhouette."""
    rows_with_mask = np.where(mask.any(axis=1))[0]
    if len(rows_with_mask) < 10:
        return None
    top_row = rows_with_mask[0]
    bot_row = rows_with_mask[-1]
    mask_height = bot_row - top_row
    if mask_height < 20:
        return None
    head_bottom = top_row + max(1, int(mask_height * HEAD_TOP_FRACTION))
    head_region = mask[top_row:head_bottom, :]
    head_cols = np.where(head_region.any(axis=0))[0]
    if len(head_cols) == 0:
        return None
    return float((head_cols[0] + head_cols[-1]) / 2)


def _open_label_map_encoder(
    output_path: str,
    fps: float,
    vid_w: int,
    vid_h: int,
) -> subprocess.Popen:
    """Open a streaming FFV1 encoder. Write grayscale label maps to stdin."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{vid_w}x{vid_h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-pix_fmt",
        "gray",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def run_stage2(
    db: AnalysisDB,
    video_path: str,
    audio_stream: int = 0,
    device: str = "cuda",
    store_masks: bool = True,
    tracking_fps: float = 1.0,
    masks_dir: str | None = None,
):
    """Run SAM3 person tracking for every shot.

    Args:
        db: Analysis database.
        video_path: Path to source video.
        audio_stream: Unused, for interface consistency.
        device: CUDA device.
        store_masks: Whether to store masks (as FFV1 video files).
        tracking_fps: Frames per second to track at.
        masks_dir: Directory for mask video files. Auto-derived if None.
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
        f"at {tracking_fps:.1f}fps"
    )

    vid_w, vid_h = _get_video_dimensions(video_path)

    # Create masks directory
    if masks_dir is None:
        masks_dir = str(
            db.db_file.parent / db.db_file.stem.replace(".analysis", ".masks")
        )
    Path(masks_dir).mkdir(parents=True, exist_ok=True)

    # Load SAM3
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from sam3.model_builder import build_sam3_video_predictor

    sam3 = build_sam3_video_predictor()
    sam3.model.to(device)

    try:
        _process_shots(
            db,
            sam3,
            video_path,
            remaining,
            store_masks,
            tracking_fps,
            vid_w,
            vid_h,
            masks_dir,
        )
    finally:
        sam3.model.to("cpu")
        import torch

        torch.cuda.empty_cache()


def _process_shots(
    db: AnalysisDB,
    sam3,
    video_path: str,
    shots: list[dict],
    store_masks: bool,
    tracking_fps: float,
    vid_w: int,
    vid_h: int,
    masks_dir: str,
):
    total_tracks = 0

    for shot_idx, shot in enumerate(shots):
        shot_id = shot["shot_id"]
        shot_duration = shot["end_sec"] - shot["start_sec"]
        total_frames = int(shot_duration * tracking_fps) + 1

        if total_frames == 0:
            db.mark_progress(STAGE, str(shot_id), "done")
            continue

        shot_tracks = []
        total_label_frames = 0

        # Open streaming FFV1 encoder for this shot's mask video
        mask_path = os.path.join(masks_dir, f"shot_{shot_id:04d}.mkv")
        encoder = None
        if store_masks:
            encoder = _open_label_map_encoder(mask_path, tracking_fps, vid_w, vid_h)

        # Process in chunks — decode to GPU tensors, run SAM3, write FFV1
        dt = 1.0 / tracking_fps
        chunk_idx = 0
        frames_processed = 0

        try:
            while frames_processed < total_frames:
                chunk_size = min(MAX_FRAMES_PER_CHUNK, total_frames - frames_processed)
                chunk_start_sec = shot["start_sec"] + frames_processed * dt
                chunk_end_sec = min(chunk_start_sec + chunk_size * dt, shot["end_sec"])

                images_tensor, timestamps = _decode_frames_to_gpu(
                    video_path,
                    chunk_start_sec,
                    chunk_end_sec,
                    tracking_fps,
                    vid_w,
                    vid_h,
                    max_frames=chunk_size,
                )

                if images_tensor is None:
                    break

                obj_id_offset = chunk_idx * 1000

                tracks, label_maps = _track_persons_in_shot(
                    sam3,
                    images_tensor,
                    timestamps,
                    shot_id,
                    vid_w,
                    vid_h,
                    obj_id_offset,
                )
                shot_tracks.extend(tracks)

                # Stream label maps to encoder
                if encoder is not None:
                    for lm in label_maps:
                        encoder.stdin.write(lm.tobytes())
                        total_label_frames += 1
                    del label_maps

                frames_processed += len(timestamps)
                chunk_idx += 1
                del images_tensor
        finally:
            if encoder is not None:
                encoder.stdin.close()
                encoder.wait()

        if shot_tracks:
            db.insert_person_tracks(shot_tracks)
            total_tracks += len(shot_tracks)

        if store_masks and total_label_frames > 0:
            db.insert_mask_video(
                {
                    "shot_id": shot_id,
                    "video_path": os.path.basename(mask_path),
                    "fps": tracking_fps,
                    "width": vid_w,
                    "height": vid_h,
                    "frame_count": total_label_frames,
                }
            )

        db.mark_progress(STAGE, str(shot_id), "done")

        if (shot_idx + 1) % 50 == 0 or shot_idx == len(shots) - 1:
            log.info(
                f"Stage 2: {shot_idx + 1}/{len(shots)} shots done, "
                f"{total_tracks} track rows total"
            )


def _init_sam3_state_from_tensor(sam3_model, images_tensor, orig_h, orig_w):
    """Initialize SAM3 inference state from a pre-processed GPU tensor.

    Bypasses load_resource_as_video_frames — tensor is already on GPU,
    resized to 1008x1008, and normalized.
    """
    inference_state = {}
    inference_state["image_size"] = sam3_model.image_size
    inference_state["num_frames"] = len(images_tensor)
    inference_state["orig_height"] = orig_h
    inference_state["orig_width"] = orig_w
    inference_state["constants"] = {}
    sam3_model._construct_initial_input_batch(inference_state, images_tensor)
    inference_state["tracker_inference_states"] = []
    inference_state["tracker_metadata"] = {}
    inference_state["feature_cache"] = {}
    inference_state["cached_frame_outputs"] = {}
    inference_state["action_history"] = []
    inference_state["is_image_only"] = False
    return inference_state


def _track_persons_in_shot(
    sam3,
    images_tensor: torch.Tensor,
    timestamps: list[float],
    shot_id: int,
    vid_w: int,
    vid_h: int,
    obj_id_offset: int = 0,
) -> tuple[list[dict], list[np.ndarray]]:
    """Run SAM3 text="person" on pre-processed GPU frames.

    images_tensor: [N, 3, 1008, 1008] float16 on CUDA.
    """
    if images_tensor is None or len(images_tensor) == 0:
        return [], []

    # Initialize SAM3 session directly from tensor (bypass PIL path)
    import uuid

    inference_state = _init_sam3_state_from_tensor(
        sam3.model, images_tensor, vid_h, vid_w
    )
    session_id = str(uuid.uuid4())
    sam3._ALL_INFERENCE_STATES[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": __import__("time").time(),
    }

    try:
        response = sam3.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": "person",
            }
        )

        outputs = response.get("outputs")
        if outputs is None or len(outputs.get("out_obj_ids", [])) == 0:
            # No detections — return empty label maps (all background)
            n_frames = len(images_tensor)
            empty_maps = [
                np.zeros((vid_h, vid_w), dtype=np.uint8) for _ in range(n_frames)
            ]
            return [], empty_maps

        per_frame: dict[int, dict[int, np.ndarray]] = {}

        for obj_id, mask in zip(
            outputs["out_obj_ids"], outputs["out_binary_masks"], strict=True
        ):
            per_frame.setdefault(0, {})[int(obj_id)] = mask

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

        # Build track rows + label maps
        track_rows = []
        label_maps = []

        for fi in range(len(images_tensor)):
            label_map = np.zeros((vid_h, vid_w), dtype=np.uint8)

            if fi in per_frame:
                t = timestamps[fi]
                for obj_id, mask in per_frame[fi].items():
                    bbox = _bbox_from_mask(mask)
                    if bbox[2] - bbox[0] < 1 and bbox[3] - bbox[1] < 1:
                        continue

                    head_cx = _head_cx_from_mask(mask)
                    effective_id = obj_id + obj_id_offset

                    track_rows.append(
                        {
                            "shot_id": shot_id,
                            "sam3_obj_id": effective_id,
                            "frame_sec": t,
                            "bbox_x1": bbox[0],
                            "bbox_y1": bbox[1],
                            "bbox_x2": bbox[2],
                            "bbox_y2": bbox[3],
                            "head_cx": head_cx,
                            "mask_rle": None,
                        }
                    )

                    # Label map: pixel = effective_id + 1 (0 = background)
                    # Clamp to uint8 range
                    label_val = min(effective_id + 1, 255)
                    label_map[mask > 0] = label_val

            label_maps.append(label_map)

        return track_rows, label_maps

    finally:
        sam3.handle_request({"type": "close_session", "session_id": session_id})
