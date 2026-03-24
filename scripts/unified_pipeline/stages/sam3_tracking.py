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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from unified_pipeline.db import AnalysisDB

log = logging.getLogger(__name__)

STAGE = "stage2"

MAX_FRAMES_PER_CHUNK = 200
HEAD_TOP_FRACTION = 0.10
SAM3_IMAGE_SIZE = 1008
SAM3_MEAN = 0.5
SAM3_STD = 0.5


def _get_video_dimensions(video_path: str) -> tuple[int, int]:
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


def _decode_frames_to_gpu(
    video_path,
    start_sec,
    end_sec,
    fps,
    vid_w,
    vid_h,
    max_frames=0,
):
    """Decode frames via ffmpeg pipe → GPU tensors, pre-processed for SAM3."""
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


def _bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _head_cx_from_mask(mask):
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


def _open_label_map_encoder(output_path, fps, vid_w, vid_h):
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

    if masks_dir is None:
        masks_dir = str(
            db.db_file.parent / db.db_file.stem.replace(".analysis", ".masks")
        )
    Path(masks_dir).mkdir(parents=True, exist_ok=True)

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
        torch.cuda.empty_cache()


def _process_shots(
    db,
    sam3,
    video_path,
    shots,
    store_masks,
    tracking_fps,
    vid_w,
    vid_h,
    masks_dir,
):
    """Process shots with prefetch decode.

    Decodes next chunk in a background thread while SAM3 runs on current chunk.
    """
    total_tracks = 0
    dt = 1.0 / tracking_fps

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="decode") as pool:
        for shot_idx, shot in enumerate(shots):
            shot_id = shot["shot_id"]
            shot_duration = shot["end_sec"] - shot["start_sec"]
            total_frames = int(shot_duration * tracking_fps) + 1

            if total_frames == 0:
                db.mark_progress(STAGE, str(shot_id), "done")
                continue

            shot_tracks = []
            total_label_frames = 0

            mask_path = os.path.join(masks_dir, f"shot_{shot_id:04d}.mkv")
            encoder = None
            if store_masks:
                encoder = _open_label_map_encoder(mask_path, tracking_fps, vid_w, vid_h)

            chunk_idx = 0
            frames_processed = 0

            # Submit first decode
            cs = min(MAX_FRAMES_PER_CHUNK, total_frames)
            future = pool.submit(
                _decode_frames_to_gpu,
                video_path,
                shot["start_sec"],
                min(shot["start_sec"] + cs * dt, shot["end_sec"]),
                tracking_fps,
                vid_w,
                vid_h,
                cs,
            )

            try:
                while frames_processed < total_frames:
                    images_tensor, timestamps = future.result()

                    n_decoded = len(timestamps) if timestamps else 0
                    next_fp = frames_processed + n_decoded

                    # Prefetch next chunk
                    next_future = None
                    if next_fp < total_frames:
                        next_cs = min(MAX_FRAMES_PER_CHUNK, total_frames - next_fp)
                        next_start = shot["start_sec"] + next_fp * dt
                        next_end = min(next_start + next_cs * dt, shot["end_sec"])
                        next_future = pool.submit(
                            _decode_frames_to_gpu,
                            video_path,
                            next_start,
                            next_end,
                            tracking_fps,
                            vid_w,
                            vid_h,
                            next_cs,
                        )

                    if images_tensor is None:
                        break

                    tracks, label_maps = _track_persons_in_shot(
                        sam3,
                        images_tensor,
                        timestamps,
                        shot_id,
                        vid_w,
                        vid_h,
                        chunk_idx * 1000,
                    )
                    shot_tracks.extend(tracks)

                    if encoder is not None:
                        for lm in label_maps:
                            encoder.stdin.write(lm.tobytes())
                            total_label_frames += 1
                        del label_maps

                    frames_processed = next_fp
                    chunk_idx += 1
                    del images_tensor

                    if next_future is None:
                        break
                    future = next_future
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
    """Initialize SAM3 inference state from a pre-processed GPU tensor."""
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
    images_tensor,
    timestamps,
    shot_id,
    vid_w,
    vid_h,
    obj_id_offset=0,
):
    """Run SAM3 text='person' on pre-processed GPU frames."""
    if images_tensor is None or len(images_tensor) == 0:
        return [], []

    import time
    import uuid

    inference_state = _init_sam3_state_from_tensor(
        sam3.model, images_tensor, vid_h, vid_w
    )
    session_id = str(uuid.uuid4())
    sam3._ALL_INFERENCE_STATES[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": time.time(),
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

                    label_val = min(effective_id + 1, 255)
                    label_map[mask > 0] = label_val

            label_maps.append(label_map)

        return track_rows, label_maps

    finally:
        sam3.handle_request({"type": "close_session", "session_id": session_id})
