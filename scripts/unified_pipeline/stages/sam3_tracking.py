"""Stage 2: SAM3 person tracking across all shots.

Multiple independent SAM3 instances run in parallel threads, each with
its own model copy on GPU. Frames decoded via ffmpeg pipe → GPU tensors.
Masks stored as FFV1 label-map videos. head_cx precomputed from masks.
"""

import logging
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Queue

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


def _get_video_dimensions(video_path):
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
    video_path, start_sec, end_sec, fps, vid_w, vid_h, max_frames=0
):
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
    tensors, timestamps = [], []
    try:
        i = 0
        while not max_frames or i < max_frames:
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
    top_row, bot_row = rows_with_mask[0], rows_with_mask[-1]
    mask_height = bot_row - top_row
    if mask_height < 20:
        return None
    head_bottom = top_row + max(1, int(mask_height * HEAD_TOP_FRACTION))
    head_cols = np.where(mask[top_row:head_bottom, :].any(axis=0))[0]
    if len(head_cols) == 0:
        return None
    return float((head_cols[0] + head_cols[-1]) / 2)


def _open_label_map_encoder(output_path, fps, vid_w, vid_h):
    return subprocess.Popen(
        [
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
        ],
        stdin=subprocess.PIPE,
    )


def _process_one_shot(
    sam3, video_path, shot, store_masks, tracking_fps, vid_w, vid_h, masks_dir
):
    """Process a single shot end-to-end with one SAM3 instance."""
    shot_id = shot["shot_id"]
    shot_duration = shot["end_sec"] - shot["start_sec"]
    total_frames = int(shot_duration * tracking_fps) + 1
    dt = 1.0 / tracking_fps

    if total_frames == 0:
        return shot_id, [], None

    shot_tracks = []
    total_label_frames = 0
    mask_path = os.path.join(masks_dir, f"shot_{shot_id:04d}.mkv")
    encoder = (
        _open_label_map_encoder(mask_path, tracking_fps, vid_w, vid_h)
        if store_masks
        else None
    )

    chunk_idx = 0
    fp = 0
    try:
        while fp < total_frames:
            cs = min(MAX_FRAMES_PER_CHUNK, total_frames - fp)
            cs_sec = shot["start_sec"] + fp * dt
            ce_sec = min(cs_sec + cs * dt, shot["end_sec"])

            images_tensor, timestamps = _decode_frames_to_gpu(
                video_path,
                cs_sec,
                ce_sec,
                tracking_fps,
                vid_w,
                vid_h,
                cs,
            )
            if images_tensor is None:
                break

            # Init SAM3 session
            inference_state = {}
            inference_state["image_size"] = sam3.model.image_size
            inference_state["num_frames"] = len(images_tensor)
            inference_state["orig_height"] = vid_h
            inference_state["orig_width"] = vid_w
            inference_state["constants"] = {}
            with torch.inference_mode(), torch.amp.autocast("cuda"):
                sam3.model._construct_initial_input_batch(
                    inference_state, images_tensor
                )
            inference_state["tracker_inference_states"] = []
            inference_state["tracker_metadata"] = {}
            inference_state["feature_cache"] = {}
            inference_state["cached_frame_outputs"] = {}
            inference_state["action_history"] = []
            inference_state["is_image_only"] = False

            session_id = str(uuid.uuid4())
            sam3._ALL_INFERENCE_STATES[session_id] = {
                "state": inference_state,
                "session_id": session_id,
                "start_time": time.time(),
            }

            try:
                with torch.inference_mode(), torch.amp.autocast("cuda"):
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
                    # No detections — write empty label maps
                    if encoder is not None:
                        for _ in range(len(images_tensor)):
                            encoder.stdin.write(
                                np.zeros((vid_h, vid_w), dtype=np.uint8).tobytes()
                            )
                            total_label_frames += 1
                    fp += len(timestamps)
                    chunk_idx += 1
                    del images_tensor
                    continue

                per_frame: dict[int, dict[int, np.ndarray]] = {}
                for obj_id, mask in zip(
                    outputs["out_obj_ids"], outputs["out_binary_masks"], strict=True
                ):
                    per_frame.setdefault(0, {})[int(obj_id)] = mask

                with torch.inference_mode(), torch.amp.autocast("cuda"):
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
                            frame_out["out_obj_ids"],
                            frame_out["out_binary_masks"],
                            strict=True,
                        ):
                            per_frame.setdefault(fi, {})[int(obj_id)] = mask

                obj_id_offset = chunk_idx * 1000
                for fi in range(len(images_tensor)):
                    label_map = np.zeros((vid_h, vid_w), dtype=np.uint8)
                    if fi in per_frame:
                        t = timestamps[fi]
                        for obj_id, mask in per_frame[fi].items():
                            bbox = _bbox_from_mask(mask)
                            if bbox[2] - bbox[0] < 1 and bbox[3] - bbox[1] < 1:
                                continue
                            eid = obj_id + obj_id_offset
                            shot_tracks.append(
                                {
                                    "shot_id": shot_id,
                                    "sam3_obj_id": eid,
                                    "frame_sec": t,
                                    "bbox_x1": bbox[0],
                                    "bbox_y1": bbox[1],
                                    "bbox_x2": bbox[2],
                                    "bbox_y2": bbox[3],
                                    "head_cx": _head_cx_from_mask(mask),
                                    "mask_rle": None,
                                }
                            )
                            label_map[mask > 0] = min(eid + 1, 255)
                    if encoder is not None:
                        encoder.stdin.write(label_map.tobytes())
                        total_label_frames += 1

            finally:
                sam3.handle_request({"type": "close_session", "session_id": session_id})

            fp += len(timestamps)
            chunk_idx += 1
            del images_tensor
    finally:
        if encoder is not None:
            encoder.stdin.close()
            encoder.wait()

    mask_info = None
    if store_masks and total_label_frames > 0:
        mask_info = {
            "shot_id": shot_id,
            "video_path": os.path.basename(mask_path),
            "fps": tracking_fps,
            "width": vid_w,
            "height": vid_h,
            "frame_count": total_label_frames,
        }
    return shot_id, shot_tracks, mask_info


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

    vid_w, vid_h = _get_video_dimensions(video_path)
    if masks_dir is None:
        masks_dir = str(
            db.db_file.parent / db.db_file.stem.replace(".analysis", ".masks")
        )
    Path(masks_dir).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Determine worker count: each SAM3 instance ~10GB VRAM, keep 10GB headroom
    total_vram = torch.cuda.get_device_properties(0).total_memory
    n_workers = min(4, max(1, int((total_vram - 10 * 1024**3) / (10 * 1024**3))))

    log.info(
        f"Stage 2: tracking {len(remaining)}/{len(shots)} shots "
        f"at {tracking_fps:.1f}fps with {n_workers} independent SAM3 instances"
    )

    # Build N independent SAM3 predictors, each with own model weights
    from sam3.model_builder import build_sam3_video_predictor

    sam3_instances = []
    for i in range(n_workers):
        sam3 = build_sam3_video_predictor()
        sam3.model.to(device)
        # Give each instance its own session dict to avoid cross-thread interference
        sam3._ALL_INFERENCE_STATES = {}
        vram_gb = torch.cuda.memory_allocated(0) / 1e9
        log.info(f"SAM3 instance {i} loaded, total VRAM used: {vram_gb:.1f}GB")
        sam3_instances.append(sam3)

    # Thread-safe DB writes
    db_lock = threading.Lock()
    shot_queue: Queue = Queue()
    for s in remaining:
        shot_queue.put(s)

    total_tracks = 0
    shots_done = 0

    def _worker(worker_id, sam3_inst):
        nonlocal total_tracks, shots_done
        while True:
            try:
                shot = shot_queue.get_nowait()
            except Empty:
                return
            try:
                shot_id, tracks, mask_info = _process_one_shot(
                    sam3_inst,
                    video_path,
                    shot,
                    store_masks,
                    tracking_fps,
                    vid_w,
                    vid_h,
                    masks_dir,
                )
                with db_lock:
                    if tracks:
                        db.insert_person_tracks(tracks)
                        total_tracks += len(tracks)
                    if mask_info:
                        db.insert_mask_video(mask_info)
                    db.mark_progress(STAGE, str(shot_id), "done")
                    shots_done += 1
                    if shots_done % 50 == 0:
                        log.info(
                            f"Stage 2: {shots_done}/{len(remaining)} shots, "
                            f"{total_tracks} tracks"
                        )
            except Exception:
                log.exception(f"Worker {worker_id} error on shot {shot['shot_id']}")
                with db_lock:
                    db.mark_progress(STAGE, str(shot["shot_id"]), "error")

    try:
        threads = []
        for i, inst in enumerate(sam3_instances):
            t = threading.Thread(target=_worker, args=(i, inst), name=f"sam3-{i}")
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        log.info(
            f"Stage 2 complete: {shots_done}/{len(remaining)} shots, {total_tracks} tracks"
        )
    finally:
        for inst in sam3_instances:
            inst.model.to("cpu")
        torch.cuda.empty_cache()
