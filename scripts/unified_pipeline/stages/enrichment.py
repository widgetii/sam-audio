"""Stage 4: TalkNet ASD scoring on SAM3 person tracks.

For each person track within a shot:
  - Interpolate 1fps bboxes to 25fps
  - Extract face crops from video at 25fps
  - Extract shot audio as 16kHz mono WAV
  - Run TalkNet ASD: face video + audio → per-frame speaking scores
  - Average scores per 1fps keyframe and store in DB
"""

import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np
from scipy import signal as scipy_signal

from unified_pipeline.db import AnalysisDB

log = logging.getLogger(__name__)

STAGE = "stage4"

# Face region = upper portion of person bbox
FACE_REGION_RATIO = 0.45
CROP_SCALE = 0.40  # Padding around face center (matches TalkNet demo)
MIN_FACE_SIZE = 48
TALKNET_FPS = 25  # TalkNet expects 25fps face crops


def _extract_audio_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    audio_stream: int = 0,
):
    """Extract audio segment as 16kHz mono WAV."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(start_sec),
        "-t",
        str(end_sec - start_sec),
        "-i",
        video_path,
        "-map",
        f"0:{audio_stream}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def _interpolate_bbox_to_25fps(
    keyframes: list[dict], shot_start: float, shot_end: float
) -> list[tuple[float, tuple[float, float, float, float]]]:
    """Interpolate 1fps bbox keyframes to 25fps within shot bounds.

    Returns list of (timestamp, (x1, y1, x2, y2)).
    """
    if not keyframes:
        return []

    keyframes = sorted(keyframes, key=lambda k: k["frame_sec"])
    times = [k["frame_sec"] for k in keyframes]
    bboxes = [
        (k["bbox_x1"], k["bbox_y1"], k["bbox_x2"], k["bbox_y2"]) for k in keyframes
    ]

    result = []
    dt = 1.0 / TALKNET_FPS
    t = max(shot_start, times[0] - 0.5)
    t_end = min(shot_end, times[-1] + 0.5)

    while t <= t_end:
        # Find surrounding keyframes
        before_idx = 0
        for i, kt in enumerate(times):
            if kt <= t:
                before_idx = i
        after_idx = min(before_idx + 1, len(times) - 1)

        if times[before_idx] == times[after_idx] or before_idx == after_idx:
            bbox = bboxes[before_idx]
        else:
            frac = (t - times[before_idx]) / (times[after_idx] - times[before_idx])
            frac = max(0.0, min(1.0, frac))
            b, a = bboxes[before_idx], bboxes[after_idx]
            bbox = tuple(bv + (av - bv) * frac for bv, av in zip(b, a, strict=True))

        result.append((t, bbox))
        t += dt

    return result


def _extract_face_crops_25fps(
    video_path: str,
    track_bboxes: list[tuple[float, tuple[float, float, float, float]]],
) -> list[np.ndarray]:
    """Extract face crops at 25fps from video using interpolated bboxes.

    Uses ffmpeg to decode the relevant segment, then crops each frame.
    """
    if not track_bboxes:
        return []

    start_sec = track_bboxes[0][0]
    end_sec = track_bboxes[-1][0]
    duration = end_sec - start_sec + 1.0 / TALKNET_FPS

    # Decode video segment at 25fps
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
        "-r",
        str(TALKNET_FPS),
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-vsync",
        "cfr",
        "pipe:1",
    ]

    # Get video dimensions
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
    vid_w, vid_h = [int(x) for x in probe.stdout.strip().split(",")]
    frame_bytes = vid_w * vid_h * 3

    # Smooth the bbox centers and sizes (like TalkNet demo)
    centers_x = []
    centers_y = []
    sizes = []
    for _, bbox in track_bboxes:
        x1, y1, x2, y2 = bbox
        # Face region: upper portion
        body_h = y2 - y1
        face_y2 = y1 + body_h * FACE_REGION_RATIO
        face_cx = (x1 + x2) / 2
        face_cy = (y1 + face_y2) / 2
        face_s = max(x2 - x1, face_y2 - y1) / 2
        centers_x.append(face_cx)
        centers_y.append(face_cy)
        sizes.append(face_s)

    # Median filter for smoothing (kernel=13 as in TalkNet demo)
    kernel = min(13, len(sizes) if len(sizes) % 2 == 1 else max(1, len(sizes) - 1))
    if kernel >= 3:
        centers_x = scipy_signal.medfilt(centers_x, kernel_size=kernel).tolist()
        centers_y = scipy_signal.medfilt(centers_y, kernel_size=kernel).tolist()
        sizes = scipy_signal.medfilt(sizes, kernel_size=kernel).tolist()

    decoder = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    crops = []
    frame_idx = 0
    try:
        while frame_idx < len(track_bboxes):
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(vid_h, vid_w, 3)

            cx = centers_x[frame_idx]
            cy = centers_y[frame_idx]
            bs = sizes[frame_idx]

            if bs < MIN_FACE_SIZE / 2:
                crops.append(np.zeros((224, 224, 3), dtype=np.uint8))
                frame_idx += 1
                continue

            cs = CROP_SCALE
            bsi = int(bs * (1 + 2 * cs))
            # Pad frame to handle edge cases
            padded = np.pad(
                frame, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110
            )
            my = cy + bsi
            mx = cx + bsi
            y_start = int(my - bs)
            y_end = int(my + bs * (1 + 2 * cs))
            x_start = int(mx - bs * (1 + cs))
            x_end = int(mx + bs * (1 + cs))
            face = padded[y_start:y_end, x_start:x_end]

            if face.size == 0:
                crops.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                crops.append(cv2.resize(face, (224, 224)))

            frame_idx += 1
    finally:
        decoder.stdout.close()
        decoder.wait()

    return crops


def run_stage4(
    db: AnalysisDB,
    video_path: str,
    audio_stream: int = 0,
    device: str = "cuda",
    talknet_model_path: str | None = None,
    talknet_root: str | None = None,
    ddffnet_model_path: str | None = None,
):
    """Run TalkNet ASD on SAM3 person tracks.

    Args:
        db: Analysis database.
        video_path: Source video.
        audio_stream: Audio stream index for English track.
        device: CUDA device.
        talknet_model_path: Path to TalkNet pretrained model.
        talknet_root: Path to TalkNet source directory.
        ddffnet_model_path: Path to DDFFNet pretrained model (unused for now).
    """
    shots = db.get_shots()
    if not shots:
        raise RuntimeError("Stage 4: no shots in DB")

    progress = db.get_progress(STAGE)
    remaining = []
    for s in shots:
        sid = str(s["shot_id"])
        if progress.get(sid) != "done":
            tracks = db.get_person_tracks(shot_id=s["shot_id"])
            if tracks:
                remaining.append(s)
            else:
                db.mark_progress(STAGE, sid, "done")

    if not remaining:
        log.info("Stage 4: all shots already enriched, skipping")
        return

    log.info(f"Stage 4: enriching {len(remaining)} shots with ASD scoring")

    # Load TalkNet
    talknet = None
    if talknet_model_path:
        talknet = _load_talknet(talknet_model_path, talknet_root, device)

    if talknet is None:
        log.warning("Stage 4: no TalkNet model available, skipping")
        for s in remaining:
            db.mark_progress(STAGE, str(s["shot_id"]), "done")
        return

    for shot_idx, shot in enumerate(remaining):
        shot_id = shot["shot_id"]
        tracks = db.get_person_tracks(shot_id=shot_id)

        # Group tracks by sam3_obj_id
        by_obj: dict[int, list[dict]] = {}
        for t in tracks:
            by_obj.setdefault(t["sam3_obj_id"], []).append(t)

        speaker_rows = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract audio for the whole shot
            audio_path = os.path.join(tmpdir, "audio.wav")
            try:
                _extract_audio_segment(
                    video_path,
                    shot["start_sec"],
                    shot["end_sec"],
                    audio_path,
                    audio_stream,
                )
            except subprocess.CalledProcessError:
                log.debug(f"Shot {shot_id}: audio extraction failed, skipping")
                db.mark_progress(STAGE, str(shot_id), "done")
                continue

            for obj_id, obj_tracks in by_obj.items():
                obj_tracks.sort(key=lambda t: t["frame_sec"])

                # Skip full-frame detections (background)
                sample = obj_tracks[0]
                w = sample["bbox_x2"] - sample["bbox_x1"]
                h = sample["bbox_y2"] - sample["bbox_y1"]
                if w > 1600 and h > 900:
                    continue

                # Interpolate bboxes to 25fps
                track_25fps = _interpolate_bbox_to_25fps(
                    obj_tracks, shot["start_sec"], shot["end_sec"]
                )
                if len(track_25fps) < TALKNET_FPS:  # Less than 1 second
                    continue

                # Extract face crops
                face_crops = _extract_face_crops_25fps(video_path, track_25fps)
                if len(face_crops) < TALKNET_FPS:
                    continue

                # Run TalkNet
                try:
                    audio_offset = track_25fps[0][0] - shot["start_sec"]
                    scores = talknet.score_track(
                        face_crops, audio_path, audio_offset=audio_offset
                    )
                except Exception as e:
                    log.debug(f"TalkNet error on shot {shot_id} obj {obj_id}: {e}")
                    continue

                # Map 25fps scores back to 1fps keyframes
                for kf in obj_tracks:
                    t_offset = kf["frame_sec"] - track_25fps[0][0]
                    frame_idx = int(t_offset * TALKNET_FPS)
                    frame_idx = max(0, min(frame_idx, len(scores) - 1))

                    # Average over ±0.5s window around the keyframe
                    win_start = max(0, frame_idx - TALKNET_FPS // 2)
                    win_end = min(len(scores), frame_idx + TALKNET_FPS // 2)
                    avg_score = float(np.mean(scores[win_start:win_end]))

                    speaker_rows.append(
                        {
                            "shot_id": shot_id,
                            "sam3_obj_id": obj_id,
                            "frame_sec": kf["frame_sec"],
                            "asd_score": avg_score,
                        }
                    )

        if speaker_rows:
            db.insert_speaker_scores(speaker_rows)

        db.mark_progress(STAGE, str(shot_id), "done")

        if (shot_idx + 1) % 10 == 0 or shot_idx == len(remaining) - 1:
            log.info(f"Stage 4: {shot_idx + 1}/{len(remaining)} shots enriched")


def _load_talknet(model_path: str, talknet_root: str | None, device: str):
    """Load TalkNet ASD model."""
    try:
        from unified_pipeline.vendor.talknet import TalkNetASD

        if talknet_root is None:
            log.warning("TalkNet root directory not specified")
            return None

        model = TalkNetASD(model_path, talknet_root=talknet_root, device=device)
        return model
    except Exception as e:
        log.warning(f"TalkNet load failed: {e}")
        return None
