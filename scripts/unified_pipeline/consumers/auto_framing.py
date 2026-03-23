"""Consumer 6A: Auto-framing — DB → interpolate 1fps tracks → score → crop → ffmpeg.

Reads person_tracks, speaker_scores, blur_scores from SQLite.
Interpolates 1fps data to native video fps for smooth crop positioning.
Scores each person using speaking/blur/class weights, picks main subject,
outputs vertical (9:16) crop positions, and optionally renders via ffmpeg.
"""

import logging
import subprocess
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from unified_pipeline.db import AnalysisDB, rle_to_mask

log = logging.getLogger(__name__)

# Scoring weights (ported from auto_framing/main.py:choose_main_detections)
W_SPEAKING = 30401
W_BLUR = -301
W_CLASS = 301  # bonus for face vs body-only


@dataclass
class PersonFrame:
    """A person detection at a specific timestamp."""

    sam3_obj_id: int
    shot_id: int
    frame_sec: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    character_id: int | None = None
    asd_score: float = 0.0
    blur_score: float = 0.5
    has_face: bool = False
    head_cx: float | None = None  # head horizontal center from mask silhouette


def _head_cx_from_mask(mask_rle: bytes, top_fraction: float = 0.10) -> float | None:
    """Compute head horizontal center from the top portion of a mask silhouette.

    Decodes the RLE mask, finds the topmost region (top_fraction of mask height),
    and returns the horizontal center of that region. Returns None if mask is
    empty or too small.
    """
    mask = rle_to_mask(mask_rle)
    rows_with_mask = np.where(mask.any(axis=1))[0]
    if len(rows_with_mask) < 10:
        return None

    top_row = rows_with_mask[0]
    bot_row = rows_with_mask[-1]
    mask_height = bot_row - top_row
    if mask_height < 20:
        return None

    head_bottom = top_row + max(1, int(mask_height * top_fraction))
    head_region = mask[top_row:head_bottom, :]
    head_cols = np.where(head_region.any(axis=0))[0]
    if len(head_cols) == 0:
        return None

    return float((head_cols[0] + head_cols[-1]) / 2)


def query_person_data(
    db: AnalysisDB, start_sec: float, end_sec: float
) -> list[PersonFrame]:
    """Query all person data for a time range, joining tracks + scores + identity."""
    tracks = db.get_tracks_with_identity(time_range=(start_sec, end_sec))

    # Index speaker/blur scores by (shot_id, sam3_obj_id, frame_sec)
    speaker_scores = db.get_speaker_scores(time_range=(start_sec, end_sec))
    speaker_idx: dict[tuple, float] = {}
    for s in speaker_scores:
        key = (s["shot_id"], s["sam3_obj_id"], round(s["frame_sec"], 3))
        speaker_idx[key] = s["asd_score"]

    blur_scores = db.get_blur_scores(time_range=(start_sec, end_sec))
    blur_idx: dict[tuple, float] = {}
    for b in blur_scores:
        key = (b["shot_id"], b["sam3_obj_id"], round(b["frame_sec"], 3))
        blur_idx[key] = b["blur_score"]

    # Face detection presence (for scoring bonus)
    face_dets = db.get_face_detections()
    face_set: set[tuple[int, int]] = set()
    for f in face_dets:
        face_set.add((f["shot_id"], f["sam3_obj_id"]))

    # Fetch mask_rle for head_cx computation
    mask_rows = db.conn.execute(
        "SELECT shot_id, sam3_obj_id, frame_sec, mask_rle "
        "FROM person_tracks WHERE frame_sec BETWEEN ? AND ? AND mask_rle IS NOT NULL",
        (start_sec, end_sec),
    ).fetchall()
    mask_idx: dict[tuple, bytes] = {}
    for m in mask_rows:
        key = (m[0], m[1], round(m[2], 3))
        mask_idx[key] = m[3]

    result = []
    for t in tracks:
        sec_key = round(t["frame_sec"], 3)
        key = (t["shot_id"], t["sam3_obj_id"], sec_key)

        # Compute head_cx from mask silhouette
        head_cx = None
        mask_data = mask_idx.get(key)
        if mask_data:
            head_cx = _head_cx_from_mask(mask_data)

        result.append(
            PersonFrame(
                sam3_obj_id=t["sam3_obj_id"],
                shot_id=t["shot_id"],
                frame_sec=t["frame_sec"],
                bbox=(t["bbox_x1"], t["bbox_y1"], t["bbox_x2"], t["bbox_y2"]),
                character_id=t.get("character_id"),
                asd_score=speaker_idx.get(key, 0.0),
                blur_score=blur_idx.get(key, 0.5),
                has_face=(t["shot_id"], t["sam3_obj_id"]) in face_set,
                head_cx=head_cx,
            )
        )
    return result


def interpolate_bbox(
    before: tuple[float, float, float, float],
    after: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    """Linear interpolation between two bboxes. t in [0, 1]."""
    return tuple(b + (a - b) * t for b, a in zip(before, after, strict=True))


def _is_fullframe_bbox(
    bbox: tuple[float, float, float, float],
    video_width: int,
    video_height: int,
    threshold: float = 0.85,
) -> bool:
    """Return True if bbox covers most of the frame (background detection)."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return (w / video_width) > threshold and (h / video_height) > threshold


def interpolate_tracks_to_fps(
    persons: list[PersonFrame],
    target_fps: float,
    start_sec: float,
    end_sec: float,
    video_width: int = 1920,
    video_height: int = 1080,
) -> dict[float, list[PersonFrame]]:
    """Interpolate 1fps person tracks to target_fps.

    Returns {timestamp: [PersonFrame, ...]} at every frame.
    Only interpolates within each track's keyframe time range (no extrapolation
    beyond the first/last keyframe). Filters out full-frame background detections.
    """
    # Group by (shot_id, obj_id)
    tracks: dict[tuple[int, int], list[PersonFrame]] = defaultdict(list)
    for p in persons:
        # Skip full-frame background detections
        if _is_fullframe_bbox(p.bbox, video_width, video_height):
            continue
        tracks[(p.shot_id, p.sam3_obj_id)].append(p)

    # Sort and compute time bounds per track
    track_bounds: dict[tuple[int, int], tuple[float, float]] = {}
    for key in tracks:
        tracks[key].sort(key=lambda p: p.frame_sec)
        kfs = tracks[key]
        # Allow 0.5s padding beyond first/last keyframe (half the 1fps interval)
        track_bounds[key] = (kfs[0].frame_sec - 0.5, kfs[-1].frame_sec + 0.5)

    result: dict[float, list[PersonFrame]] = defaultdict(list)
    dt = 1.0 / target_fps

    t = start_sec
    while t <= end_sec:
        t_round = round(t, 6)
        for (shot_id, obj_id), keyframes in tracks.items():
            # Only interpolate within track's time range
            lo, hi = track_bounds[(shot_id, obj_id)]
            if t_round < lo or t_round > hi:
                continue

            # Find surrounding keyframes
            before = None
            after = None
            for kf in keyframes:
                if kf.frame_sec <= t_round:
                    before = kf
                if kf.frame_sec >= t_round and after is None:
                    after = kf

            if before is None and after is None:
                continue

            if before is None:
                before = after
            if after is None:
                after = before

            if before.frame_sec == after.frame_sec:
                interp_bbox = before.bbox
                interp_head_cx = before.head_cx
            else:
                frac = (t_round - before.frame_sec) / (
                    after.frame_sec - before.frame_sec
                )
                frac = max(0.0, min(1.0, frac))
                interp_bbox = interpolate_bbox(before.bbox, after.bbox, frac)
                # Interpolate head_cx if both keyframes have it
                if before.head_cx is not None and after.head_cx is not None:
                    interp_head_cx = (
                        before.head_cx + (after.head_cx - before.head_cx) * frac
                    )
                else:
                    interp_head_cx = before.head_cx or after.head_cx

            result[t_round].append(
                PersonFrame(
                    sam3_obj_id=obj_id,
                    shot_id=shot_id,
                    frame_sec=t_round,
                    bbox=interp_bbox,
                    character_id=before.character_id,
                    asd_score=before.asd_score,
                    blur_score=before.blur_score,
                    has_face=before.has_face,
                    head_cx=interp_head_cx,
                )
            )

        t += dt

    return dict(result)


def score_person(p: PersonFrame) -> float:
    """Score a person for main-subject selection."""
    speaking = W_SPEAKING * p.asd_score
    blur = W_BLUR * (1.0 - p.blur_score)
    cls = W_CLASS * (1.0 if p.has_face else 0.0)
    # Bbox area as tiebreaker (prefer larger)
    area = (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])
    return speaking + blur + cls + area * 0.01


def choose_main_person(persons: list[PersonFrame]) -> PersonFrame | None:
    """Choose the main subject from a list of persons at one timestamp."""
    if not persons:
        return None
    return max(persons, key=score_person)


def compute_crop_positions(
    db: AnalysisDB,
    start_sec: float,
    end_sec: float,
    video_width: int,
    video_height: int,
    target_fps: float = 24.0,
    output_aspect: tuple[int, int] = (9, 16),
) -> list[dict]:
    """Compute per-frame crop positions for vertical video.

    Returns list of {timestamp, crop_x, crop_y, crop_w, crop_h, character_id}.
    """
    persons = query_person_data(db, start_sec, end_sec)
    interp = interpolate_tracks_to_fps(
        persons, target_fps, start_sec, end_sec, video_width, video_height
    )

    aspect = output_aspect[0] / output_aspect[1]
    crop_h = video_height
    crop_w = int(crop_h * aspect)
    if crop_w > video_width:
        crop_w = video_width
        crop_h = int(crop_w / aspect)

    crops = []
    prev_x = video_width // 2 - crop_w // 2  # default center

    for t in sorted(interp.keys()):
        main = choose_main_person(interp[t])
        if main is not None:
            # Center crop on head (from mask), fall back to bbox center
            if main.head_cx is not None:
                target_cx = main.head_cx
            else:
                target_cx = (main.bbox[0] + main.bbox[2]) / 2
            crop_x = int(target_cx - crop_w / 2)
            crop_x = max(0, min(crop_x, video_width - crop_w))
        else:
            crop_x = prev_x

        # Smooth: blend with previous position
        crop_x = int(0.7 * crop_x + 0.3 * prev_x)
        prev_x = crop_x

        crop_y = max(0, (video_height - crop_h) // 2)

        crops.append(
            {
                "timestamp": t,
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "character_id": main.character_id if main else None,
            }
        )

    return crops


def render_vertical_video(
    video_path: str,
    crops: list[dict],
    output_path: str,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    audio_stream: int = 0,
    target_fps: float = 24.0,
):
    """Render vertical video with per-frame dynamic cropping.

    Decodes the source segment with ffmpeg, crops each frame in Python using
    the precomputed crop positions, and pipes back to ffmpeg for encoding
    with the original audio track.
    """
    if not crops:
        log.warning("No crop positions to render")
        return

    crop_w = crops[0]["crop_w"]
    crop_h = crops[0]["crop_h"]

    # Build sorted lookup: [(timestamp, crop_x, crop_y), ...]
    crop_keys = np.array([c["timestamp"] for c in crops])
    crop_xs = np.array([c["crop_x"] for c in crops])
    crop_ys = np.array([c["crop_y"] for c in crops])

    duration = (end_sec - start_sec) if end_sec else crops[-1]["timestamp"] - start_sec

    # --- Decoder: ffmpeg → raw RGB frames to stdout ---
    decode_cmd = [
        "ffmpeg",
        "-loglevel",
        "warning",
        "-ss",
        str(start_sec),
        "-t",
        str(duration),
        "-i",
        video_path,
        "-map",
        "0:v:0",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-vsync",
        "cfr",
        "pipe:1",
    ]

    # --- Encoder: raw RGB from stdin → H.264 + audio ---
    encode_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        # Raw video input from pipe
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{crop_w}x{crop_h}",
        "-r",
        str(target_fps),
        "-i",
        "pipe:0",
        # Audio from original file
        "-ss",
        str(start_sec),
        "-t",
        str(duration),
        "-i",
        video_path,
        "-map",
        "0:v:0",
        "-map",
        f"1:{audio_stream}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        output_path,
    ]

    log.info(
        f"Rendering {crop_w}x{crop_h} vertical video to {output_path} "
        f"({duration:.1f}s, {len(crops)} crop keyframes)"
    )

    # Get source video dimensions from first crop's implied frame size
    # We need source width/height to read raw frames — get via ffprobe
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
    src_w, src_h = [int(x) for x in probe.stdout.strip().split(",")]
    frame_bytes = src_w * src_h * 3  # RGB24

    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    frame_idx = 0
    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(src_h, src_w, 3)

            # Compute timestamp relative to start, find nearest crop position
            t = frame_idx / target_fps
            idx = int(np.searchsorted(crop_keys, t + start_sec, side="right")) - 1
            idx = max(0, min(idx, len(crop_keys) - 1))

            cx = int(crop_xs[idx])
            cy = int(crop_ys[idx])

            # Crop the frame
            cropped = frame[cy : cy + crop_h, cx : cx + crop_w]
            encoder.stdin.write(cropped.tobytes())

            frame_idx += 1
            if frame_idx % 1000 == 0:
                log.info(f"  rendered {frame_idx} frames ({t:.1f}s)")
    finally:
        decoder.stdout.close()
        encoder.stdin.close()
        decoder.wait()
        encoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError(f"Encoder failed with return code {encoder.returncode}")

    log.info(f"Done: {frame_idx} frames rendered to {output_path}")
