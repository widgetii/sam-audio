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

from unified_pipeline.db import AnalysisDB

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

    # Face detection presence
    face_dets = db.get_face_detections()
    face_set: set[tuple[int, int]] = set()
    for f in face_dets:
        face_set.add((f["shot_id"], f["sam3_obj_id"]))

    result = []
    for t in tracks:
        sec_key = round(t["frame_sec"], 3)
        key = (t["shot_id"], t["sam3_obj_id"], sec_key)
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


def interpolate_tracks_to_fps(
    persons: list[PersonFrame],
    target_fps: float,
    start_sec: float,
    end_sec: float,
) -> dict[float, list[PersonFrame]]:
    """Interpolate 1fps person tracks to target_fps.

    Returns {timestamp: [PersonFrame, ...]} at every frame.
    """
    # Group by (shot_id, obj_id)
    tracks: dict[tuple[int, int], list[PersonFrame]] = defaultdict(list)
    for p in persons:
        tracks[(p.shot_id, p.sam3_obj_id)].append(p)

    for key in tracks:
        tracks[key].sort(key=lambda p: p.frame_sec)

    result: dict[float, list[PersonFrame]] = defaultdict(list)
    dt = 1.0 / target_fps

    t = start_sec
    while t <= end_sec:
        t_round = round(t, 6)
        for (shot_id, obj_id), keyframes in tracks.items():
            if not keyframes:
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
            else:
                frac = (t_round - before.frame_sec) / (
                    after.frame_sec - before.frame_sec
                )
                frac = max(0.0, min(1.0, frac))
                interp_bbox = interpolate_bbox(before.bbox, after.bbox, frac)

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
    interp = interpolate_tracks_to_fps(persons, target_fps, start_sec, end_sec)

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
            # Center crop on main person's horizontal center
            person_cx = (main.bbox[0] + main.bbox[2]) / 2
            crop_x = int(person_cx - crop_w / 2)
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
    target_fps: float = 24.0,
):
    """Render vertical video using ffmpeg with crop positions.

    Uses a crop filter with sendcmd to change positions per frame.
    For simplicity, uses the average crop position per second.
    """
    if not crops:
        log.warning("No crop positions to render")
        return

    # Group by second and average
    by_sec: dict[int, list[dict]] = defaultdict(list)
    for c in crops:
        by_sec[int(c["timestamp"])].append(c)

    crop_w = crops[0]["crop_w"]
    crop_h = crops[0]["crop_h"]

    # Build sendcmd script for crop position changes
    lines = []
    for sec in sorted(by_sec.keys()):
        avg_x = int(np.mean([c["crop_x"] for c in by_sec[sec]]))
        avg_y = int(np.mean([c["crop_y"] for c in by_sec[sec]]))
        lines.append(f"{sec} [enter] crop x {avg_x};")
        lines.append(f"{sec} [enter] crop y {avg_y};")

    # TODO: use sendcmd for dynamic crop per second with the lines above
    avg_x = int(np.mean([c["crop_x"] for c in crops]))
    avg_y = int(np.mean([c["crop_y"] for c in crops]))

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-i",
        video_path,
        "-vf",
        f"crop={crop_w}:{crop_h}:{avg_x}:{avg_y}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "copy",
        output_path,
    ]
    log.info(f"Rendering vertical video to {output_path}")
    subprocess.run(cmd, check=True)
