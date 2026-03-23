"""Stage 4: TalkNet ASD + DDFFNet blur scoring on SAM3 person tracks.

Runs at 1fps using SAM3 person bboxes from Stage 2.
For each person track at each keyframe:
  - Crop face region (upper portion of person bbox)
  - TalkNet ASD: face video crops + audio → speaker probability
  - DDFFNet: face crop → blur/sharpness score
"""

import logging
import os
import subprocess
import tempfile

import numpy as np

from unified_pipeline.db import AnalysisDB

log = logging.getLogger(__name__)

STAGE = "stage4"

# Face region = upper 40% of person bbox (head area)
FACE_REGION_RATIO = 0.4


def _extract_face_crop(
    frame: np.ndarray, bbox: tuple[float, float, float, float], min_size: int = 48
) -> np.ndarray | None:
    """Crop the face region (upper portion) from person bbox."""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2_full = min(h, int(bbox[3]))

    # Upper portion for face
    body_h = y2_full - y1
    y2 = y1 + int(body_h * FACE_REGION_RATIO)
    y2 = min(y2, y2_full)

    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None

    return frame[y1:y2, x1:x2]


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
        f"0:a:{audio_stream}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def run_stage4(
    db: AnalysisDB,
    video_path: str,
    audio_stream: int = 0,
    device: str = "cuda",
    talknet_model_path: str | None = None,
    ddffnet_model_path: str | None = None,
):
    """Run TalkNet ASD and DDFFNet blur on SAM3 person tracks.

    Args:
        db: Analysis database.
        video_path: Source video.
        audio_stream: Audio stream index for English track.
        device: CUDA device.
        talknet_model_path: Path to TalkNet pretrained model.
        ddffnet_model_path: Path to DDFFNet pretrained model.
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

    log.info(f"Stage 4: enriching {len(remaining)} shots with ASD + blur")

    import cv2

    # Load TalkNet
    talknet = None
    if talknet_model_path:
        talknet = _load_talknet(talknet_model_path, device)

    # Load DDFFNet
    ddffnet = None
    if ddffnet_model_path:
        ddffnet = _load_ddffnet(ddffnet_model_path, device)

    if talknet is None and ddffnet is None:
        log.warning(
            "Stage 4: no TalkNet or DDFFNet models provided, skipping enrichment"
        )
        for s in remaining:
            db.mark_progress(STAGE, str(s["shot_id"]), "done")
        return

    for shot_idx, shot in enumerate(remaining):
        shot_id = shot["shot_id"]
        tracks = db.get_person_tracks(shot_id=shot_id)

        # Group tracks by frame_sec
        by_sec: dict[float, list[dict]] = {}
        for t in tracks:
            by_sec.setdefault(t["frame_sec"], []).append(t)

        speaker_rows = []
        blur_rows = []

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
                audio_path = None

            for sec, frame_tracks in sorted(by_sec.items()):
                # Extract frame
                frame_path = os.path.join(tmpdir, "frame.jpg")
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-ss",
                            str(sec),
                            "-i",
                            video_path,
                            "-frames:v",
                            "1",
                            "-qscale:v",
                            "2",
                            frame_path,
                        ],
                        check=True,
                    )
                    frame = cv2.imread(frame_path)
                except (subprocess.CalledProcessError, Exception):
                    continue

                if frame is None:
                    continue

                for track in frame_tracks:
                    bbox = (
                        track["bbox_x1"],
                        track["bbox_y1"],
                        track["bbox_x2"],
                        track["bbox_y2"],
                    )
                    face_crop = _extract_face_crop(frame, bbox)
                    if face_crop is None:
                        continue

                    # DDFFNet blur
                    if ddffnet is not None:
                        blur_val = _run_ddffnet(ddffnet, frame, bbox, device)
                        blur_rows.append(
                            {
                                "shot_id": shot_id,
                                "sam3_obj_id": track["sam3_obj_id"],
                                "frame_sec": sec,
                                "blur_score": blur_val,
                            }
                        )

                    # TalkNet ASD
                    if talknet is not None and audio_path is not None:
                        asd_val = _run_talknet(
                            talknet,
                            face_crop,
                            audio_path,
                            sec - shot["start_sec"],
                            device,
                        )
                        speaker_rows.append(
                            {
                                "shot_id": shot_id,
                                "sam3_obj_id": track["sam3_obj_id"],
                                "frame_sec": sec,
                                "asd_score": asd_val,
                            }
                        )

        if speaker_rows:
            db.insert_speaker_scores(speaker_rows)
        if blur_rows:
            db.insert_blur_scores(blur_rows)

        db.mark_progress(STAGE, str(shot_id), "done")

        if (shot_idx + 1) % 50 == 0 or shot_idx == len(remaining) - 1:
            log.info(f"Stage 4: {shot_idx + 1}/{len(remaining)} shots enriched")


# ---------------------------------------------------------------------------
# Model loaders — these will be adapted when vendoring TalkNet/DDFFNet
# ---------------------------------------------------------------------------


def _load_talknet(model_path: str, device: str):
    """Load TalkNet ASD model. Expects model_path to a pretrained checkpoint."""
    try:
        from unified_pipeline.vendor.talknet import TalkNetASD

        model = TalkNetASD(model_path, device=device)
        return model
    except ImportError:
        log.warning("TalkNet vendor module not found — ASD scoring disabled")
        return None


def _load_ddffnet(model_path: str, device: str):
    """Load DDFFNet blur detection model."""
    try:
        from unified_pipeline.vendor.ddffnet import DDFFNet

        model = DDFFNet(model_path, device=device)
        return model
    except ImportError:
        log.warning("DDFFNet vendor module not found — blur scoring disabled")
        return None


def _run_talknet(
    model, face_crop: np.ndarray, audio_path: str, offset_sec: float, device: str
) -> float:
    """Run TalkNet on a single face crop + audio, return speaking probability."""
    try:
        score = model.score_face(face_crop, audio_path, offset_sec)
        return float(score)
    except Exception as e:
        log.debug(f"TalkNet error: {e}")
        return 0.0


def _run_ddffnet(
    model, frame: np.ndarray, bbox: tuple[float, float, float, float], device: str
) -> float:
    """Run DDFFNet on the person region, return blur score (0=blurry, 1=sharp)."""
    try:
        score = model.score_region(frame, bbox)
        return float(score)
    except Exception as e:
        log.debug(f"DDFFNet error: {e}")
        return 0.5
