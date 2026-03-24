"""Pipeline orchestrator — runs stages 0-5 in order with resume support."""

import logging
import subprocess
import time
from pathlib import Path

from unified_pipeline.db import AnalysisDB, db_path_for_video

log = logging.getLogger(__name__)


def get_video_fps(video_path: str) -> float:
    """Get video frame rate via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    fps_str = result.stdout.strip()
    if "/" in fps_str:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    return float(fps_str)


def run_pipeline(
    video_path: str,
    audio_stream: int = 0,
    device: str = "cuda",
    stages: str = "0,1,2,3,4,5",
    db_file: str | None = None,
    store_masks: bool = True,
    cluster_threshold: float = 0.4,
    rms_threshold_db: float = -50.0,
    center_threshold_db: float = -40.0,
    talknet_model: str | None = None,
    talknet_root: str | None = None,
    ddffnet_model: str | None = None,
    tracking_fps: float | None = None,
):
    """Run the unified video analysis pipeline.

    Args:
        video_path: Path to source video file.
        audio_stream: Audio stream index (e.g. 6 for English in Aliens).
        device: CUDA device string.
        stages: Comma-separated stage numbers to run (e.g. "0,1,2,3,4,5").
        db_file: Override DB path (default: auto from video name).
        store_masks: Store RLE masks in DB (needed for SAM-Audio visual sep).
        cluster_threshold: Face clustering distance threshold.
        rms_threshold_db: SAM-Audio dialogue threshold.
        center_threshold_db: Center channel dialogue threshold.
        talknet_model: Path to TalkNet model checkpoint.
        talknet_root: Path to TalkNet source directory.
        ddffnet_model: Path to DDFFNet model checkpoint.
        tracking_fps: SAM3 tracking fps (default: native video fps).
    """
    video_path = str(Path(video_path).resolve())
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if db_file is None:
        db_file = str(db_path_for_video(video_path))

    log.info(f"Pipeline: {video_path}")
    log.info(f"Database: {db_file}")

    db = AnalysisDB(db_file)
    stage_set = {int(s.strip()) for s in stages.split(",")}

    fps = get_video_fps(video_path)
    log.info(f"Video FPS: {fps:.3f}")

    t0 = time.time()

    try:
        if 0 in stage_set:
            _timed("Stage 0: Shot detection", lambda: _run_stage0(db, video_path, fps))

        if 1 in stage_set:
            _timed("Stage 1: Scene grouping", lambda: _run_stage1(db, video_path))

        if 2 in stage_set:
            tfps = tracking_fps or fps
            _timed(
                f"Stage 2: SAM3 tracking @ {tfps}fps",
                lambda: _run_stage2(
                    db, video_path, audio_stream, device, store_masks, tfps, db_file
                ),
            )

        if 3 in stage_set:
            _timed(
                "Stage 3: Face identity",
                lambda: _run_stage3(db, video_path, cluster_threshold),
            )

        if 4 in stage_set:
            _timed(
                "Stage 4: ASD + blur",
                lambda: _run_stage4(
                    db,
                    video_path,
                    audio_stream,
                    device,
                    talknet_model,
                    talknet_root,
                    ddffnet_model,
                ),
            )

        if 5 in stage_set:
            _timed(
                "Stage 5: Audio analysis",
                lambda: _run_stage5(
                    db,
                    video_path,
                    audio_stream,
                    device,
                    rms_threshold_db,
                    center_threshold_db,
                ),
            )

    finally:
        elapsed = time.time() - t0
        log.info(f"Pipeline complete in {elapsed / 60:.1f} min")
        _print_db_stats(db)
        db.close()


def _timed(label: str, fn):
    log.info(f"--- {label} ---")
    t = time.time()
    fn()
    log.info(f"--- {label}: {time.time() - t:.1f}s ---")


def _run_stage0(db, video_path, fps):
    from unified_pipeline.stages.shot_detection import run_stage0

    run_stage0(db, video_path, fps)


def _run_stage1(db, video_path):
    from unified_pipeline.stages.shot_detection import run_stage1

    run_stage1(db, video_path)


def _run_stage2(
    db, video_path, audio_stream, device, store_masks, tracking_fps, db_file
):
    from pathlib import Path

    from unified_pipeline.stages.sam3_tracking import run_stage2

    masks_dir = str(
        Path(db_file).parent / Path(db_file).stem.replace(".analysis", ".masks")
    )
    run_stage2(
        db, video_path, audio_stream, device, store_masks, tracking_fps, masks_dir
    )


def _run_stage3(db, video_path, cluster_threshold):
    from unified_pipeline.stages.identity import run_stage3

    run_stage3(db, video_path, cluster_threshold)


def _run_stage4(
    db, video_path, audio_stream, device, talknet_model, talknet_root, ddffnet_model
):
    from unified_pipeline.stages.enrichment import run_stage4

    run_stage4(
        db, video_path, audio_stream, device, talknet_model, talknet_root, ddffnet_model
    )


def _run_stage5(
    db, video_path, audio_stream, device, rms_threshold_db, center_threshold_db
):
    from unified_pipeline.stages.audio_analysis import run_stage5

    run_stage5(
        db, video_path, audio_stream, device, rms_threshold_db, center_threshold_db
    )


def _print_db_stats(db: AnalysisDB):
    tables = [
        "shots",
        "scenes",
        "person_tracks",
        "track_identities",
        "characters",
        "face_detections",
        "speaker_scores",
        "blur_scores",
        "dialogue_segments",
        "character_audio",
    ]
    log.info("Database stats:")
    for table in tables:
        try:
            count = db.count_rows(table)
            if count > 0:
                log.info(f"  {table}: {count} rows")
        except Exception:
            pass
