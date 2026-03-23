"""Stage 0: Shot detection via av1an.  Stage 1: Scene grouping."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scene_detector import detect_shots_av1an, group_shots_into_scenes

from unified_pipeline.db import AnalysisDB

log = logging.getLogger(__name__)


def run_stage0(db: AnalysisDB, video_path: str, fps: float):
    """Detect shots with av1an and store in DB."""
    if db.count_rows("shots") > 0:
        log.info("Stage 0: shots already in DB, skipping")
        return

    log.info("Stage 0: detecting shots with av1an...")
    shot_objects = detect_shots_av1an(video_path)
    log.info(f"Stage 0: found {len(shot_objects)} shots")

    # Store Shot objects for stage 1, convert to DB rows
    _shot_objects_cache.clear()
    _shot_objects_cache.extend(shot_objects)

    shots = []
    for shot in shot_objects:
        shots.append(
            {
                "shot_id": shot.index,
                "start_frame": shot.start_frame,
                "end_frame": shot.end_frame,
                "start_sec": shot.start_sec,
                "end_sec": shot.end_sec,
            }
        )

    db.insert_shots(shots)
    db.mark_progress("stage0", "shots", "done")
    log.info(f"Stage 0: stored {len(shots)} shots in DB")


# Cache Shot objects between stage 0 and stage 1 within the same process
_shot_objects_cache: list = []


def run_stage1(db: AnalysisDB, video_path: str | None = None):
    """Group shots into scenes and store in DB."""
    if db.count_rows("scenes") > 0:
        log.info("Stage 1: scenes already in DB, skipping")
        return

    db_shots = db.get_shots()
    if not db_shots:
        raise RuntimeError("Stage 1: no shots in DB — run stage 0 first")

    # Reconstruct Shot objects if not cached (e.g. resuming from a different process)
    if _shot_objects_cache:
        shot_objects = list(_shot_objects_cache)
    else:
        from scene_detector import Shot

        shot_objects = []
        for s in db_shots:
            shot_objects.append(
                Shot(
                    index=s["shot_id"],
                    start_sec=s["start_sec"],
                    end_sec=s["end_sec"],
                    start_frame=s["start_frame"],
                    end_frame=s["end_frame"],
                )
            )

    log.info("Stage 1: grouping shots into scenes...")
    scene_objects = group_shots_into_scenes(shot_objects)

    scenes = []
    for scene in scene_objects:
        shot_ids = [shot.index for shot in scene.shots]
        scenes.append(
            {
                "scene_id": scene.scene_id,
                "start_sec": scene.start_sec,
                "end_sec": scene.end_sec,
                "shot_ids": shot_ids,
            }
        )

    db.insert_scenes(scenes)
    db.mark_progress("stage1", "scenes", "done")
    log.info(f"Stage 1: stored {len(scenes)} scenes in DB")
