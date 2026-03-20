"""Shot boundary detection (via av1an) and scene grouping with character propagation."""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Shot:
    """A single camera shot (between two cuts)."""

    index: int
    start_sec: float
    end_sec: float
    start_frame: int = 0
    end_frame: int = 0
    character_ids: set[int] = field(default_factory=set)  # detected by face
    propagated_character_ids: set[int] = field(
        default_factory=set
    )  # inherited from scene
    face_detections: list = field(default_factory=list)  # FaceDetection objects
    scene_id: int = -1

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def all_characters(self) -> set[int]:
        return self.character_ids | self.propagated_character_ids


@dataclass
class Scene:
    """A group of consecutive shots forming a logical scene."""

    scene_id: int
    shots: list[Shot]

    @property
    def start_sec(self) -> float:
        return self.shots[0].start_sec if self.shots else 0.0

    @property
    def end_sec(self) -> float:
        return self.shots[-1].end_sec if self.shots else 0.0

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def confirmed_characters(self) -> set[int]:
        """Union of all detected + propagated characters across shots."""
        chars = set()
        for shot in self.shots:
            chars |= shot.all_characters
        return chars

    has_dialogue: bool = False


def detect_shots_av1an(
    video_path: str,
    output_path: str | None = None,
) -> list[Shot]:
    """Detect shot boundaries using av1an --sc-only.

    Args:
        video_path: Path to the video file.
        output_path: Optional path to save the scenes JSON. If None, uses a tempfile.

    Returns:
        List of Shot objects with frame-accurate boundaries.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        output_path = tmp.name
        tmp.close()

    logger.info(f"Running av1an shot detection on {video_path}")
    cmd = [
        "av1an",
        "--sc-only",
        "-i",
        video_path,
        "--scenes",
        output_path,
        "-x",
        "0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"av1an failed: {result.stderr}")

    return load_shots_from_json(output_path, video_path)


def load_shots_from_json(json_path: str, video_path: str | None = None) -> list[Shot]:
    """Load shot boundaries from av1an JSON output.

    av1an outputs: {"scenes": [[start_frame, end_frame], ...], "frames": total_frames}
    We need the video FPS to convert frames to seconds.
    """
    with open(json_path) as f:
        data = json.load(f)

    # Get FPS from video
    fps = _get_video_fps(video_path) if video_path else 23.976

    scenes_data = data.get("scenes", data.get("frames", []))

    shots = []
    for i, scene in enumerate(scenes_data):
        if isinstance(scene, list) and len(scene) == 2:
            start_frame, end_frame = scene
        elif isinstance(scene, dict):
            start_frame = scene.get("start_frame", scene.get("start", 0))
            end_frame = scene.get("end_frame", scene.get("end", 0))
        else:
            continue

        shots.append(
            Shot(
                index=i,
                start_sec=start_frame / fps,
                end_sec=end_frame / fps,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )

    logger.info(f"Loaded {len(shots)} shots from {json_path} (fps={fps:.3f})")
    return shots


def _get_video_fps(video_path: str) -> float:
    """Get video FPS using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        rate_str = data["streams"][0]["r_frame_rate"]
        num, den = rate_str.split("/")
        return float(num) / float(den)
    except Exception:
        logger.warning("Could not detect FPS, defaulting to 23.976")
        return 23.976


def group_shots_into_scenes(
    shots: list[Shot],
    max_gap_sec: float = 2.0,
    max_scene_duration: float = 300.0,
) -> list[Scene]:
    """Group consecutive shots into scenes based on character overlap and temporal proximity.

    Merge rules (all must hold):
    1. Inter-shot gap < max_gap_sec
    2. Character overlap is non-empty (or either shot has no detections)
    3. Merged scene duration < max_scene_duration
    """
    if not shots:
        return []

    scenes = []
    current_shots = [shots[0]]

    for i in range(1, len(shots)):
        prev = shots[i - 1]
        curr = shots[i]

        gap = curr.start_sec - prev.end_sec
        current_duration = curr.end_sec - current_shots[0].start_sec

        # Check character overlap
        prev_chars = prev.character_ids
        curr_chars = curr.character_ids
        has_overlap = bool(prev_chars & curr_chars) or not prev_chars or not curr_chars

        should_merge = (
            gap < max_gap_sec and has_overlap and current_duration < max_scene_duration
        )

        if should_merge:
            current_shots.append(curr)
        else:
            scene_id = len(scenes)
            scene = Scene(scene_id=scene_id, shots=current_shots)
            for s in current_shots:
                s.scene_id = scene_id
            scenes.append(scene)
            current_shots = [curr]

    # Final scene
    scene_id = len(scenes)
    scene = Scene(scene_id=scene_id, shots=current_shots)
    for s in current_shots:
        s.scene_id = scene_id
    scenes.append(scene)

    logger.info(f"Grouped {len(shots)} shots into {len(scenes)} scenes")
    return scenes


def propagate_characters(scenes: list[Scene]):
    """Propagate character IDs within each scene.

    If a character is detected by face in any shot of a scene,
    propagate them to all other shots in the same scene.
    """
    for scene in scenes:
        all_chars = set()
        for shot in scene.shots:
            all_chars |= shot.character_ids

        for shot in scene.shots:
            shot.propagated_character_ids = all_chars - shot.character_ids

    total_propagated = sum(
        len(shot.propagated_character_ids) for scene in scenes for shot in scene.shots
    )
    logger.info(f"Propagated {total_propagated} character-shot assignments")


def generate_scene_chunks(
    scenes: list[Scene],
    max_chunk_seconds: float = 90.0,
) -> list[dict]:
    """Generate scene-aligned audio chunks for processing.

    Scenes <= max_chunk_seconds -> single chunk.
    Longer scenes -> split at internal shot boundaries.

    Returns list of chunk dicts with:
        scene_id, start_sec, end_sec, shot_indices, characters
    """
    chunks = []
    chunk_idx = 0

    for scene in scenes:
        if scene.duration <= max_chunk_seconds:
            chunks.append(
                {
                    "chunk_index": chunk_idx,
                    "scene_id": scene.scene_id,
                    "start_sec": scene.start_sec,
                    "end_sec": scene.end_sec,
                    "shot_indices": [s.index for s in scene.shots],
                    "characters": sorted(scene.confirmed_characters),
                }
            )
            chunk_idx += 1
        else:
            # Split at shot boundaries
            current_start = scene.shots[0].start_sec
            current_shots = []
            for shot in scene.shots:
                proposed_end = shot.end_sec
                if proposed_end - current_start > max_chunk_seconds and current_shots:
                    chunks.append(
                        {
                            "chunk_index": chunk_idx,
                            "scene_id": scene.scene_id,
                            "start_sec": current_start,
                            "end_sec": current_shots[-1].end_sec,
                            "shot_indices": [s.index for s in current_shots],
                            "characters": sorted(scene.confirmed_characters),
                        }
                    )
                    chunk_idx += 1
                    current_start = shot.start_sec
                    current_shots = [shot]
                else:
                    current_shots.append(shot)

            if current_shots:
                chunks.append(
                    {
                        "chunk_index": chunk_idx,
                        "scene_id": scene.scene_id,
                        "start_sec": current_start,
                        "end_sec": current_shots[-1].end_sec,
                        "shot_indices": [s.index for s in current_shots],
                        "characters": sorted(scene.confirmed_characters),
                    }
                )
                chunk_idx += 1

    logger.info(
        f"Generated {len(chunks)} scene-aligned chunks from {len(scenes)} scenes"
    )
    return chunks


def save_shots_and_scenes(
    shots: list[Shot],
    scenes: list[Scene],
    output_path: str,
):
    """Save shot and scene data to JSON for inspection/resume."""
    data = {
        "shots": [
            {
                "index": s.index,
                "start_sec": round(s.start_sec, 3),
                "end_sec": round(s.end_sec, 3),
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "character_ids": sorted(s.character_ids),
                "propagated_character_ids": sorted(s.propagated_character_ids),
                "scene_id": s.scene_id,
            }
            for s in shots
        ],
        "scenes": [
            {
                "scene_id": sc.scene_id,
                "start_sec": round(sc.start_sec, 3),
                "end_sec": round(sc.end_sec, 3),
                "duration": round(sc.duration, 3),
                "num_shots": len(sc.shots),
                "shot_indices": [s.index for s in sc.shots],
                "confirmed_characters": sorted(sc.confirmed_characters),
                "has_dialogue": sc.has_dialogue,
            }
            for sc in scenes
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved shots/scenes to {output_path}")


def load_shots_and_scenes(input_path: str) -> tuple[list[Shot], list[Scene]]:
    """Load previously saved shots and scenes from JSON."""
    with open(input_path) as f:
        data = json.load(f)

    shots = []
    for sd in data["shots"]:
        shots.append(
            Shot(
                index=sd["index"],
                start_sec=sd["start_sec"],
                end_sec=sd["end_sec"],
                start_frame=sd.get("start_frame", 0),
                end_frame=sd.get("end_frame", 0),
                character_ids=set(sd["character_ids"]),
                propagated_character_ids=set(sd["propagated_character_ids"]),
                scene_id=sd["scene_id"],
            )
        )

    shot_by_index = {s.index: s for s in shots}
    scenes = []
    for scd in data["scenes"]:
        scene_shots = [shot_by_index[i] for i in scd["shot_indices"]]
        scene = Scene(scene_id=scd["scene_id"], shots=scene_shots)
        scene.has_dialogue = scd.get("has_dialogue", False)
        scenes.append(scene)

    return shots, scenes
