"""Consumer 6B: Dialogue timeline — DB → per-character speaking timeline + transcription."""

import logging
from collections import defaultdict
from dataclasses import dataclass

from unified_pipeline.db import AnalysisDB

log = logging.getLogger(__name__)


@dataclass
class DialogueLine:
    character_id: int | None
    character_name: str | None
    start_sec: float
    end_sec: float
    text: str | None = None


def get_dialogue_timeline(db: AnalysisDB) -> list[DialogueLine]:
    """Build ordered dialogue timeline from DB.

    Merges consecutive dialogue segments attributed to the same character.
    """
    segments = db.get_dialogue_segments(dialogue_only=True)
    characters = {c["character_id"]: c for c in db.get_characters()}

    # Merge consecutive segments with same character
    lines: list[DialogueLine] = []
    current: DialogueLine | None = None

    for seg in segments:
        char_id = seg.get("character_id")
        char_name = (
            characters[char_id]["name"] if char_id and char_id in characters else None
        )

        if (
            current is not None
            and current.character_id == char_id
            and seg["start_sec"] - current.end_sec < 1.5  # gap tolerance
        ):
            current.end_sec = seg["end_sec"]
        else:
            if current is not None:
                lines.append(current)
            current = DialogueLine(
                character_id=char_id,
                character_name=char_name,
                start_sec=seg["start_sec"],
                end_sec=seg["end_sec"],
            )

    if current is not None:
        lines.append(current)

    return lines


def get_character_speaking_time(db: AnalysisDB) -> dict[int, float]:
    """Get total speaking time per character in seconds."""
    segments = db.get_dialogue_segments(dialogue_only=True)
    speaking: dict[int, float] = defaultdict(float)

    for seg in segments:
        char_id = seg.get("character_id")
        if char_id is not None:
            speaking[char_id] += seg["end_sec"] - seg["start_sec"]

    return dict(speaking)


def print_dialogue_summary(db: AnalysisDB):
    """Print a human-readable dialogue summary."""
    segments = db.get_dialogue_segments()
    dialogue_segs = [s for s in segments if s["has_dialogue"]]
    characters = {c["character_id"]: c for c in db.get_characters()}

    total = len(segments)
    dialogue = len(dialogue_segs)
    attributed = sum(1 for s in dialogue_segs if s.get("character_id") is not None)

    print("\nDialogue Summary:")
    print(f"  Total seconds analyzed: {total}")
    print(f"  Dialogue seconds: {dialogue} ({100 * dialogue / max(total, 1):.1f}%)")
    print(
        f"  Attributed to character: {attributed} ({100 * attributed / max(dialogue, 1):.1f}%)"
    )

    speaking = get_character_speaking_time(db)
    if speaking:
        print("\nPer-character speaking time:")
        for char_id in sorted(speaking, key=speaking.get, reverse=True):
            name = characters.get(char_id, {}).get("name", f"Character {char_id}")
            secs = speaking[char_id]
            print(f"  {name}: {secs:.0f}s ({secs / 60:.1f}m)")

    timeline = get_dialogue_timeline(db)
    print(f"\nDialogue lines: {len(timeline)}")
    unattributed = sum(1 for line in timeline if line.character_id is None)
    print(f"  Attributed: {len(timeline) - unattributed}")
    print(f"  Unattributed: {unattributed}")
