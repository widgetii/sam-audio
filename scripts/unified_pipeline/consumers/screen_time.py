"""Consumer 6C: Screen time — DB → per-character presence + co-occurrence matrix."""

import logging
from collections import defaultdict

import numpy as np

from unified_pipeline.db import AnalysisDB

log = logging.getLogger(__name__)


def get_screen_time(db: AnalysisDB) -> dict[int, float]:
    """Get total screen time per character in seconds (from 1fps person tracks)."""
    rows = db.conn.execute("""
        SELECT ti.character_id, COUNT(DISTINCT pt.frame_sec) as seconds
        FROM person_tracks pt
        JOIN track_identities ti ON pt.shot_id = ti.shot_id AND pt.sam3_obj_id = ti.sam3_obj_id
        WHERE ti.character_id IS NOT NULL
        GROUP BY ti.character_id
    """).fetchall()
    return {r["character_id"]: float(r["seconds"]) for r in rows}


def get_presence_timeline(
    db: AnalysisDB, character_id: int
) -> list[tuple[float, float]]:
    """Get time intervals where character is on screen.

    Returns list of (start_sec, end_sec) merged intervals.
    """
    rows = db.conn.execute(
        """
        SELECT DISTINCT pt.frame_sec
        FROM person_tracks pt
        JOIN track_identities ti ON pt.shot_id = ti.shot_id AND pt.sam3_obj_id = ti.sam3_obj_id
        WHERE ti.character_id = ?
        ORDER BY pt.frame_sec
    """,
        (character_id,),
    ).fetchall()

    if not rows:
        return []

    timestamps = [r["frame_sec"] for r in rows]

    # Merge consecutive seconds (gap <= 2s) into intervals
    intervals = []
    start = timestamps[0]
    end = timestamps[0]

    for t in timestamps[1:]:
        if t - end <= 2.0:
            end = t
        else:
            intervals.append((start, end + 1.0))
            start = t
            end = t

    intervals.append((start, end + 1.0))
    return intervals


def get_co_occurrence_matrix(db: AnalysisDB) -> tuple[list[int], np.ndarray]:
    """Compute co-occurrence matrix: how many seconds two characters share screen.

    Returns (character_ids, matrix) where matrix[i][j] = shared seconds.
    """
    # Get all (frame_sec, character_id) pairs
    rows = db.conn.execute("""
        SELECT DISTINCT pt.frame_sec, ti.character_id
        FROM person_tracks pt
        JOIN track_identities ti ON pt.shot_id = ti.shot_id AND pt.sam3_obj_id = ti.sam3_obj_id
        WHERE ti.character_id IS NOT NULL
        ORDER BY pt.frame_sec
    """).fetchall()

    # Group by frame_sec
    by_sec: dict[float, set[int]] = defaultdict(set)
    char_set: set[int] = set()
    for r in rows:
        by_sec[r["frame_sec"]].add(r["character_id"])
        char_set.add(r["character_id"])

    char_ids = sorted(char_set)
    idx = {c: i for i, c in enumerate(char_ids)}
    n = len(char_ids)
    matrix = np.zeros((n, n), dtype=np.float64)

    for chars in by_sec.values():
        chars_list = sorted(chars)
        for i_c in chars_list:
            for j_c in chars_list:
                matrix[idx[i_c]][idx[j_c]] += 1.0

    return char_ids, matrix


def print_screen_time_summary(db: AnalysisDB):
    """Print a human-readable screen time summary."""
    screen_time = get_screen_time(db)
    characters = {c["character_id"]: c for c in db.get_characters()}

    total_duration = (
        db.conn.execute("SELECT MAX(end_sec) FROM shots").fetchone()[0] or 0
    )

    print(f"\nScreen Time Summary (total: {total_duration / 60:.1f} min):")

    for char_id in sorted(screen_time, key=screen_time.get, reverse=True):
        name = characters.get(char_id, {}).get("name", f"Character {char_id}")
        secs = screen_time[char_id]
        pct = 100 * secs / max(total_duration, 1)
        print(f"  {name}: {secs:.0f}s ({secs / 60:.1f}m, {pct:.1f}%)")

    # Co-occurrence
    char_ids, matrix = get_co_occurrence_matrix(db)
    if len(char_ids) > 1:
        print("\nTop co-occurrences:")
        pairs = []
        for i in range(len(char_ids)):
            for j in range(i + 1, len(char_ids)):
                pairs.append((char_ids[i], char_ids[j], matrix[i][j]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        for c1, c2, secs in pairs[:10]:
            n1 = characters.get(c1, {}).get("name", f"Char {c1}")
            n2 = characters.get(c2, {}).get("name", f"Char {c2}")
            print(f"  {n1} + {n2}: {secs:.0f}s ({secs / 60:.1f}m)")
