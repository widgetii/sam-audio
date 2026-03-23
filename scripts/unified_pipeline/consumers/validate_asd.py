"""Validate auto-framing with vs without ASD scores.

Compares crop positions computed from the same DB with speaker_scores present
(TalkNet ASD) vs absent, showing how the main-subject selection changes.
"""

import sys

from unified_pipeline.consumers.auto_framing import (
    choose_main_person,
    interpolate_tracks_to_fps,
    query_person_data,
)
from unified_pipeline.db import AnalysisDB


def compare_main_subjects(
    db: AnalysisDB,
    start_sec: float,
    end_sec: float,
    video_width: int = 1920,
    video_height: int = 1080,
):
    """Compare main subject selection with/without ASD scores."""
    persons = query_person_data(db, start_sec, end_sec)

    # Count how many have non-zero ASD scores
    with_asd = sum(1 for p in persons if p.asd_score != 0.0)
    print(f"Persons in range: {len(persons)}, with ASD scores: {with_asd}")

    if with_asd == 0:
        print("No ASD scores found — run Stage 4 first")
        return

    interp = interpolate_tracks_to_fps(
        persons, 1.0, start_sec, end_sec, video_width, video_height
    )

    same = 0
    different = 0
    asd_wins = 0
    total = 0

    for t in sorted(interp.keys()):
        people = interp[t]
        if not people:
            continue

        # With ASD (normal)
        main_asd = choose_main_person(people)

        # Without ASD (zero out scores)
        for p in people:
            p.asd_score = 0.0
        main_no_asd = choose_main_person(people)
        # Restore (not needed since we're done with this timestamp)

        total += 1
        if main_asd and main_no_asd:
            if main_asd.sam3_obj_id == main_no_asd.sam3_obj_id:
                same += 1
            else:
                different += 1
                asd_cx = (main_asd.bbox[0] + main_asd.bbox[2]) / 2
                no_asd_cx = (main_no_asd.bbox[0] + main_no_asd.bbox[2]) / 2
                shift = abs(asd_cx - no_asd_cx)
                if shift > 100:
                    asd_wins += 1
                    print(
                        f"  t={t:.1f}s: ASD picks obj {main_asd.sam3_obj_id} "
                        f"(cx={asd_cx:.0f}, asd={main_asd.asd_score:.2f}) "
                        f"vs no-ASD obj {main_no_asd.sam3_obj_id} "
                        f"(cx={no_asd_cx:.0f}) — shift {shift:.0f}px"
                    )

    print(f"\nTotal 1fps frames: {total}")
    print(f"Same subject: {same} ({100 * same / total:.1f}%)")
    print(f"Different subject: {different} ({100 * different / total:.1f}%)")
    print(f"Significant shifts (>100px): {asd_wins}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python -m unified_pipeline.consumers.validate_asd <db_path> <start_sec> <end_sec>"
        )
        sys.exit(1)

    db = AnalysisDB(sys.argv[1])
    start = float(sys.argv[2])
    end = float(sys.argv[3])
    compare_main_subjects(db, start, end)
    db.close()
