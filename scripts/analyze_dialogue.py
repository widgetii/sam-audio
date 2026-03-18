"""Analyze dialogue detection results for quality investigation."""

import json
import sys


def analyze(path):
    d = json.load(open(path))
    source_dur = d["source"]["duration_seconds"]
    p = d["processing"]

    print(f"=== {path} ===")
    print(f"Source: {source_dur:.0f}s, window: {p['window_seconds']}s, overlap: {p['overlap_seconds']}s")
    print(f"Chunks: {p['num_chunks']}, multi-speaker: {p['num_multi_speaker_chunks']}")
    print(f"min_face_detections: {p.get('min_face_detections', 'N/A')}")
    print()

    # Characters
    print("Characters:")
    for c in d["characters"]:
        print(f"  ID {c['character_id']}: screen={c['total_screen_time_seconds']}s speaking={c['total_speaking_seconds']}s")
    print()

    # Raw segment sums (with overlap double-counting)
    raw_dialogue = 0
    raw_total = 0
    for chunk in d["chunks"]:
        for seg in chunk["segments"]:
            dur = seg["end_time"] - seg["start_time"]
            raw_total += dur
            if seg["has_dialogue"]:
                raw_dialogue += dur

    print(f"Raw segment sums (with overlap): {raw_dialogue:.0f}s dialogue / {raw_total:.0f}s total")
    print(f"Overlap inflation: {raw_total / source_dur:.2f}x")

    # Dedup by start_time (same as build_timeline - later chunk wins)
    all_segments = {}
    for chunk in d["chunks"]:
        for seg in chunk["segments"]:
            all_segments[seg["start_time"]] = seg

    deduped_dialogue = sum(
        seg["end_time"] - seg["start_time"]
        for seg in all_segments.values()
        if seg["has_dialogue"]
    )
    deduped_total = sum(
        seg["end_time"] - seg["start_time"]
        for seg in all_segments.values()
    )

    total_segs = sum(len(chunk["segments"]) for chunk in d["chunks"])
    unique_segs = len(all_segments)

    print(f"Deduped: {deduped_dialogue:.0f}s dialogue / {deduped_total:.0f}s total")
    print(f"Deduped dialogue %: {deduped_dialogue/source_dur*100:.1f}%")
    print(f"Segment entries: {total_segs}, unique timestamps: {unique_segs}, overwritten: {total_segs - unique_segs}")
    print()

    # Analyze overlap regions specifically
    # With 5s overlap, each chunk boundary has 5 seconds shared with the next
    # The dedup keeps the LATER chunk's version (dict overwrite)
    # Check: do earlier vs later chunks disagree on dialogue in overlaps?
    overlap_secs = p["overlap_seconds"]
    overlap_agrees = 0
    overlap_disagrees = 0
    overlap_lost_dialogue = 0  # earlier said yes, later said no
    overlap_gained_dialogue = 0  # earlier said no, later said yes

    chunks = d["chunks"]
    for i in range(len(chunks) - 1):
        curr_end = chunks[i]["end_time"]
        next_start = chunks[i + 1]["start_time"]
        overlap_start = next_start
        overlap_end = curr_end

        if overlap_end <= overlap_start:
            continue

        # Find segments in overlap from both chunks
        curr_segs = {
            seg["start_time"]: seg
            for seg in chunks[i]["segments"]
            if overlap_start <= seg["start_time"] < overlap_end
        }
        next_segs = {
            seg["start_time"]: seg
            for seg in chunks[i + 1]["segments"]
            if overlap_start <= seg["start_time"] < overlap_end
        }

        for t in curr_segs:
            if t in next_segs:
                c_dial = curr_segs[t]["has_dialogue"]
                n_dial = next_segs[t]["has_dialogue"]
                if c_dial == n_dial:
                    overlap_agrees += 1
                else:
                    overlap_disagrees += 1
                    if c_dial and not n_dial:
                        overlap_lost_dialogue += 1
                    elif not c_dial and n_dial:
                        overlap_gained_dialogue += 1

    print(f"Overlap region analysis:")
    print(f"  Segments in overlap: {overlap_agrees + overlap_disagrees}")
    print(f"  Agree: {overlap_agrees}, Disagree: {overlap_disagrees}")
    print(f"  Lost dialogue (earlier=yes, later=no): {overlap_lost_dialogue}")
    print(f"  Gained dialogue (earlier=no, later=yes): {overlap_gained_dialogue}")
    print()

    # Check which chunks were skipped by min_face_detections
    if "char_detection_counts" in chunks[0]:
        skipped_chunks = 0
        skipped_chars = 0
        min_det = p.get("min_face_detections", 3)
        for chunk in chunks:
            if not chunk.get("needs_visual_pass"):
                continue
            counts = chunk.get("char_detection_counts", {})
            eligible = [
                cid for cid in chunk["visible_characters"]
                if counts.get(cid, counts.get(str(cid), 0)) >= min_det
            ]
            total_vis = len(chunk["visible_characters"])
            skipped_in_chunk = total_vis - len(eligible)
            skipped_chars += skipped_in_chunk
            if len(eligible) < 2:
                skipped_chunks += 1

            # Show chunks that got fully skipped
            if len(eligible) < 2 and chunk.get("has_any_dialogue"):
                has_sep = "character_separation" in chunk
                print(f"  Skipped chunk {chunk['chunk_index']}: "
                      f"{chunk['start_time']:.0f}-{chunk['end_time']:.0f}s, "
                      f"{total_vis} visible chars, {len(eligible)} eligible, "
                      f"has_separation={has_sep}")

        print(f"\nMin face detection filter (threshold={min_det}):")
        print(f"  Skipped characters: {skipped_chars}")
        print(f"  Fully skipped chunks (< 2 eligible): {skipped_chunks}")

    # RMS threshold analysis - distribution of target_rms_db values
    rms_values = []
    for chunk in chunks:
        for seg in chunk["segments"]:
            rms_values.append(seg["target_rms_db"])

    rms_values.sort()
    print(f"\nRMS distribution (all segments):")
    print(f"  Min: {rms_values[0]:.1f}, Max: {rms_values[-1]:.1f}")
    print(f"  Median: {rms_values[len(rms_values)//2]:.1f}")
    # Count near threshold (-40 dB)
    near_threshold = sum(1 for r in rms_values if -45 <= r <= -35)
    above = sum(1 for r in rms_values if r > -40)
    below = sum(1 for r in rms_values if r <= -40)
    print(f"  Above -40dB (dialogue): {above} ({above/len(rms_values)*100:.1f}%)")
    print(f"  Below -40dB (silence): {below} ({below/len(rms_values)*100:.1f}%)")
    print(f"  Near threshold (-45 to -35): {near_threshold} ({near_threshold/len(rms_values)*100:.1f}%)")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        analyze(path)
