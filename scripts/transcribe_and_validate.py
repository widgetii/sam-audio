"""Transcribe character dialogue and cross-validate against subtitles.

Uses Whisper for word-level transcription of the English audio track,
attributes words to characters via the pipeline's per-character speaking
segments, then compares against English subtitles from the MKV.

Usage:
    uv run python scripts/transcribe_and_validate.py \
        --input /data/huggingface/Aliens.1080p.mkv \
        --pipeline-json workspace/v2/dialogue_v2_unified.json \
        --audio-stream 6 --subtitle-stream 9 \
        --output workspace/v2/transcription.json
"""

import argparse
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


# --- Subtitle extraction ---


def extract_subtitles(video_path: str, stream_index: int) -> list[dict]:
    """Extract subtitles from MKV as list of {start, end, text}."""
    tmp = tempfile.NamedTemporaryFile(suffix=".ass", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-map",
            f"0:{stream_index}",
            tmp.name,
        ],
        capture_output=True,
        check=True,
    )
    subs = parse_ass(tmp.name)
    Path(tmp.name).unlink()
    return subs


def parse_ass(path: str) -> list[dict]:
    """Parse ASS subtitle file into list of {start, end, text, speaker}."""
    subs = []
    in_events = False
    format_fields = None

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[Events]"):
                in_events = True
                continue
            if not in_events:
                continue
            if line.startswith("Format:"):
                format_fields = [f.strip() for f in line.split(":", 1)[1].split(",")]
                continue
            if not line.startswith("Dialogue:"):
                continue

            parts = line.split(":", 1)[1].split(",", len(format_fields) - 1)
            if len(parts) < len(format_fields):
                continue

            field_map = dict(zip(format_fields, parts, strict=True))
            start = ass_time_to_sec(field_map.get("Start", "").strip())
            end = ass_time_to_sec(field_map.get("End", "").strip())
            raw_text = field_map.get("Text", "").strip()
            # Strip ASS formatting tags
            text = re.sub(r"\{[^}]*\}", "", raw_text)
            text = text.replace("\\N", " ").replace("\\n", " ").strip()
            if not text:
                continue

            speaker = field_map.get("Name", "").strip() or None
            subs.append({"start": start, "end": end, "text": text, "speaker": speaker})

    subs.sort(key=lambda s: s["start"])
    return subs


def ass_time_to_sec(t: str) -> float:
    """Convert ASS timestamp 'H:MM:SS.cc' to seconds."""
    try:
        parts = t.split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return 0.0


# --- Whisper transcription ---


def transcribe_audio(
    video_path: str, audio_stream: int, model_size: str = "base"
) -> list[dict]:
    """Transcribe English audio with word-level timestamps."""
    # Extract audio to temp WAV
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-map",
            f"0:{audio_stream}",
            "-ac",
            "1",
            "-ar",
            "16000",
            tmp.name,
        ],
        capture_output=True,
        check=True,
    )

    logger.info(f"Transcribing with faster-whisper ({model_size})...")
    from faster_whisper import WhisperModel

    whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = whisper_model.transcribe(
        tmp.name,
        language="en",
        word_timestamps=True,
        vad_filter=True,
    )

    words = []
    for segment in segments:
        for word in segment.words:
            words.append(
                {
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "text": word.word.strip(),
                    "probability": round(word.probability, 3),
                }
            )

    del whisper_model
    Path(tmp.name).unlink()
    logger.info(f"Transcribed {len(words)} words")
    return words


# --- Attribution ---


def attribute_words(
    words: list[dict],
    per_character: dict,
    dialogue_timeline: list[list[float]],
) -> list[dict]:
    """Assign each word a character_id (if in per-character segments) and
    an in_dialogue flag (if in the full dialogue timeline).

    per_character: {char_id_str: {"segments": [[start, end], ...]}}
    dialogue_timeline: [[start, end], ...]
    """
    # Build character intervals
    char_intervals = []
    for char_id_str, data in per_character.items():
        char_id = int(char_id_str)
        for seg in data.get("segments", []):
            char_intervals.append((seg[0], seg[1], char_id))
    char_intervals.sort()

    # Sort dialogue timeline
    dial_intervals = sorted(dialogue_timeline)

    for word in words:
        mid = (word["start"] + word["end"]) / 2
        # Character attribution (multi-speaker chunks only)
        word["character_id"] = None
        for seg_start, seg_end, char_id in char_intervals:
            if seg_start <= mid <= seg_end:
                word["character_id"] = char_id
                break
        # Full dialogue timeline (all detected dialogue)
        word["in_dialogue"] = False
        for seg_start, seg_end in dial_intervals:
            if seg_start <= mid <= seg_end:
                word["in_dialogue"] = True
                break
            if seg_start > mid:
                break

    return words


def build_dialogue_lines(words: list[dict]) -> list[dict]:
    """Group consecutive words into dialogue lines.

    Uses in_dialogue to include all detected dialogue, character_id for
    attribution when available (multi-speaker chunks).
    """
    lines = []
    current = None

    for word in words:
        if not word.get("in_dialogue") and word.get("character_id") is None:
            if current:
                lines.append(current)
                current = None
            continue

        cid = word.get("character_id")  # may be None for single-speaker
        gap = word["start"] - current["end"] if current else 999

        # Break line on character change or gap > 1s
        if current and (current["character_id"] != cid or gap > 1.0):
            lines.append(current)
            current = None

        if current is None:
            current = {
                "character_id": cid,
                "start": word["start"],
                "end": word["end"],
                "text": word["text"],
            }
        else:
            current["text"] += " " + word["text"]
            current["end"] = word["end"]

    if current:
        lines.append(current)

    for line in lines:
        line["text"] = line["text"].strip()

    return lines


# --- Cross-validation ---


def cross_validate(
    character_lines: list[dict], subtitles: list[dict], tolerance: float = 2.0
) -> dict:
    """Compare character lines against subtitles.

    For each subtitle, find the best-matching character line within tolerance.
    """
    matched_subs = 0
    unmatched_subs = []
    sub_matches = []

    for sub in subtitles:
        sub_mid = (sub["start"] + sub["end"]) / 2
        best_line = None
        best_overlap = 0

        for line in character_lines:
            # Check temporal overlap
            overlap_start = max(sub["start"], line["start"])
            overlap_end = min(sub["end"], line["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap or (
                overlap == 0
                and best_line is None
                and abs(sub_mid - (line["start"] + line["end"]) / 2) < tolerance
            ):
                best_line = line
                best_overlap = overlap

        if best_line and best_overlap > 0:
            matched_subs += 1
            sub_matches.append(
                {
                    "subtitle": sub["text"],
                    "sub_time": f"{sub['start']:.1f}-{sub['end']:.1f}",
                    "transcribed": best_line["text"],
                    "trans_time": f"{best_line['start']:.1f}-{best_line['end']:.1f}",
                    "character_id": best_line["character_id"],
                    "sub_speaker": sub.get("speaker"),
                }
            )
        else:
            unmatched_subs.append(
                {
                    "text": sub["text"],
                    "time": f"{sub['start']:.1f}-{sub['end']:.1f}",
                    "speaker": sub.get("speaker"),
                }
            )

    return {
        "total_subtitles": len(subtitles),
        "matched": matched_subs,
        "unmatched": len(unmatched_subs),
        "recall": round(matched_subs / max(len(subtitles), 1) * 100, 1),
        "matches_sample": sub_matches[:30],
        "unmatched_sample": unmatched_subs[:20],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe character dialogue and validate against subtitles"
    )
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument(
        "--pipeline-json", required=True, help="Pipeline output JSON from Stage 3/4"
    )
    parser.add_argument("--audio-stream", type=int, required=True)
    parser.add_argument("--subtitle-stream", type=int, required=True)
    parser.add_argument(
        "--whisper-model", default="base", help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--output", default="transcription.json", help="Output JSON path"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # Load pipeline results
    with open(args.pipeline_json) as f:
        pipeline = json.load(f)

    per_character = pipeline.get("per_character", {})
    dialogue_timeline = pipeline.get("dialogue_timeline", [])
    characters = pipeline.get("characters", [])
    logger.info(
        f"Loaded pipeline results: {len(characters)} characters, "
        f"{sum(len(v.get('segments', [])) for v in per_character.values())} character segments, "
        f"{len(dialogue_timeline)} dialogue segments"
    )

    # Extract subtitles
    logger.info(f"Extracting English subtitles (stream {args.subtitle_stream})")
    subtitles = extract_subtitles(args.input, args.subtitle_stream)
    logger.info(f"Extracted {len(subtitles)} subtitle lines")

    # Transcribe
    words = transcribe_audio(args.input, args.audio_stream, args.whisper_model)

    # Attribute words to characters + full dialogue timeline
    words = attribute_words(words, per_character, dialogue_timeline)
    char_attributed = sum(1 for w in words if w["character_id"] is not None)
    dial_attributed = sum(1 for w in words if w["in_dialogue"])
    logger.info(
        f"Words: {len(words)} total, {dial_attributed} in dialogue, "
        f"{char_attributed} attributed to characters"
    )

    # Build all dialogue lines (character when known, None for single-speaker)
    all_lines = build_dialogue_lines(words)
    char_lines = [ln for ln in all_lines if ln["character_id"] is not None]
    unattr_lines = [ln for ln in all_lines if ln["character_id"] is None]
    logger.info(
        f"Dialogue lines: {len(all_lines)} total, "
        f"{len(char_lines)} with character, {len(unattr_lines)} unattributed"
    )

    # Per-character summary
    from collections import Counter

    char_line_counts = Counter(ln["character_id"] for ln in char_lines)
    for cid, count in char_line_counts.most_common():
        char_words = sum(
            len(ln["text"].split()) for ln in char_lines if ln["character_id"] == cid
        )
        logger.info(f"  Character {cid}: {count} lines, {char_words} words")

    # Cross-validate: per-character lines only
    logger.info("Cross-validating character lines against subtitles...")
    char_validation = cross_validate(char_lines, subtitles)
    logger.info(
        f"Character recall: {char_validation['recall']}% "
        f"({char_validation['matched']}/{char_validation['total_subtitles']})"
    )

    # Cross-validate: ALL dialogue lines (character + unattributed)
    logger.info("Cross-validating ALL dialogue lines against subtitles...")
    full_validation = cross_validate(all_lines, subtitles)
    logger.info(
        f"Full dialogue recall: {full_validation['recall']}% "
        f"({full_validation['matched']}/{full_validation['total_subtitles']})"
    )

    # Build output
    output = {
        "source": pipeline.get("source", {}),
        "characters": characters,
        "dialogue_lines": all_lines,
        "subtitles": {"count": len(subtitles), "stream_index": args.subtitle_stream},
        "transcription": {
            "model": args.whisper_model,
            "total_words": len(words),
            "in_dialogue": dial_attributed,
            "with_character": char_attributed,
        },
        "validation": {
            "character_only": char_validation,
            "full_dialogue": full_validation,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote results to {args.output}")

    # Print results
    print("\n=== Recall ===")
    print(f"  Character lines vs subtitles: {char_validation['recall']}%")
    print(f"  All dialogue vs subtitles:    {full_validation['recall']}%")

    print("\n=== Sample matches (all dialogue) ===")
    for m in full_validation["matches_sample"][:20]:
        cid = m["character_id"]
        label = f"char {cid}" if cid is not None else "unknown"
        print(
            f"  [{m['sub_time']}] {label}: "
            f'"{m["subtitle"][:60]}" → "{m["transcribed"][:60]}"'
        )

    if full_validation["unmatched_sample"]:
        print(f"\n=== Unmatched subtitles ({full_validation['unmatched']}) ===")
        for u in full_validation["unmatched_sample"][:10]:
            print(f'  [{u["time"]}] {u.get("speaker", "?")}: "{u["text"][:60]}"')


if __name__ == "__main__":
    main()
