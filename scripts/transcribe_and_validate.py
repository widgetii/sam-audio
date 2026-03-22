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


def attribute_words_to_characters(words: list[dict], per_character: dict) -> list[dict]:
    """Assign each transcribed word to a character based on pipeline segments.

    per_character: {char_id_str: {"segments": [[start, end], ...]}}
    """
    # Build flat list of (start, end, char_id)
    char_intervals = []
    for char_id_str, data in per_character.items():
        char_id = int(char_id_str)
        for seg in data.get("segments", []):
            char_intervals.append((seg[0], seg[1], char_id))
    char_intervals.sort()

    for word in words:
        mid = (word["start"] + word["end"]) / 2
        word["character_id"] = None
        for seg_start, seg_end, char_id in char_intervals:
            if seg_start <= mid <= seg_end:
                word["character_id"] = char_id
                break

    return words


def build_character_lines(words: list[dict]) -> list[dict]:
    """Group consecutive words by character into dialogue lines."""
    lines = []
    current = None

    for word in words:
        cid = word.get("character_id")
        if cid is None:
            if current:
                lines.append(current)
                current = None
            continue

        if (
            current
            and current["character_id"] == cid
            and word["start"] - current["end"] < 1.0
        ):
            current["text"] += " " + word["text"]
            current["end"] = word["end"]
            current["words"].append(word)
        else:
            if current:
                lines.append(current)
            current = {
                "character_id": cid,
                "start": word["start"],
                "end": word["end"],
                "text": word["text"],
                "words": [word],
            }

    if current:
        lines.append(current)

    # Clean up — drop word details from output
    for line in lines:
        del line["words"]
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
    characters = pipeline.get("characters", [])
    logger.info(
        f"Loaded pipeline results: {len(characters)} characters, "
        f"{sum(len(v.get('segments', [])) for v in per_character.values())} speaking segments"
    )

    # Extract subtitles
    logger.info(f"Extracting English subtitles (stream {args.subtitle_stream})")
    subtitles = extract_subtitles(args.input, args.subtitle_stream)
    logger.info(f"Extracted {len(subtitles)} subtitle lines")

    # Transcribe
    words = transcribe_audio(args.input, args.audio_stream, args.whisper_model)

    # Attribute words to characters
    words = attribute_words_to_characters(words, per_character)
    attributed = sum(1 for w in words if w["character_id"] is not None)
    logger.info(f"Attributed {attributed}/{len(words)} words to characters")

    # Build character lines
    character_lines = build_character_lines(words)
    logger.info(f"Built {len(character_lines)} character dialogue lines")

    # Per-character summary
    from collections import Counter

    char_line_counts = Counter(ln["character_id"] for ln in character_lines)
    for cid, count in char_line_counts.most_common():
        char_words = sum(
            len(ln["text"].split())
            for ln in character_lines
            if ln["character_id"] == cid
        )
        logger.info(f"  Character {cid}: {count} lines, {char_words} words")

    # Cross-validate against subtitles
    logger.info("Cross-validating against subtitles...")
    validation = cross_validate(character_lines, subtitles)
    logger.info(
        f"Subtitle recall: {validation['recall']}% "
        f"({validation['matched']}/{validation['total_subtitles']})"
    )

    # Build output
    output = {
        "source": pipeline.get("source", {}),
        "characters": characters,
        "character_lines": character_lines,
        "subtitles": {"count": len(subtitles), "stream_index": args.subtitle_stream},
        "transcription": {
            "model": args.whisper_model,
            "total_words": len(words),
            "attributed_words": attributed,
        },
        "validation": validation,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote results to {args.output}")

    # Print sample matches
    print("\n=== Sample matches (subtitle → transcription) ===")
    for m in validation["matches_sample"][:15]:
        print(
            f"  [{m['sub_time']}] char {m['character_id']}: "
            f'"{m["subtitle"]}" → "{m["transcribed"]}"'
        )

    if validation["unmatched_sample"]:
        print(f"\n=== Unmatched subtitles ({validation['unmatched']}) ===")
        for u in validation["unmatched_sample"][:10]:
            print(f'  [{u["time"]}] {u.get("speaker", "?")}: "{u["text"]}"')


if __name__ == "__main__":
    main()
