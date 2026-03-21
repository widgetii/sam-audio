"""
Extract subtitle text at filmstrip timestamps using Whisper.
Uses the same audio track as the pipeline (track 0 = Russian MVO).

Run on GPU machine: python3 docs/extract_subtitles.py
Output: docs/real_frames/subtitles.json
"""

import json
import os
import subprocess
import tempfile
import whisper

VIDEO_PATH = "/data/huggingface/Aliens.1986.mkv"
# Filmstrip timestamps (same as in extract_real_frames.py)
# Filmstrip timestamps within the timeline window (80:00-90:00)
TIMESTAMPS = [4800, 4840, 4880, 4920, 4960, 5000, 5040, 5080,
              5120, 5160, 5200, 5240, 5280, 5320, 5360, 5400]

# Audio track 0 = Russian (same as pipeline uses via torchaudio.load)
AUDIO_TRACK = 0
CLIP_DURATION = 10  # seconds around each timestamp

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_frames")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model("medium")

results = {}
tmpdir = tempfile.mkdtemp()

for ts in TIMESTAMPS:
    start = max(0, ts - 3)
    clip_path = os.path.join(tmpdir, f"clip_{ts}.wav")

    # Extract audio clip using ffmpeg (track 0 = first audio = Russian)
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start), "-t", str(CLIP_DURATION),
        "-i", VIDEO_PATH, "-map", f"0:a:{AUDIO_TRACK}",
        "-ac", "1", "-ar", "16000", clip_path,
    ], capture_output=True)

    if not os.path.exists(clip_path):
        print(f"  t={ts}s: no audio extracted")
        results[str(ts)] = {"text": "", "language": "unknown"}
        continue

    # Transcribe
    result = model.transcribe(clip_path, language=None)
    text = result["text"].strip()
    lang = result.get("language", "unknown")

    # Filter Whisper hallucinations (common on silent/ambient segments)
    HALLUCINATIONS = ["thanks for watching", "subscribe", "thank you for watching",
                      "like and subscribe", "see you next time"]
    if any(h in text.lower() for h in HALLUCINATIONS):
        print(f"  t={ts}s ({ts//60}:{ts%60:02d}): [{lang}] HALLUCINATION filtered: {text[:80]}")
        text = ""

    print(f"  t={ts}s ({ts//60}:{ts%60:02d}): [{lang}] {text[:80]}")
    results[str(ts)] = {"text": text, "language": lang}

    os.remove(clip_path)

# Save
out_path = os.path.join(OUT_DIR, "subtitles.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
