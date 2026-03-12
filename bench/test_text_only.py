#!/usr/bin/env python3
"""Test text_only mode VRAM usage vs full model on sam-audio-small."""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam_audio import SAMAudio, SAMAudioProcessor

AUDIO_PATH = Path.home() / "chapter_2.mkv"
MODEL = "facebook/sam-audio-small"
PROMPT = "speech"
DEVICE = torch.device("cuda")


def load_audio(path, sr=48_000, max_seconds=0):
    wav, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    if max_seconds > 0:
        wav = wav[:, : int(max_seconds * sr)]
    return wav


def test_config(label, audio, text_only, candidates, max_chunk_tokens):
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"  audio: {audio.shape[-1]/48000:.1f}s, text_only={text_only}, "
          f"candidates={candidates}, chunks={max_chunk_tokens}")
    print(f"{'=' * 60}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Load
    t0 = time.perf_counter()
    kwargs = {"text_only": True} if text_only else {}
    model = SAMAudio.from_pretrained(MODEL, **kwargs).eval().to(DEVICE)
    processor = SAMAudioProcessor.from_pretrained(MODEL)
    torch.cuda.synchronize()
    load_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  After load: {load_mem:.2f} GB  ({time.perf_counter()-t0:.1f}s)")

    # Prepare batch
    batch = processor(descriptions=[PROMPT], audios=[audio]).to(DEVICE)

    # Run
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model.separate(
            batch, reranking_candidates=candidates, max_chunk_tokens=max_chunk_tokens
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    print(f"  Inference: {elapsed:.2f}s")
    print(f"  Peak VRAM: {peak_mem:.2f} GB")
    print(f"  Target shape: {result.target[0].shape}")

    del model, batch, result
    torch.cuda.empty_cache()
    return {"label": label, "peak_gb": peak_mem, "time_s": elapsed}


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    audio_30s = load_audio(str(AUDIO_PATH), max_seconds=30)
    audio_full = load_audio(str(AUDIO_PATH))
    print(f"Audio 30s: {audio_30s.shape}")
    print(f"Audio full: {audio_full.shape} ({audio_full.shape[-1]/48000:.1f}s)")

    results = []

    # 1. Full model, 30s audio (baseline)
    results.append(test_config(
        "FULL model - 30s", audio_30s,
        text_only=False, candidates=2, max_chunk_tokens=500,
    ))

    # 2. text_only, 30s audio
    results.append(test_config(
        "TEXT_ONLY - 30s", audio_30s,
        text_only=True, candidates=1, max_chunk_tokens=500,
    ))

    # 3. text_only, full audio (~203s)
    results.append(test_config(
        "TEXT_ONLY - full (~203s)", audio_full,
        text_only=True, candidates=1, max_chunk_tokens=500,
    ))

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Config':<30} {'Peak VRAM':>10} {'Time':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['label']:<30} {r['peak_gb']:>8.2f} GB {r['time_s']:>7.2f}s")

    # Would it fit in 16 GB?
    for r in results:
        fits = "YES" if r["peak_gb"] < 16.0 else "NO"
        print(f"  {r['label']}: fits 16GB? {fits} ({r['peak_gb']:.2f} GB)")


if __name__ == "__main__":
    main()
