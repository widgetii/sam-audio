#!/usr/bin/env python3
"""Simulate 6 GB and 16 GB VRAM limits on A100 to verify text_only fits consumer GPUs."""

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


def load_audio(path, sr=48_000, max_seconds=0):
    wav, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    if max_seconds > 0:
        wav = wav[:, : int(max_seconds * sr)]
    return wav


def run_with_limit(label, audio, vram_limit_gb):
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"  VRAM limit: {vram_limit_gb} GB, audio: {audio.shape[-1]/48000:.1f}s")
    print(f"{'=' * 60}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Set memory fraction to simulate limited GPU
    total_mem = torch.cuda.get_device_properties(0).total_memory
    fraction = (vram_limit_gb * 1024**3) / total_mem
    fraction = min(fraction, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    effective_gb = total_mem * fraction / (1024**3)
    print(f"  Memory fraction: {fraction:.4f} ({effective_gb:.2f} GB effective)")

    try:
        # Load model
        t0 = time.perf_counter()
        model = SAMAudio.from_pretrained(MODEL, text_only=True).eval().to("cuda")
        processor = SAMAudioProcessor.from_pretrained(MODEL)
        torch.cuda.synchronize()
        load_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  After load: {load_mem:.2f} GB ({time.perf_counter()-t0:.1f}s)")

        # Run
        batch = processor(descriptions=[PROMPT], audios=[audio]).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.separate(batch, reranking_candidates=1, max_chunk_tokens=500)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / (1024**3)

        print(f"  Inference: {elapsed:.2f}s")
        print(f"  Peak VRAM: {peak:.2f} GB")
        print(f"  Target shape: {result.target[0].shape}")
        print(f"  RESULT: SUCCESS within {vram_limit_gb} GB limit")

        del model, batch, result

    except torch.cuda.OutOfMemoryError as e:
        print(f"  RESULT: OOM — does NOT fit in {vram_limit_gb} GB")
        print(f"  Error: {e}")

    finally:
        torch.cuda.empty_cache()
        # Reset to full memory for next test
        torch.cuda.set_per_process_memory_fraction(1.0, device=0)


def main():
    assert torch.cuda.is_available()
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total:.1f} GB)")

    audio_30s = load_audio(str(AUDIO_PATH), max_seconds=30)
    audio_full = load_audio(str(AUDIO_PATH))
    print(f"Audio 30s: {audio_30s.shape}")
    print(f"Audio full: {audio_full.shape} ({audio_full.shape[-1]/48000:.1f}s)")

    # Test 1: 8 GB limit, 30s — should fit (measured 6.97 GB)
    run_with_limit("text_only 30s @ 8 GB limit", audio_30s, 8.0)

    # Test 2: 7 GB limit, 30s — razor thin (measured 6.97 GB)
    run_with_limit("text_only 30s @ 7 GB limit", audio_30s, 7.0)

    # Test 3: 16 GB limit, 203s — should be close (measured 17.89 GB)
    run_with_limit("text_only 203s @ 16 GB limit", audio_full, 16.0)

    # Test 4: 20 GB limit, 203s — should fit
    run_with_limit("text_only 203s @ 20 GB limit", audio_full, 20.0)


if __name__ == "__main__":
    main()
