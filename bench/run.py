#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Benchmark all SAM-Audio model variants: latency, memory, and quality metrics."""

import json
import sys
import time
from pathlib import Path

import torch
import torchaudio

# Add project root and eval/ to path so we can import metrics
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from sam_audio import SAMAudio, SAMAudioProcessor

MODELS = [
    "facebook/sam-audio-small",
    "facebook/sam-audio-base",
    "facebook/sam-audio-large",
    "facebook/sam-audio-small-tv",
    "facebook/sam-audio-base-tv",
    "facebook/sam-audio-large-tv",
]

AUDIO_PATH = Path.home() / "chapter_2.mkv"
PROMPT = "speech"
RERANKING_CANDIDATES = 2
NUM_TIMED_RUNS = 3
MAX_DURATION_S = 30  # Trim audio to avoid OOM on large models
DEVICE = torch.device("cuda")
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

# Reduce fragmentation on large allocations
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_audio(path: str, sr: int = 48_000, max_seconds: float = 0) -> torch.Tensor:
    wav, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    if max_seconds > 0:
        max_samples = int(max_seconds * sr)
        wav = wav[:, :max_samples]
    return wav


def timed_separate(model, batch, candidates):
    """Run model.separate() with accurate GPU timing. Returns elapsed seconds."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = model.separate(batch, reranking_candidates=candidates)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0


def benchmark_model(model_name: str, audio: torch.Tensor):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Load model + processor
    t0 = time.perf_counter()
    model = SAMAudio.from_pretrained(model_name).eval().to(DEVICE)
    processor = SAMAudioProcessor.from_pretrained(model_name)
    torch.cuda.synchronize()
    load_time = time.perf_counter() - t0
    print(f"  Load time: {load_time:.2f}s")

    # Prepare batch
    batch = processor(descriptions=[PROMPT], audios=[audio])
    batch = batch.to(DEVICE)

    # Warm-up run (discarded)
    print("  Warm-up run...")
    torch.cuda.reset_peak_memory_stats()
    result, warmup_time = timed_separate(model, batch, RERANKING_CANDIDATES)
    print(f"  Warm-up time: {warmup_time:.2f}s")

    # Timed runs
    times = []
    for i in range(NUM_TIMED_RUNS):
        _, elapsed = timed_separate(model, batch, RERANKING_CANDIDATES)
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

    target_shape = list(result.target[0].shape)
    residual_shape = list(result.residual[0].shape)

    print(f"  Mean inference: {mean_time:.2f}s +/- {std_time:.3f}s")
    print(f"  Peak GPU memory: {peak_mem_gb:.2f} GB")
    print(f"  Target shape: {target_shape}, Residual shape: {residual_shape}")

    record = {
        "model": model_name,
        "load_time_s": round(load_time, 3),
        "warmup_time_s": round(warmup_time, 3),
        "inference_mean_s": round(mean_time, 3),
        "inference_std_s": round(std_time, 4),
        "inference_runs_s": [round(t, 3) for t in times],
        "peak_gpu_memory_gb": round(peak_mem_gb, 3),
        "target_shape": target_shape,
        "residual_shape": residual_shape,
    }

    # Keep result and input audio on CPU for metrics
    target_wav = result.target[0].cpu()
    input_wav = audio.mean(0) if audio.ndim > 1 else audio

    # Free GPU memory before metrics
    del model, batch, result
    torch.cuda.empty_cache()

    return record, target_wav, input_wav


def compute_quality_metrics(
    records: list[dict],
    target_wavs: list[torch.Tensor],
    input_wavs: list[torch.Tensor],
):
    """Compute CLAP, Judge, and Aesthetic metrics for all models."""
    from metrics import Aesthetic, Judge

    print(f"\n{'='*60}")
    print("Computing quality metrics...")
    print(f"{'='*60}")

    device = DEVICE

    valid = [(i, rec) for i, rec in enumerate(records) if target_wavs[i] is not None]
    if not valid:
        print("  No successful runs to compute metrics for.")
        return

    # CLAP - construct directly to avoid bug in CLAP.__init__ (device passed as checkpoint)
    print("  Loading CLAP metric...")
    from sam_audio.ranking.clap import get_model as get_clap_model
    from tempfile import TemporaryDirectory
    from torchcodec.encoders import AudioEncoder

    clap_model = get_clap_model(device=str(device))
    for i, rec in valid:
        with TemporaryDirectory() as tdir, torch.inference_mode():
            wav = target_wavs[i]
            fpath = f"{tdir}/hyp.wav"
            encoder = AudioEncoder(
                samples=wav.cpu()[None] if wav.ndim == 1 else wav.cpu(),
                sample_rate=48_000,
            )
            encoder.to_file(fpath)
            audio_embs = clap_model.get_audio_embedding_from_filelist([fpath], use_tensor=True)
            text_embs = clap_model.get_text_embedding([PROMPT], use_tensor=True)
            sim = (audio_embs @ text_embs.T)[0, 0].item()
        rec["clap_similarity"] = round(sim, 4)
        print(f"  {rec['model']}: CLAP={rec['clap_similarity']:.4f}")
    del clap_model
    torch.cuda.empty_cache()

    # Judge
    print("  Loading Judge metric...")
    judge = Judge(device=device)
    for i, rec in valid:
        judge_result = judge(
            input_wavs=[input_wavs[i]],
            target_wavs=[target_wavs[i]],
            descriptions=[PROMPT],
        )
        rec["judge_overall"] = round(judge_result["JudgeOverall"][0], 4)
        rec["judge_precision"] = round(judge_result["JudgePrecision"][0], 4)
        rec["judge_recall"] = round(judge_result["JudgeRecall"][0], 4)
        rec["judge_faithfulness"] = round(judge_result["JudgeFaithfulness"][0], 4)
        print(
            f"  {rec['model']}: Judge={rec['judge_overall']:.4f} "
            f"(P={rec['judge_precision']:.4f} R={rec['judge_recall']:.4f} F={rec['judge_faithfulness']:.4f})"
        )
    del judge
    torch.cuda.empty_cache()

    # Aesthetic
    print("  Loading Aesthetic metric...")
    aes = Aesthetic(device=device)
    for i, rec in valid:
        aes_result = aes(target_wavs=[target_wavs[i]])
        rec["aesthetic_pq"] = round(aes_result["ProductionQuality"][0], 4)
        rec["aesthetic_ce"] = round(aes_result["ContentEnjoyment"][0], 4)
        print(
            f"  {rec['model']}: AesPQ={rec['aesthetic_pq']:.4f} AesCE={rec['aesthetic_ce']:.4f}"
        )
    del aes
    torch.cuda.empty_cache()


def print_summary_table(records):
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    # Header
    header = (
        f"{'Model':<30} {'Load':>5} {'Infer':>7} {'Mem':>6} "
        f"{'CLAP':>6} {'Judge':>6} {'AesPQ':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in records:
        short_name = r["model"].replace("facebook/sam-audio-", "")
        if "error" in r:
            print(f"{short_name:<30}   {'OOM':>50}")
            continue
        infer_str = f"{r['inference_mean_s']:.2f}s"
        load_str = f"{r['load_time_s']:.1f}s"
        mem_str = f"{r['peak_gpu_memory_gb']:.1f}GB"
        clap_str = f"{r.get('clap_similarity', 0):.3f}"
        judge_str = f"{r.get('judge_overall', 0):.3f}"
        aes_str = f"{r.get('aesthetic_pq', 0):.3f}"
        print(
            f"{short_name:<30} {load_str:>5} {infer_str:>7} {mem_str:>6} "
            f"{clap_str:>6} {judge_str:>6} {aes_str:>6}"
        )


def main():
    assert torch.cuda.is_available(), "CUDA required for benchmarking"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Audio: {AUDIO_PATH}")
    print(f"Prompt: '{PROMPT}'")
    print(f"Reranking candidates: {RERANKING_CANDIDATES}")
    print(f"Timed runs per model: {NUM_TIMED_RUNS}")

    # Load audio once, trimmed to MAX_DURATION_S
    audio = load_audio(str(AUDIO_PATH), max_seconds=MAX_DURATION_S)
    print(f"Audio shape: {list(audio.shape)}, duration: {audio.shape[-1]/48000:.1f}s")

    records = []
    target_wavs = []
    input_wavs = []

    for model_name in MODELS:
        try:
            record, target_wav, input_wav = benchmark_model(model_name, audio)
            records.append(record)
            target_wavs.append(target_wav)
            input_wavs.append(input_wav)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            torch.cuda.empty_cache()
            records.append({"model": model_name, "error": "OOM"})
            target_wavs.append(None)
            input_wavs.append(None)

    # Quality metrics (all at once to share model loads)
    compute_quality_metrics(records, target_wavs, input_wavs)

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    print_summary_table(records)


if __name__ == "__main__":
    main()
