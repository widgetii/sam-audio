# SAM-Audio Benchmark Observations

Benchmark of all 6 SAM-Audio model variants on a single NVIDIA A100 80GB PCIe.

## Setup

- **GPU**: NVIDIA A100 80GB PCIe
- **Input**: 30s clip from `chapter_2.mkv` (1080p H.264, AC3 48kHz stereo, downmixed to mono)
- **Prompt**: "speech" (text-only, no visual prompts)
- **Reranking candidates**: 2
- **Timing**: 1 warm-up run + 3 timed runs per model, `torch.cuda.synchronize()` barriers
- **Quality metrics**: CLAP similarity, Judge (overall/precision/recall/faithfulness), Aesthetic (PQ/CE)

## Results

| Model | Infer (s) | Std (s) | Peak VRAM (GB) | CLAP | Judge | Judge P | Judge R | Judge F | Aes PQ | Aes CE |
|-----------|-----------|---------|----------------|-------|-------|---------|---------|---------|--------|--------|
| small | 6.49 | 0.011 | 34.1 | 0.232 | 3.94 | 3.91 | 4.92 | 4.94 | 6.87 | 3.49 |
| base | 10.14 | 0.043 | 36.6 | 0.265 | 4.85 | 4.88 | 4.99 | 4.99 | 6.83 | 3.46 |
| large | 19.76 | 0.136 | 43.3 | 0.284 | 4.72 | 4.77 | 4.99 | 4.95 | 6.62 | 3.40 |
| small-tv | 6.53 | 0.014 | 34.1 | 0.239 | 4.55 | 4.61 | 4.97 | 4.91 | 6.46 | 3.50 |
| base-tv | 10.19 | 0.036 | 36.6 | 0.347 | 4.89 | 4.91 | 5.00 | 4.98 | 7.06 | 3.48 |
| large-tv | 19.84 | 0.148 | 43.3 | 0.378 | 4.30 | 4.31 | 4.87 | 4.99 | 6.80 | 3.59 |

## Observations

### Latency

- Inference time scales roughly linearly with model size: small ~6.5s, base ~10.1s, large ~19.8s for 30 seconds of audio.
- The ratio is approximately 1x / 1.56x / 3.05x (small / base / large), suggesting the ODE solver dominates and its cost grows with transformer width.
- TV variants add negligible overhead (~0.04s) over their non-TV counterparts when using text-only prompts. The visual encoder path is not exercised without video input.
- Variance is very low across runs (std < 0.15s), indicating deterministic compute-bound behavior with no scheduling jitter.

### Memory

- Three distinct memory tiers: 34.1 GB (small), 36.6 GB (base), 43.3 GB (large).
- TV and non-TV variants of the same size use identical peak VRAM, confirming the visual encoder adds no memory when unused.
- With 2 reranking candidates, even the small model uses 34 GB, which would not fit on a 24 GB consumer GPU (RTX 3090/4090). A single candidate (no reranking) would be needed for those cards.
- The full 203.5s audio caused OOM on all models, even small. The codec decoder's conv1d tried to allocate 14 GB on top of an already-loaded model. For long-form audio, chunked processing would be required.

### Quality: CLAP (text-audio alignment)

- TV variants consistently outscore their non-TV counterparts: small 0.232 vs 0.239 (+3%), base 0.265 vs 0.347 (+31%), large 0.284 vs 0.378 (+33%).
- The TV advantage grows with model size, suggesting the TV training objective improves the model's understanding of semantic content even for text-only prompts.
- `large-tv` achieves the highest CLAP score (0.378), but `base-tv` offers the best CLAP-per-second ratio (0.347 / 10.19s = 0.034/s vs 0.378 / 19.84s = 0.019/s).

### Quality: Judge (separation quality)

- `base-tv` achieves the highest Judge overall score (4.89), followed closely by `base` (4.85).
- Surprisingly, `large` and `large-tv` score lower on Judge than `base` variants. This may indicate overfitting in the larger models on this particular input, or that the Judge model's training data is better aligned with base-sized outputs.
- Recall is uniformly high across all models (4.87-5.00), meaning they all capture most of the target speech. The differentiation comes from precision (ability to exclude non-speech sounds).
- `small` is the clear outlier with the lowest Judge score (3.94), driven by low precision (3.91).

### Quality: Aesthetic

- Aesthetic scores are relatively flat across models (PQ: 6.46-7.06, CE: 3.40-3.59), suggesting production quality doesn't vary dramatically between variants for speech separation.
- `base-tv` leads in Production Quality (7.06), while `large-tv` leads in Content Enjoyment (3.59).

### Best model per use case

- **Best quality overall**: `base-tv` — highest Judge (4.89), strong CLAP (0.347), best Aesthetic PQ (7.06).
- **Best quality/latency trade-off**: `base-tv` — 10.2s inference with top-tier quality across all metrics.
- **Fastest acceptable quality**: `small-tv` — 6.5s with decent Judge (4.55) and CLAP (0.239).
- **Best text-audio alignment**: `large-tv` — highest CLAP (0.378) but 2x the latency of `base-tv` for only +9% CLAP improvement.

### Methodology notes

- Load times (80-150s) are dominated by model download/caching on first run and HuggingFace Hub overhead. They are not representative of production warm-start latency.
- This benchmark uses a single audio sample with one prompt. Results may not generalize to other content types (music, SFX) or longer durations.
- Reranking with 2 candidates roughly doubles memory vs 1 candidate. Production deployments on smaller GPUs should use `reranking_candidates=1`.
