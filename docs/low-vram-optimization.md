# Fitting SAM-Audio into 16 GB VRAM

This document traces the step-by-step optimization of SAM-Audio from its original 60+ GB peak VRAM down to under 7 GB for 30-second audio, making it runnable on consumer GPUs (RTX 4080, 3080 Ti, etc.).

All benchmarks were run on NVIDIA A100 80GB PCIe with `facebook/sam-audio-small` (1275M params, 5.1 GB checkpoint), prompt "speech", 48kHz audio from a 203.5s audiobook chapter.

## Starting point: 34–61 GB

The unmodified model with FP32 precision, 2 reranking candidates:

| Audio duration | Peak VRAM | Inference time | Notes |
|---------------|-----------|----------------|-------|
| 30s | 34.1 GB | 6.5s | FP32, 2 candidates |
| 203s | OOM | — | Crashed on 80 GB A100 |

The 34 GB for just 30s of audio is driven by loading every component — vision encoder (671M), span predictor (~1B), CLAP+Judge rankers (~2B), ImageBind ranker (~1.2B) — even when using text-only prompts. Activations in FP32 add further overhead.

## Iteration 1: BF16 autocast

**Commit:** `d97aaff` — Add BF16 autocast to benchmark

Wrapping `model.separate()` with `torch.autocast("cuda", dtype=torch.bfloat16)` halves the memory used by activations during the ODE solve (32 evaluations × 12 transformer layers). Weights remain FP32 on load but compute happens in BF16.

| Audio duration | Peak VRAM | Inference time | Change |
|---------------|-----------|----------------|--------|
| 203s | 60.8 GB | 35.4s | Now fits A100 (was OOM without BF16) |

BF16 alone enabled 203s audio to run but is still far from 16 GB.

## Iteration 2: Chunked codec decode

**Commit:** `c4dcab0` — Add chunked codec decode and ODE state fix for long audio OOM

The DACVAE codec's `decode()` processes all tokens at once, causing a memory spike for long audio. Added `decode_chunked()` to `sam_audio/model/codec.py` which processes `max_chunk_tokens=500` tokens at a time (~20.8s per chunk), keeping codec decode memory constant regardless of audio length.

Also fixed an issue where `odeint` accumulated intermediate states across all timesteps. Using the `[-1]` index on the result to keep only the final state avoids storing the full ODE trajectory.

This iteration didn't change peak VRAM for 30s audio (codec decode isn't the bottleneck at that length) but prevented OOM on long audio during the decode phase.

## Iteration 3: Text-only loading mode (this PR)

**Files changed:** `sam_audio/model/config.py`, `sam_audio/model/model.py`, `sam_audio/model/base.py`

### The insight

For text-prompted separation (the most common use case), only 3 of the 8 model components are needed:

| Component | Params | FP32 size | Needed for text-only? |
|-----------|--------|-----------|----------------------|
| transformer (DiT) | 492M | 1.97 GB | Yes |
| audio_codec (DACVAE) | 108M | 0.43 GB | Yes |
| text_encoder (T5-base) | 220M | 0.88 GB | Yes (once, then offloaded) |
| proj/align/embed layers | ~3M | 0.02 GB | Yes |
| vision_encoder (PE-Core-L14) | 671M | 2.69 GB | **No** |
| visual_ranker (ImageBind) | ~1.2B | ~4.8 GB | **No** |
| text_ranker (CLAP+Judge) | ~1B+ | ~4+ GB | **No** |
| span_predictor (pe-a-frame-large) | ~1B+ | ~4+ GB | **No** |

Strippable components total ~16+ GB FP32. Core model is only ~3.3 GB FP32 / ~1.65 GB BF16.

### Changes

1. **`SAMAudioConfig.text_only`** — New boolean flag (default `False`). When `True`:
   - `SAMAudio.__init__()` skips creating `vision_encoder`, `visual_ranker`, `text_ranker`, and `span_predictor`
   - `load_state_dict()` filters out `vision_encoder.*` weights from checkpoint (the other skipped components are loaded from HuggingFace separately and were already filtered)
   - `_get_video_features()` returns zeros (same path as when `video is None`)

2. **T5 CPU offload** — After text encoding in `_get_forward_args()`, T5 is moved to CPU and CUDA cache is cleared. This frees ~0.44 GB during the ODE solve phase where only the transformer, codec, and activations need GPU memory.

3. **Config passthrough in `base.py`** — `_from_pretrained()` now passes `model_kwargs` to the config constructor if the key is a valid config parameter (not just if it already exists in the JSON file). This lets callers do:
   ```python
   SAMAudio.from_pretrained("facebook/sam-audio-small", text_only=True)
   ```

### Results

| Config | Peak VRAM | Inference | Load time | Fits 16 GB? |
|--------|-----------|-----------|-----------|-------------|
| Full model, 30s | 29.37 GB | 5.52s | 168s | No |
| **text_only, 30s** | **6.97 GB** | **2.76s** | **9s** | **Yes** |
| **text_only, 203s** | **17.89 GB** | **20.92s** | **9s** | No (by 1.89 GB) |

### VRAM breakdown (text_only, 30s, BF16)

| Component | Estimated VRAM |
|-----------|---------------|
| Transformer weights (BF16) | ~0.98 GB |
| Audio codec weights (BF16) | ~0.22 GB |
| Proj/align/embed layers | ~0.01 GB |
| T5 (offloaded after encoding) | 0 GB |
| ODE activations (32 evals × 12 layers) | ~3-4 GB |
| Codec decode (chunked) | ~0.5 GB |
| Allocator overhead | ~1-2 GB |
| **Total measured** | **6.97 GB** |

## Summary of all optimizations

| Step | Technique | VRAM impact | Effort |
|------|-----------|-------------|--------|
| BF16 autocast | `torch.autocast("cuda", dtype=torch.bfloat16)` | ~2× reduction in activation memory | 1 line |
| Chunked codec decode | Process 500 tokens at a time | Prevents OOM on long audio decode | `decode_chunked()` in codec.py |
| Text-only mode | Skip vision/rankers/span predictor | ~22 GB saved (30s: 29→7 GB) | Config flag + conditional init |
| T5 CPU offload | Move to CPU after encoding | ~0.44 GB saved during ODE | 3 lines |

**Combined result: 34.1 GB → 6.97 GB (4.9× reduction) for 30s audio.**

## Usage

```python
import torch
from sam_audio import SAMAudio, SAMAudioProcessor

# Load in text-only mode — skips vision encoder, rankers, span predictor
model = SAMAudio.from_pretrained("facebook/sam-audio-small", text_only=True)
model = model.eval().to("cuda")

processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-small")
batch = processor(descriptions=["speech"], audios=[audio_tensor]).to("cuda")

# BF16 autocast halves activation memory; chunked decode prevents codec OOM
with torch.autocast("cuda", dtype=torch.bfloat16):
    result = model.separate(batch, reranking_candidates=1, max_chunk_tokens=500)

target_audio = result.target[0].cpu()
```

## Remaining options for 203s on 16 GB

The 203s case peaks at 17.89 GB — 1.89 GB over the target. Potential next steps:

- **Fewer ODE steps** — Reduce from 32 to 16 steps (`step_size: 2/16`). Trades quality for ~halved activation peak.
- **Smaller chunk size** — Reduce `max_chunk_tokens` below 500.
- **Limit audio length** — Audio under ~2.5 minutes fits comfortably in 16 GB.
- **INT8/INT4 quantization** — Not needed for 30s target, but could close the gap for long audio.

## Correctness

`text_only=True` with `reranking_candidates=1` produces identical output to the full model with the same settings — the same code path runs, just with fewer components loaded. The skipped components (vision encoder, rankers, span predictor) are only used for visual prompts and multi-candidate reranking, neither of which applies in text-only single-candidate mode.
