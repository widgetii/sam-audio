# Dialogue Pipeline Rounds 2-3: Performance Investigation

## Overview

Rounds 2-3 attempted to optimize the Pass 2 bottleneck identified in Round 1. The key finding: **the original bottleneck analysis was wrong**. Frame extraction is not the dominant cost — `model.separate()` is.

## Round 2: Batch Range Access (Regression)

### Change
Replaced per-frame `get_frame_played_at()` with `get_frames_played_in_range(start, end)` in both `extract_chunk_frames()` and Pass 0 face detection.

### Result: 55% slower on Pass 2

`get_frames_played_in_range()` decodes ALL frames between start and end timestamps. For a 180s chunk at 25fps, that's ~4500 frames. The vision encoder (PerceptionEncoder/CLIP) processes every frame in batches of 300, so:

- Round 1: 500 frames = 2 CLIP batches
- Round 2: 4500 frames = 15 CLIP batches

Chunks with 7 characters took ~17:49 each vs Round 1's ~11:30.

Pass 0 face detection also regressed from 17min to 20min because range access decoded many unnecessary intermediate frames between 2-second sample points.

### Commit: d94db12 (reverted in Round 3)

## Round 3: Batch Timestamp Access + Reduced Frames

### Changes
1. Switched to `get_frames_played_at(list_of_timestamps)` — only decodes requested frames
2. Reduced `max_frames` from 500 to 150 (SAMAudioProcessor resamples via `torch.linspace` anyway)
3. Fixed Pass 0 to also use `get_frames_played_at()` batch API

### Result: No meaningful speedup

| Metric | Round 1 | Round 3 |
|--------|---------|---------|
| Pass 0 (face detection) | 17 min | 17 min |
| Pass 1 (text-only) | 30 min | 30 min |
| Pass 2 per-character time | ~152s | ~152s |
| Pass 2 projected total | 5h 45min | ~5h 45min |

The per-character processing time is **identical** at ~152s regardless of whether we extract 500 or 150 frames.

### Commit: a28634e

## Why the Round 1 Analysis Was Wrong

Round 1 attributed 70% of Pass 2 time to "video frame extraction". This was a misattribution. The actual time breakdown per character:

| Operation | Actual Time | Notes |
|-----------|-------------|-------|
| `model.separate()` total | ~150s | Includes everything below |
| -- Vision encoder (CLIP) | ~40-60s | PerceptionEncoder processes all frames |
| -- ODE solver (16 steps) | ~60-80s | Transformer self-attention on 4500 audio tokens |
| -- Audio codec decode | ~10-15s | Chunked decoding with max_chunk_tokens=500 |
| Frame extraction + masking | ~2-5s | Negligible with batch API |

The "7 minutes of frame extraction" in Round 1 was actually dominated by CLIP processing the 500 frames inside `model.separate()`. Reducing to 150 frames saves one CLIP batch (~40s) per character, but the ODE solver (which doesn't depend on frame count) still takes ~70s. The net savings per character are small relative to the total.

### Why per-character time didn't change

With batch_size=300 in PerceptionEncoder:
- 500 frames: 2 CLIP batches, ~80s total
- 150 frames: 1 CLIP batch, ~40s total
- Savings: ~40s per character

But each `model.separate()` call also runs the ODE solver (16 diffusion steps through the transformer). The transformer uses self-attention on **4500 audio tokens** (180s × 48kHz / 1920 hop_length). This is O(4500^2) per step and takes ~70s regardless of frame count.

So reducing frames saves ~40s out of ~150s = ~27%. But our measurements show 0% improvement. This suggests either:
1. The CLIP time was overestimated (GPU was already saturated with other work)
2. The 150→500 reduction falls within a single CLIP batch boundary on A100
3. Some other overhead masked the savings

## The Real Bottleneck: model.separate()

For a 180s chunk with N characters, Pass 2 takes ~150 × N seconds. The cost is entirely inside `model.separate()`:

```
model.separate(batch)
  -> _get_forward_args(batch)
       -> _get_audio_features()     # DACVAE encode, ~3s
       -> text_encoder()            # T5, ~2s (CPU offloaded)
       -> _get_video_features()     # PerceptionEncoder/CLIP, ~40-60s
  -> odeint(vector_field, ...)      # 16 diffusion steps, ~60-80s each step ~4s
  -> audio_codec.decode_chunked()   # DACVAE decode, ~10s
```

This runs ONCE PER CHARACTER because each character has different masked video frames, producing different CLIP features and thus different separation results.

## Confirmed Performance Profile (A100 80GB)

| Phase | Time | Bottleneck |
|-------|------|------------|
| Pass 0: Face detection | 17 min | CPU (InsightFace without CUDA provider) |
| Pass 0: Clustering | 2 sec | CPU (sklearn) |
| Pass 1: Text-only | 30 min | GPU (model.separate × 49 chunks) |
| Pass 2: Visual separation | ~5h 45min | GPU (model.separate × ~136 char-chunks) |
| **Total** | **~6h 33min** | |

Pass 2 processes ~136 character-chunks (35 chunks × avg 3.9 chars). At 152s each = 20,672s = 344 min = 5h44m.

## Optimization Opportunities (Actual)

### 1. Shorter chunk windows (high impact)

The ODE solver uses transformer self-attention on audio tokens. Token count = duration × 48000 / 1920.

| Window | Audio tokens | Attention O(n^2) | Estimated per-char |
|--------|-------------|------------------|--------------------|
| 180s | 4500 | 20.25M | 152s |
| 90s | 2250 | 5.06M | ~60-80s |
| 60s | 1500 | 2.25M | ~40-55s |

Halving the window to 90s could reduce ODE solver time by ~4x, but doubles the number of chunks (more overhead, more overlap regions).

**Estimated impact**: 90s windows → ~98 chunks, ~70 need visual pass, avg 3.9 chars, ~70s/char = ~300 min Pass 2. Total pipeline: ~6h → ~4h.

### 2. Skip low-visibility characters per chunk (medium impact)

Currently, every character detected in a chunk's time range gets a full `model.separate()` call. Characters with only 1-2 face detections in a 180s window (brief appearances) could be skipped.

**Estimated impact**: Reducing avg chars/chunk from 3.9 to 2.5 → ~35% reduction in Pass 2 time.

### 3. Cache audio features across characters (low impact)

Audio codec encoding (`_get_audio_features`) and text encoding are identical for all characters in a chunk (same audio, same empty text prompt). Currently recomputed for each character.

Would require modifying `model.separate()` or calling lower-level APIs directly. Saves ~5s per character.

### 4. GPU-accelerated face detection (low impact on total)

Install `onnxruntime-gpu` to use CUDA for InsightFace. Would reduce Pass 0 from 17min to ~2min. Small impact on total time but meaningful for iteration speed.

### 5. Pre-compute all vision features (high complexity, high impact)

Run CLIP once per chunk on unmasked frames, then apply masks in feature space. This would eliminate N-1 CLIP forward passes per chunk. However, masking operates at pixel level before CLIP — the features of a masked image differ from masking features of an unmasked image. Would need architectural changes or approximation.

## Open Questions

1. **Does frame count affect separation quality?** We reduced from 500 to 150 without checking output quality. The SAMAudioProcessor resamples via linspace, so 150 may be fine, but needs verification.

2. **What's the minimum effective window size?** Shorter windows reduce attention cost quadratically but increase chunk count and overlap. Is 90s or 60s viable without quality degradation?

3. **Can we batch multiple characters in one model.separate() call?** The model supports batch processing, but VRAM (80GB already 98% used for single character) is the constraint.

4. **Would flash attention help?** The transformer in model.py doesn't appear to use flash attention. Enabling it could speed up the ODE solver's attention computation on long sequences.
