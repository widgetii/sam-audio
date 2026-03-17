# Dialogue Pipeline Round 1: Implementation & Findings

## Overview

First end-to-end run of the movie dialogue extraction pipeline with character tracking on "Aliens" (1986), 2h17m, A100 80GB GPU.

## Implementation

Two new files:
- `scripts/face_tracker.py` — Face detection (InsightFace/RetinaFace), ArcFace embeddings, agglomerative clustering, bbox mask generation
- `scripts/dialogue_metadata.py` — Three-pass pipeline: preprocessing, text-only dialogue detection, visual per-character separation

Dependencies added to `pyproject.toml` under `[dependency-groups] dialogue`.

## Run Configuration

```
Model: facebook/sam-audio-base-tv
Window: 180s, Overlap: 10s
Face sampling: every 2.0s (CPU-based InsightFace)
No SAM3 (bbox masks only), No Judge scoring
Max characters: 8
```

## Timing Results (A100 80GB)

| Phase | Duration | Details |
|-------|----------|---------|
| Pass 0: Face detection | 17 min | 4115 frames on CPU (InsightFace lacks CUDA provider) |
| Pass 0: Clustering | 2 sec | Agglomerative, cosine, threshold=0.6 |
| Pass 1: Text-only | 30 min | 49 chunks × 37s/chunk |
| Pass 2: Visual separation | **5h 45min** | 35 chunks × avg 10min/chunk |
| **Total** | **6h 33min** | |

**Pass 2 was 3.4x slower than estimated** (103 min estimated vs 345 min actual).

## Key Findings

### 1. Pass 2 is dominated by video frame extraction, not model inference

Each Pass 2 chunk calls `get_frame_played_at()` up to 500 times individually to extract video frames. For 35 chunks, that's ~17,500 individual seek+decode operations on the mkv container. This is the primary bottleneck.

Model inference (`model.separate()`) takes ~37s per character. A chunk with 4 characters = ~2.5 min of GPU time. But frame extraction for that same chunk takes ~7 min of I/O time.

**Fix**: Use `get_frames_played_in_range(start_sec, end_sec)` — a single call that returns all frames in a time range with sequential decoding, then subsample from the result.

### 2. 71% of chunks triggered visual pass (vs estimated 20-30%)

35 out of 49 chunks had 2+ visible characters with dialogue. Aliens is a dialogue-heavy movie with many multi-character scenes. The "selective" optimization is less effective than expected for this genre.

### 3. Character speaking time exceeds screen time

Several characters have `speaking_seconds > screen_time_seconds` (e.g., Char 0: 1402s screen, 3729s speaking). This is because:
- Screen time is measured from face detection (sampled at 2s intervals)
- Speaking time accumulates across all chunks where the character is "visible" — even if their face wasn't detected in every sampled frame
- The 1-second RMS analysis marks segments as "has_dialogue" based on energy threshold, which may be too sensitive

### 4. Face clustering produced 8 characters with non-sequential IDs

Cluster IDs from agglomerative clustering: 0, 2, 9, 17, 31, 45, 80, 91. These are raw cluster labels, not reindexed. Character 0 is the most prominent (likely Ripley).

### 5. InsightFace CPU fallback works but is slow

The `onnxruntime` package on the test machine doesn't have `CUDAExecutionProvider`. Face detection falls back to CPU, making Pass 0 take 17 min instead of ~2 min. Installing `onnxruntime-gpu` would fix this.

### 6. Judge model loaded despite --no-judge

The log shows `facebook/sam-audio-judge` being fetched during initialization even with `--no-judge`. Investigation shows this is loaded by the `JudgeRanker` within `sam-audio-base-tv`'s ranker system, not by our judge scoring code. The `--no-judge` flag correctly prevents our explicit judge scoring. Not a bug — just model initialization overhead.

### 7. SSH sessions time out during long runs

The SSH connection monitoring the pipeline timed out after ~3 hours. The pipeline itself continued running fine (it's a background process). The `--resume` support proved essential for monitoring.

## Output Statistics

```
Source: Aliens.1986.mkv, 8228.9s (2h17m)
Dialogue: 41.7% of movie (3432s)
Dialogue segments: 891
Gaps (>= 3s): 236, average 19.2s
Characters tracked: 8
Output JSON: 7.2 MB
```

## Performance Bottleneck Analysis

Time breakdown per Pass 2 chunk (180s window, avg 3.9 chars):

| Operation | Time | % |
|-----------|------|---|
| Video frame extraction (500 × get_frame_played_at) | ~7 min | 70% |
| Model inference (3.9 × model.separate @ 37s) | ~2.5 min | 25% |
| Mask generation + overhead | ~0.5 min | 5% |

The frame extraction is called **once per chunk** (shared across characters), but it still dominates because of the per-frame seeking overhead in mkv containers.

## Open Questions

1. **Are bbox masks sufficient, or does SAM3 segmentation meaningfully improve separation quality?** The current run uses padded bounding box masks. SAM3 would provide precise silhouette masks but adds ~15 min preprocessing.

2. **Is the RMS threshold (-40 dB) appropriate?** Characters show more speaking time than screen time, suggesting the threshold may be too permissive. Need to compare against manual annotations.

3. **Should we reduce max_frames for Pass 2?** Currently capped at 500 frames per chunk. The vision encoder processes at its own rate and the processor resamples frames via `torch.linspace`. We may be extracting far more frames than needed.

4. **How to handle overlapping dialogue?** When two characters speak simultaneously in the same 1-second window, both are marked as speaking. This is technically correct but may confuse downstream consumers.

5. **Should character IDs be reindexed to 0..N-1?** Current cluster labels are arbitrary integers. Reindexing would be cleaner for the output JSON.

6. **Is the face detection sampling interval (2s) too sparse?** Characters in fast-cut scenes may be missed. But increasing frequency proportionally increases CPU time.

7. **VRAM usage**: GPU used 56.7 GB of 80 GB. The pipeline should work on 40 GB GPUs with `max_chunk_tokens=250` but this hasn't been tested.

## Next Steps

1. **Fix video frame extraction bottleneck** — Switch from per-frame `get_frame_played_at()` to batch `get_frames_played_in_range()` + subsampling
2. **Reduce frame count** — Determine minimum frames needed for effective visual prompting
3. **Reindex character IDs** — Clean up cluster labels in output
4. **Validate separation quality** — Listen to a few separated chunks manually
5. **Test with onnxruntime-gpu** — Accelerate face detection in Pass 0
