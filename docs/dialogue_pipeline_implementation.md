# Dialogue Pipeline: Implementation Details & Optimization History

## Overview

End-to-end movie dialogue extraction pipeline using SAM-Audio with visual prompting for per-character speech isolation. Processes full-length movies (tested on "Aliens" 1986, 2h17m) on A100 80GB GPU.

## Architecture

```
Movie file (mkv/mp4)
  |
  Pass 0: Preprocessing
  |  - Extract mono audio at 48kHz
  |  - Face detection (InsightFace RetinaFace + ArcFace) every 2s
  |  - Agglomerative clustering (cosine, threshold=0.6) -> character IDs
  |  - Optional SAM3 segmentation masks (--no-sam3 for bbox fallback)
  |
  Pass 1: Text-only dialogue detection
  |  - Chunk audio into windows (default 90s, 5s overlap)
  |  - SAM-Audio text_only, prompt="speech"
  |  - 1-second RMS analysis -> has_dialogue per segment
  |  - Identify multi-speaker chunks (2+ faces with dialogue)
  |
  Pass 2: Visual per-character separation (selective)
  |  - For each multi-speaker chunk x each eligible character:
  |    - Generate face mask (bbox padded 20% or SAM3)
  |    - SAM-Audio full model with masked video frames
  |    - RMS analysis -> per-character speaking timeline
  |
  Output: dialogue_metadata.json
    - Per-character speaking segments with timestamps
    - Quality scores (optional Judge model)
    - Gaps suitable for audio description
```

## Files

| File | Purpose |
|------|---------|
| `scripts/dialogue_metadata.py` | Main pipeline script (CLI entry point) |
| `scripts/face_tracker.py` | Face detection, embedding, clustering, mask generation |

## Key Technical Details

### Audio Token Count

The SAM-Audio model operates on audio tokens where:
```
tokens = duration_seconds * 48000 / hop_length
hop_length = 1920 (product of encoder_rates: 2*8*10*12)
```

This gives 25 tokens/second. A 90s chunk = 2250 tokens, a 180s chunk = 4500 tokens.

### model.separate() Call Stack

Each `model.separate()` call is the dominant cost. Internal breakdown:

```
model.separate(batch)
  -> _get_forward_args(batch)
       -> _get_audio_features()     # DACVAE encode, ~3s
       -> text_encoder()            # T5, ~2s (CPU offloaded)
       -> _get_video_features()     # PerceptionEncoder/CLIP, ~40-60s
  -> odeint(vector_field, ...)      # 16 diffusion steps, ~60-80s
  -> audio_codec.decode_chunked()   # DACVAE decode, ~10s
```

The ODE solver uses midpoint method (`step_size=2/32`, 16 steps over [0,1]). Each step runs the full DiT transformer with self-attention on ALL audio tokens — cost is **O(n^2)** in token count.

### Vision Encoder (CLIP) Processing

`PerceptionEncoder` (in `vision_encoder.py`) processes ALL video frames through CLIP in batches of 300. Critically, `SAMAudioProcessor.load_video()` resamples input frames to match the audio token count via `torch.linspace`:

```python
idx = torch.linspace(0, video.size(0)-1, int(size)).round().long()
```

This means **reducing input frame count does NOT reduce CLIP processing** — the processor upsamples back to token count. Whether you provide 150 or 500 input frames, CLIP always processes `duration * 25` frames.

### Mask Convention

Binary masks where target=0, background=non-zero. Applied by `processor.mask_videos()` which does `video * mask.eq(0)`, zeroing out everything except the target character.

### Face Detection & Clustering

- InsightFace (RetinaFace detection + ArcFace 512-dim embeddings)
- Samples every 2s (CPU-only, ~17-20 min for 2h movie)
- Agglomerative clustering with cosine distance, threshold=0.6
- Cluster IDs are raw labels (not reindexed to 0..N-1)

### Video Frame Extraction

Uses torchcodec `VideoDecoder` with three APIs:
- `get_frame_played_at(seconds)` — single frame seek+decode
- `get_frames_played_at(seconds_list)` — batch decode at specific timestamps (used)
- `get_frames_played_in_range(start, stop)` — decodes ALL frames in range (avoid)

### Resume Support

Progress saved to `{output}.progress.json` after each chunk. Stores `pass1_completed`, `pass2_completed` indices and full results. Essential for multi-hour runs where SSH may timeout.

## CLI Interface

```bash
uv run python scripts/dialogue_metadata.py \
  --input /mnt/data/video-sources/Aliens.1986.mkv \
  --output aliens_dialogue.json \
  --checkpoint facebook/sam-audio-base-tv \
  --window-seconds 90 \
  --overlap-seconds 5 \
  --rms-threshold-db -40 \
  --min-gap-seconds 3.0 \
  --max-chunk-tokens 500 \
  --max-characters 8 \
  --min-face-detections 3 \
  --no-judge \
  --no-sam3 \
  --save-audio --audio-output-dir ./separated/ \
  --resume
```

## Optimization History

### Round 1: Baseline (commit 316134e)

First end-to-end run. Results on Aliens (A100 80GB):

| Phase | Time |
|-------|------|
| Pass 0 (face detection) | 17 min |
| Pass 1 (text-only) | 30 min |
| Pass 2 (visual separation) | 5h 45min |
| **Total** | **6h 33min** |

Initial analysis attributed 70% of Pass 2 time to video frame extraction (500 individual `get_frame_played_at()` calls per chunk). **This analysis was wrong** — the time was actually inside `model.separate()`.

### Round 2: Batch Range Access (commit d94db12) — REGRESSION

Replaced per-frame extraction with `get_frames_played_in_range(start, end)`.

**Result: 55% slower on Pass 2.** `get_frames_played_in_range()` decodes ALL frames between timestamps. For a 180s chunk at 25fps = ~4500 frames. CLIP then processes all 4500 in 15 batches of 300 (vs 2 batches for 500 frames). Reverted.

### Round 3: Batch Timestamp Access + Reduced Frames (commit a28634e)

- Switched to `get_frames_played_at(list_of_timestamps)` — only decodes requested frames
- Reduced `max_frames` from 500 to 150
- Fixed Pass 0 to use batch API too

**Result: No meaningful speedup.** Per-character time: 152s in both R1 and R3.

**Root cause discovered**: Frame extraction takes ~2-5s per character. The real cost is `model.separate()` at ~150s per character, split between CLIP vision encoding (~40-60s) and ODE solver (~60-80s). Reducing input frames from 500 to 150 saves nothing because `SAMAudioProcessor.load_video()` resamples back to the audio token count anyway — CLIP always processes the same number of frames.

### Round 4: Shorter Chunk Windows (commit e29475a)

Reduced default window from 180s to 90s (overlap from 10s to 5s).

| Metric | R3 (180s) | R4 (90s) | Change |
|--------|-----------|----------|--------|
| Pass 2 time | 338 min | 257 min | **-24%** |
| Total time | 388 min | 305 min | **-22%** |
| Per-character time | 152s | ~75s | **-51%** |
| Total char-chunks | 136 | 207 | +52% |

**Why it works**: Transformer self-attention is O(n^2) in token count:
- 180s chunk: 4500 tokens, attention = O(4500^2) = 20.25M
- 90s chunk: 2250 tokens, attention = O(2250^2) = 5.06M = **4x reduction**

Per-character time halved (2x), but more chunks means same characters appear in more windows (1.5x more char-chunks). Net: 2x / 1.5x = **1.3x faster** on Pass 2.

**Quality tradeoff**: Dialogue detection dropped from 41.0% to 36.8%. Needs investigation — may be due to shorter context making ambiguous speech harder to identify, or more overlap boundary effects.

### Round 5: Min Visibility Threshold (commit d06e2b4)

Added `--min-face-detections` (default 3). Characters with fewer than N face detections in a chunk window are skipped from the expensive `model.separate()` call. If fewer than 2 characters remain eligible, the entire chunk's visual pass is skipped.

Rationale: A character with 1-2 face detections in a 90s window (= 2-4 seconds of visibility at 2s sampling interval) is likely a brief appearance. Running a full 75s `model.separate()` call for such fleeting appearances is wasteful.

| Metric | R4 (no filter) | R5 (min_face_detections=3) | Change |
|--------|---------------|---------------------------|--------|
| Pass 0 | 20 min | 19.6 min | Same |
| Pass 1 | 27.5 min | 27.5 min | Same |
| Pass 2 | 257 min | **149 min** | **-42%** |
| **Total** | **305 min (5.1h)** | **197 min (3.3h)** | **-36%** |
| Char-chunks | 207 | 130 eligible / 197 total | -37% |
| Dialogue % | 36.8% | 33.1% | -10% relative |

Pass 2 dropped from 257 min to 149 min — closely matching the 34% char-chunk reduction (130/197). Some chunks were skipped entirely when fewer than 2 characters remained eligible after filtering.

**Cumulative speedup from R1 baseline: 6.5h -> 3.3h (49% reduction).**

Dialogue detection dropped further (33.1% vs 36.8% in R4). This additional drop may be from skipping low-visibility characters who were speaking off-screen, or from chunks being skipped entirely when filtering leaves < 2 eligible characters. Needs investigation to determine if this is real lost dialogue or noise reduction.

## Performance Profile (A100 80GB, Aliens 2h17m)

Best known configuration (90s windows, min_face_detections=3):

| Phase | Time | Bottleneck |
|-------|------|------------|
| Pass 0: Face detection | 20 min | CPU (InsightFace without CUDA provider) |
| Pass 0: Clustering | 2 sec | CPU (sklearn) |
| Pass 1: Text-only | 27.5 min | GPU (model.separate x 97 chunks) |
| Pass 2: Visual separation | 149 min | GPU (model.separate x 130 eligible char-chunks) |
| **Total** | **197 min (3.3h)** | |

VRAM usage: ~57 GB of 80 GB for single character separation.

### Optimization Progress

| Round | Total Time | Pass 2 Time | Key Change |
|-------|-----------|-------------|------------|
| R1 (baseline) | 6h 33min | 5h 45min | — |
| R2 (range API) | ~10h | ~9h | REGRESSION: decoded all frames |
| R3 (batch API) | 6h 30min | 5h 45min | No effect: wrong bottleneck |
| R4 (90s windows) | 5h 6min | 4h 17min | O(n^2) attention reduction |
| R5 (min visibility) | **3h 17min** | **2h 29min** | Skip low-detection characters |

## Remaining Optimization Opportunities

### 1. Window size tuning (medium impact)

120s windows as a compromise between 90s (faster) and 180s (better quality):
- O(3000^2) = 9M attention vs O(2250^2) = 5.06M vs O(4500^2) = 20.25M
- Fewer chunks = less char-chunk inflation

### 2. Cache audio features across characters (low impact)

Audio codec encoding and T5 text encoding are identical for all characters in a chunk (same audio, same empty text prompt). Currently recomputed for each character. Would save ~5s per character but requires modifying `model.separate()` internals.

### 3. GPU-accelerated face detection (low total impact)

Install `onnxruntime-gpu` for CUDA-based InsightFace. Would reduce Pass 0 from 20 min to ~2 min. Small impact on total but meaningful for iteration speed.

### 4. Flash attention (high potential impact)

The DiT transformer in `model.py` doesn't use flash attention. Enabling it could significantly reduce the ODE solver's O(n^2) attention cost, especially on long sequences.

### 5. Pre-compute vision features (high complexity)

Run CLIP once per chunk on unmasked frames, then apply masks in feature space. Would eliminate N-1 CLIP forward passes per chunk. However, masking operates at pixel level before CLIP — masked image features differ from masked features of unmasked images. Would need architectural changes.

### 6. Batch multiple characters (VRAM-limited)

The model supports batch processing, but VRAM is already at ~57/80 GB for single character. Would need gradient checkpointing or model offloading to fit 2+ characters.

## Quality Investigation: Dialogue Detection Regression

### Reported regression: 41% (R1) -> 37% (R4) -> 33% (R5)

Investigation found the apparent regression is mostly explained by two factors:

### 1. Overlap double-counting bug (fixed)

The `dialogue_percentage` calculation summed ALL segments including overlapping regions between consecutive chunks, double-counting ~480 segments (96 overlaps × 5 seconds). Corrected R5 value: **31.4%** (not 33.1%). This bug affected all rounds equally, inflating all reported values.

### 2. Model non-determinism (primary cause of R4 vs R5 difference)

The diffusion model's ODE solver is non-deterministic — each run produces slightly different separation even with identical inputs. The RMS value distribution shows **15.3% of segments** (1,330 out of 8,709) fall within ±5dB of the -40dB threshold. The 304s difference between R4 (3027s) and R5 (2723s) represents ~304 segments flipping across the threshold — about 41% of the 749 segments within ±3dB of -40dB. The min_face_detections filter has zero effect on dialogue detection since it only affects Pass 2, not Pass 1.

### 3. RMS distribution is bimodal with a noisy middle

```
RMS Histogram (5dB buckets, Aliens R5):
  -75dB: 1346  ████████████████████ (silence)
  -70dB:  528  ████████
  -65dB:  872  █████████████
  -60dB:  645  ██████████
  -55dB:  497  ███████
  -50dB:  519  ████████
  -45dB:  919  ██████████████
  -40dB:  669  ██████████         <-- threshold
  -35dB:  648  ██████████
  -30dB: 1294  ███████████████████ (dialogue)
  -25dB:  753  ███████████
```

Clear silence peak (-75 to -65dB) and dialogue peak (-30 to -25dB), but significant density in the threshold zone (-45 to -35dB). This makes the binary classification sensitive to small noise variations.

### 4. Real regression from shorter windows (R1 vs R4) is smaller than reported

The 180s→90s window change does cause a real regression because shorter context gives the model less information. But the reported 4pp difference (41%→37%) is amplified by different overlap amounts (10s vs 5s creates different double-counting) and by model non-determinism between runs.

### Conclusions

- **R4→R5 regression is not real** — same Pass 1 pipeline, difference is entirely model non-determinism
- **R1→R4 regression is real but small** — shorter windows reduce context, but the magnitude is uncertain due to the double-counting bug and non-determinism
- **Binary -40dB threshold is fragile** — 15% of segments are borderline. Consider using a confidence margin, smoothing, or multi-threshold approach

## Key Lessons Learned

1. **Profile before optimizing**: The Round 1 analysis was completely wrong about where time was spent. Frame extraction appeared slow but was actually negligible compared to model inference.

2. **Understand the full data flow**: Reducing input frames to `extract_chunk_frames()` had zero effect because `SAMAudioProcessor.load_video()` resamples frames back to audio token count. The optimization targeted the wrong layer.

3. **O(n^2) attention scaling is the real lever**: Halving the chunk duration gives 4x reduction in per-step attention cost. This is the only optimization that produced meaningful speedup.

4. **torchcodec API matters**: `get_frames_played_in_range()` decodes every frame sequentially (expensive for sparse sampling). `get_frames_played_at(timestamps)` only decodes requested frames. Wrong API choice caused a 55% regression.

5. **Model non-determinism matters for evaluation**: Comparing runs requires accounting for diffusion model noise. Single-run percentage comparisons are unreliable when 15% of segments are near the detection threshold.
