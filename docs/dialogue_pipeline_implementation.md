# Dialogue Pipeline: Implementation Details & Optimization History

## Overview

End-to-end movie dialogue extraction pipeline using SAM-Audio with visual prompting for per-character speech isolation. Processes full-length movies (tested on "Aliens" 1986, 2h17m) on A100 80GB GPU.

Two pipeline versions exist:
- **v1** (`scripts/dialogue_metadata.py`): Fixed 90s windows, face-only identity. Working baseline, ~3.3h on A100.
- **v2** (`scripts/dialogue_pipeline_v2.py`): Scene-aware chunking, multi-modal identity (face + voice). Addresses the Burke detection problem.

## Architecture — v2 (Scene-Aware Multi-Modal)

```
Movie file (mkv/mp4, 1080p)
  |
  Stage 0: Shot Boundary Detection
  |  av1an --sc-only -> shot boundaries (camera cuts)
  |  Output: list of Shot(start, end) — ~1500-3000 per 2h movie
  |
  Stage 1: Dense Face Detection + Clustering
  |  InsightFace every 0.5s (4x denser), 1080p source, det_thresh=0.3
  |  Global agglomerative clustering -> CharacterProfile IDs
  |  Output: per-shot character presence with face embeddings
  |
  Stage 2: Scene Grouping + Character Propagation
  |  Group consecutive shots into scenes by character overlap + temporal proximity
  |  Propagate: if Burke seen in shots 1,5 of scene -> present in shots 2-4 too
  |  Output: scenes with confirmed character sets (detected + propagated)
  |
  Stage 3: Dialogue Detection (scene-aligned chunks)
  |  SAM-Audio text_only on scene-aligned audio (split at shot boundaries, not arbitrary)
  |  Identify scenes with dialogue + 2+ characters
  |
  Stage 4: Visual Separation + Voice Fingerprinting
  |  SAM-Audio visual separation per character per scene-chunk
  |  Extract speaker embeddings (ECAPA-TDNN 192-dim) from separated audio
  |  Build incremental voice profiles per character
  |  Voice-based discovery: match unattributed speech to known voice profiles
  |
  Stage 5: Reconciliation + Timeline Assembly
  |  Merge face-only and voice-only character identities
  |  Assemble per-character speaking timeline (v2 output format)
  |
  Output: dialogue_metadata_v2.json
    - Per-character speaking segments with timestamps
    - Scene/shot structure
    - Multi-modal identity sources per character
```

## Architecture — v1 (Fixed Window)

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
| `scripts/dialogue_pipeline_v2.py` | **v2** main pipeline script (5-stage, scene-aware) |
| `scripts/scene_detector.py` | Shot detection (av1an), scene grouping, character propagation |
| `scripts/voice_tracker.py` | ECAPA-TDNN speaker embedding extraction + voice profiles |
| `scripts/character_profile.py` | CharacterProfile dataclass, FaceDetection, identity fusion |
| `scripts/dialogue_metadata.py` | **v1** pipeline script (fixed windows, kept for reference) |
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

## v2 Design Rationale

### Problems with v1

1. **Arbitrary chunking**: Fixed 90s windows split mid-scene. No character state flows between chunks.
2. **Face-only identity**: Character identity relies 100% on InsightFace. When faces are missed (Burke undetected in entire 81:30-83:00 confrontation), the character vanishes from the speaking timeline.
3. **No voice signal**: The pipeline discards the most reliable identity cue for dialogue — the voice itself.

### Key changes in v2

- **1080p source** (`Aliens.1080p.mkv`) instead of 720x480 DVD — faces are ~6x more pixels before InsightFace downscales.
- **4x denser sampling** (0.5s vs 2s) — 16,500 frames vs 4,100 for a 2h17m movie.
- **Lower detection threshold** (`det_thresh=0.3` vs 0.5) — accept more candidates, clustering filters noise.
- **Shot-aware sampling** — minimum 2 frames per shot, even for 1-second shots.
- **Scene-aligned chunks** — no arbitrary mid-scene splits, no overlap deduplication needed.
- **Character propagation** — Burke in shots 1 and 5 of a scene → propagated to shots 2-4.
- **Voice fingerprinting** — ECAPA-TDNN 192-dim speaker embeddings from clean SAM-Audio separations.
- **Voice discovery** — unattributed dialogue matched to known voice profiles by cosine similarity.

### Identity fusion

```
1. Face match exists → use it (high precision)
2. Face + voice agree → boost confidence
3. Only voice match → use it (0.8x confidence penalty)
4. Face and voice disagree → trust face
5. No match → unknown character placeholder
```

## CLI Interface — v2

```bash
uv run python scripts/dialogue_pipeline_v2.py \
  --input /data/huggingface/Aliens.1080p.mkv \
  --output aliens_dialogue_v2.json \
  --workspace ./workspace/v2 \
  --checkpoint facebook/sam-audio-base-tv \
  --audio-stream 6 \
  --face-det-threshold 0.3 \
  --sample-interval 0.5 \
  --max-scene-gap 2.0 \
  --max-scene-duration 300 \
  --window-seconds 90 \
  --rms-threshold-db -40 \
  --max-characters 8 \
  --voice-quality-threshold 5.0 \
  --enable-voice-discovery \
  --no-sam3 \
  --resume
```

### Shot detection options

```bash
# Run av1an locally (if not on GPU machine)
av1an --sc-only -i /data/huggingface/Aliens.1080p.mkv --scenes shots.json -x 0

# Then pass pre-computed shots to pipeline
--shots-json shots.json
```

## CLI Interface — v1

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

Pass 2 dropped from 257 min to 149 min — closely matching the 34% char-chunk reduction (130/197). Some chunks were skipped entirely when fewer than 2 characters remained eligible after filtering.

### Round 6: Confirmation Run with Dedup Fix (commit f2200ae)

Fixed overlap double-counting bug in dialogue_percentage calculation. Re-ran with identical settings to R5 to validate non-determinism hypothesis.

| Metric | R5 | R6 | Notes |
|--------|----|----|-------|
| Pass 2 | 149 min | 156 min | R6 had more char-chunks |
| **Total** | **197 min (3.3h)** | **203 min (3.4h)** | |
| Multi-speaker chunks | 62 | 67 | Different clustering |
| Eligible char-chunks | 130/197 | 137/212 | Different clustering |
| Dialogue % (reported) | 33.1% (buggy) | **33.6%** (correct) | |
| Dialogue % (deduped) | 31.4% | 33.6% | +2.2pp non-determinism |

R6 confirms: the R5→R6 dialogue difference (31.4% vs 33.6%) is purely model non-determinism — same pipeline, same settings, 2.2pp variation. Face clustering is also non-deterministic (different multi-speaker chunk counts). Pass 2 time scales proportionally with eligible char-chunks (~68s each).

**Cumulative speedup from R1 baseline: 6.5h -> 3.3h (49% reduction).**

### Round 7: Flash Attention Mask Skip (no commit — negative result)

Added `and not key_padding_mask.all()` to the attention mask condition in `Attention.forward()` (`transformer.py:155`). When the padding mask is trivially all-True (batch_size=1, no padding), `attn_mask=None` is passed to `F.scaled_dot_product_attention`, which should allow PyTorch SDPA to dispatch to FlashAttention2 instead of the math backend.

| Metric | R6 | R7 | Notes |
|--------|----|----|-------|
| Pass 0 | 20 min | 19.9 min | Same |
| Pass 1 | 27.5 min | 27.3 min | Same |
| Pass 2 | 156 min | 159 min | Same (per-char ~68.7s vs ~68.3s) |
| **Total** | **203 min (3.4h)** | **206 min (3.4h)** | |
| Multi-speaker chunks | 67 | 67 | Same clustering this run |
| Eligible char-chunks | 137/212 | 139/212 | +2 |
| Dialogue % | 33.6% | 34.2% | +0.6pp (within ±2pp noise) |

**Result: No speedup.** The mask skip is correct but insufficient. The real blocker is that **the model runs in float32**. FlashAttention2 requires float16 or bfloat16 — with float32, PyTorch SDPA always uses the math backend regardless of whether `attn_mask` is None or not.

To actually enable flash attention, the model would need `model.half()` or `model.bfloat16()`, or inference under `torch.autocast`. This would be a separate, higher-risk change requiring validation that half-precision doesn't degrade separation quality.

The code change is kept as it's harmless (avoids constructing a redundant mask tensor), but it alone provides no performance benefit.

## Performance Profile (A100 80GB, Aliens 2h17m)

Best known configuration (90s windows, min_face_detections=3):

| Phase | Time | Bottleneck |
|-------|------|------------|
| Pass 0: Face detection | 20 min | CPU (InsightFace without CUDA provider) |
| Pass 0: Clustering | 2 sec | CPU (sklearn) |
| Pass 1: Text-only | 27.5 min | GPU (model.separate x 97 chunks) |
| Pass 2: Visual separation | ~150-156 min | GPU (model.separate x ~130-137 eligible char-chunks) |
| **Total** | **~200 min (3.3-3.4h)** | |

VRAM usage: ~57 GB of 80 GB for single character separation.

### Optimization Progress

| Round | Total Time | Pass 2 Time | Dialogue % | Key Change |
|-------|-----------|-------------|------------|------------|
| R1 (baseline) | 6h 33min | 5h 45min | ~41%* | — |
| R2 (range API) | ~10h | ~9h | — | REGRESSION: decoded all frames |
| R3 (batch API) | 6h 30min | 5h 45min | ~41%* | No effect: wrong bottleneck |
| R4 (90s windows) | 5h 6min | 4h 17min | ~37%* | O(n^2) attention reduction |
| R5 (min visibility) | 3h 17min | 2h 29min | 31.4%** | Skip low-detection characters |
| R6 (dedup fix) | **3h 23min** | **2h 36min** | **33.6%** | Fix overlap double-counting |
| R7 (flash attn mask) | 3h 26min | 2h 39min | 34.2% | No effect: model uses float32 |

\* R1-R4 values inflated by overlap double-counting bug (fixed in R6).
\*\* R5 reported 33.1% but was 31.4% after manual dedup correction.

## Known Issues

### SAM3 video predictor OOMs on full-length movies

When running with SAM3 mask generation (the default), Pass 2 gets OOM-killed. SAM3's `start_session(resource_path=video_file)` loads all video frames into memory. For a 2h17m movie at 24fps that's ~197K frames — far too much for system RAM.

**Observed**: frame loading reaches ~9% (17K/197K frames), slows from 480 it/s to 1.2 it/s as memory fills, then process is killed by OS. No error in log — silent OOM kill.

**Workaround**: `--no-sam3` flag uses padded bounding box masks instead.

**Possible fixes**:
1. **Chunk-level sessions** — extract only the chunk's frames (~2160 for 90s at 24fps) to a temp dir, use that as SAM3 resource path
2. **Frame range API** — if SAM3 supports limiting which frames to load
3. **Lazy frame loading** — patch SAM3 to stream frames instead of loading all upfront

**References**: `scripts/face_tracker.py:222` (`_generate_sam3_masks`), `scripts/dialogue_metadata.py:370-374` (SAM3 init)

### SAM-Audio produces silence on pre-separated center channel audio

**Experiment** (Round 8): Tried using the English 5.1 center channel (dialogue-only by convention) as SAM-Audio input instead of the full audio mix, combined with visual face prompts for per-character separation.

**Setup**:
- Extract center channel: `ffmpeg -i movie.mkv -map 0:a:2 -af "pan=mono|c0=FC" -ar 48000 -ac 1 center.wav`
- Detect faces in 80:00-90:00 window, cluster, generate bbox masks
- Run SAM-Audio (sam-audio-base-tv) with `masked_video` + center channel audio
- Transcribe separated audio with faster-whisper

**Result**: SAM-Audio returned near-silence for almost all characters. Out of 8 chunks × ~4 characters each, only **2 lines** survived Whisper transcription. The separated audio energy was below the VAD threshold for >95% of character-chunks.

**Root cause**: SAM-Audio is trained on mixed audio (speech + music + SFX + ambient). Its diffusion model learns to extract a target sound from a complex soundscape. When the input is already a clean dialogue track (center channel), there's minimal non-target signal for the model to suppress. The model appears to interpret the near-absence of "other sounds" as a signal that the target is also absent, producing silence.

**Additional issues encountered**:
- **Face clustering fragmentation**: 300 frames (2s sampling × 10min) produced 37-143 clusters depending on threshold. CPU-only InsightFace on low-resolution (720×480) video yields inconsistent embeddings across lighting/angle changes.
- **Cross-scene face matching**: Cosine similarity between face embeddings from different scenes (e.g., reference at 14:20 vs target at 82:00) drops below 0.4, making automated cluster-to-character mapping unreliable.

**Conclusion**: SAM-Audio visual separation requires mixed audio input. For per-character transcription from a clean dialogue track, use word-level Whisper timestamps + speaking timeline attribution instead (see `docs/transcribe_wordlevel.py`).

**Files**: `docs/transcribe_visual_separation.py` (experiment), `docs/transcribe_wordlevel.py` (working alternative)

### InsightFace misses faces in dialogue scenes → characters absent from speaking timeline

**Problem**: Burke (Paul Reiser) speaks extensively in the confrontation scene at 81:30-83:00 but has **zero face detections** in that time range. His speaking timeline jumps from 79:32 straight to 124:40 — a 45-minute gap.

**Evidence from `dialogue_metadata.json`**:
```
Chunk 57 (80:45-82:15): visible = [Gorman(1), Ripley(24), Bishop(10), Vasquez(1)]  — no Burke
Chunk 58 (82:10-83:40): visible = [Ripley(22), Hicks(20)]                          — no Burke
```

Burke is not detected even once across ~45 sampled frames (90s ÷ 2s interval) in these chunks.

**Impact**: Since Burke has 0 detections, he is excluded from `visible_characters`, SAM-Audio visual separation is never run for him, and he gets no speaking segments. The speaking timeline then has no Burke data for the confrontation scene. Downstream, the word-level transcription script (`transcribe_wordlevel.py`) cannot attribute any words to Burke because there are no Burke speaking segments to match against.

**Likely causes**:
- Low source resolution (720×480) — faces may be too small for RetinaFace at `det_size=(640,640)`
- Profile angles, partial occlusion, or motion blur during dialogue
- InsightFace running CPU-only (no CUDA provider) — uses lower-precision inference
- `det_thresh=0.5` may be too strict for challenging frames
- 2s sampling interval may miss frames where Burke is clearly visible

**Potential fixes**:
1. Lower `det_thresh` (e.g., 0.3) to catch more marginal detections
2. Increase sampling rate (0.5s instead of 2s) for more frame coverage
3. Upscale frames before face detection (e.g., 2x super-resolution)
4. Use a different face detector (e.g., YOLO-Face, MediaPipe) as fallback
5. Manual character registration: provide reference face crops for characters, then match by embedding similarity without relying on exclusive speaking segments

**References**: `scripts/face_tracker.py:46-82` (detect_faces), `scripts/dialogue_metadata.py:391-415` (Pass 0 face detection loop)

## Remaining Optimization Opportunities

### 1. Window size tuning (medium impact)

120s windows as a compromise between 90s (faster) and 180s (better quality):
- O(3000^2) = 9M attention vs O(2250^2) = 5.06M vs O(4500^2) = 20.25M
- Fewer chunks = less char-chunk inflation

### 2. Cache audio features across characters (low impact)

Audio codec encoding and T5 text encoding are identical for all characters in a chunk (same audio, same empty text prompt). Currently recomputed for each character. Would save ~5s per character but requires modifying `model.separate()` internals.

### 3. GPU-accelerated face detection (low total impact)

Install `onnxruntime-gpu` for CUDA-based InsightFace. Would reduce Pass 0 from 20 min to ~2 min. Small impact on total but meaningful for iteration speed.

### 4. Flash attention via half-precision inference (high potential impact)

R7 confirmed that removing the attention mask alone doesn't help — the model runs in float32, and FlashAttention2 requires float16/bfloat16. To unlock flash attention, the model needs `model.bfloat16()` or `torch.autocast('cuda', dtype=torch.bfloat16)`. This could significantly reduce the ODE solver's O(n^2) attention cost but requires validating that half-precision doesn't degrade separation quality.

### 5. Pre-compute vision features (high complexity)

Run CLIP once per chunk on unmasked frames, then apply masks in feature space. Would eliminate N-1 CLIP forward passes per chunk. However, masking operates at pixel level before CLIP — masked image features differ from masked features of unmasked images. Would need architectural changes.

### 6. Batch multiple characters (VRAM-limited)

The model supports batch processing, but VRAM is already at ~57/80 GB for single character. Would need gradient checkpointing or model offloading to fit 2+ characters.

## Quality Investigation: Dialogue Detection Regression

### Reported regression: 41% (R1) -> 37% (R4) -> 33% (R5)

Investigation and R6 confirmation run found the apparent regression is mostly explained by two factors:

### 1. Overlap double-counting bug (fixed)

The `dialogue_percentage` calculation summed ALL segments including overlapping regions between consecutive chunks, double-counting ~480 segments (96 overlaps × 5 seconds). Corrected R5 value: **31.4%** (not 33.1%). This bug affected all rounds equally, inflating all reported values. Fixed in commit f2200ae, confirmed correct in R6 (33.6% reported = 33.6% deduped).

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

### R6 Confirmation

R6 re-ran with identical settings to R5 (with the dedup fix). Results:
- R5 deduped: 31.4%, R6: 33.6% — a **2.2pp difference** from pure non-determinism
- Face clustering also varies: R5 got 62 multi-speaker chunks, R6 got 67
- Same character IDs (18, 20, 336, 32, 2, 22, 37, 44) but different detection counts

### Conclusions

- **R4→R5 regression was not real** — confirmed by R6 re-run with 2.2pp variation from non-determinism alone
- **R1→R4 regression is real but small** — shorter windows reduce context, but the magnitude is uncertain due to the double-counting bug and non-determinism
- **Binary -40dB threshold is fragile** — 15-17% of segments are borderline. Consider using a confidence margin, smoothing, or multi-threshold approach
- **Run-to-run variance is ~2pp** — any single-run comparison smaller than this is noise

## Key Lessons Learned

1. **Profile before optimizing**: The Round 1 analysis was completely wrong about where time was spent. Frame extraction appeared slow but was actually negligible compared to model inference.

2. **Understand the full data flow**: Reducing input frames to `extract_chunk_frames()` had zero effect because `SAMAudioProcessor.load_video()` resamples frames back to audio token count. The optimization targeted the wrong layer.

3. **O(n^2) attention scaling is the real lever**: Halving the chunk duration gives 4x reduction in per-step attention cost. This is the only optimization that produced meaningful speedup.

4. **torchcodec API matters**: `get_frames_played_in_range()` decodes every frame sequentially (expensive for sparse sampling). `get_frames_played_at(timestamps)` only decodes requested frames. Wrong API choice caused a 55% regression.

5. **Model non-determinism matters for evaluation**: Comparing runs requires accounting for diffusion model noise. Single-run percentage comparisons are unreliable when 15% of segments are near the detection threshold.
