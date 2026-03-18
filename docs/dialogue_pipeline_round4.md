# Dialogue Pipeline Round 4: Shorter Chunk Windows

## Overview

Round 4 reduced the default chunk window from 180s to 90s (overlap from 10s to 5s). This directly reduces the transformer's self-attention cost in the ODE solver, achieving a **22% overall speedup** (6.5h -> 5.1h).

## Change

Default `--window-seconds` changed from 180 to 90, `--overlap-seconds` from 10 to 5.

Commit: e29475a

## Results (A100 80GB, Aliens 2h17m)

| Metric | R3 (180s windows) | R4 (90s windows) | Change |
|--------|-------------------|-------------------|--------|
| Pass 0 (face detection) | 20 min | 20 min | Same |
| Pass 1 (text-only) | 30 min | 27.5 min | -8% |
| Pass 2 (visual separation) | 338 min | 257 min | **-24%** |
| **Total** | **388 min (6.5h)** | **305 min (5.1h)** | **-22%** |
| Per-character time | 152s | ~75s | **-51%** |
| Total char-chunks | 136 | 207 | +52% |
| Chunks total | 49 | 97 | +98% |
| Multi-speaker chunks | 35 (71%) | 65 (67%) | +86% |

## Why It Works

### Transformer attention is O(n^2)

The ODE solver runs 16 diffusion steps, each calling the full transformer on all audio tokens:
- 180s chunk: 4500 tokens, attention = O(4500^2) = 20.25M
- 90s chunk: 2250 tokens, attention = O(2250^2) = 5.06M

This is a **4x reduction** in attention compute per ODE step.

### Vision encoder processes fewer frames

The SAMAudioProcessor resamples video frames to match audio token count:
- 180s: 4500 frames through CLIP (15 batches of 300)
- 90s: 2250 frames through CLIP (8 batches of 300)

**1.9x reduction** in CLIP processing per character.

### Net effect: per-char 2x faster, but more char-chunks

Per-character time dropped from 152s to 75s (2x). However, shorter windows mean characters span more chunks — total char-chunks increased from 136 to 207 (1.5x). Net Pass 2 improvement: 2x / 1.5x = **1.3x faster**.

## Output Quality Comparison

| Metric | R3 (180s) | R4 (90s) |
|--------|-----------|----------|
| Dialogue percentage | 41.0% | 36.8% |
| Dialogue seconds | 3371s | 3027s |
| Gaps detected | 243 | 336 |
| Characters | 8 | 8 |

R4 detects **~10% less dialogue** than R3. This may be due to:
1. Shorter context makes it harder for the model to identify speech in ambiguous sections
2. More overlap regions = more deduplication boundary effects
3. The 1-second RMS segments at chunk boundaries behave differently

This needs investigation — the quality tradeoff may not be acceptable.

## Open Questions

1. **Is 37% vs 41% dialogue detection a meaningful quality regression?** Need manual comparison on specific scenes to determine if R4 misses real dialogue or R3 has false positives.

2. **Would 120s windows be a better tradeoff?** Halfway between 90s and 180s, potentially better quality retention with most of the speedup: O(3000^2) = 9M attention vs O(4500^2) = 20.25M.

3. **Can we reduce char-chunk inflation?** Characters that appear in consecutive 90s chunks could potentially be processed once with a merged frame range, reducing the 207 char-chunks closer to R3's 136.

4. **Per-character minimum visibility threshold**: Skipping characters with <3 face detections in a chunk window would reduce char-chunks further without meaningful quality loss.
