# SAM-Audio Benchmark Results

All benchmarks run on NVIDIA A100 80GB PCIe. Audio source: 203.5s Russian-dubbed movie clip (Aliens, 1986) — 3 speakers (nurse, Ripley, Burke) + ambient SFX. Prompt: `"speech"`. Sample rate: 48kHz.

## Full model (BF16 autocast, 2 candidates, chunked decode)

203s audio, all components loaded (vision encoder, ImageBind ranker, CLAP+Judge ranker, span predictor):

| Model | Inference | Peak VRAM | Load time | Status |
|-------|-----------|-----------|-----------|--------|
| sam-audio-small | 33.88s | 60.86 GB | 96s | OK |
| sam-audio-base | 49.20s | 64.57 GB | 108s | OK |
| sam-audio-large | — | >80 GB | 140s | OOM |
| sam-audio-small-tv | 34.14s | 60.86 GB | 107s | OK |
| sam-audio-base-tv | 49.36s | 64.57 GB | 75s | OK |
| sam-audio-large-tv | — | >80 GB | 151s | OOM |

## Text-only mode (BF16 autocast, 1 candidate, chunked decode, T5 CPU offload)

Skips vision encoder (2.69 GB), ImageBind (~4.8 GB), CLAP+Judge (~4+ GB), span predictor (~4+ GB).

### VRAM scaling with audio duration (small model)

| Duration | Tokens | Inference | Peak VRAM |
|----------|--------|-----------|-----------|
| 30s | ~750 | 3.93s | 6.97 GB |
| 60s | ~1500 | 5.08s | 7.15 GB |
| 120s | ~3000 | 10.81s | 11.63 GB |
| 203s | ~5075 | 21.01s | 17.89 GB |

### Model size comparison (30s audio)

| Model | Inference | Peak VRAM |
|-------|-----------|-----------|
| sam-audio-small | 3.96s | 6.97 GB |
| sam-audio-base | 3.09s | 10.64 GB |
| sam-audio-large | 4.37s | 20.60 GB |

## Quality comparison (text-only, 30s, prompt "speech")

### Metrics

| Model | CLAP | Judge Overall | Precision | Recall | Faithfulness |
|-------|------|---------------|-----------|--------|--------------|
| small | 0.2179 | 3.8861 | 3.9733 | 4.7848 | 4.9083 |
| base | 0.3041 | 4.4488 | 4.5404 | 4.9570 | 4.9644 |
| large | 0.2889 | 4.4136 | 4.4825 | 4.9434 | 4.8936 |

Base has the best quality scores overall. The small-to-base jump is significant (+14% Judge Overall), while base-to-large shows no improvement.

### Cross-model output similarity (cosine)

| Pair | Cosine similarity |
|------|-------------------|
| small vs base | 0.9059 |
| small vs large | 0.9254 |
| base vs large | 0.9194 |

All models produce ~90% similar outputs — they all extract the same speech, but base/large produce cleaner separation.

### Signal characteristics

| Model | Target RMS | Residual RMS | Target/Residual ratio |
|-------|------------|--------------|----------------------|
| small | 0.0014 | 0.0045 | 0.31 |
| base | 0.0014 | 0.0045 | 0.30 |
| large | 0.0015 | 0.0046 | 0.32 |

## GPU compatibility (text-only mode)

| GPU VRAM | Max audio duration | Recommended model |
|----------|-------------------|-------------------|
| 8 GB | ~30-40s | small |
| 12 GB | ~90-100s | small or base (30s) |
| 16 GB | ~150-160s | small; base up to ~60s |
| 24 GB | 203s+ | any size |
| 80 GB | 203s+ (full model) | any size, with rankers |

## Recommendations

- **Best value**: `sam-audio-base` with `text_only=True` on a 12-16 GB GPU — best quality at moderate VRAM
- **Most accessible**: `sam-audio-small` with `text_only=True` — runs on 8 GB GPUs, 30s audio
- **Maximum quality**: `sam-audio-base` full model on 80 GB GPU — adds reranking for best candidate selection

## Reproduction

```python
import torch
from sam_audio import SAMAudio, SAMAudioProcessor

model = SAMAudio.from_pretrained("facebook/sam-audio-small", text_only=True)
model = model.eval().to("cuda")
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-small")

batch = processor(descriptions=["speech"], audios=[audio_tensor]).to("cuda")

with torch.autocast("cuda", dtype=torch.bfloat16):
    result = model.separate(batch, reranking_candidates=1, max_chunk_tokens=500)
```

## Test environment

- GPU: NVIDIA A100 80GB PCIe
- PyTorch: 2.6.0+cu124
- Python: 3.11
- Branch: `rnd`
- Date: 2026-03-17
