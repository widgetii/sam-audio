# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM-Audio (Segment Anything Audio) is a Meta Research model for isolating and separating sounds in audio using text, visual, or time-span prompts. Built on Perception-Encoder Audio-Visual (PE-AV) and diffusion transformers (DiT). Python 3.11+.

## Common Commands

```bash
# Install
pip install .

# Lint
ruff check .

# Format check
ruff format --check .

# Auto-fix lint issues
ruff check --fix .

# Auto-format
ruff format .

# Run evaluation
python eval/main.py --setting sfx speech music --batch-size 1 --candidates 8

# Multi-GPU evaluation
torchrun --nproc_per_node=8 eval/main.py
```

There are no unit tests in this repo. CI runs `ruff format --check` and `ruff check` only.

## Architecture

### Core Pipeline

`SAMAudioProcessor` prepares inputs (audio at 48kHz, text via T5 tokenizer, video frames) into `Batch` objects. `SAMAudio` runs diffusion-based separation via its `separate()` method, returning `SeparationResult` with `target`, `residual`, and `noise` waveforms.

### Key Modules (`sam_audio/`)

- **`processor.py`** — Input preprocessing: audio loading/resampling, text tokenization, video frame extraction, batching. Also `SAMAudioJudgeProcessor`.
- **`model/model.py`** — `SAMAudio` class: orchestrates codec encoding, transformer diffusion, ODE solving, optional span prediction, and candidate re-ranking.
- **`model/transformer.py`** — DiT (Diffusion Transformer) with cross-attention to text/vision embeddings.
- **`model/config.py`** — Dataclass configs: `SAMAudioConfig`, `DACVAEConfig`, `T5EncoderConfig`, `PerceptionEncoderConfig`, `TransformerConfig`.
- **`model/codec.py`** — DACVAE wrapper for audio encode/decode.
- **`model/vision_encoder.py`** — Perception-Encoder wrapper for video frame encoding with mask support.
- **`model/judge.py`** — `SAMAudioJudgeModel` for quality scoring (precision, recall, faithfulness).
- **`model/base.py`** — `BaseModel` with HuggingFace Hub integration (`ModelHubMixin`).
- **`ranking/`** — Pluggable ranker system: `ClapRanker`, `JudgeRanker`, `ImageBindRanker`, `EnsembleRanker`. Factory via `create_ranker()`.

### Evaluation (`eval/`)

- **`main.py`** — CLI evaluation script with argparse. Computes Judge, CLAP, ImageBind, and Aesthetic metrics.
- **`dataset/`** — Dataset loaders for SAM-Audio Benchmark and MUSDB.
- **`metrics/`** — Metric implementations wrapping the ranking models.

### Model Variants

Three sizes: `sam-audio-small`, `sam-audio-base`, `sam-audio-large`. TV variants (`-tv` suffix) are optimized for visual prompting. All hosted on HuggingFace (`facebook/sam-audio-*`).

### Prompt Types

1. **Text**: natural language description (noun/verb phrases)
2. **Visual**: video frames + binary masks identifying target objects
3. **Span**: time ranges `[("+", start, end), ...]` where target sounds occur

## Ruff Configuration

Target: py311. Enabled rules: B, C, E, W, F, I. Ignored: E501 (line length), E731, C901, B006.
