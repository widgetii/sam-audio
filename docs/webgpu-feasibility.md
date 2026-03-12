# SAM-Audio WebGPU Feasibility Study

## Summary

Running SAM-Audio (small, text_only) in the browser via WebGPU is feasible but challenging. The core architecture — codec encoder, diffusion transformer, codec decoder — can be exported to ONNX and executed via ONNX Runtime Web with WebGPU backend. Key constraints are the 8 GB WebGPU buffer limit and the iterative ODE solver requiring 32 model evaluations per separation.

## Browser WebGPU Support

| Browser | Status | Buffer Limit |
|---------|--------|-------------|
| Chrome 113+ | Stable | 8 GB (device-dependent) |
| Edge 113+ | Stable | 8 GB |
| Firefox | Nightly only | Experimental |
| Safari | Behind flag | Limited |

WebGPU provides compute shader access with performance roughly 60-80% of native Vulkan. The 8 GB buffer limit is a hard constraint — all model weights + activations + intermediate tensors must fit.

## Memory Budget

SAM-Audio small (text_only, FP16):

| Component | FP32 Size | FP16 Size |
|-----------|-----------|-----------|
| T5-base encoder | 850 MB | 425 MB |
| DiT (transformer + proj + align) | ~600 MB | ~300 MB |
| DACVAE (encoder + decoder) | ~200 MB | ~100 MB |
| **Total weights** | **~1.65 GB** | **~825 MB** |
| Peak activations (30s audio) | ~2 GB | ~1 GB |
| **Peak total** | **~3.65 GB** | **~1.8 GB** |

FP16 fits well within the 8 GB limit. INT8 quantization would further reduce weights to ~415 MB. Audio duration is the main scaling factor — 30s audio should work, longer audio may require chunking.

## Framework Options

### ONNX Runtime Web (Recommended)

- Mature WebGPU backend (`wasm` + `webgpu` execution providers)
- Direct ONNX model loading — matches our Phase 1 export format
- Used by Transformers.js under the hood
- Supports FP16 and INT8/INT4 quantized models
- Active development, good operator coverage for transformer architectures

### Transformers.js

- Higher-level API built on ONNX Runtime Web
- Has T5 encoder support out of the box
- Would need custom pipeline for DiT + ODE loop
- Good for the T5 component, less useful for custom model components

### Apache TVM (WebGPU)

- Compiles models to optimized WebGPU shaders
- Potentially better performance than ONNX Runtime Web
- Used by WebLLM project for LLM inference
- Higher integration complexity, less mature ecosystem
- Worth evaluating in Phase 2 if ONNX Runtime Web performance is insufficient

## Component-by-Component Export Analysis

### T5 Encoder

**Status: Straightforward**

Standard HuggingFace T5EncoderModel. Well-tested ONNX export path via `torch.onnx.export` or `optimum`. Called once per separation — not performance critical.

### DiT Forward (ODE Step Function)

**Status: Exportable with workarounds**

The ODE solver (`torchdiffeq.odeint`) uses a Python closure, so the full pipeline cannot be a single ONNX graph. Instead, we export the closure body (`SAMAudio.forward()`) and reimplement the ODE loop in JavaScript.

Concerns handled:
- `einops.rearrange` — traces correctly with concrete shapes during export
- `AlignModalities` — has `if tgt is None` branch; always pass a tensor (zeros) so tracing takes the non-None path
- `RMSNorm` with `.float()` cast — traces fine, produces correct ONNX ops
- `RotaryEmbedding` with dynamic slicing `freqs_cis[:, :seqlen]` — supported in opset 17
- `scaled_dot_product_attention` — ONNX opset 17 supports this natively

### DACVAE Codec

**Status: Exportable with preprocessing**

Requires `torch.nn.utils.remove_weight_norm()` on all weight-normed layers before tracing. Split into encoder and decoder for flexibility (chunked decode calls decoder multiple times). The `cudnn.flags(enabled=False)` context manager is handled by disabling cuDNN globally during export.

### ODE Solver (JavaScript)

**Status: Must reimplement**

The midpoint method is simple enough to implement in JS:

```javascript
// Midpoint method: 16 steps, dt = 2/32
let y = noise;
const dt = 2/32;
for (let step = 0; step < 16; step++) {
    const t = step * dt;
    const k1 = await ditSession.run({ noisy_audio: y, time: [t], ...args });
    const y_mid = add(y, scale(k1, dt/2));
    const k2 = await ditSession.run({ noisy_audio: y_mid, time: [t + dt/2], ...args });
    y = add(y, scale(k2, dt));
}
```

This is 32 DiT evaluations per separation. Each evaluation involves GPU compute + CPU-GPU synchronization, which is the main performance bottleneck in browser.

## Precedents

### Stable Diffusion in Browser (WebSD)

- Runs full SD 1.5 pipeline in Chrome via ONNX Runtime Web
- Similar architecture: text encoder + iterative denoiser + decoder
- ~50 steps of UNet evaluation (we need 32 DiT evaluations)
- Achieves ~15-30 seconds per image on modern GPUs
- Demonstrates feasibility of iterative diffusion models in WebGPU

### Whisper in Browser

- Runs via Transformers.js + ONNX Runtime Web
- Encoder-decoder architecture with ONNX export
- INT8 quantization for smaller models
- Real-time factor ~0.3x for whisper-small on WebGPU

### WebLLM

- Runs LLaMA-7B in browser via TVM WebGPU
- Demonstrates large model deployment feasibility
- Uses INT4 quantization to fit in memory

## Estimated Browser Performance

For SAM-Audio small, text_only, 30s audio, FP16:

| Step | Native GPU (A100) | WebGPU (Estimated) |
|------|-------------------|-------------------|
| T5 encode | ~50ms | ~200-500ms |
| DACVAE encode | ~20ms | ~100-200ms |
| DiT x32 steps | ~2s | ~10-30s |
| DACVAE decode | ~30ms | ~100-300ms |
| **Total** | **~2.1s** | **~10-31s** |

The DiT evaluation dominates. Each of the 32 steps requires a full transformer forward pass plus CPU-GPU sync overhead. WebGPU compute is roughly 3-5x slower than native CUDA, and sync overhead adds ~5-10ms per step.

For a consumer laptop GPU (RTX 3060 mobile / Apple M2):
- Expect 15-45 seconds per 30s audio separation
- Acceptable for a "process and wait" UX, not real-time

## Quantization Opportunities (Phase 2)

| Format | Weight Size | Expected Quality | WebGPU Support |
|--------|------------|-----------------|----------------|
| FP16 | 825 MB | Baseline | Full |
| INT8 (dynamic) | ~415 MB | ~0.1 dB degradation | ONNX RT Web |
| INT4 (GPTQ/AWQ) | ~210 MB | ~0.5 dB degradation | TVM WebGPU |

INT8 is the sweet spot — halves model size with minimal quality loss. INT4 would enable running on 4 GB WebGPU devices but needs quality validation.

## Risks

1. **ONNX operator coverage**: Some PyTorch ops may not have ONNX equivalents. The `einops` patterns and custom rope implementation are the most likely to cause issues during export. Mitigation: Phase 1 validates this.

2. **Numerical precision**: FP16 accumulation in WebGPU may differ from PyTorch's mixed-precision behavior. The ODE solver amplifies small errors over 32 steps. Mitigation: validate with cosine similarity > 0.999 tolerance.

3. **Memory fragmentation**: WebGPU memory allocation is less flexible than CUDA. Large contiguous buffers may fail even if total memory is sufficient. Mitigation: chunked processing, smaller batch sizes.

4. **Browser tab limits**: Browsers may kill tabs using excessive GPU memory. No reliable way to query available GPU memory from JavaScript. Mitigation: conservative memory estimates, graceful error handling.

5. **Cross-browser compatibility**: WebGPU behavior varies across browsers and GPU drivers. Testing matrix is large. Mitigation: target Chrome first, expand later.

## Roadmap

### Phase 1: ONNX Export PoC (Current)
- Export 3 ONNX models from SAMAudio small text_only
- Validate numerical equivalence with Python ONNX Runtime
- Document export issues and workarounds

### Phase 2: Browser Runtime
- Load ONNX models in browser with ONNX Runtime Web (WebGPU backend)
- Implement JavaScript ODE solver (midpoint method)
- INT8 quantization of DiT and T5
- Basic web UI for file upload + text prompt

### Phase 3: Optimization
- INT4 quantization evaluation
- Reduced ODE steps (16 → 8) with quality trade-off analysis
- Streaming decode for progressive audio output
- Service Worker for background processing
- IndexedDB caching for model weights

## Conclusion

Browser deployment of SAM-Audio small is technically feasible with today's WebGPU. The main challenges are export correctness (solved in Phase 1) and inference latency (32 DiT evaluations at ~0.5-1s each in WebGPU). The resulting 15-45 second processing time is acceptable for an offline tool. Memory requirements (~1.8 GB FP16) fit within browser limits with room to spare.
