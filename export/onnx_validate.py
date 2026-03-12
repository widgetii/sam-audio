# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Validate ONNX exports against PyTorch inference.

Runs the same input through both pipelines and compares outputs:
  1. PyTorch: SAMAudio.separate() end-to-end
  2. ONNX: Manual orchestration of exported components + Python ODE loop

Usage:
    uv run python export/onnx_validate.py [--model-dir export/onnx_models] [--model-id facebook/sam-audio-small]
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import transformers

from sam_audio.model.model import SAMAudio


def load_pytorch_model(model_id: str, device: str = "cpu") -> SAMAudio:
    """Load SAMAudio in text_only mode for reference inference."""
    print(f"Loading PyTorch model {model_id}...")
    model = SAMAudio.from_pretrained(model_id, text_only=True)
    model = model.to(device).eval()
    return model


def create_ort_session(path: Path) -> ort.InferenceSession:
    """Create an ONNX Runtime session."""
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


def pytorch_inference(
    model: SAMAudio,
    waveform: torch.Tensor,
    text: str,
    noise: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Run PyTorch inference and return generated features (pre-decode)."""
    # Encode audio
    audio_features = model._get_audio_features(waveform)
    text_features, text_mask = model.text_encoder([text])
    B, T, _ = audio_features.shape
    masked_video_features = audio_features.new_zeros(B, model._vision_encoder_dim, T)
    audio_pad_mask = torch.ones(B, T, dtype=torch.bool, device=device)

    forward_args = {
        "audio_features": audio_features,
        "text_features": text_features,
        "text_mask": text_mask,
        "masked_video_features": masked_video_features,
        "anchor_ids": None,
        "anchor_alignment": None,
        "audio_pad_mask": audio_pad_mask,
    }

    # ODE solve
    def vector_field(t, noisy_audio):
        return model.forward(
            noisy_audio=noisy_audio,
            time=t.expand(noisy_audio.size(0)),
            **forward_args,
        )

    from torchdiffeq import odeint

    generated_features = odeint(
        vector_field,
        noise,
        torch.tensor([0.0, 1.0], device=device),
        method="midpoint",
        options={"step_size": 2 / 32},
    )[-1]

    return generated_features


def onnx_inference(
    model_dir: Path,
    waveform: torch.Tensor,
    text: str,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Run ONNX inference with Python ODE loop."""
    # Load sessions
    t5_session = create_ort_session(model_dir / "t5_encoder.onnx")
    dit_session = create_ort_session(model_dir / "dit_forward.onnx")
    enc_session = create_ort_session(model_dir / "dacvae_encoder.onnx")

    # 1. Encode audio with DACVAE
    waveform_np = waveform.numpy()
    enc_result = enc_session.run(None, {"waveform": waveform_np})
    features_np = enc_result[0]  # [B, C, T]
    # Transpose to [B, T, C] and double (target + residual channels)
    audio_features_np = np.transpose(features_np, (0, 2, 1))
    audio_features_np = np.concatenate([audio_features_np, audio_features_np], axis=2)

    B, T, C2 = audio_features_np.shape

    # 2. Encode text with T5
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir / "t5_tokenizer")
    encoded = tokenizer(
        [text], truncation=True, max_length=512, padding="longest", return_tensors="np"
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    t5_result = t5_session.run(
        None, {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    text_features_np = t5_result[0]  # [B, S, 768]
    text_mask_np = attention_mask.astype(bool)

    # 3. Prepare remaining inputs
    masked_video_features_np = np.zeros((B, 1024, T), dtype=np.float32)
    audio_pad_mask_np = np.ones((B, T), dtype=bool)

    # 4. ODE loop (midpoint method, step_size=2/32 over [0,1])
    dt = 2.0 / 32
    num_steps = 16  # midpoint with step_size=2/32 over interval [0,1]: 16 steps
    y = noise.numpy().copy()

    forward_inputs = {
        "audio_features": audio_features_np.astype(np.float32),
        "text_features": text_features_np.astype(np.float32),
        "masked_video_features": masked_video_features_np,
        "text_mask": text_mask_np,
        "audio_pad_mask": audio_pad_mask_np,
    }

    for step in range(num_steps):
        t_val = step * dt
        # k1 = f(t, y)
        inputs = {
            "noisy_audio": y.astype(np.float32),
            "time": np.array([t_val], dtype=np.float32),
            **forward_inputs,
        }
        k1 = dit_session.run(None, inputs)[0]

        # y_mid = y + k1 * dt/2
        y_mid = y + k1 * (dt / 2)

        # k2 = f(t + dt/2, y_mid)
        inputs_mid = {
            "noisy_audio": y_mid.astype(np.float32),
            "time": np.array([t_val + dt / 2], dtype=np.float32),
            **forward_inputs,
        }
        k2 = dit_session.run(None, inputs_mid)[0]

        # y = y + k2 * dt
        y = y + k2 * dt

    return torch.from_numpy(y)


def compare_outputs(pytorch_out: torch.Tensor, onnx_out: torch.Tensor):
    """Compare PyTorch and ONNX outputs with multiple metrics."""
    pytorch_flat = pytorch_out.float().flatten()
    onnx_flat = onnx_out.float().flatten()

    max_abs_err = (pytorch_flat - onnx_flat).abs().max().item()
    mean_abs_err = (pytorch_flat - onnx_flat).abs().mean().item()

    cos_sim = torch.nn.functional.cosine_similarity(
        pytorch_flat.unsqueeze(0), onnx_flat.unsqueeze(0)
    ).item()

    print("\n=== Numerical Comparison (generated features, pre-decode) ===")
    print(f"  Max absolute error:  {max_abs_err:.6f}")
    print(f"  Mean absolute error: {mean_abs_err:.6f}")
    print(f"  Cosine similarity:   {cos_sim:.6f}")
    print()

    # Evaluate pass/fail
    passed = max_abs_err < 1e-2 and cos_sim > 0.999
    if passed:
        print("PASSED: Outputs are numerically equivalent within tolerance.")
    else:
        print("FAILED: Outputs diverge beyond acceptable tolerance.")
        if max_abs_err >= 1e-2:
            print(f"  Max abs error {max_abs_err:.6f} >= threshold 1e-2")
        if cos_sim <= 0.999:
            print(f"  Cosine similarity {cos_sim:.6f} <= threshold 0.999")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX vs PyTorch")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="export/onnx_models",
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/sam-audio-small",
        help="HuggingFace model ID for PyTorch reference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for PyTorch inference",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=5.0,
        help="Duration of test audio in seconds (shorter = faster)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="a dog barking",
        help="Text prompt for separation",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    device = args.device

    # Load PyTorch model
    model = load_pytorch_model(args.model_id, device=device)

    # Create synthetic test audio
    sample_rate = model.sample_rate
    num_samples = int(args.duration_sec * sample_rate)
    # Align to hop_length
    hop = model.audio_codec.hop_length
    num_samples = (num_samples // hop) * hop
    waveform = torch.randn(1, 1, num_samples, device=device)

    print(f"Test audio: {args.duration_sec}s, {num_samples} samples, {sample_rate} Hz")
    print(f"Text prompt: '{args.text}'")

    # Encode with PyTorch to get matching shapes for noise
    with torch.inference_mode():
        audio_features = model._get_audio_features(waveform)
    B, T, C2 = audio_features.shape
    print(f"Audio features shape: [{B}, {T}, {C2}]")

    # Fixed noise for reproducibility
    torch.manual_seed(42)
    noise = torch.randn(B, T, C2, device=device)

    # Run PyTorch inference
    print("\nRunning PyTorch inference...")
    with torch.inference_mode():
        pytorch_out = pytorch_inference(model, waveform, args.text, noise, device)

    # Move model off device to free memory before ONNX
    model.to("cpu")

    # Run ONNX inference (always on CPU)
    print("Running ONNX inference...")
    onnx_out = onnx_inference(model_dir, waveform.cpu(), args.text, noise.cpu())

    # Compare
    passed = compare_outputs(pytorch_out.cpu(), onnx_out.cpu())
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
