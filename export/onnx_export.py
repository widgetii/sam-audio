# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Export SAMAudio components to ONNX format.

Exports 3 separate ONNX models that can be orchestrated externally:
  1. t5_encoder.onnx       — text encoding (called once)
  2. dit_forward.onnx      — one ODE step (called N times by solver)
  3. dacvae_encoder.onnx   — audio waveform → features
  4. dacvae_decoder.onnx   — features → audio waveform

Usage:
    uv run python export/onnx_export.py [--output-dir export/onnx_models] [--model-id facebook/sam-audio-small]
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from sam_audio.model.model import SAMAudio


def remove_weight_norm_recursive(module: nn.Module):
    """Remove weight_norm from all submodules (required for ONNX tracing)."""
    for child in module.modules():
        if hasattr(child, "weight_g"):
            torch.nn.utils.remove_weight_norm(child)


class DitForwardWrapper(nn.Module):
    """Wraps SAMAudio.forward() as a standalone module for ONNX export.

    In text_only mode with no spans, anchor_ids and anchor_alignment are None,
    so embed_anchors is a no-op. We bake this into the wrapper.
    """

    def __init__(self, model: SAMAudio):
        super().__init__()
        self.proj = model.proj
        self.align_masked_video = model.align_masked_video
        self.embed_anchors = model.embed_anchors
        self.memory_proj = model.memory_proj
        self.timestep_emb = model.timestep_emb
        self.transformer = model.transformer

    def forward(
        self,
        noisy_audio: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        time: torch.Tensor,
        masked_video_features: torch.Tensor,
        text_mask: torch.Tensor,
        audio_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        # align_inputs (no anchors in text_only mode)
        x = torch.cat(
            [noisy_audio, torch.zeros_like(audio_features), audio_features], dim=2
        )
        projected = self.proj(x)
        aligned = self.align_masked_video(projected, masked_video_features)
        # embed_anchors with anchor_ids=None is identity — skip

        # memory projection + timestep embedding
        timestep_emb = self.timestep_emb(time, pos=time).unsqueeze(1)
        memory = self.memory_proj(text_features) + timestep_emb

        return self.transformer(
            aligned,
            time,
            padding_mask=audio_pad_mask,
            memory=memory,
            memory_padding_mask=text_mask,
        )


class DACVAEEncoderWrapper(nn.Module):
    """Wraps the DACVAE encoder path for ONNX export."""

    def __init__(self, codec):
        super().__init__()
        self.encoder = codec.encoder
        self.quantizer_in_proj = codec.quantizer.in_proj
        self.hop_length = codec.hop_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Note: Input waveform must already be padded to a multiple of hop_length.
        # The padding logic uses data-dependent control flow that can't be traced.
        z = self.encoder(waveform)
        mean, _ = self.quantizer_in_proj(z).chunk(2, dim=1)
        return mean


class DACVAEDecoderWrapper(nn.Module):
    """Wraps the DACVAE decoder path for ONNX export."""

    def __init__(self, codec):
        super().__init__()
        self.quantizer_out_proj = codec.quantizer.out_proj
        self.decoder = codec.decoder

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        emb = self.quantizer_out_proj(features)
        return self.decoder(emb)


def load_model(model_id: str, device: str = "cpu") -> SAMAudio:
    """Load SAMAudio in text_only mode."""
    print(f"Loading model {model_id} (text_only=True)...")
    model = SAMAudio.from_pretrained(model_id, text_only=True)
    model = model.to(device).eval()
    return model


def export_t5_encoder(model: SAMAudio, output_dir: Path, device: str = "cpu"):
    """Export T5 encoder to ONNX."""
    print("Exporting T5 encoder...")
    t5_model = model.text_encoder.model.eval()

    B, S = 1, 16
    input_ids = torch.randint(0, 100, (B, S), device=device)
    attention_mask = torch.ones(B, S, dtype=torch.long, device=device)

    output_path = output_dir / "t5_encoder.onnx"
    torch.onnx.export(
        t5_model,
        (input_ids, attention_mask),
        str(output_path),
        opset_version=18,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        },
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  -> {output_path} ({size_mb:.1f} MB)")
    return output_path


def export_dit_forward(model: SAMAudio, output_dir: Path, device: str = "cpu"):
    """Export DiT forward pass (one ODE step) to ONNX."""
    print("Exporting DiT forward...")
    wrapper = DitForwardWrapper(model).to(device).eval()

    # Compute realistic trace shapes from the model config.
    # The TorchScript tracer bakes shape-dependent padding as constants,
    # so we must trace with the same T that will be used at inference.
    # Default validation uses 5s audio at 48kHz with hop_length=1920 → T=125.
    hop = model.audio_codec.hop_length
    sample_rate = model.audio_codec.sample_rate
    trace_duration_sec = 5.0
    num_samples = int(trace_duration_sec * sample_rate)
    num_samples = (num_samples // hop) * hop
    T = num_samples // hop

    C = model.audio_codec.quantizer.in_proj.out_features // 2
    B, S = 1, 16

    noisy_audio = torch.randn(B, T, 2 * C, device=device)
    audio_features = torch.randn(B, T, 2 * C, device=device)
    text_features = torch.randn(B, S, 768, device=device)
    time = torch.tensor([0.5], device=device)
    masked_video_features = torch.zeros(B, 1024, T, device=device)
    text_mask = torch.ones(B, S, dtype=torch.bool, device=device)
    audio_pad_mask = torch.ones(B, T, dtype=torch.bool, device=device)

    output_path = output_dir / "dit_forward.onnx"
    # Use legacy TorchScript exporter — the dynamo exporter can't handle
    # dynamic padding in Patcher's Conv1d and RotaryEmbedding slicing
    torch.onnx.export(
        wrapper,
        (
            noisy_audio,
            audio_features,
            text_features,
            time,
            masked_video_features,
            text_mask,
            audio_pad_mask,
        ),
        str(output_path),
        opset_version=18,
        dynamo=False,
        input_names=[
            "noisy_audio",
            "audio_features",
            "text_features",
            "time",
            "masked_video_features",
            "text_mask",
            "audio_pad_mask",
        ],
        output_names=["velocity"],
        dynamic_axes={
            "noisy_audio": {0: "batch", 1: "seq_len"},
            "audio_features": {0: "batch", 1: "seq_len"},
            "text_features": {0: "batch", 1: "text_len"},
            "time": {0: "batch"},
            "masked_video_features": {0: "batch", 2: "seq_len"},
            "text_mask": {0: "batch", 1: "text_len"},
            "audio_pad_mask": {0: "batch", 1: "seq_len"},
            "velocity": {0: "batch", 1: "seq_len"},
        },
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  -> {output_path} ({size_mb:.1f} MB)")
    return output_path


def export_dacvae(model: SAMAudio, output_dir: Path, device: str = "cpu"):
    """Export DACVAE encoder and decoder to separate ONNX files."""
    print("Exporting DACVAE encoder + decoder...")

    # Remove weight_norm before export (required for tracing)
    remove_weight_norm_recursive(model.audio_codec)

    # Disable cuDNN (DACVAE uses this context manager in forward)
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    try:
        # --- Encoder ---
        encoder_wrapper = DACVAEEncoderWrapper(model.audio_codec).to(device).eval()
        hop = model.audio_codec.hop_length
        sample_rate = model.audio_codec.sample_rate
        B = 1
        # Trace with 5s audio (matching default validation duration)
        samples = int(5.0 * sample_rate)
        samples = (samples // hop) * hop
        waveform = torch.randn(B, 1, samples, device=device)

        encoder_path = output_dir / "dacvae_encoder.onnx"
        torch.onnx.export(
            encoder_wrapper,
            (waveform,),
            str(encoder_path),
            opset_version=18,
            dynamo=False,
            input_names=["waveform"],
            output_names=["features"],
            dynamic_axes={
                "waveform": {0: "batch", 2: "samples"},
                "features": {0: "batch", 2: "time"},
            },
        )
        size_mb = encoder_path.stat().st_size / (1024 * 1024)
        print(f"  -> {encoder_path} ({size_mb:.1f} MB)")

        # --- Decoder ---
        decoder_wrapper = DACVAEDecoderWrapper(model.audio_codec).to(device).eval()
        C_latent = model.audio_codec.quantizer.in_proj.out_features // 2
        T = samples // hop
        features = torch.randn(B, C_latent, T, device=device)

        decoder_path = output_dir / "dacvae_decoder.onnx"
        torch.onnx.export(
            decoder_wrapper,
            (features,),
            str(decoder_path),
            opset_version=18,
            dynamo=False,
            input_names=["features"],
            output_names=["waveform"],
            dynamic_axes={
                "features": {0: "batch", 2: "time"},
                "waveform": {0: "batch", 2: "samples"},
            },
        )
        size_mb = decoder_path.stat().st_size / (1024 * 1024)
        print(f"  -> {decoder_path} ({size_mb:.1f} MB)")
    finally:
        torch.backends.cudnn.enabled = prev_cudnn

    return encoder_path, decoder_path


def main():
    parser = argparse.ArgumentParser(description="Export SAMAudio to ONNX")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="export/onnx_models",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/sam-audio-small",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export tracing (cpu recommended)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_id, device=args.device)

    # Save tokenizer for later use in validation
    print("Saving T5 tokenizer...")
    tokenizer = model.text_encoder.tokenizer
    tokenizer.save_pretrained(output_dir / "t5_tokenizer")

    # Export all components
    t5_path = export_t5_encoder(model, output_dir, device=args.device)
    dit_path = export_dit_forward(model, output_dir, device=args.device)
    enc_path, dec_path = export_dacvae(model, output_dir, device=args.device)

    print("\nExport complete! Files:")
    for p in [t5_path, dit_path, enc_path, dec_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
