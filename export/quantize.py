# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Quantize ONNX models to FP16 for browser deployment.

Converts FP32 ONNX models to FP16, roughly halving model size.
Also merges external data files into single ONNX files for easier serving.

Usage:
    uv run python export/quantize.py [--input-dir export/onnx_models] [--output-dir export/onnx_models_fp16]
"""

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


def convert_to_fp16(input_path: Path, output_path: Path):
    """Convert ONNX model from FP32 to FP16."""
    print(f"  Loading {input_path.name}...")
    model = onnx.load(str(input_path))

    from onnx import numpy_helper

    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            data = numpy_helper.to_array(initializer)
            data_fp16 = data.astype("float16")
            new_init = numpy_helper.from_array(data_fp16, name=initializer.name)
            initializer.CopyFrom(new_init)

    onnx.save(model, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  -> {output_path} ({size_mb:.1f} MB)")


def quantize_int8(input_path: Path, output_path: Path):
    """Quantize ONNX model to INT8 (dynamic quantization)."""
    print(f"  Preprocessing {input_path.name} for quantization...")
    preprocessed = input_path.parent / f"{input_path.stem}_preprocessed.onnx"

    try:
        quant_pre_process(str(input_path), str(preprocessed))
    except Exception as e:
        print(f"  Warning: preprocessing failed ({e}), using original model")
        preprocessed = input_path

    print("  Quantizing to INT8...")
    quantize_dynamic(str(preprocessed), str(output_path))

    # Clean up preprocessed file
    if preprocessed != input_path and preprocessed.exists():
        preprocessed.unlink()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  -> {output_path} ({size_mb:.1f} MB)")


def merge_external_data(input_path: Path, output_path: Path):
    """Load ONNX model with external data and save as single file."""
    print(f"  Merging external data for {input_path.name}...")
    model = onnx.load(str(input_path), load_external_data=True)
    onnx.save(
        model,
        str(output_path),
        save_as_external_data=False,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  -> {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="export/onnx_models",
        help="Input directory with FP32 ONNX models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="export/onnx_models_web",
        help="Output directory for quantized models",
    )
    parser.add_argument(
        "--dit-format",
        choices=["fp16", "int8"],
        default="fp16",
        help="Quantization format for DiT (largest model)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "t5_encoder": input_dir / "t5_encoder.onnx",
        "dit_forward": input_dir / "dit_forward.onnx",
        "dacvae_encoder": input_dir / "dacvae_encoder.onnx",
        "dacvae_decoder": input_dir / "dacvae_decoder.onnx",
    }

    for name, path in models.items():
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue

        print(f"\nProcessing {name}:")
        out_path = output_dir / f"{name}.onnx"

        # T5 uses external data files — merge first
        external_data = path.parent / f"{path.name}.data"
        if external_data.exists():
            merged = output_dir / f"{name}_merged.onnx"
            merge_external_data(path, merged)
            path = merged

        if name == "dit_forward" and args.dit_format == "int8":
            quantize_int8(path, out_path)
        else:
            convert_to_fp16(path, out_path)

        # Clean up merged temp file
        merged_path = output_dir / f"{name}_merged.onnx"
        if merged_path.exists() and merged_path != out_path:
            merged_path.unlink()

    # Copy tokenizer
    tokenizer_src = input_dir / "t5_tokenizer"
    tokenizer_dst = output_dir / "t5_tokenizer"
    if tokenizer_src.exists() and not tokenizer_dst.exists():
        import shutil

        shutil.copytree(str(tokenizer_src), str(tokenizer_dst))
        print(f"\nCopied tokenizer to {tokenizer_dst}")

    print("\nQuantization complete!")
    print(f"Output directory: {output_dir}")
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.onnx"))
    print(f"Total ONNX size: {total_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
