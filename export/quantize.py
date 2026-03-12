# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Quantize ONNX models to FP16 for browser deployment.

Converts FP32 ONNX models to FP16, roughly halving model size.
Handles large models (>2GB) via ONNX external data format.

Usage:
    uv run python export/quantize.py [--input-dir export/onnx_models] [--output-dir export/onnx_models_web]
"""

import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper


def convert_to_fp16(input_path: Path, output_path: Path, load_external=False):
    """Convert ONNX model from FP32 to FP16."""
    print(f"  Loading {input_path.name}...")
    model = onnx.load(str(input_path), load_external_data=load_external)

    total_params = 0
    converted_params = 0
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            data = numpy_helper.to_array(initializer)
            total_params += data.size
            data_fp16 = data.astype("float16")
            new_init = numpy_helper.from_array(data_fp16, name=initializer.name)
            initializer.CopyFrom(new_init)
            converted_params += data.size

    print(f"  Converted {converted_params:,} / {total_params:,} FP32 params to FP16")

    # Estimate output size to decide on external data
    est_size_mb = sum(
        numpy_helper.to_array(init).nbytes for init in model.graph.initializer
    ) / (1024 * 1024)

    if est_size_mb > 1800:
        # Save with external data for large models (protobuf 2GB limit)
        data_path = output_path.name + ".data"
        print(f"  Saving with external data ({data_path})...")
        onnx.save(
            model,
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path,
        )
        data_file = output_path.parent / data_path
        total_mb = (output_path.stat().st_size + data_file.stat().st_size) / (
            1024 * 1024
        )
        print(f"  -> {output_path} + {data_path} ({total_mb:.1f} MB total)")
    else:
        onnx.save(model, str(output_path))
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  -> {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models to FP16")
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
        help="Output directory for FP16 models",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        "t5_encoder",
        "dit_forward",
        "dacvae_encoder",
        "dacvae_decoder",
    ]

    for name in models:
        path = input_dir / f"{name}.onnx"
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue

        print(f"\nProcessing {name}:")
        out_path = output_dir / f"{name}.onnx"

        # Check for external data (T5 dynamo exporter creates .onnx.data files)
        has_external = (path.parent / f"{path.name}.data").exists()
        convert_to_fp16(path, out_path, load_external=has_external)

    # Copy tokenizer
    tokenizer_src = input_dir / "t5_tokenizer"
    tokenizer_dst = output_dir / "t5_tokenizer"
    if tokenizer_src.exists() and not tokenizer_dst.exists():
        import shutil

        shutil.copytree(str(tokenizer_src), str(tokenizer_dst))
        print(f"\nCopied tokenizer to {tokenizer_dst}")

    print("\nQuantization complete!")
    print(f"Output directory: {output_dir}")

    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
