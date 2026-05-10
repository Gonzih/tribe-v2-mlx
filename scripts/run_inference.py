#!/usr/bin/env python3
"""
CLI for TRIBE v2 MLX fMRI prediction.

Usage
-----
    # Predict from a video (requires converted MLX weights in ./weights/mlx)
    python scripts/run_inference.py --video path/to/video.mp4

    # Predict from a static image
    python scripts/run_inference.py --image path/to/frame.jpg

    # Save output to a numpy file
    python scripts/run_inference.py --video clip.mp4 --output preds.npy

    # Specify weights directory and subject
    python scripts/run_inference.py --video clip.mp4 \
        --weights-dir ./weights/mlx \
        --subject 0

Pipeline weights location (expected after running convert_to_mlx.py):
    ./weights/mlx/vjepa2-vitg-8bit.safetensors
    ./weights/mlx/dinov2-large-8bit.safetensors
    ./weights/mlx/wav2vec-bert-8bit.safetensors
    ./weights/mlx/llama-3.2-3b-4bit/  (or mlx-community/Llama-3.2-3B-Instruct-4bit)
    ./weights/tribe/best.ckpt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRIBE v2 MLX fMRI prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", metavar="PATH", help="Input video file")
    input_group.add_argument("--image", metavar="PATH", help="Input image file")

    parser.add_argument("--caption", default=None, help="Optional text caption for the video")
    parser.add_argument("--weights-dir", default="./weights/mlx", help="MLX weights directory")
    parser.add_argument("--tribe-ckpt", default=None, help="Path to TRIBE best.ckpt")
    parser.add_argument("--subject", type=int, default=0, help="Subject ID (default 0)")
    parser.add_argument("--output", default=None, help="Save predictions to .npy file")
    parser.add_argument("--no-vjepa2", action="store_true", help="Skip V-JEPA2 encoder")
    parser.add_argument("--no-dinov2", action="store_true", help="Skip DINOv2 encoder")
    parser.add_argument("--no-audio", action="store_true", help="Skip Wav2Vec-BERT encoder")
    parser.add_argument("--no-llama", action="store_true", help="Skip LLaMA encoder")

    args = parser.parse_args()

    # Validate input
    input_path = args.video or args.image
    if not Path(input_path).exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"TRIBE v2 MLX Inference")
    print(f"  Input:       {input_path}")
    print(f"  Weights dir: {args.weights_dir}")
    print(f"  Subject:     {args.subject}")

    # Load pipeline
    from tribe_v2_mlx.pipeline import TribeV2MLXPipeline

    pipeline = TribeV2MLXPipeline.from_weights(
        weights_dir=args.weights_dir,
        tribe_ckpt=args.tribe_ckpt,
        subject_id=args.subject,
    )

    # Optionally disable encoders
    if args.no_vjepa2:
        pipeline.vjepa2 = None
    if args.no_dinov2:
        pipeline.dinov2 = None
    if args.no_audio:
        pipeline.w2v_bert = None
    if args.no_llama:
        pipeline.llama = None

    # Run inference
    print("\nRunning inference …")
    if args.video:
        preds = pipeline.predict(args.video, caption=args.caption)
    else:
        preds = pipeline.predict_image(args.image, caption=args.caption)

    print(f"\nPredictions shape: {preds.shape}")
    print(f"  n_timepoints:    {preds.shape[0]}")
    print(f"  n_vertices:      {preds.shape[1]}")
    print(f"  value range:     [{preds.min():.3f}, {preds.max():.3f}]")

    # Save output
    if args.output:
        out_path = Path(args.output)
        np.save(str(out_path), preds)
        print(f"\nSaved predictions → {out_path.resolve()}")
    else:
        print("\nTip: use --output preds.npy to save predictions to a file.")


if __name__ == "__main__":
    main()
