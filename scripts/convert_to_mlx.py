#!/usr/bin/env python3
"""
Convert all TRIBE v2 encoder weights to MLX safetensors format.

Reads PyTorch weights from ./weights/ (downloaded by download_weights.py)
and writes quantised MLX weights to ./weights/mlx/.

Usage
-----
    HF_TOKEN=hf_... python scripts/convert_to_mlx.py
    python scripts/convert_to_mlx.py --bits 8 --skip-llama
    python scripts/convert_to_mlx.py --weights-dir /path/to/weights --bits 4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TRIBE v2 encoders to MLX")
    parser.add_argument("--weights-dir", default="./weights")
    parser.add_argument(
        "--bits", type=int, default=8, choices=[4, 8],
        help="Quantization bits (default 8; LLaMA uses 4-bit regardless)",
    )
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--skip-vjepa2", action="store_true")
    parser.add_argument("--skip-dinov2", action="store_true")
    parser.add_argument("--skip-wav2vec-bert", action="store_true")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    wdir = Path(args.weights_dir)
    mlx_dir = wdir / "mlx"
    mlx_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []

    # ── V-JEPA2 ──────────────────────────────────────────────────────────────
    if not args.skip_vjepa2:
        hf_path = str(wdir / "vjepa2")
        if Path(hf_path).exists():
            print("\n── Converting V-JEPA2 ViT-g ──")
            try:
                from tribe_v2_mlx.conversion.vjepa2 import convert_vjepa2
                convert_vjepa2(
                    hf_model_id=hf_path,
                    output_path=str(mlx_dir / f"vjepa2-vitg-{args.bits}bit.safetensors"),
                    quantize_bits=args.bits,
                    hf_token=token,
                )
            except Exception as e:
                msg = f"V-JEPA2 conversion failed: {e}"
                print(f"WARNING: {msg}", file=sys.stderr)
                errors.append(msg)
        else:
            print(f"SKIP V-JEPA2: {hf_path} not found (run download_weights.py first)")

    # ── DINOv2 ───────────────────────────────────────────────────────────────
    if not args.skip_dinov2:
        hf_path = str(wdir / "dinov2")
        if Path(hf_path).exists():
            print("\n── Converting DINOv2-Large ──")
            try:
                from tribe_v2_mlx.conversion.dinov2 import convert_dinov2
                convert_dinov2(
                    hf_model_id=hf_path,
                    output_path=str(mlx_dir / f"dinov2-large-{args.bits}bit.safetensors"),
                    quantize_bits=args.bits,
                    hf_token=token,
                )
            except Exception as e:
                msg = f"DINOv2 conversion failed: {e}"
                print(f"WARNING: {msg}", file=sys.stderr)
                errors.append(msg)
        else:
            print(f"SKIP DINOv2: {hf_path} not found")

    # ── Wav2Vec-BERT 2.0 ─────────────────────────────────────────────────────
    if not args.skip_wav2vec_bert:
        hf_path = str(wdir / "wav2vec_bert")
        if Path(hf_path).exists():
            print("\n── Converting Wav2Vec-BERT 2.0 ──")
            try:
                from tribe_v2_mlx.conversion.wav2vec_bert import convert_wav2vec_bert
                convert_wav2vec_bert(
                    hf_model_id=hf_path,
                    output_path=str(mlx_dir / f"wav2vec-bert-{args.bits}bit.safetensors"),
                    quantize_bits=args.bits,
                    hf_token=token,
                )
            except Exception as e:
                msg = f"Wav2Vec-BERT conversion failed: {e}"
                print(f"WARNING: {msg}", file=sys.stderr)
                errors.append(msg)
        else:
            print(f"SKIP Wav2Vec-BERT: {hf_path} not found")

    # ── LLaMA 3.2-3B ─────────────────────────────────────────────────────────
    if not args.skip_llama:
        llama_out = str(mlx_dir / "llama-3.2-3b-4bit")
        if not Path(llama_out).exists():
            print("\n── Converting LLaMA 3.2-3B (4-bit, via mlx_lm) ──")
            try:
                from tribe_v2_mlx.conversion.llama import convert_llama
                # Try the pre-converted community model first
                try:
                    import mlx_lm
                    mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
                    print(
                        "Using pre-converted mlx-community/Llama-3.2-3B-Instruct-4bit — "
                        "skipping local conversion."
                    )
                except Exception:
                    convert_llama(
                        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
                        output_path=llama_out,
                        quantize_bits=4,
                        hf_token=token,
                    )
            except Exception as e:
                msg = f"LLaMA conversion failed: {e}"
                print(f"WARNING: {msg}", file=sys.stderr)
                errors.append(msg)
        else:
            print(f"\nLLaMA MLX checkpoint already exists at {llama_out}")

    print("\n" + "=" * 60)
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"✓ Conversion complete.  MLX weights saved to {mlx_dir.resolve()}")


if __name__ == "__main__":
    main()
