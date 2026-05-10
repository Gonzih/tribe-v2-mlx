#!/usr/bin/env python3
"""
Download all TRIBE v2 model weights to ./weights/.

Components downloaded:
  facebook/tribev2         → ./weights/tribe/best.ckpt + config.yaml
  facebook/vjepa2-vitg-fpc64-256 → ./weights/vjepa2/
  facebook/dinov2-large    → ./weights/dinov2/
  facebook/w2v-bert-2.0    → ./weights/wav2vec_bert/

LLaMA 3.2-3B is handled by mlx_lm and cached in the HuggingFace hub cache.

Usage
-----
    HF_TOKEN=hf_... python scripts/download_weights.py
    python scripts/download_weights.py --weights-dir /path/to/weights
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def download_component(repo_id: str, local_dir: str, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    print(f"\n── Downloading {repo_id} ──")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token,
            ignore_patterns=["*.bin.index.json", "flax_model*", "tf_model*"],
        )
        print(f"   → {local_dir}")
    except Exception as e:
        print(f"   WARNING: Failed to download {repo_id}: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TRIBE v2 model weights")
    parser.add_argument("--weights-dir", default="./weights", help="Root weights directory")
    parser.add_argument("--token", default=None, help="HuggingFace token (overrides HF_TOKEN)")
    parser.add_argument(
        "--skip-tribe", action="store_true",
        help="Skip TRIBE v2 checkpoint (requires Meta-internal packages to be useful)"
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print(
            "WARNING: No HuggingFace token provided. "
            "Set HF_TOKEN env var or use --token for gated models.",
            file=sys.stderr,
        )

    wdir = Path(args.weights_dir)
    wdir.mkdir(parents=True, exist_ok=True)

    # TRIBE v2 transformer checkpoint
    if not args.skip_tribe:
        download_component("facebook/tribev2", str(wdir / "tribe"), token)

    # Vision encoders
    download_component("facebook/vjepa2-vitg-fpc64-256", str(wdir / "vjepa2"), token)
    download_component("facebook/dinov2-large", str(wdir / "dinov2"), token)

    # Audio encoder
    download_component("facebook/w2v-bert-2.0", str(wdir / "wav2vec_bert"), token)

    print("\n✓ All downloads complete.")
    print(f"  Weights directory: {wdir.resolve()}")
    print(
        "\nNote: LLaMA 3.2-3B will be downloaded automatically when you run "
        "scripts/convert_to_mlx.py (it uses mlx_lm which caches in ~/.cache/huggingface/)."
    )


if __name__ == "__main__":
    main()
