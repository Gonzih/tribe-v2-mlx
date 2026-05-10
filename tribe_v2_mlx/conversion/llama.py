"""
Set up LLaMA 3.2-3B for MLX inference.

Unlike the other encoders, LLaMA conversion is handled natively by mlx_lm.
This module provides:
  1. convert_llama() — wraps mlx_lm.convert for first-time setup
  2. verify_llama()  — quick smoke test that hidden state extraction works
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def convert_llama(
    hf_model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
    output_path: str = "./weights/mlx/llama-3.2-3b-4bit",
    quantize_bits: int = 4,
    hf_token: Optional[str] = None,
) -> None:
    """
    Convert LLaMA 3.2-3B from HuggingFace to a local 4-bit MLX checkpoint.

    Alternatively, use the pre-converted community model:
      mlx-community/Llama-3.2-3B-Instruct-4bit

    Parameters
    ----------
    hf_model_id : HuggingFace model ID
    output_path : local directory for the MLX checkpoint
    quantize_bits : 4 (default) or 8
    hf_token : falls back to HF_TOKEN env var
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    # mlx_lm.convert handles LLaMA fully automatically
    try:
        from mlx_lm import convert as mlx_convert
    except ImportError as e:
        raise ImportError("mlx_lm is required: pip install mlx-lm") from e

    Path(output_path).mkdir(parents=True, exist_ok=True)

    print(f"Converting {hf_model_id} → {output_path} at {quantize_bits}-bit …")
    mlx_convert(
        hf_path=hf_model_id,
        mlx_path=output_path,
        quantize=True,
        q_bits=quantize_bits,
        q_group_size=64,
        hf_token=hf_token,
    )
    print(f"LLaMA MLX checkpoint saved → {output_path}")


def verify_llama(
    model_path: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
) -> bool:
    """
    Load the MLX LLaMA model and run a quick hidden-state extraction test.

    Returns True if the model runs successfully.
    """
    from tribe_v2_mlx.models.llama import MLXLlamaExtractor

    print(f"Loading {model_path} for verification …")
    extractor = MLXLlamaExtractor.load(model_path)

    test_text = "The cat sat on the mat."
    hidden = extractor.extract_text_features(test_text, layer_indices=[0, 14, 27])

    for layer_idx, arr in hidden.items():
        print(f"  Layer {layer_idx}: shape={arr.shape}")

    assert all(arr.ndim == 3 for arr in hidden.values()), "Expected (1, T, D) tensors."
    print("LLaMA verification passed.")
    return True
