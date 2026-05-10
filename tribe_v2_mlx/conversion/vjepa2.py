"""
Convert V-JEPA2 ViT-g weights from HuggingFace safetensors to MLX format.

Source: facebook/vjepa2-vitg-fpc64-256
Output: vjepa2-vitg-{bits}bit.safetensors

Actual safetensors key layout (verified from model.safetensors inspection):
  encoder.embeddings.patch_embeddings.proj.{weight,bias}
  encoder.layer.N.norm1.{weight,bias}
  encoder.layer.N.norm2.{weight,bias}
  encoder.layer.N.attention.{query,key,value}.{weight,bias}
  encoder.layer.N.attention.proj.{weight,bias}
  encoder.layer.N.mlp.fc1.{weight,bias}
  encoder.layer.N.mlp.fc2.{weight,bias}
  encoder.layernorm.{weight,bias}
  predictor.*  (skip — not needed for encoding)

Architecture (from config.json):
  hidden_size=1408, num_hidden_layers=40, num_attention_heads=22,
  mlp_dim=6144, patch_size=16, tubelet_size=2, image_size=256

Key shape notes:
  patch_embeddings.proj.weight: (1408, 3, 2, 16, 16) — Conv3d format
    → reshape to (1408, 1536) for our Linear-based TubeletEmbedding
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def _hf_key_to_mlx(hf_key: str) -> Optional[str]:
    k = hf_key

    # Patch embedding (Conv3d weight handled by reshape in convert function)
    if k == "encoder.embeddings.patch_embeddings.proj.weight":
        return "patch_embed.proj.weight"
    if k == "encoder.embeddings.patch_embeddings.proj.bias":
        return "patch_embed.proj.bias"

    # Final layer norm
    if k == "encoder.layernorm.weight":
        return "norm.weight"
    if k == "encoder.layernorm.bias":
        return "norm.bias"

    # Transformer blocks
    m = re.match(r"encoder\.layer\.(\d+)\.(.*)", k)
    if not m:
        return None  # skip predictor.* and other keys

    idx, rest = int(m.group(1)), m.group(2)
    prefix = f"blocks.{idx}"

    mappings = {
        "norm1.weight": f"{prefix}.norm1.weight",
        "norm1.bias": f"{prefix}.norm1.bias",
        "norm2.weight": f"{prefix}.norm2.weight",
        "norm2.bias": f"{prefix}.norm2.bias",
        # Attention (V-JEPA2 safetensors: encoder.layer.N.attention.{query,key,value,proj})
        "attention.query.weight": f"{prefix}.attn.query_proj.weight",
        "attention.query.bias": f"{prefix}.attn.query_proj.bias",
        "attention.key.weight": f"{prefix}.attn.key_proj.weight",
        "attention.key.bias": f"{prefix}.attn.key_proj.bias",
        "attention.value.weight": f"{prefix}.attn.value_proj.weight",
        "attention.value.bias": f"{prefix}.attn.value_proj.bias",
        "attention.proj.weight": f"{prefix}.attn.out_proj.weight",
        "attention.proj.bias": f"{prefix}.attn.out_proj.bias",
        # MLP
        "mlp.fc1.weight": f"{prefix}.mlp.fc1.weight",
        "mlp.fc1.bias": f"{prefix}.mlp.fc1.bias",
        "mlp.fc2.weight": f"{prefix}.mlp.fc2.weight",
        "mlp.fc2.bias": f"{prefix}.mlp.fc2.bias",
    }
    return mappings.get(rest)


def convert_vjepa2(
    hf_model_id: str = "facebook/vjepa2-vitg-fpc64-256",
    output_path: str = "./weights/mlx/vjepa2-vitg-8bit.safetensors",
    quantize_bits: int = 8,
    hf_token: Optional[str] = None,
) -> None:
    """
    Convert V-JEPA2 ViT-g weights to MLX safetensors.

    Loads directly from model.safetensors (no AutoModel needed).
    Handles Conv3d → Linear reshape for patch embedding.
    """
    import os

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    # Find the safetensors file
    model_path = Path(hf_model_id)
    if model_path.is_dir():
        sf_path = model_path / "model.safetensors"
    else:
        # Download from hub
        from huggingface_hub import hf_hub_download
        sf_path = Path(hf_hub_download(
            repo_id=hf_model_id, filename="model.safetensors", token=hf_token
        ))

    print(f"Loading V-JEPA2 weights from {sf_path} …")
    hf_weights = mx.load(str(sf_path))

    unmapped = []
    mlx_weights: dict[str, mx.array] = {}

    for hf_key, arr in hf_weights.items():
        mlx_key = _hf_key_to_mlx(hf_key)
        if mlx_key is None:
            unmapped.append(hf_key)
            continue

        # Conv3d patch embed weight: (1408, 3, 2, 16, 16) → (1408, 1536)
        if mlx_key == "patch_embed.proj.weight" and arr.ndim == 5:
            arr = arr.reshape(arr.shape[0], -1)

        mlx_weights[mlx_key] = arr.astype(mx.float32)

    if unmapped:
        print(f"  Skipped {len(unmapped)} unmapped keys (predictor + unknown): {unmapped[:3]} …")
    print(f"Mapped {len(mlx_weights)} parameters to MLX layout.")

    from tribe_v2_mlx.models.vjepa2 import MLXVJepa2, vjepa2_vitg_config
    model = MLXVJepa2(vjepa2_vitg_config())
    model.load_weights(list(mlx_weights.items()), strict=False)
    mx.eval(model.parameters())

    if quantize_bits in (4, 8):
        print(f"Quantizing to {quantize_bits}-bit …")
        nn.quantize(
            model,
            bits=quantize_bits,
            class_predicate=lambda _, m: (
                isinstance(m, nn.Linear)
                and m.weight.shape[-1] % 64 == 0
                and m.weight.shape[-1] >= 64
            ),
        )
        mx.eval(model.parameters())

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: v for k, v in tree_flatten(model.parameters()) if isinstance(v, mx.array)}
    mx.save_safetensors(str(out), flat)
    print(f"Saved MLX weights → {out}  ({len(flat)} tensors)")
