"""
Convert DINOv2-Large weights from HuggingFace to MLX safetensors.

Source: facebook/dinov2-large
Output: dinov2-large-{bits}bit.safetensors

HuggingFace Dinov2Model key layout:
  embeddings.patch_embeddings.projection.{weight,bias}
  embeddings.cls_token
  embeddings.register_tokens
  embeddings.position_embeddings
  encoder.layer.N.norm1.{weight,bias}
  encoder.layer.N.attention.attention.{query,key,value}.{weight,bias}
  encoder.layer.N.attention.output.dense.{weight,bias}
  encoder.layer.N.norm2.{weight,bias}
  encoder.layer.N.mlp.{fc1,fc2}.{weight,bias}   (or intermediate/output)
  layernorm.{weight,bias}
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _hf_key_to_mlx(hf_key: str) -> Optional[str]:
    k = hf_key
    for prefix in ("dinov2.", "model.", ""):
        pass  # no prefix stripping needed for Dinov2Model

    # Patch embedding
    if k == "embeddings.patch_embeddings.projection.weight":
        return "patch_embed.proj.weight"
    if k == "embeddings.patch_embeddings.projection.bias":
        return "patch_embed.proj.bias"
    if k == "embeddings.cls_token":
        return "cls_token"
    if k == "embeddings.register_tokens":
        return "register_tokens"
    if k == "embeddings.position_embeddings":
        return "pos_embed"

    # Transformer blocks
    m = re.match(r"encoder\.layer\.(\d+)\.(.*)", k)
    if not m:
        if k in ("layernorm.weight",):
            return "norm.weight"
        if k in ("layernorm.bias",):
            return "norm.bias"
        return None

    idx, rest = int(m.group(1)), m.group(2)
    prefix = f"blocks.{idx}"

    mappings = {
        "norm1.weight": f"{prefix}.norm1.weight",
        "norm1.bias": f"{prefix}.norm1.bias",
        "norm2.weight": f"{prefix}.norm2.weight",
        "norm2.bias": f"{prefix}.norm2.bias",
        # HF attention
        "attention.attention.query.weight": f"{prefix}.attn.query_proj.weight",
        "attention.attention.query.bias": f"{prefix}.attn.query_proj.bias",
        "attention.attention.key.weight": f"{prefix}.attn.key_proj.weight",
        "attention.attention.key.bias": f"{prefix}.attn.key_proj.bias",
        "attention.attention.value.weight": f"{prefix}.attn.value_proj.weight",
        "attention.attention.value.bias": f"{prefix}.attn.value_proj.bias",
        "attention.output.dense.weight": f"{prefix}.attn.out_proj.weight",
        "attention.output.dense.bias": f"{prefix}.attn.out_proj.bias",
        # MLP (DINOv2 uses fc1/fc2 directly)
        "mlp.fc1.weight": f"{prefix}.mlp.fc1.weight",
        "mlp.fc1.bias": f"{prefix}.mlp.fc1.bias",
        "mlp.fc2.weight": f"{prefix}.mlp.fc2.weight",
        "mlp.fc2.bias": f"{prefix}.mlp.fc2.bias",
        # Some HF versions use intermediate/output naming
        "intermediate.dense.weight": f"{prefix}.mlp.fc1.weight",
        "intermediate.dense.bias": f"{prefix}.mlp.fc1.bias",
        "output.dense.weight": f"{prefix}.mlp.fc2.weight",
        "output.dense.bias": f"{prefix}.mlp.fc2.bias",
    }
    return mappings.get(rest)


def convert_dinov2(
    hf_model_id: str = "facebook/dinov2-large",
    output_path: str = "./weights/mlx/dinov2-large-8bit.safetensors",
    quantize_bits: int = 8,
    hf_token: Optional[str] = None,
) -> None:
    """
    Download DINOv2-Large from HuggingFace, convert to MLX, optionally quantize.
    """
    import os
    from transformers import AutoModel

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading {hf_model_id} …")
    hf_model = AutoModel.from_pretrained(hf_model_id, token=hf_token)
    hf_state = hf_model.state_dict()

    unmapped = []
    mlx_weights: dict[str, mx.array] = {}
    for hf_key, tensor in hf_state.items():
        mlx_key = _hf_key_to_mlx(hf_key)
        if mlx_key is None:
            unmapped.append(hf_key)
            continue
        mlx_weights[mlx_key] = mx.array(tensor.float().numpy())

    if unmapped:
        print(f"  Skipped {len(unmapped)} unmapped keys: {unmapped[:5]} …")
    print(f"Mapped {len(mlx_weights)} parameters.")

    from tribe_v2_mlx.models.dinov2 import MLXDINOv2Large, DINOv2Config
    model = MLXDINOv2Large(DINOv2Config())
    model.load_weights(list(mlx_weights.items()), strict=False)
    mx.eval(model.parameters())

    if quantize_bits in (4, 8):
        print(f"Quantizing to {quantize_bits}-bit …")
        nn.quantize(
            model,
            bits=quantize_bits,
            class_predicate=lambda _, m: (
                isinstance(m, nn.Linear) and m.weight.shape[-1] >= 64
            ),
        )
        mx.eval(model.parameters())

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    weights_dict = dict(model.parameters())
    mx.save_safetensors(str(out), {k: v for k, v in weights_dict.items() if v is not None})
    print(f"Saved → {out}")
