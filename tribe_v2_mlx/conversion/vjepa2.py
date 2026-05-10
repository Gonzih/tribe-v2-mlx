"""
Convert V-JEPA2 ViT-g weights from HuggingFace to MLX safetensors.

Source: facebook/vjepa2-vitg-fpc64-256
Output: vjepa2-vitg-{bits}bit.safetensors

HuggingFace key → MLX key mapping:
  vjepa.{embeddings.patch_embeddings.projection} → patch_embed.proj
  vjepa.{encoder.layer.N.layernorm_before}       → blocks.N.norm1
  vjepa.{encoder.layer.N.attention.attention.query} → blocks.N.attn.query_proj
  vjepa.{encoder.layer.N.attention.attention.key}   → blocks.N.attn.key_proj
  vjepa.{encoder.layer.N.attention.attention.value} → blocks.N.attn.value_proj
  vjepa.{encoder.layer.N.attention.output.dense}    → blocks.N.attn.out_proj
  vjepa.{encoder.layer.N.layernorm_after}           → blocks.N.norm2
  vjepa.{encoder.layer.N.intermediate.dense}        → blocks.N.mlp.fc1
  vjepa.{encoder.layer.N.output.dense}              → blocks.N.mlp.fc2
  vjepa.{layernorm}                                  → norm
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _hf_key_to_mlx(hf_key: str) -> Optional[str]:
    """
    Map a HuggingFace V-JEPA2 parameter name to the MLX module tree.

    Returns None if the key should be skipped (e.g. pooler, decoder).
    """
    # Strip top-level model prefixes
    k = hf_key
    for prefix in ("vjepa.", "model.", "vision_model."):
        if k.startswith(prefix):
            k = k[len(prefix):]

    # Patch embedding
    if k == "embeddings.patch_embeddings.projection.weight":
        return "patch_embed.proj.weight"
    if k == "embeddings.patch_embeddings.projection.bias":
        return "patch_embed.proj.bias"

    # Positional / CLS embeddings
    if "position_embeddings" in k:
        return "pos_embed"
    if "cls_token" in k:
        return "cls_token"

    # Transformer blocks
    m = re.match(r"encoder\.layer\.(\d+)\.(.*)", k)
    if not m:
        # Final layernorm
        if k in ("layernorm.weight", "norm.weight"):
            return "norm.weight"
        if k in ("layernorm.bias", "norm.bias"):
            return "norm.bias"
        return None

    idx, rest = int(m.group(1)), m.group(2)
    prefix = f"blocks.{idx}"

    mappings = {
        "layernorm_before.weight": f"{prefix}.norm1.weight",
        "layernorm_before.bias": f"{prefix}.norm1.bias",
        "layernorm_after.weight": f"{prefix}.norm2.weight",
        "layernorm_after.bias": f"{prefix}.norm2.bias",
        "attention.attention.query.weight": f"{prefix}.attn.query_proj.weight",
        "attention.attention.query.bias": f"{prefix}.attn.query_proj.bias",
        "attention.attention.key.weight": f"{prefix}.attn.key_proj.weight",
        "attention.attention.key.bias": f"{prefix}.attn.key_proj.bias",
        "attention.attention.value.weight": f"{prefix}.attn.value_proj.weight",
        "attention.attention.value.bias": f"{prefix}.attn.value_proj.bias",
        "attention.output.dense.weight": f"{prefix}.attn.out_proj.weight",
        "attention.output.dense.bias": f"{prefix}.attn.out_proj.bias",
        "intermediate.dense.weight": f"{prefix}.mlp.fc1.weight",
        "intermediate.dense.bias": f"{prefix}.mlp.fc1.bias",
        "output.dense.weight": f"{prefix}.mlp.fc2.weight",
        "output.dense.bias": f"{prefix}.mlp.fc2.bias",
    }
    return mappings.get(rest)


def convert_vjepa2(
    hf_model_id: str = "facebook/vjepa2-vitg-fpc64-256",
    output_path: str = "./weights/mlx/vjepa2-vitg-8bit.safetensors",
    quantize_bits: int = 8,
    hf_token: Optional[str] = None,
) -> None:
    """
    Download V-JEPA2 ViT-g from HuggingFace, convert to MLX, optionally quantize.

    Parameters
    ----------
    hf_model_id : HuggingFace model ID
    output_path : where to save the MLX safetensors file
    quantize_bits : 4 or 8 for quantization; 0 or None for fp16
    hf_token : HuggingFace token (falls back to HF_TOKEN env var)
    """
    import os
    from transformers import AutoModel

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading {hf_model_id} from HuggingFace …")
    hf_model = AutoModel.from_pretrained(
        hf_model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    hf_state = hf_model.state_dict()

    print("Inspecting HuggingFace keys …")
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

    print(f"Mapped {len(mlx_weights)} parameters to MLX layout.")

    # Build model and load weights
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
                isinstance(m, nn.Linear) and m.weight.shape[-1] >= 64
            ),
        )
        mx.eval(model.parameters())

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    weights_dict = dict(model.parameters())
    mx.save_safetensors(str(out), {k: v for k, v in weights_dict.items() if v is not None})
    print(f"Saved MLX weights → {out}")
