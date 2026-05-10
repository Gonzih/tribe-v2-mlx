"""
Convert DINOv2-Large weights from HuggingFace safetensors to MLX format.

Source: facebook/dinov2-large
Output: dinov2-large-{bits}bit.safetensors

Actual safetensors key layout (verified from model.safetensors inspection):
  embeddings.cls_token: (1, 1, 1024)
  embeddings.patch_embeddings.projection.{weight,bias}
    weight: (1024, 3, 14, 14) — Conv2d format → reshape to (1024, 588)
  embeddings.position_embeddings: (1, 1370, 1024)  [1 CLS + 1369 patches @ 518px]
  embeddings.mask_token  (skip)
  encoder.layer.N.norm1.{weight,bias}
  encoder.layer.N.norm2.{weight,bias}
  encoder.layer.N.attention.attention.{query,key,value}.{weight,bias}
  encoder.layer.N.attention.output.dense.{weight,bias}
  encoder.layer.N.mlp.fc1.{weight,bias}
  encoder.layer.N.mlp.fc2.{weight,bias}
  encoder.layer.N.layer_scale1.lambda1  (skip — layer scale)
  encoder.layer.N.layer_scale2.lambda1  (skip — layer scale)
  layernorm.{weight,bias}

Architecture (from config.json):
  hidden_size=1024, num_hidden_layers=24, patch_size=14, image_size=518,
  num_register_tokens=0 (no register tokens in facebook/dinov2-large)
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

    # Patch embedding
    if k == "embeddings.patch_embeddings.projection.weight":
        return "patch_embed.proj.weight"  # Conv2d → Linear via reshape
    if k == "embeddings.patch_embeddings.projection.bias":
        return "patch_embed.proj.bias"

    # Special tokens and positional embeddings
    if k == "embeddings.cls_token":
        return "cls_token"
    if k == "embeddings.register_tokens":
        return "register_tokens"
    if k == "embeddings.position_embeddings":
        return "pos_embed"

    # Skip mask_token (not used in our model)
    if k == "embeddings.mask_token":
        return None

    # Final norm
    if k == "layernorm.weight":
        return "norm.weight"
    if k == "layernorm.bias":
        return "norm.bias"

    # Transformer blocks
    m = re.match(r"encoder\.layer\.(\d+)\.(.*)", k)
    if not m:
        return None

    idx, rest = int(m.group(1)), m.group(2)
    prefix = f"blocks.{idx}"

    mappings = {
        "norm1.weight": f"{prefix}.norm1.weight",
        "norm1.bias": f"{prefix}.norm1.bias",
        "norm2.weight": f"{prefix}.norm2.weight",
        "norm2.bias": f"{prefix}.norm2.bias",
        # Attention (DINOv2: double .attention. nesting)
        "attention.attention.query.weight": f"{prefix}.attn.query_proj.weight",
        "attention.attention.query.bias": f"{prefix}.attn.query_proj.bias",
        "attention.attention.key.weight": f"{prefix}.attn.key_proj.weight",
        "attention.attention.key.bias": f"{prefix}.attn.key_proj.bias",
        "attention.attention.value.weight": f"{prefix}.attn.value_proj.weight",
        "attention.attention.value.bias": f"{prefix}.attn.value_proj.bias",
        "attention.output.dense.weight": f"{prefix}.attn.out_proj.weight",
        "attention.output.dense.bias": f"{prefix}.attn.out_proj.bias",
        # MLP (DINOv2 safetensors use fc1/fc2 directly)
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
    # Skip layer_scale parameters (not in our model)
    if rest.startswith("layer_scale"):
        return None
    return mappings.get(rest)


def convert_dinov2(
    hf_model_id: str = "facebook/dinov2-large",
    output_path: str = "./weights/mlx/dinov2-large-8bit.safetensors",
    quantize_bits: int = 8,
    hf_token: Optional[str] = None,
) -> None:
    """
    Convert DINOv2-Large weights to MLX safetensors.

    Uses the real model config: image_size=518, num_register_tokens=0.
    The patch embedding Conv2d weight is reshaped to fit our Linear-based implementation.
    """
    import os

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    model_path = Path(hf_model_id)
    if model_path.is_dir():
        sf_path = model_path / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        sf_path = Path(hf_hub_download(
            repo_id=hf_model_id, filename="model.safetensors", token=hf_token
        ))

    print(f"Loading DINOv2-Large weights from {sf_path} …")
    hf_weights = mx.load(str(sf_path))

    unmapped = []
    mlx_weights: dict[str, mx.array] = {}

    for hf_key, arr in hf_weights.items():
        mlx_key = _hf_key_to_mlx(hf_key)
        if mlx_key is None:
            unmapped.append(hf_key)
            continue

        # Conv2d patch embed: (1024, 3, 14, 14) → (1024, 588) for Linear
        if mlx_key == "patch_embed.proj.weight" and arr.ndim == 4:
            arr = arr.reshape(arr.shape[0], -1)

        mlx_weights[mlx_key] = arr.astype(mx.float32)

    if unmapped:
        print(f"  Skipped {len(unmapped)} unmapped keys: {unmapped[:5]} …")
    print(f"Mapped {len(mlx_weights)} parameters to MLX layout.")

    # Use real model config: image_size=518, no register tokens
    from tribe_v2_mlx.models.dinov2 import MLXDINOv2Large, DINOv2Config
    config = DINOv2Config(image_size=518, num_register_tokens=0)
    model = MLXDINOv2Large(config)
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
    # Skip zero-size arrays (e.g. register_tokens when num_register_tokens=0)
    flat = {
        k: v
        for k, v in tree_flatten(model.parameters())
        if isinstance(v, mx.array) and v.size > 0
    }
    mx.save_safetensors(str(out), flat)
    print(f"Saved → {out}  ({len(flat)} tensors)")
