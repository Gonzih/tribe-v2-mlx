"""
Convert Wav2Vec-BERT 2.0 weights from HuggingFace safetensors to MLX format.

Source: facebook/w2v-bert-2.0
Output: wav2vec-bert-{bits}bit.safetensors

Actual safetensors key layout (verified from model.safetensors, transformers 5.8):
  feature_extractor.conv_layers.N.conv.weight  (Conv1d: out, in, k)
  feature_extractor.conv_layers.N.layer_norm.{weight,bias}
  feature_projection.layer_norm.{weight,bias}
  feature_projection.projection.{weight,bias}
  encoder.pos_conv_embed.conv.{weight,bias}
  encoder.layer_norm.{weight,bias}
  encoder.layers.N.ffn1_layer_norm.{weight,bias}
  encoder.layers.N.ffn1.intermediate_dense.{weight,bias}
  encoder.layers.N.ffn1.output_dense.{weight,bias}
  encoder.layers.N.self_attn_layer_norm.{weight,bias}
  encoder.layers.N.self_attn.{linear_q,linear_k,linear_v,linear_out}.{weight,bias}
  encoder.layers.N.self_attn.distance_embedding.weight  (skip)
  encoder.layers.N.conv_module.layer_norm.{weight,bias}
  encoder.layers.N.conv_module.pointwise_conv1.weight  (Conv1d: 2048, 1024, 1)
  encoder.layers.N.conv_module.depthwise_conv.weight   (Conv1d: 1024, 1, 31)
  encoder.layers.N.conv_module.depthwise_layer_norm.{weight,bias}
  encoder.layers.N.conv_module.pointwise_conv2.weight  (Conv1d: 1024, 1024, 1)
  encoder.layers.N.ffn2_layer_norm.{weight,bias}
  encoder.layers.N.ffn2.intermediate_dense.{weight,bias}
  encoder.layers.N.ffn2.output_dense.{weight,bias}
  encoder.layers.N.final_layer_norm.{weight,bias}

Conv1d weight note (PyTorch → MLX):
  PyTorch Conv1d: (out, in/groups, kernel)
  MLX Conv1d:     (out, kernel, in/groups)
  → transpose dims 1 and 2

  Pointwise Conv1d (kernel=1): (out, in, 1) → squeeze(-1) for our Linear implementation
  Depthwise Conv1d: (dim, 1, k) → MLX: (dim, k, 1)
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

    # Conv feature extractor (Conv1d weights handled in convert function)
    m = re.match(r"feature_extractor\.conv_layers\.(\d+)\.conv\.(weight|bias)", k)
    if m:
        idx, wb = int(m.group(1)), m.group(2)
        return f"feature_extractor.conv_layers.{idx}.{wb}"
    # Skip layer_norm inside feature extractor
    if re.match(r"feature_extractor\.conv_layers\.\d+\.layer_norm\.", k):
        return None

    # Skip masked_spec_embed
    if k == "masked_spec_embed":
        return None

    # Feature projection
    if k == "feature_projection.layer_norm.weight":
        return "feature_projection.layer_norm.weight"
    if k == "feature_projection.layer_norm.bias":
        return "feature_projection.layer_norm.bias"
    if k == "feature_projection.projection.weight":
        return "feature_projection.projection.weight"
    if k == "feature_projection.projection.bias":
        return "feature_projection.projection.bias"

    # Positional conv embed
    if k == "encoder.pos_conv_embed.conv.weight":
        return "pos_conv_embed.weight"
    if k == "encoder.pos_conv_embed.conv.bias":
        return "pos_conv_embed.bias"

    # Encoder layer norm
    if k == "encoder.layer_norm.weight":
        return "layer_norm.weight"
    if k == "encoder.layer_norm.bias":
        return "layer_norm.bias"

    # Encoder layers
    m = re.match(r"encoder\.layers\.(\d+)\.(.*)", k)
    if not m:
        return None

    idx, rest = int(m.group(1)), m.group(2)
    pfx = f"encoder_layers.{idx}"

    mappings = {
        # FFN1 (first feed-forward sublayer, pre-attention)
        "ffn1_layer_norm.weight": f"{pfx}.ff1.layer_norm.weight",
        "ffn1_layer_norm.bias": f"{pfx}.ff1.layer_norm.bias",
        "ffn1.intermediate_dense.weight": f"{pfx}.ff1.fc1.weight",
        "ffn1.intermediate_dense.bias": f"{pfx}.ff1.fc1.bias",
        "ffn1.output_dense.weight": f"{pfx}.ff1.fc2.weight",
        "ffn1.output_dense.bias": f"{pfx}.ff1.fc2.bias",

        # Self-attention
        "self_attn_layer_norm.weight": f"{pfx}.attn_norm.weight",
        "self_attn_layer_norm.bias": f"{pfx}.attn_norm.bias",
        "self_attn.linear_q.weight": f"{pfx}.attn.query_proj.weight",
        "self_attn.linear_q.bias": f"{pfx}.attn.query_proj.bias",
        "self_attn.linear_k.weight": f"{pfx}.attn.key_proj.weight",
        "self_attn.linear_k.bias": f"{pfx}.attn.key_proj.bias",
        "self_attn.linear_v.weight": f"{pfx}.attn.value_proj.weight",
        "self_attn.linear_v.bias": f"{pfx}.attn.value_proj.bias",
        "self_attn.linear_out.weight": f"{pfx}.attn.out_proj.weight",
        "self_attn.linear_out.bias": f"{pfx}.attn.out_proj.bias",

        # Conv module (weights needing special shape handling)
        "conv_module.layer_norm.weight": f"{pfx}.conv_module.layer_norm.weight",
        "conv_module.layer_norm.bias": f"{pfx}.conv_module.layer_norm.bias",
        "conv_module.pointwise_conv1.weight": f"{pfx}.conv_module.pointwise_conv1.weight",
        # pointwise_conv1 has no bias in w2v-bert-2.0
        "conv_module.depthwise_conv.weight": f"{pfx}.conv_module.depthwise_conv.weight",
        "conv_module.depthwise_conv.bias": f"{pfx}.conv_module.depthwise_conv.bias",
        # depthwise_layer_norm maps to our batch_norm (both are LayerNorm)
        "conv_module.depthwise_layer_norm.weight": f"{pfx}.conv_module.batch_norm.weight",
        "conv_module.depthwise_layer_norm.bias": f"{pfx}.conv_module.batch_norm.bias",
        "conv_module.pointwise_conv2.weight": f"{pfx}.conv_module.pointwise_conv2.weight",
        # pointwise_conv2 has no bias in w2v-bert-2.0

        # FFN2 (second feed-forward sublayer, post-conv)
        "ffn2_layer_norm.weight": f"{pfx}.ff2.layer_norm.weight",
        "ffn2_layer_norm.bias": f"{pfx}.ff2.layer_norm.bias",
        "ffn2.intermediate_dense.weight": f"{pfx}.ff2.fc1.weight",
        "ffn2.intermediate_dense.bias": f"{pfx}.ff2.fc1.bias",
        "ffn2.output_dense.weight": f"{pfx}.ff2.fc2.weight",
        "ffn2.output_dense.bias": f"{pfx}.ff2.fc2.bias",

        # Final layer norm
        "final_layer_norm.weight": f"{pfx}.final_norm.weight",
        "final_layer_norm.bias": f"{pfx}.final_norm.bias",
    }
    # Skip distance embedding (relative position)
    if rest == "self_attn.distance_embedding.weight":
        return None
    return mappings.get(rest)


def convert_wav2vec_bert(
    hf_model_id: str = "facebook/w2v-bert-2.0",
    output_path: str = "./weights/mlx/wav2vec-bert-8bit.safetensors",
    quantize_bits: int = 8,
    hf_token: Optional[str] = None,
) -> None:
    """
    Convert Wav2Vec-BERT 2.0 weights to MLX safetensors.

    Conv1d weight shapes are transposed from PyTorch (out, in, k) to MLX (out, k, in).
    Pointwise Conv1d (kernel=1) are squeezed to Linear shape.
    The convolutional feature extractor and depthwise conv are kept in fp16.
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

    print(f"Loading Wav2Vec-BERT 2.0 weights from {sf_path} …")
    hf_weights = mx.load(str(sf_path))

    unmapped = []
    mlx_weights: dict[str, mx.array] = {}

    for hf_key, arr in hf_weights.items():
        mlx_key = _hf_key_to_mlx(hf_key)
        if mlx_key is None:
            unmapped.append(hf_key)
            continue

        arr = arr.astype(mx.float32)

        # Handle Conv1d weight transpositions
        # PyTorch Conv1d: (out, in/groups, kernel) → MLX Conv1d: (out, kernel, in/groups)
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                # Pointwise Conv1d (kernel=1) — used as Linear in our model → squeeze
                arr = arr[..., 0]  # (out, in)
            else:
                # Regular or depthwise Conv1d — transpose to MLX format
                arr = arr.transpose(0, 2, 1)  # (out, kernel, in/groups)

        mlx_weights[mlx_key] = arr

    if unmapped:
        print(f"  Skipped {len(unmapped)} unmapped keys: {unmapped[:5]} …")
    print(f"Mapped {len(mlx_weights)} parameters to MLX layout.")

    from tribe_v2_mlx.models.wav2vec_bert import MLXWav2VecBert, W2VBertConfig
    model = MLXWav2VecBert(W2VBertConfig())
    model.load_weights(list(mlx_weights.items()), strict=False)
    mx.eval(model.parameters())

    if quantize_bits in (4, 8):
        print(f"Quantizing to {quantize_bits}-bit (Linear only, skip Conv1d) …")
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
    print(f"Saved → {out}  ({len(flat)} tensors)")
