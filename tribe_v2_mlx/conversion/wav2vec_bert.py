"""
Convert Wav2Vec-BERT 2.0 weights from HuggingFace to MLX safetensors.

Source: facebook/w2v-bert-2.0
Output: wav2vec-bert-{bits}bit.safetensors

HuggingFace Wav2Vec2BertModel key layout:
  feature_extractor.conv_layers.N.layer_norm.{weight,bias}
  feature_extractor.conv_layers.N.conv.{weight,bias}
  feature_projection.layer_norm.{weight,bias}
  feature_projection.projection.{weight,bias}
  encoder.pos_conv_embed.conv.{weight,bias}
  encoder.layer_norm.{weight,bias}
  encoder.layers.N.self_attn_layer_norm.{weight,bias}
  encoder.layers.N.self_attn.{linear_q,linear_k,linear_v,linear_out}.{weight,bias}
  encoder.layers.N.conv_module.depthwise_conv.{weight,bias}
  encoder.layers.N.conv_module.pointwise_conv1.{weight,bias}
  encoder.layers.N.conv_module.pointwise_conv2.{weight,bias}
  encoder.layers.N.conv_module.batch_norm.{weight,bias}
  encoder.layers.N.conv_module.layer_norm.{weight,bias}
  encoder.layers.N.feed_forward.{intermediate_dense,output_dense}.{weight,bias}
  encoder.layers.N.final_layer_norm.{weight,bias}
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _hf_key_to_mlx(hf_key: str) -> Optional[str]:
    k = hf_key

    # Conv feature extractor
    m = re.match(r"feature_extractor\.conv_layers\.(\d+)\.conv\.(weight|bias)", k)
    if m:
        return f"feature_extractor.conv_layers.{m.group(1)}.weight" if m.group(2) == "weight" \
            else f"feature_extractor.conv_layers.{m.group(1)}.bias"
    # skip layer_norm inside conv feature extractor (small, keep as-is)

    # Feature projection
    if k == "feature_projection.layer_norm.weight":
        return "feature_projection.layer_norm.weight"
    if k == "feature_projection.layer_norm.bias":
        return "feature_projection.layer_norm.bias"
    if k == "feature_projection.projection.weight":
        return "feature_projection.projection.weight"
    if k == "feature_projection.projection.bias":
        return "feature_projection.projection.bias"

    # Pos conv embed
    if k.startswith("encoder.pos_conv_embed.conv."):
        suffix = k[len("encoder.pos_conv_embed.conv."):]
        return f"pos_conv_embed.{suffix}"

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
        # attention norm
        "self_attn_layer_norm.weight": f"{pfx}.attn_norm.weight",
        "self_attn_layer_norm.bias": f"{pfx}.attn_norm.bias",
        # attention projections (HF Wav2Vec2Bert uses linear_q/k/v/out)
        "self_attn.linear_q.weight": f"{pfx}.attn.query_proj.weight",
        "self_attn.linear_q.bias": f"{pfx}.attn.query_proj.bias",
        "self_attn.linear_k.weight": f"{pfx}.attn.key_proj.weight",
        "self_attn.linear_k.bias": f"{pfx}.attn.key_proj.bias",
        "self_attn.linear_v.weight": f"{pfx}.attn.value_proj.weight",
        "self_attn.linear_v.bias": f"{pfx}.attn.value_proj.bias",
        "self_attn.linear_out.weight": f"{pfx}.attn.out_proj.weight",
        "self_attn.linear_out.bias": f"{pfx}.attn.out_proj.bias",
        # Conformer conv module
        "conv_module.layer_norm.weight": f"{pfx}.conv_module.layer_norm.weight",
        "conv_module.layer_norm.bias": f"{pfx}.conv_module.layer_norm.bias",
        "conv_module.pointwise_conv1.weight": f"{pfx}.conv_module.pointwise_conv1.weight",
        "conv_module.pointwise_conv1.bias": f"{pfx}.conv_module.pointwise_conv1.bias",
        "conv_module.depthwise_conv.weight": f"{pfx}.conv_module.depthwise_conv.weight",
        "conv_module.depthwise_conv.bias": f"{pfx}.conv_module.depthwise_conv.bias",
        "conv_module.batch_norm.weight": f"{pfx}.conv_module.batch_norm.weight",
        "conv_module.batch_norm.bias": f"{pfx}.conv_module.batch_norm.bias",
        "conv_module.pointwise_conv2.weight": f"{pfx}.conv_module.pointwise_conv2.weight",
        "conv_module.pointwise_conv2.bias": f"{pfx}.conv_module.pointwise_conv2.bias",
        # Feed-forward 1
        "feed_forward.intermediate_dense.weight": f"{pfx}.ff1.fc1.weight",
        "feed_forward.intermediate_dense.bias": f"{pfx}.ff1.fc1.bias",
        "feed_forward.output_dense.weight": f"{pfx}.ff1.fc2.weight",
        "feed_forward.output_dense.bias": f"{pfx}.ff1.fc2.bias",
        # Feed-forward 2 (final)
        "final_feed_forward.intermediate_dense.weight": f"{pfx}.ff2.fc1.weight",
        "final_feed_forward.intermediate_dense.bias": f"{pfx}.ff2.fc1.bias",
        "final_feed_forward.output_dense.weight": f"{pfx}.ff2.fc2.weight",
        "final_feed_forward.output_dense.bias": f"{pfx}.ff2.fc2.bias",
        # Final layer norms
        "final_layer_norm.weight": f"{pfx}.final_norm.weight",
        "final_layer_norm.bias": f"{pfx}.final_norm.bias",
    }
    return mappings.get(rest)


def convert_wav2vec_bert(
    hf_model_id: str = "facebook/w2v-bert-2.0",
    output_path: str = "./weights/mlx/wav2vec-bert-8bit.safetensors",
    quantize_bits: int = 8,
    hf_token: Optional[str] = None,
) -> None:
    """
    Download Wav2Vec-BERT 2.0 from HuggingFace, convert to MLX, optionally quantize.

    Notes
    -----
    - The convolutional feature extractor (7 strided Conv1d) is kept in fp16 to
      preserve waveform feature quality (small matrices, quantization-sensitive).
    - Conformer Conv1d (depthwise) is also excluded from quantization.
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
                isinstance(m, nn.Linear) and m.weight.shape[-1] >= 128
            ),
        )
        mx.eval(model.parameters())

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    weights_dict = dict(model.parameters())
    mx.save_safetensors(str(out), {k: v for k, v in weights_dict.items() if v is not None})
    print(f"Saved → {out}")
