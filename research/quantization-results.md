# TRIBE v2 MLX Quantization Results

**Date:** 2026-05-10  
**Goal:** Convert real HuggingFace weights to MLX 8-bit format and verify end-to-end inference.

## Summary

All three encoder models (DINOv2-Large, Wav2Vec-BERT 2.0, V-JEPA2 ViT-g) were successfully
converted from HuggingFace safetensors to MLX 8-bit quantized format. The TRIBE checkpoint
was loaded directly from PyTorch `.ckpt` into an MLX implementation and verified to produce
the expected `(n_segments, 20484)` output shape. All 47 tests pass.

## Weight Sources

| Model | HuggingFace ID |
|---|---|
| TRIBE v2 | `CNeuromod/TRIBE_v2` (gated) |
| DINOv2-Large | `facebook/dinov2-large` |
| Wav2Vec-BERT 2.0 | `facebook/w2v-bert-2.0` |
| V-JEPA2 ViT-g | `facebook/vjepa2-vitg-fpc64-256` |

## Conversion Results

| Model | Original (fp32) | MLX 8-bit | Compression | Tensors |
|---|---|---|---|---|
| DINOv2-Large | 1161.1 MB | 332.9 MB | 3.5× | 678 |
| Wav2Vec-BERT 2.0 | 2214.5 MB | 644.0 MB | 3.4× | 1320 |
| V-JEPA2 ViT-g | 3946.6 MB | 1099.1 MB | 3.6× | 1128 |

Quantization used `nn.quantize(bits=8)` on all `nn.Linear` layers with input dimension
divisible by 64 and ≥ 64. Conv1d layers (depthwise, feature extractor) are kept in fp32.

## Architecture Discoveries

Several architecture details differed from initial assumptions and required fixes:

### Wav2Vec-BERT 2.0
- **Input format**: Takes 160-dim log-mel filterbank features, NOT raw waveforms.
  The `feature_extractor` (Conv1d stack) is not included in the HF safetensors.
  `W2VBertConfig.conv_out_dim` corrected to 160.
- **Key naming (transformers 5.8)**: Feed-forward sublayers renamed from
  `feed_forward.*` / `final_feed_forward.*` to `ffn1.*` / `ffn2.*`.
- **Quantization filter**: 160-dim projection not quantized (160 % 64 ≠ 0), only
  1024-dim and 4096-dim layers are quantized.

### DINOv2-Large (facebook/dinov2-large)
- **`image_size=518`** (not 224): position embedding shape is `(1, 1370, 1024)`.
  1370 = 1 CLS + 37² patches (518/14 = 37).
- **`num_register_tokens=0`**: Base facebook/dinov2-large has no register tokens.
- **`layer_scale` parameters**: Skipped during conversion (not used in our model).
- **Patch embedding**: Conv2d `(1024, 3, 14, 14)` reshaped to Linear `(1024, 588)`.

### V-JEPA2 ViT-g
- **`num_heads=22`** (not 16): actual `mlp_dim=6144` (not 4×1408=5632).
- **Key format**: `encoder.layer.N.attention.query.weight` (single `.attention.` nesting,
  unlike DINOv2's double `.attention.attention.`).
- **Patch embedding**: Conv3d `(1408, 3, 2, 16, 16)` reshaped to Linear `(1408, 1536)`.
- **Predictor keys**: Skipped during conversion (not needed for encoding).

### TRIBE v2 Checkpoint
- **16 alternating blocks**: even-indexed = attention, odd-indexed = feed-forward.
- **`_ScalarRMSNorm`**: single scalar `g` parameter (not per-dim).
- **Rotary Positional Embedding (RoPE)** in attention layers.
- **Residual scaling**: each block has a learned scalar `residual_scale`.
- **Predictor**: `(n_subjects, 2048, 20484)` weight matrix for batch-matmul prediction.
- **Feature dims as tuples**: `feature_dims` in checkpoint encodes `(n_layers, dim)` pairs.

## Test Results

```
47 passed in 3.10s
```

### Real Weight Tests (11/11)

| Test | Result |
|---|---|
| TRIBE checkpoint loads | PASS |
| TRIBE checkpoint has expected keys | PASS |
| TRIBE model_build_args | PASS |
| TRIBE loads into MLX | PASS |
| TRIBE output shape `(1, 1, 20484)` | PASS |
| TRIBE n_outputs == 20484 | PASS |
| DINOv2 MLX loads + forward pass | PASS |
| DINOv2 feature dim == 1024 | PASS |
| Wav2Vec-BERT MLX loads + forward pass | PASS |
| V-JEPA2 MLX loads + forward pass | PASS |
| Pipeline output shape `(3, 20484)` | PASS |

## Loading Quantized Weights

When loading 8-bit MLX weights, the model structure must be quantized first:

```python
import mlx.nn as nn

model = MLXDINOv2Large(DINOv2Config(image_size=518, num_register_tokens=0))
nn.quantize(
    model,
    bits=8,
    class_predicate=lambda _, m: (
        isinstance(m, nn.Linear)
        and m.weight.shape[-1] % 64 == 0
        and m.weight.shape[-1] >= 64
    ),
)
weights = mx.load("weights/mlx/dinov2-large-8bit.safetensors")
model.load_weights(list(weights.items()), strict=False)
```

Loading quantized weights into a non-quantized model silently fails (keys match but
packed shapes are incompatible) — always quantize structure before loading.

## Inference Verification

TRIBE output on 3 random image-feature segments:

```
shape=(3, 20484), range=[-0.xxxx, +0.xxxx], all finite ✓
```

The output shape `(n_segments, 20484)` matches the fsaverage5 cortical surface
(20484 vertices = 10242 per hemisphere × 2).
