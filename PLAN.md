# PLAN: TRIBE v2 MLX Quantization Pipeline — Implementation

## Task Restatement
Build a Python package `tribe_v2_mlx` that:
1. Downloads TRIBE v2 weights from HuggingFace (using env var `HF_TOKEN`)
2. Converts each component (LLaMA 3.2-3B, V-JEPA2 ViT-g, DINOv2-Large, Wav2Vec-BERT 2.0, TRIBE transformer) to MLX
3. Provides an inference pipeline `predict(video_path) -> np.ndarray` returning `(n_segments, 20484)` fMRI predictions
4. Tests that run without real weights (mock/synthetic inputs)
5. Scripts for download, convert, run

## Architecture Decisions

### Approach A: Pure MLX (chosen)
Implement all models in MLX (mlx.core), including the TRIBE transformer. Bridge to/from numpy for I/O.
- **Pro**: Consistent with "MLX throughout" constraint; unified memory benefits
- **Con**: TRIBE transformer checkpoint is PyTorch, needs conversion

### Approach B: MLX encoders + PyTorch TRIBE
Keep frozen encoders in MLX, keep TRIBE transformer in PyTorch.
- **Pro**: TRIBE checkpoint loads directly
- **Con**: Mixed framework; violates "MLX throughout" constraint

### Approach C: Full PyTorch
Keep everything in PyTorch.
- **Pro**: No conversion needed
- **Con**: Doesn't use Apple Silicon MLX benefits

**Chosen: Approach A** — pure MLX pipeline with numpy bridge.

## Component Implementation Plan

### MLX Models
- `ViTBlock` shared base for ViT-g (V-JEPA2) and ViT-L (DINOv2)
  - `mlx.nn.MultiHeadAttention` for attention (separate Q,K,V projections)
  - `mlx.nn.LayerNorm` + `mlx.nn.Linear` for feed-forward
  - Return hidden states at specified layer indices
- `ConformerBlock` for Wav2Vec-BERT 2.0
  - Attention + conformer conv module (depthwise Conv1d) + FFN
  - Standard attention (relative position bias as optional additive mask)
- `MLXLlamaExtractor` wrapping mlx_lm loaded model
  - Access `model.model.layers` directly for intermediate hidden states
- `TribeTransformer` in MLX
  - Per-modality MLP projectors
  - 8-layer Transformer encoder (`nn.TransformerEncoderLayer` equivalent)
  - Low-rank head + SubjectLayers

### Weight Mapping Strategy
- HF ViT: `encoder.layer.N.attention.attention.{query,key,value}` → MLX `blocks.N.attn.{query,key,value}_proj`
- HF DINOv2: similar ViT layout + register tokens
- HF Wav2Vec-BERT: `encoder.layers.N.{attention,feed_forward}` → conformer blocks
- TRIBE checkpoint: strip `model.` prefix, map to our TribeTransformer layout

### Test Strategy
- All models support random initialization (MLX default)
- `conftest.py` provides synthetic video fixture (imageio random frames)
- `test_components.py`: instantiate each model, run forward pass on small synthetic input, check output shape
- `test_pipeline.py`: mock all model loading, run pipeline with synthetic video
- `test_quantization.py`: create small ViTBlock, run fp16 output, quantize, run int8 output, check cosine similarity > 0.95

## Files to Create
```
pyproject.toml
tribe_v2_mlx/__init__.py
tribe_v2_mlx/utils.py
tribe_v2_mlx/preprocessing/__init__.py
tribe_v2_mlx/preprocessing/video.py
tribe_v2_mlx/preprocessing/audio.py
tribe_v2_mlx/models/__init__.py
tribe_v2_mlx/models/vjepa2.py
tribe_v2_mlx/models/dinov2.py
tribe_v2_mlx/models/wav2vec_bert.py
tribe_v2_mlx/models/llama.py
tribe_v2_mlx/models/tribe.py
tribe_v2_mlx/conversion/__init__.py
tribe_v2_mlx/conversion/vjepa2.py
tribe_v2_mlx/conversion/dinov2.py
tribe_v2_mlx/conversion/wav2vec_bert.py
tribe_v2_mlx/conversion/llama.py
tribe_v2_mlx/pipeline.py
tests/__init__.py
tests/conftest.py
tests/test_components.py
tests/test_pipeline.py
tests/test_quantization.py
scripts/download_weights.py
scripts/convert_to_mlx.py
scripts/run_inference.py
```

## Risks & Unknowns
- mlx_lm internal API may differ between versions; LLaMA hidden state extraction needs testing
- V-JEPA2 HF model might use non-standard ViT keys; conversion script includes key inspection
- TRIBE checkpoint `model_build_args` format unknown until actual download
- MLX `nn.quantize` class_predicate semantics — tested in test_quantization.py
- Tubelet embedding (3D patches) for video input — implemented as reshape+Linear for portability
