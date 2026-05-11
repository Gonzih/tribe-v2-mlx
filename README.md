# tribe-v2-mlx

**TRIBE v2 (Meta's multimodal brain encoder) running fully local on Apple Silicon via MLX. First known Apple Silicon port.**

TRIBE v2 predicts BOLD fMRI responses across the fsaverage5 cortical surface — 20,484 vertices, 1 prediction per second of video — from video, audio, image, and text. The original requires 40 GB dedicated NVIDIA VRAM. This port runs on M-series unified memory with a 3.5–3.6× smaller frozen encoder footprint. No cloud. No API. No CUDA.

Output shape: `(n_segments, 20484)` — BOLD signal across both hemispheres of fsaverage5, with 5-second hemodynamic offset baked in.

---

## Architecture

Four frozen encoders extract multimodal features; a small learned TRIBE transformer maps them to cortical predictions.

| Component | Role | Params | fp32 on disk | MLX 8-bit | Compression |
|---|---|---|---|---|---|
| V-JEPA2 ViT-g | Video encoder | 1.0 B | 3,946.6 MB | 1,099.1 MB | **3.6×** |
| Wav2Vec-BERT 2.0 | Audio encoder | ~600 M | 2,214.5 MB | 644.0 MB | **3.4×** |
| DINOv2-Large | Image encoder | 307 M | 1,161.1 MB | 332.9 MB | **3.5×** |
| LLaMA 3.2-3B | Text encoder | 3.0 B | — | ~1,700 MB (4-bit) | — |
| TRIBE transformer | fMRI predictor | ~50 M | 709 MB (`best.ckpt`) | loaded directly | fp16 |

**Total frozen encoder footprint: ~8.3 GB fp32 → ~2.1 GB MLX quantized.**

The TRIBE transformer is loaded directly from Meta's `best.ckpt` PyTorch checkpoint — no conversion needed, weight file is 709 MB.

### TRIBE transformer internals

The checkpoint structure is completely different from what the paper describes:

- **16 alternating blocks**: even-indexed = attention, odd-indexed = feed-forward (not 8 uniform transformer blocks)
- **`_ScalarRMSNorm`**: single learned scalar `g`, not a per-dimension scale vector
- **RoPE**: rotary positional embeddings in all attention layers
- **`residual_scale`**: each block has a learned scalar multiplied onto the residual path
- **Predictor**: `(n_subjects, 2048, 20484)` weight matrix — subject-specific batch-matmul, not a single shared head
- **`feature_dims` in checkpoint**: stored as `(n_layers, dim)` tuples, not flat ints

### Per-modality projections

```
Text:   (B, 6, 3072, T)  → layer_cat → (B, 18432, T) → MLP → (B, T, 384)
Video:  (B, 2, 1408, T)  → layer_cat → (B, 2816, T)  → MLP → (B, T, 384)
Image:  (B, 1, 1024, T)                               → MLP → (B, T, 384)
Audio:  (B, 2, 1024, T)  → layer_cat → (B, 2048, T)  → MLP → (B, T, 384)

Concat: (B, T, 1536) → 8-layer Transformer → low-rank head (→ 2048) → SubjectLayers → (B, 20484, T)
```

---

## What TRIBE v2 actually predicts

Not virality scores. Neural engagement — which brain regions activate in response to video content.

High activations in reward/salience circuits (nucleus accumbens, anterior insula) correlate with viewer engagement under the Knutson AIM framework. TRIBE was trained on paired (video, fMRI) data from subjects watching natural video; the cortical surface output is a per-vertex regression of BOLD signal.

---

## Engineering discoveries

These took time to find. None are documented anywhere.

### Wav2Vec-BERT 2.0 takes log-mel, not raw waveforms
The HuggingFace model card implies raw waveform input (like Wav2Vec 1.x). Wrong. The deployed HF safetensors do **not** include the Conv1d feature extractor stack. Input is 160-dim log-mel filterbank features. `W2VBertConfig.conv_out_dim` must be set to 160.

### V-JEPA2 has 22 attention heads, not 16
Published ViT-g specs say 16 heads (1408 dim ÷ 88 head dim). Actual checkpoint: `num_heads=22`, `mlp_dim=6144` (not 4×1408=5632). Both differ from standard ViT-g. Read the weights, don't trust the paper.

### DINOv2-Large: image_size=518, num_register_tokens=0
Position embedding shape is `(1, 1370, 1024)` — that's 1 CLS + 37² patches (518/14=37). Not 224. Not 256. The `facebook/dinov2-large` base checkpoint has zero register tokens despite DINOv2v2 adding them in later variants.

### MLX quantized weights: quantize structure before loading
Loading 8-bit MLX weights into an unquantized model silently fails — key names match but packed tensor shapes are incompatible. Always quantize the model structure first, then call `load_weights()`:

```python
model = MLXDINOv2Large(config)
nn.quantize(model, bits=8, class_predicate=lambda _, m: (
    isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0 and m.weight.shape[-1] >= 64
))
weights = mx.load("weights/mlx/dinov2-large-8bit.safetensors")
model.load_weights(list(weights.items()), strict=False)
```

Not documented in MLX docs. Costs hours to debug.

### V-JEPA2 Conv3d patch embed → Linear
The HF checkpoint has a Conv3d patch embedding with shape `(1408, 3, 2, 16, 16)` (output_channels, in_channels, temporal_kernel, h, w). MLX has no Conv3d. Reshape to `Linear(1408, 1536)` where 1536 = 3×2×16×16 — functionally identical for non-strided tubelet embedding.

### PyTorch Conv1d weight layout transpose
PyTorch stores Conv1d weights as `(out_channels, in_channels, kernel_size)`. MLX expects `(out_channels, kernel_size, in_channels)`. All Conv1d weights in the Wav2Vec-BERT conformer need `.transpose(0, 2, 1)` before conversion.

### transformers 5.8 Wav2Vec-BERT key rename
Feed-forward sublayer keys changed between transformers versions:
- Old: `feed_forward.*`, `final_feed_forward.*`  
- New (v5.8): `ffn1.*`, `ffn2.*`

---

## Quick start

```bash
git clone https://github.com/gonzih/tribe-v2-mlx
cd tribe-v2-mlx
pip install -e .

# Download weights (~8 GB, requires HF token for gated TRIBE repo)
HF_TOKEN=your_token python scripts/download_weights.py

# Convert to MLX 8-bit
HF_TOKEN=your_token python scripts/convert_to_mlx.py

# Run inference
python scripts/run_inference.py --video your_video.mp4
```

**HF token note:** TRIBE v2 weights are hosted at `CNeuromod/TRIBE_v2` (gated). Request access at HuggingFace before downloading.

### Python API

```python
from tribe_v2_mlx import TribeV2MLXPipeline

pipeline = TribeV2MLXPipeline.from_weights("./weights/mlx")
preds = pipeline.predict("video.mp4")       # (n_segments, 20484)
preds = pipeline.predict_image("frame.jpg") # (1, 20484)
```

---

## Weight layout after conversion

```
weights/
  tribe/
    best.ckpt                         # 709 MB — TRIBE transformer, loaded as-is
  mlx/
    vjepa2-vitg-8bit.safetensors      # 1,099 MB — 1128 tensors
    dinov2-large-8bit.safetensors     #   333 MB — 678 tensors
    wav2vec-bert-8bit.safetensors     #   644 MB — 1320 tensors
    llama-3.2-3b-4bit/                # ~1,700 MB — via mlx-community
```

---

## Quantization

`nn.quantize(bits=8)` on all `nn.Linear` layers where `input_dim % 64 == 0` and `input_dim >= 64`. Conv1d layers (depthwise conformer blocks, feature extractor) stay in fp32 — their weight matrices are too small and strided convolutions degrade measurably at 8-bit.

LLaMA 3.2-3B at 4-bit via `mlx-community/Llama-3.2-3B-4bit` — well-validated, no custom conversion needed.

TRIBE transformer stays in fp16. It's 50M params, 709 MB, and quantizing the learned brain-prediction head risks R² degradation for minimal memory gain.

---

## Memory footprint (M4 Max 48 GB)

Sequential loading — each encoder is freed after feature extraction:

| Encoder | Peak during extraction |
|---|---|
| LLaMA 3.2-3B 4-bit | ~1.7 GB |
| V-JEPA2 ViT-g 8-bit | ~1.1 GB |
| Wav2Vec-BERT 8-bit | ~0.6 GB |
| DINOv2-Large 8-bit | ~0.3 GB |

**Peak at any point during inference: ~1.7 GB** (LLaMA load). Total resident when all encoders freed and TRIBE loaded: ~0.7 GB. The original TRIBE pipeline requires all encoders loaded simultaneously on a 40 GB NVIDIA GPU.

---

## Tests

47 tests, all passing. Unit + integration + real weight loading.

```bash
# All tests (synthetic data only — no weights needed)
python -m pytest tests/ -v

# Real weight tests (requires downloaded weights)
python -m pytest tests/test_real_weights.py -v -m requires_weights
```

```
47 passed in 3.10s
```

| Test file | Tests | Coverage |
|---|---|---|
| `test_components.py` | 18 | Model components, weight loading, quantization |
| `test_pipeline.py` | 10 | End-to-end pipeline with synthetic video |
| `test_quantization.py` | 8 | Quantization correctness, shape preservation |
| `test_real_weights.py` | 11 | Real checkpoint loading, output shape `(1, 1, 20484)` |

MLX default device: `Device(gpu, 0)` — Metal GPU, confirmed on M4 Max.

---

## Requirements

- Apple M2/M3/M4 Pro/Max/Ultra, 32 GB+ unified memory (tested on M4 Max 48 GB)
- Python 3.10+
- MLX 0.17+

```bash
pip install -e .
```

---

## Repository

```
tribe_v2_mlx/
  models/          # MLX implementations: DINOv2, Wav2Vec-BERT, V-JEPA2, TRIBE transformer
  conversion/      # PyTorch → MLX weight conversion scripts per encoder
  pipeline.py      # TribeV2MLXPipeline — end-to-end inference
  preprocessing/   # Video frame sampling, log-mel extraction
scripts/
  download_weights.py   # HF download with ignore_patterns for large legacy .pt files
  convert_to_mlx.py     # Converts all encoders, saves safetensors
  run_inference.py      # CLI inference entry point
research/
  tribe-v2-mlx-research.md   # Architecture research, conversion strategy
  quantization-results.md    # Actual compression numbers, architecture discoveries
```
