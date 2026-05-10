# TRIBE v2 MLX Quantization — Feasibility Research

**Target hardware:** Apple M4 Max, 48 GB unified memory  
**Framework:** MLX v0.31.2 (May 2026)  
**Goal:** Run TRIBE v2 fMRI brain-response prediction locally, with quantized frozen encoders

---

## 1. TRIBE v2 Architecture Map

TRIBE v2 is a multimodal brain-encoding model that combines four frozen feature extractors with a learned TRIBE transformer that maps multimodal features to fMRI cortical surface predictions.

### 1.1 System Overview

```
Input (video / audio / text)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  Feature Extraction (frozen, pre-cached to disk)            │
│                                                              │
│  Text ──► LLaMA 3.2-3B          3B params  ~6 GB fp16      │
│  Audio ─► Wav2Vec-BERT 2.0    600M params  ~1.2 GB fp16    │
│  Video ─► V-JEPA2 ViT-g         1B params  ~2 GB fp16      │
│  Image ─► DINOv2-Large         307M params  ~0.6 GB fp16    │
└──────────────────────────────────────────────────────────────┘
        │  (B, n_layers, feature_dim, T) per modality
        ▼
┌─────────────────────────────────────────────────────────────┐
│  FmriEncoderModel (TRIBE transformer, learned)              │
│                                                             │
│  Per-modality MLP projectors  → (B, T, hidden//n_mod)      │
│  Concat all modalities        → (B, T, 1152)               │
│  8-layer Transformer encoder                               │
│  Low-rank head (→ 2048)                                    │
│  Per-subject SubjectLayers predictor                       │
│  AdaptiveAvgPool → (B, n_vertices, T)                      │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Output: (n_timesteps, ~20 484) on fsaverage5 cortical mesh
```

### 1.2 Component Details

| Component | HF / URL | Architecture | Params | fp16 size | Layers extracted | Output dim |
|---|---|---|---|---|---|---|
| LLaMA 3.2-3B | `meta-llama/Llama-3.2-3B` | Decoder-only LLM | 3.0 B | 6.0 GB | 6 (at 0, 0.2, 0.4, 0.6, 0.8, 1.0 depth) | 3 072 |
| V-JEPA2 ViT-g | `facebook/vjepa2-vitg-fpc64-256` | ViT-Giant, 1B | 1.0 B | 2.0 GB | 2 (at 0.75, 1.0 depth) | 1 408 |
| DINOv2-Large | `facebook/dinov2-large` | ViT-Large | 307 M | 0.6 GB | 1 (at depth 2/3) | 1 024 |
| Wav2Vec-BERT 2.0 | `facebook/w2v-bert-2.0` | Conformer encoder | ~600 M | 1.2 GB | 2 (at 0.75, 1.0 depth) | 1 024 |
| FmriEncoderModel | `facebook/tribev2` (best.ckpt) | 8-layer Transformer | ~50 M | 0.1 GB | N/A (TRIBE itself) | ~20 484 vertices |

**Total frozen encoder footprint (fp16):** ~9.8 GB  
**Total with TRIBE transformer:** ~9.9 GB  
**M4 Max 48 GB headprint:** ample — ~38 GB free for OS, activations, and video buffers

### 1.3 TRIBE Transformer Details (from `tribev2/model.py`)

```python
# Production config (tribev2/grids/defaults.py)
FmriEncoder(
    hidden=1152,                  # Transformer hidden dim
    encoder=TransformerEncoder(depth=8),
    low_rank_head=2048,           # bottleneck linear before subject head
    extractor_aggregation="cat",  # concat modality features along hidden dim
    layer_aggregation="cat",      # concat extracted layers along feature dim
    combiner=None,                # no extra MLP; direct concat → Transformer
    subject_layers=SubjectLayers(subject_dropout=0.1),
    modality_dropout=0.3,         # 30% modality dropout during training
    time_pos_embedding=True,
    max_seq_len=1024,
)
```

Input to TRIBE per modality: `(B, n_layers, feature_dim, T)`  
- Text: `(B, 6, 3072, T)` → after layer_cat → `(B, 18432, T)` → projector → `(B, T, 384)`  
- Video: `(B, 2, 1408, T)` → after layer_cat → `(B, 2816, T)` → projector → `(B, T, 384)`  
- Image: `(B, 1, 1024, T)` → projector → `(B, T, 384)`  
- Audio: `(B, 2, 1024, T)` → after layer_cat → `(B, 2048, T)` → projector → `(B, T, 384)`  
- After concat: `(B, T, 1536)` → Transformer → `(B, T, 1152)` → low-rank → `(B, 2048, T)` → SubjectLayers → `(B, 20484, T)`

### 1.4 Inference Entry Point (`tribev2/demo_utils.py`)

```python
from tribev2 import TribeModel

# Loads config.yaml + best.ckpt from HuggingFace
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

df = model.get_events_dataframe(video_path="path/to/video.mp4")
preds, segments = model.predict(events=df)
# preds.shape == (n_timesteps, 20484)  — on fsaverage5 cortical mesh
```

Checkpoint loading sequence:
1. `hf_hub_download("facebook/tribev2", "config.yaml")` and `"best.ckpt"`
2. `torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)`
3. Read `ckpt["model_build_args"]` → `{feature_dims, n_outputs, n_output_timesteps}`
4. `brain_model_config.build(**build_args)` → `FmriEncoderModel`
5. Strip `"model."` prefix from `state_dict`, `load_state_dict(strict=True, assign=True)`
6. `.to(device).eval()`

Feature extraction is done by `Data.get_loaders()` which calls `extractor.prepare(events)` per modality, caches activations to disk, then frees the GPU. This cache-to-disk pattern is the key hook for MLX integration.

---

## 2. MLX Framework Capabilities

MLX v0.31.2 (Apple ML Research, May 2026) provides:

- **Unified memory** — arrays live in shared CPU/GPU space; no data-copy overhead
- **Lazy evaluation** — compute graph compiled only when materialized
- **`mlx.nn`** — full suite: `Linear`, `LayerNorm`, `RMSNorm`, `MultiHeadAttention`, `Conv1d`, `Conv2d`, `Embedding`, `GELU`, `SiLU`, etc.
- **Native 4-bit and 8-bit quantized linear** via `mlx.nn.QuantizedLinear`
- **`mlx_lm.convert`** — automated HuggingFace → MLX conversion for registered decoder-only LLMs

### 2.1 Existing MLX Precedents Relevant to TRIBE v2

| MLX Example | Relevant For | Precedent |
|---|---|---|
| `llms/llama` + `mlx_lm.convert` | LLaMA 3.2-3B text encoder | Drop-in via `mlx-community/Llama-3.2-3B` |
| `segment_anything/` (SAM ViT-H, 632M) | V-JEPA2 ViT-g (1B), DINOv2-Large | ViT-class vision encoder conversion template |
| `clip/` (CLIP ViT) | DINOv2-Large ViT-Large | ViT conversion pattern |
| `whisper/` (audio Transformer encoder) | Wav2Vec-BERT 2.0 audio encoder | Audio encoder conversion pattern |
| `bert/` (BERT encoder) | Conformer layers in Wav2Vec-BERT | Encoder-only Transformer template |

---

## 3. Quantization Feasibility — Per Component

### 3.1 LLaMA 3.2-3B (Text Encoder)

**Quantization feasibility: HIGH**

- `mlx-community/Llama-3.2-3B` already exists and runs on Apple Silicon
- `mlx_lm.convert` handles this fully automatically
- TRIBE only needs intermediate hidden states (not next-token logits), so quantization accuracy loss is on representations, not predictions

**Conversion:**
```bash
mlx_lm.convert \
  --hf-path meta-llama/Llama-3.2-3B \
  --mlx-path ./mlx-models/llama-3.2-3b-4bit \
  -q --q-bits 4 --q-group-size 64
```

Or use the pre-converted community model directly:
```python
import mlx_lm
model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-4bit")
# Extract hidden states at specified layer indices
```

**Size estimates:**

| Precision | Size |
|---|---|
| fp16 | 6.0 GB |
| 8-bit | 3.0 GB |
| 4-bit | 1.5 GB |

**Risk:** Intermediate layer representations at 4-bit may diverge from fp16 baseline. 6 layers are extracted (not just the last), so quantization noise can compound across layers 0–1.0. Recommend 8-bit as the conservative default; validate Pearson R² against fp16 baseline before deploying 4-bit.

---

### 3.2 V-JEPA2 ViT-g (Video Encoder, Dynamic)

**Quantization feasibility: MEDIUM — requires custom conversion**

- **No existing MLX port.** `mlx_lm.convert` does not handle ViT architectures.
- Architecture is a standard ViT-Giant: patch embedding + positional embedding + N×(LayerNorm + MultiHeadAttention + MLP) blocks + layer norm
- ViT-g: ~1B params, 1408 hidden dim, 40 layers, 16 heads
- Precedent: `mlx-examples/segment_anything/` implements SAM's ViT-H (632M, 1280 hidden) — directly analogous

**Checkpoint available:** `https://dl.fbaipublicfiles.com/vjepa2/vitg.pt` (PyTorch `state_dict`)  
**HuggingFace:** `facebook/vjepa2-vitg-fpc64-256` (via `AutoModel.from_pretrained`)

**Conversion plan:**

```python
# Step 1: Load PyTorch weights
import torch
from transformers import AutoModel
vjepa2 = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
state_dict = vjepa2.state_dict()

# Step 2: Implement ViT-g in MLX (template from mlx-examples/segment_anything)
import mlx.nn as nn
import mlx.core as mx

class MLXViTGiant(nn.Module):
    def __init__(self):
        # patch_embed: Conv2d(3, 1408, kernel_size=16, stride=16)
        # pos_embed: Parameter(1, n_patches + 1, 1408)
        # blocks: 40× TransformerBlock(dim=1408, heads=16, mlp_ratio=4)
        # norm: LayerNorm(1408)

# Step 3: Map PyTorch param names → MLX param names, convert tensors
def convert_tensor(t):
    return mx.array(t.float().numpy())

# Step 4: Optionally quantize
mlx_model = nn.quantize(mlx_model, bits=8)  # or bits=4

# Step 5: Save
mx.save_safetensors("mlx-models/vjepa2-vitg-8bit.safetensors", mlx_model.parameters())
```

**Video preprocessing:** V-JEPA2 expects tubelet tokenization of video clips. TRIBE uses 4-second clips at 256px. The `HuggingFaceVideo` extractor in TRIBE currently runs this via PyTorch. For MLX inference, preprocessing (frame sampling, normalization, patch extraction) can remain in NumPy/PIL, with only the ViT forward pass moved to MLX.

**Size estimates:**

| Precision | Size |
|---|---|
| fp16 | 2.0 GB |
| 8-bit | 1.0 GB |
| 4-bit | 0.5 GB |

**Risk:** ViT-g is the largest and most complex encoder to port. TRIBE extracts only layers at 0.75 and 1.0 depth (layers 30 and 40 of 40). Quantization of deep layers in large ViTs is generally well-tolerated at 8-bit; 4-bit shows measurable degradation in feature quality for dense prediction tasks. Use 8-bit.

---

### 3.3 DINOv2-Large (Video Encoder, Static/Spatial)

**Quantization feasibility: HIGH — standard ViT, MLX CLIP example as template**

- Architecture: ViT-Large, 307M params, 1024 hidden dim, 24 layers, 16 heads
- TRIBE extracts 1 layer at depth 2/3 (layer 16)
- HuggingFace: `facebook/dinov2-large` — standard ViT accessible via `AutoModel`
- MLX ViT implementation available from `mlx-examples/clip/` (uses ViT encoder)

**Conversion plan:** Nearly identical to ViT-g above but smaller. The CLIP example's vision transformer maps almost directly:

```python
# DINOv2-Large shares the ViT-Large architecture with CLIP's ViT-L
# mlx-examples/clip/ MLX ViT is a direct template
# Only difference: DINOv2 uses register tokens + no [CLS] pool

import mlx.nn as nn
mlx_dino = nn.quantize(mlx_dino_large, bits=8)
```

**Size estimates:**

| Precision | Size |
|---|---|
| fp16 | 0.6 GB |
| 8-bit | 0.3 GB |
| 4-bit | 0.15 GB |

**Risk:** Low. DINOv2-Large is well-understood, smaller than the other encoders, and only one layer is extracted. 8-bit quantization is safe; 4-bit is likely fine too.

---

### 3.4 Wav2Vec-BERT 2.0 (Audio Encoder)

**Quantization feasibility: MEDIUM — conformer architecture requires custom MLX implementation**

- Architecture: Convolutional feature extractor (waveform → feature maps) + **Conformer** encoder (~600M params total)
- Conformer = Conv1d feature extractor + series of conformer blocks (each: LayerNorm → MultiHeadAttention → Conv module → FFN)
- HuggingFace: `facebook/w2v-bert-2.0`, class `Wav2Vec2BertModel` (supported since transformers v4.35)
- **No existing MLX port.** No `mlx-community` checkpoint.

**Key conformer block components (all expressible in MLX):**
- `mlx.nn.MultiHeadAttention` — relative position bias attention
- `mlx.nn.Conv1d` — depthwise conv in conformer conv module
- `mlx.nn.LayerNorm`, `mlx.nn.Linear`, `mlx.nn.GELU`, `mlx.nn.SiLU`

**Conversion plan:**

```python
# Step 1: Load weights
from transformers import Wav2Vec2BertModel
w2v_bert = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

# Step 2: Implement conformer in MLX
# Template: mlx-examples/whisper/ (audio Transformer encoder)
# Extra: add depthwise Conv1d conformer module (not in Whisper)

class MLXConformerBlock(nn.Module):
    def __init__(self, dim, heads, conv_kernel=31):
        self.ff1 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim), nn.SiLU(), nn.Linear(4*dim, dim))
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, heads)
        self.conv_norm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Linear(dim, 2*dim)   # GLU gate
        self.depthwise = nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel//2, groups=dim)
        self.pointwise2 = nn.Linear(dim, dim)
        self.ff2 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim), nn.SiLU(), nn.Linear(4*dim, dim))
        self.final_norm = nn.LayerNorm(dim)

# Step 3: Map HF param names → MLX tree, convert tensors
# Step 4: Quantize
mlx_w2v = nn.quantize(mlx_w2v_bert, bits=8)
```

**Size estimates:**

| Precision | Size |
|---|---|
| fp16 | 1.2 GB |
| 8-bit | 0.6 GB |
| 4-bit | 0.3 GB |

**Risk:** Conformer's depthwise convolutions have smaller weight matrices per layer than standard attention. Quantizing these to 4-bit can cause audible artifacts in audio tasks; for TRIBE's use (speech representation extraction, not audio generation), 8-bit is strongly preferred. The waveform feature extractor (small strided Conv1d stack) should remain in fp16.

---

### 3.5 FmriEncoderModel (TRIBE Transformer, Learned)

**Quantization feasibility: NOT NEEDED — keep in fp16**

- Small: ~50M parameters, 0.1 GB fp16
- This is the learned component that maps frozen features to fMRI predictions
- Quantizing it risks degrading brain response prediction R², the primary metric
- It runs on already-extracted features, so it adds minimal inference overhead
- Keep in PyTorch fp16 or MLX fp16 — no quantization

---

## 4. Memory Budget

### 4.1 M4 Max 48 GB — Component Allocation

| Component | fp16 | 8-bit | 4-bit | Recommended |
|---|---|---|---|---|
| LLaMA 3.2-3B | 6.0 GB | 3.0 GB | 1.5 GB | **4-bit** (well-validated) |
| V-JEPA2 ViT-g | 2.0 GB | 1.0 GB | 0.5 GB | **8-bit** (preserve feature quality) |
| DINOv2-Large | 0.6 GB | 0.3 GB | 0.15 GB | **8-bit** (or 4-bit, low risk) |
| Wav2Vec-BERT 2.0 | 1.2 GB | 0.6 GB | 0.3 GB | **8-bit** (conformer sensitivity) |
| FmriEncoderModel | 0.1 GB | — | — | **fp16** (no quant) |
| Video buffers (4-sec clips @ 256px) | ~0.5 GB | — | — | fp32 |
| fMRI output (20k vertices) | ~0.2 GB | — | — | fp32 |
| OS + runtime | ~8 GB | — | — | — |
| **Total (recommended)** | — | **~14 GB** | — | ✓ fits in 48 GB |
| **Total (all 4-bit)** | — | — | **~3 GB** | ✓ fits with huge margin |

With the recommended mixed strategy, peak utilization is ~14 GB — well within 48 GB, with 34 GB remaining for video frame buffers and OS headroom.

### 4.2 Sequential vs. Simultaneous Loading

TRIBE's existing data pipeline already supports sequential loading (`_free_extractor_model()` frees each encoder after caching to disk). This means all four encoders do **not** need to be resident simultaneously:

- Load encoder → extract features for entire dataset → save to disk → free encoder
- Load next encoder → repeat
- Load FmriEncoderModel → train/infer from cached features

For inference on a single video, the sequential strategy still applies: extract text features, audio features, video features in sequence (each ~1.5-6 GB peak), cache to memory/disk, then feed TRIBE. Peak memory at any moment: max(6GB LLaMA, 2GB ViT-g, 1.2GB Wav2Vec) = 6 GB + TRIBE 0.1 GB = 6.1 GB peak.

---

## 5. MLX Conversion Steps Per Component

### 5.1 LLaMA 3.2-3B — Ready to Use

```bash
# Option A: use pre-converted mlx-community checkpoint
pip install mlx-lm
python -c "import mlx_lm; m, t = mlx_lm.load('mlx-community/Llama-3.2-3B-4bit'); print('OK')"

# Option B: convert from HuggingFace yourself
mlx_lm.convert \
  --hf-path meta-llama/Llama-3.2-3B \
  --mlx-path ./mlx-models/llama-3.2-3b-4bit \
  -q --q-bits 4 --q-group-size 64
```

**Hook into TRIBE:** The `HuggingFaceText` extractor calls `model(**inputs, output_hidden_states=True)`. Replace with an MLX LLaMA that returns hidden states at the same layer indices.

```python
# In neuralset/extractors/huggingface_text.py (or a monkey-patch):
def extract_hidden_states(tokens, layer_indices):
    # Run MLX LLaMA forward, return intermediate activations
    outputs = mlx_model(tokens, output_hidden_states=True)
    return [outputs.hidden_states[i] for i in layer_indices]
```

---

### 5.2 V-JEPA2 ViT-g — Custom Conversion Required

```bash
# Step 1: Download PyTorch checkpoint
wget https://dl.fbaipublicfiles.com/vjepa2/vitg.pt -O ./checkpoints/vjepa2-vitg.pt

# OR use HuggingFace:
python -c "
from transformers import AutoModel
m = AutoModel.from_pretrained('facebook/vjepa2-vitg-fpc64-256')
m.save_pretrained('./checkpoints/vjepa2-vitg-hf')
"
```

```python
# Step 2: implement_vjepa2_mlx.py
# Reference: mlx-examples/segment_anything/mlx_sam/image_encoder.py

import mlx.core as mx
import mlx.nn as nn
import numpy as np, torch

class PatchEmbed(nn.Module):
    def __init__(self):
        self.proj = nn.Conv2d(3, 1408, kernel_size=16, stride=16)

class ViTBlock(nn.Module):
    def __init__(self, dim=1408, heads=16):
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
    def __call__(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MLXVJepa2ViTG(nn.Module):
    def __init__(self, depth=40):
        self.patch_embed = PatchEmbed()
        self.pos_embed = mx.zeros((1, 257, 1408))   # learned, loaded from ckpt
        self.blocks = [ViTBlock() for _ in range(depth)]
        self.norm = nn.LayerNorm(1408)

    def __call__(self, x, return_layers=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        hidden_states = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_layers and i in return_layers:
                hidden_states.append(x)
        return self.norm(x), hidden_states

# Step 3: weight mapping
def pt_to_mlx(pt_state_dict):
    mlx_weights = {}
    for k, v in pt_state_dict.items():
        arr = mx.array(v.float().numpy())
        # map: blocks.N.attn.proj_q.weight → blocks.N.attn.query_proj.weight  (etc.)
        mlx_weights[map_key(k)] = arr
    return mlx_weights

# Step 4: quantize and save
model = MLXVJepa2ViTG()
model.update(mlx_weights)
model = nn.quantize(model, bits=8, class_predicate=lambda _, m: isinstance(m, nn.Linear))
mx.save_safetensors("./mlx-models/vjepa2-vitg-8bit.safetensors", dict(model.parameters()))
```

**Key weight mapping challenge:** The HuggingFace V-JEPA2 model uses the standard ViT naming convention from `transformers.ViTModel`. The `blocks.N.attn` → `encoder.layer.N.attention.attention` renaming is the main mapping task. Inspecting `vjepa2.state_dict().keys()` against the MLX module tree is the first concrete implementation step.

---

### 5.3 DINOv2-Large — Custom Conversion (Simpler)

```python
# Template: mlx-examples/clip/mlx_clip/vision_transformer.py
# DINOv2-Large == ViT-L with register tokens (4 extra tokens after [CLS])
# Only difference from CLIP ViT: no final text projection, no pooling by default

from transformers import AutoModel
dino = AutoModel.from_pretrained("facebook/dinov2-large")
# state_dict keys follow standard ViT-L layout
# Extract layer at depth 2/3 = layer index 16 (of 24 total)

class MLXDINOv2Large(nn.Module):
    def __init__(self):
        self.embeddings = ...    # patch_embed + cls_token + reg_tokens + pos_embed
        self.encoder = [ViTBlock(dim=1024, heads=16) for _ in range(24)]
        self.layernorm = nn.LayerNorm(1024)

model = MLXDINOv2Large()
model = nn.quantize(model, bits=8)
```

---

### 5.4 Wav2Vec-BERT 2.0 — Custom Conversion (Complex)

```python
# Template: mlx-examples/whisper/ (audio Transformer encoder)
# Extra: conformer conv module not in Whisper — implement from scratch

from transformers import Wav2Vec2BertModel
w2v = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

# Architecture layers (from HF source):
# feature_extractor: stack of Conv1d(stride=[5,2,2,2,2,2,2]) → 1/320 sample rate reduction
# feature_projection: Linear(512→1024) + LayerNorm + Dropout
# encoder: 24× ConformerBlock(dim=1024, heads=16, conv_kernel=31)

class MLXConformerEncoder(nn.Module):
    def __init__(self, depth=24, dim=1024, heads=16):
        self.pos_conv = nn.Conv1d(dim, dim, 128, padding=64, groups=dim)
        self.blocks = [MLXConformerBlock(dim, heads) for _ in range(depth)]
        self.layer_norm = nn.LayerNorm(dim)

# Quantize only the conformer linear layers; keep Conv1d feature extractor in fp16
model = nn.quantize(model, bits=8,
    class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[0] >= 128)
```

**Important:** The waveform feature extractor (7 strided Conv1d layers) should remain in fp16/fp32. Its weights are tiny (<1 MB total) and quantizing strided convolutions at very small channel counts causes significant distortion.

---

## 6. Integration Architecture for Local M4 Max Inference

### 6.1 Proposed Pipeline

```
┌─────────────────────────────────────────────────────┐
│  video.mp4 / audio.wav / text                      │
│     │                                               │
│     ▼                                               │
│  get_events_dataframe() [unchanged — PyTorch/CPU]  │
│     │ Events DataFrame                              │
│     ▼                                               │
│  MLX Feature Extraction (sequential, disk-cached)  │
│  ┌─────────────────────────────────────────────┐   │
│  │ 1. MLXLlamaExtractor.prepare(events)        │   │
│  │    → text_features.npz (disk)               │   │
│  │    → unload (free Metal memory)             │   │
│  │ 2. MLXVJepa2Extractor.prepare(events)       │   │
│  │    → video_features.npz                     │   │
│  │    → unload                                 │   │
│  │ 3. MLXDINOv2Extractor.prepare(events)       │   │
│  │    → image_features.npz                     │   │
│  │    → unload                                 │   │
│  │ 4. MLXWav2VecBertExtractor.prepare(events)  │   │
│  │    → audio_features.npz                     │   │
│  │    → unload                                 │   │
│  └─────────────────────────────────────────────┘   │
│     │ Cached feature arrays                         │
│     ▼                                               │
│  FmriEncoderModel.forward(batch) [PyTorch fp16]    │
│     │                                               │
│     ▼                                               │
│  preds: (n_timesteps, 20484) numpy array           │
└─────────────────────────────────────────────────────┘
```

### 6.2 Extractor Interface Contract

TRIBE's existing extractor system (`neuralset.extractors.BaseExtractor`) expects:
1. `prepare(events: DataFrame)` — run feature extraction, cache to `self.folder / "{event_id}.npy"`
2. `__call__(segment: SegmentData) → Tensor` — load cached features for a segment, return `(n_layers, feature_dim, T)`

The MLX extractors need to satisfy this same interface. The simplest approach: run MLX forward pass, convert output to NumPy, save as `.npy`. The FmriEncoderModel then loads these via PyTorch `torch.from_numpy()` — zero-copy if the array is on CPU.

```python
class MLXVJepa2Extractor(BaseExtractor):
    def __init__(self, model_path="./mlx-models/vjepa2-vitg-8bit.safetensors", layers=(30, 40)):
        self.model = load_mlx_vjepa2(model_path)
        self.layer_indices = layers   # at depth 0.75 → layer 30; depth 1.0 → layer 40

    def prepare(self, events):
        for event in events[events.type == "Video"].itertuples():
            frames = load_video_frames(event.filepath, clip_duration=4.0)  # NumPy (T, H, W, 3)
            frames_mlx = mx.array(frames)
            _, hidden_states = self.model(frames_mlx, return_layers=self.layer_indices)
            features = np.stack([np.array(h) for h in hidden_states])  # (n_layers, T, feature_dim)
            np.save(self.folder / f"{event.filepath.stem}.npy", features)
```

### 6.3 Quantization Inference Mode

Since the FmriEncoderModel is small and already loads in PyTorch from `best.ckpt`, the integration strategy avoids rewriting it:

- **Frozen encoders**: MLX with 4-bit (LLaMA) or 8-bit (ViT-g, DINOv2, Wav2Vec-BERT) quantization
- **TRIBE transformer**: PyTorch fp16, loaded as-is from `best.ckpt`
- **Bridge**: NumPy arrays on CPU (MLX → NumPy → `torch.from_numpy()`)

This minimizes the scope of MLX implementation while capturing the largest memory savings (the frozen encoders are >99% of total parameter count).

---

## 7. Risk Areas and What to Test

### 7.1 Quality Risks

| Risk | Severity | Mitigation |
|---|---|---|
| LLaMA 3.2-3B 4-bit layer representations drift from fp16 | Medium | Benchmark intermediate hidden states cosine similarity; test Pearson R² on 10-min Algonauts2025 clip |
| V-JEPA2 ViT-g 8-bit degrades video features | Medium | Compare TRIBE output correlation before/after quantization on held-out fMRI data |
| Wav2Vec-BERT conformer 8-bit reduces audio representation quality | Medium | Measure phoneme-level activation similarity in early vs. late conformer layers |
| Weight key mismatch in PyTorch → MLX mapping | High (implementation) | Log all unmapped keys; assert no fp16 fallback parameters remain |
| Video tubelet tokenization misalignment | Medium | Validate that MLX patch extraction produces identical token sequences to HF preprocessor |

### 7.2 Implementation Risks

| Risk | Severity | Mitigation |
|---|---|---|
| V-JEPA2 HF model uses non-standard attention (e.g. RoPE, relative position) | High | Read `vjepa2/src/models/` source before implementing MLX ViT |
| Wav2Vec-BERT relative position attention not in `mlx.nn.MultiHeadAttention` | High | Use ALiBi or rotary embedding implementation from mlx-examples/llms/mistral |
| `nn.quantize()` not compatible with `nn.Conv1d` (conformer audio extractor) | Medium | `class_predicate` to exclude Conv1d layers from quantization |
| MLX `nn.QuantizedLinear` group size incompatibility with small matrices | Low | Use `q-group-size=32` for small matrices (<128 cols) |

### 7.3 Benchmarks to Run Before Declaring Success

1. **Cosine similarity test**: for each encoder, run 100 random inputs through PyTorch fp16 vs. MLX quantized and measure mean cosine similarity of output hidden states. Target: >0.99 for 8-bit, >0.97 for 4-bit.
2. **Brain encoding R² test**: run the full TRIBE pipeline on a 5-minute video with both fp16 PyTorch encoders and MLX quantized encoders. Measure Pearson R² on a held-out fMRI subject. Target: <2% R² drop vs. fp16 baseline.
3. **Throughput test**: measure feature extraction time per second of video on M4 Max. Target: faster than real-time (< 1s MLX inference per 1s video).

---

## 8. Recommended Implementation Sequence

1. **Week 1 — LLaMA 3.2-3B** (lowest friction): use `mlx-community/Llama-3.2-3B-4bit` directly. Write `MLXLlamaExtractor` wrapper. Validate hidden state similarity.
2. **Week 2 — DINOv2-Large** (simpler ViT): port `mlx-examples/clip/` ViT to DINOv2-Large weight layout. Write conversion script. Validate.
3. **Week 3 — V-JEPA2 ViT-g** (harder ViT, larger): extend DINOv2 conversion to ViT-g scale. Read V-JEPA2 source for attention variant. Write `MLXVJepa2Extractor`. Validate.
4. **Week 4 — Wav2Vec-BERT 2.0** (most novel — conformer): implement conformer blocks in MLX. Use whisper example as scaffolding. Validate with audio feature similarity test.
5. **Week 5 — Integration + benchmarks**: wire all four MLX extractors into TRIBE's `Data.get_loaders()` path. Run end-to-end R² benchmark. Tune quantization bits per encoder based on results.

---

## 9. Quick Reference — Key Files and URLs

| Resource | Location |
|---|---|
| TRIBE v2 GitHub | `https://github.com/facebookresearch/tribev2` |
| TRIBE v2 HuggingFace weights | `https://huggingface.co/facebook/tribev2` |
| TRIBE model code | `tribev2/model.py` (FmriEncoder, FmriEncoderModel) |
| TRIBE inference wrapper | `tribev2/demo_utils.py` (TribeModel) |
| TRIBE default config | `tribev2/grids/defaults.py` |
| V-JEPA2 GitHub | `https://github.com/facebookresearch/vjepa2` |
| V-JEPA2 ViT-g checkpoint | `https://dl.fbaipublicfiles.com/vjepa2/vitg.pt` |
| V-JEPA2 HF model | `facebook/vjepa2-vitg-fpc64-256` |
| DINOv2-Large HF | `facebook/dinov2-large` |
| Wav2Vec-BERT 2.0 HF | `facebook/w2v-bert-2.0` |
| LLaMA 3.2-3B MLX | `mlx-community/Llama-3.2-3B-4bit` |
| MLX SAM example (ViT-H template) | `https://github.com/ml-explore/mlx-examples/tree/main/segment_anything` |
| MLX Whisper example (audio template) | `https://github.com/ml-explore/mlx-examples/tree/main/whisper` |
| MLX CLIP example (ViT template) | `https://github.com/ml-explore/mlx-examples/tree/main/clip` |
| mlx_lm.convert docs | `https://github.com/ml-explore/mlx-examples/tree/main/llms` |
