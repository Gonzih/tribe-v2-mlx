# tribe-v2-mlx

MLX-accelerated TRIBE v2 fMRI prediction pipeline for Apple Silicon.

TRIBE v2 predicts BOLD fMRI responses on the fsaverage5 cortical mesh `(n_segments, 20484)` from video, audio, image, and text stimuli. This package re-implements all four frozen encoders and the TRIBE transformer in [MLX](https://github.com/ml-explore/mlx) with optional 4/8-bit quantization.

## Architecture

| Component | Framework | Precision |
|---|---|---|
| V-JEPA2 ViT-g (1408 dim, 40 layers) | MLX | 8-bit |
| DINOv2-Large (1024 dim, 24 layers) | MLX | 8-bit |
| Wav2Vec-BERT 2.0 (conformer, 24 blocks) | MLX | 8-bit (Conv1d in fp16) |
| LLaMA 3.2-3B | MLX via mlx_lm | 4-bit |
| TRIBE transformer (~50M params) | MLX | fp16 |

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- MLX 0.17+ (install via `brew install mlx` or pip)

## Installation

```bash
pip install -e ".[dev]"
```

## Running Tests

All tests run without downloading actual model weights — synthetic data only.

```bash
# Unit tests (model components)
python -m pytest tests/test_components.py -v

# Integration tests (end-to-end pipeline with synthetic video)
python -m pytest tests/test_pipeline.py -v

# Quantization quality tests
python -m pytest tests/test_quantization.py -v -m "not requires_weights"

# Full suite
python -m pytest tests/ -v
```

### Test Results (Apple Silicon M4 Max, MLX 0.31.x)

```
36 passed in 0.79s
```

| Test file | Tests | Status |
|---|---|---|
| `test_components.py` | 18 | All pass |
| `test_pipeline.py` | 10 | All pass |
| `test_quantization.py` | 8 | All pass |

MLX default device: `Device(gpu, 0)` — Metal GPU confirmed.

Tests that require actual downloaded weights are marked `@pytest.mark.requires_weights` and skipped in the standard run.

## Inference (with real weights)

```python
from tribe_v2_mlx import TribeV2MLXPipeline

pipeline = TribeV2MLXPipeline.from_weights("./weights/mlx")
preds = pipeline.predict("video.mp4")       # (n_segments, 20484)
preds = pipeline.predict_image("frame.jpg") # (1, 20484)
```

## Weight Conversion

```bash
export HF_TOKEN=<your_token>
python scripts/download_weights.py   # downloads from HuggingFace
python scripts/convert_to_mlx.py    # converts to MLX safetensors
```

Converted weights layout:
```
weights/mlx/
  vjepa2-vitg-8bit.safetensors
  dinov2-large-8bit.safetensors
  wav2vec-bert-8bit.safetensors
  llama-3.2-3b-4bit/
weights/tribe/best.ckpt
```

## Memory Footprint (M4 Max 48 GB)

Mixed precision layout keeps total frozen encoder memory ~3.5 GB:
- LLaMA 3.2-3B 4-bit: ~1.7 GB
- V-JEPA2 ViT-g 8-bit: ~1.1 GB
- DINOv2-Large 8-bit: ~0.4 GB
- Wav2Vec-BERT 8-bit: ~0.3 GB

Sequential loading keeps peak below ~6 GB.
