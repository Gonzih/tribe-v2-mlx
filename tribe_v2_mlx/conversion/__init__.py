"""
Weight conversion utilities: HuggingFace PyTorch → MLX safetensors.

Each submodule converts one encoder:
  vjepa2.py    — V-JEPA2 ViT-g (facebook/vjepa2-vitg-fpc64-256)
  dinov2.py    — DINOv2-Large  (facebook/dinov2-large)
  wav2vec_bert.py — Wav2Vec-BERT 2.0 (facebook/w2v-bert-2.0)
  llama.py     — LLaMA 3.2-3B  (mlx-community/Llama-3.2-3B-Instruct-4bit)
"""

from tribe_v2_mlx.conversion.vjepa2 import convert_vjepa2
from tribe_v2_mlx.conversion.dinov2 import convert_dinov2
from tribe_v2_mlx.conversion.wav2vec_bert import convert_wav2vec_bert
from tribe_v2_mlx.conversion.llama import convert_llama

__all__ = ["convert_vjepa2", "convert_dinov2", "convert_wav2vec_bert", "convert_llama"]
