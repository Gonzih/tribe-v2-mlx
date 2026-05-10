"""
tribe_v2_mlx — TRIBE v2 fMRI prediction pipeline on Apple Silicon via MLX.

Provides:
  - MLX model implementations for all frozen encoders
  - Weight conversion scripts (PyTorch → MLX safetensors)
  - Inference pipeline returning (n_segments, 20484) fMRI predictions
"""

from tribe_v2_mlx.pipeline import TribeV2MLXPipeline

__all__ = ["TribeV2MLXPipeline"]
__version__ = "0.1.0"
