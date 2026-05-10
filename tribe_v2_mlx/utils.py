"""Shared utilities for tribe_v2_mlx."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np


def get_hf_token() -> Optional[str]:
    """Return the HuggingFace token from the environment."""
    return os.environ.get("HF_TOKEN")


def weights_dir(base: str = "./weights") -> Path:
    """Return (and create if needed) the weights root directory."""
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def mlx_weights_dir(base: str = "./weights/mlx") -> Path:
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pt_tensor_to_mlx(tensor) -> mx.array:
    """Convert a PyTorch tensor (CPU) to an MLX array via numpy."""
    return mx.array(tensor.detach().float().numpy())


def mlx_to_numpy(x: mx.array) -> np.ndarray:
    """Materialise an MLX array to a numpy array."""
    mx.eval(x)
    return np.array(x)
