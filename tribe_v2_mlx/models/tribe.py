"""
MLX implementation of the TRIBE v2 FmriEncoderModel transformer.

Architecture (from research/tribe-v2-mlx-research.md):
  Per-modality MLP projectors: (B, T, in_dim_mod) → (B, T, mod_dim)
  Concat modalities: (B, T, hidden)
  8-layer Transformer encoder (hidden dim)
  Low-rank head: hidden → low_rank_dim (2048)
  SubjectLayers: low_rank_dim → n_outputs (20484)

Weight loading:
  Best.ckpt stores model_build_args + state_dict with "model." prefix.
  Call TribeTransformer.from_checkpoint(path) to load.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class TribeConfig:
    # Modality input dimensions after layer-concat
    # text: 6 layers × 3072 = 18432
    # video: 2 layers × 1408 = 2816
    # image: 1 layer × 1024 = 1024
    # audio: 2 layers × 1024 = 2048
    text_in_dim: int = 18432
    video_in_dim: int = 2816
    image_in_dim: int = 1024
    audio_in_dim: int = 2048

    # Projected dim per modality (hidden / n_active_modalities)
    mod_dim: int = 288  # 1152 / 4

    # Transformer
    hidden: int = 1152
    num_heads: int = 16
    depth: int = 8
    mlp_ratio: int = 4

    # Output
    low_rank_dim: int = 2048
    n_outputs: int = 20484   # fsaverage5 cortical vertices
    n_subjects: int = 8      # number of subjects in checkpoint (placeholder)
    max_seq_len: int = 1024


class ModalityProjector(nn.Module):
    """Projects concatenated-layer features to shared modality dim."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, in_dim) → (B, T, out_dim)"""
        return self.linear(self.norm(x))


class SubjectLayers(nn.Module):
    """Per-subject output projection weights."""

    def __init__(self, in_dim: int, n_outputs: int, n_subjects: int):
        super().__init__()
        # Stored as flat list; reshaped on forward
        self.n_subjects = n_subjects
        self.in_dim = in_dim
        self.n_outputs = n_outputs
        # weights shape: (n_subjects * in_dim * n_outputs,) stored as linear
        # We represent as a list of Linear layers (one per subject)
        self.layers = [nn.Linear(in_dim, n_outputs, bias=False) for _ in range(n_subjects)]

    def __call__(self, x: mx.array, subject_id: int = 0) -> mx.array:
        """x: (B, T, in_dim) → (B, T, n_outputs)"""
        layer = self.layers[subject_id % self.n_subjects]
        return layer(x)


class TribeTransformerBlock(nn.Module):
    def __init__(self, hidden: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.k_proj = nn.Linear(hidden, hidden, bias=True)
        self.v_proj = nn.Linear(hidden, hidden, bias=True)
        self.out_proj = nn.Linear(hidden, hidden, bias=True)

        self.norm2 = nn.LayerNorm(hidden)
        ff_dim = hidden * mlp_ratio
        self.ff1 = nn.Linear(hidden, ff_dim, bias=True)
        self.ff2 = nn.Linear(ff_dim, hidden, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # Self-attention
        nx = self.norm1(x)
        q = self.q_proj(nx).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(nx).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(nx).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        sa = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        x = x + self.out_proj(sa)

        # Feed-forward
        x = x + self.ff2(nn.gelu(self.ff1(self.norm2(x))))
        return x


class TribeTransformer(nn.Module):
    """
    TRIBE v2 FmriEncoderModel implemented in MLX.

    Input: dict with optional keys 'text', 'video', 'image', 'audio'.
      text:  (B, T, text_in_dim)
      video: (B, T, video_in_dim)
      image: (B, T, image_in_dim)
      audio: (B, T, audio_in_dim)

    Output: (B, T, n_outputs) — predicted fMRI responses.
    """

    def __init__(self, config: Optional[TribeConfig] = None):
        super().__init__()
        if config is None:
            config = TribeConfig()
        self.config = config

        self.text_proj = ModalityProjector(config.text_in_dim, config.mod_dim)
        self.video_proj = ModalityProjector(config.video_in_dim, config.mod_dim)
        self.image_proj = ModalityProjector(config.image_in_dim, config.mod_dim)
        self.audio_proj = ModalityProjector(config.audio_in_dim, config.mod_dim)

        # Time positional embedding
        self.time_pos_embed = nn.Embedding(config.max_seq_len, config.hidden)

        self.transformer = [
            TribeTransformerBlock(config.hidden, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        ]
        self.norm = nn.LayerNorm(config.hidden)
        self.low_rank_head = nn.Linear(config.hidden, config.low_rank_dim, bias=True)
        self.subject_layers = SubjectLayers(
            config.low_rank_dim, config.n_outputs, config.n_subjects
        )

    def __call__(
        self,
        features: Dict[str, mx.array],
        subject_id: int = 0,
    ) -> mx.array:
        """
        Parameters
        ----------
        features : dict mapping modality name → (B, T, in_dim)
        subject_id : which subject's output projection to use

        Returns
        -------
        mx.array (B, T, n_outputs)
        """
        parts: List[mx.array] = []

        if "text" in features:
            parts.append(self.text_proj(features["text"]))
        if "video" in features:
            parts.append(self.video_proj(features["video"]))
        if "image" in features:
            parts.append(self.image_proj(features["image"]))
        if "audio" in features:
            parts.append(self.audio_proj(features["audio"]))

        if not parts:
            raise ValueError("At least one modality must be provided.")

        # Concatenate modality features
        x = mx.concatenate(parts, axis=-1)  # (B, T, n_mod * mod_dim)

        # If n_mods < 4, pad to full hidden dim with zeros
        n_mods = len(parts)
        if n_mods < 4:
            pad_dim = self.config.hidden - x.shape[-1]
            if pad_dim > 0:
                x = mx.concatenate(
                    [x, mx.zeros((*x.shape[:2], pad_dim), dtype=x.dtype)], axis=-1
                )

        # Positional embedding
        T = x.shape[1]
        positions = mx.arange(T)
        x = x + self.time_pos_embed(positions)

        # Transformer
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)

        # Head
        x = self.low_rank_head(x)   # (B, T, low_rank_dim)
        x = self.subject_layers(x, subject_id)  # (B, T, n_outputs)
        return x

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        subject_id: int = 0,
    ) -> "TribeTransformer":
        """
        Load from a TRIBE v2 PyTorch checkpoint (best.ckpt).

        The checkpoint must be a dict with:
          ckpt["model_build_args"]: {feature_dims, n_outputs, n_output_timesteps}
          ckpt["state_dict"] or top-level keys with "model." prefix
        """
        import torch

        ckpt = torch.load(
            str(ckpt_path),
            map_location="cpu",
            weights_only=True,
        )

        # Extract build args
        build_args = ckpt.get("model_build_args", {})
        n_outputs = build_args.get("n_outputs", 20484)
        feature_dims = build_args.get("feature_dims", {})

        # Build config from checkpoint
        config = TribeConfig(n_outputs=n_outputs)
        if "text" in feature_dims:
            config.text_in_dim = feature_dims["text"]
        if "video" in feature_dims:
            config.video_in_dim = feature_dims["video"]
        if "image" in feature_dims:
            config.image_in_dim = feature_dims["image"]
        if "audio" in feature_dims:
            config.audio_in_dim = feature_dims["audio"]

        model = cls(config)

        # Load state dict
        state = ckpt.get("state_dict", ckpt)
        # Strip "model." prefix
        mlx_weights = {}
        for k, v in state.items():
            key = k.removeprefix("model.")
            mlx_weights[key] = mx.array(v.float().numpy())

        model.load_weights(list(mlx_weights.items()), strict=False)
        mx.eval(model.parameters())
        return model
