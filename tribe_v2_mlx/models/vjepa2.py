"""
MLX implementation of V-JEPA2 ViT-g for video feature extraction.

Architecture: ViT-Giant (1408 hidden, 40 layers, 16 heads, patch_size=16).
Processes video clips as tubelets (temporal + spatial patches).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class ViTConfig:
    hidden_dim: int = 1408
    depth: int = 40
    num_heads: int = 16
    mlp_ratio: int = 4
    patch_size: int = 16
    temporal_patch_size: int = 2  # tubelet temporal depth
    image_size: int = 256
    qkv_bias: bool = True
    # Layers to extract (depth-fraction → index, e.g. 0.75 → 30, 1.0 → 39)
    extract_layers: List[int] = field(default_factory=lambda: [30, 39])
    # Optional explicit mlp_dim override (for non-integer mlp_ratio models like V-JEPA2)
    _mlp_dim_override: Optional[int] = None

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def mlp_dim(self) -> int:
        if self._mlp_dim_override is not None:
            return self._mlp_dim_override
        return self.hidden_dim * self.mlp_ratio


class ViTAttention(nn.Module):
    """Multi-head self-attention for ViT (separate Q, K, V projections)."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5

        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.qkv_bias)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.qkv_bias)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.query_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)  # (B, H, N, D)
        k = self.key_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)
        v = self.value_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, N, N)
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        return self.out_proj(out)


class ViTMLP(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class ViTBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = ViTAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = ViTMLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TubeletEmbedding(nn.Module):
    """
    Temporal + spatial patch embedding via reshape + Linear.

    For static images (T=1), reduces to standard 2D patch embedding.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        in_features = config.temporal_patch_size * config.patch_size * config.patch_size * 3
        self.proj = nn.Linear(in_features, config.hidden_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        x: (B, T, H, W, C) float32
        returns: (B, n_tokens, hidden_dim)
        """
        B, T, H, W, C = x.shape
        pt = self.temporal_patch_size
        ph = pw = self.patch_size

        # Pad temporal dimension to multiple of pt
        if T % pt != 0:
            pad_t = pt - (T % pt)
            x = mx.concatenate(
                [x, mx.zeros((B, pad_t, H, W, C), dtype=x.dtype)], axis=1
            )
            T = T + pad_t

        nt = T // pt
        nh = H // ph
        nw = W // pw

        # Reshape into patches: (B, nt, nh, nw, pt*ph*pw*C)
        x = x.reshape(B, nt, pt, nh, ph, nw, pw, C)
        x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)          # (B, nt, nh, nw, pt, ph, pw, C)
        x = x.reshape(B, nt * nh * nw, pt * ph * pw * C)   # (B, n_tokens, features)
        return self.proj(x)                                  # (B, n_tokens, hidden_dim)


class PatchEmbedding2D(nn.Module):
    """Standard 2D patch embedding for static images."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        in_features = config.patch_size * config.patch_size * 3
        self.proj = nn.Linear(in_features, config.hidden_dim, bias=True)
        self.patch_size = config.patch_size

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C) → (B, n_patches, hidden_dim)"""
        B, H, W, C = x.shape
        ph = pw = self.patch_size
        nh, nw = H // ph, W // pw
        x = x.reshape(B, nh, ph, nw, pw, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, nh, nw, ph, pw, C)
        x = x.reshape(B, nh * nw, ph * pw * C)
        return self.proj(x)


class MLXVJepa2(nn.Module):
    """
    MLX V-JEPA2 ViT-g video encoder.

    Processes (B, T, H, W, C) video clips.
    Returns hidden states at `config.extract_layers`.

    Weight layout (for conversion from HuggingFace):
      embeddings.patch_embed.proj.{weight,bias}
      embeddings.pos_embed                        (1, n_tokens+1, hidden_dim)
      blocks.N.norm1.{weight,bias}
      blocks.N.attn.{query,key,value,out}_proj.{weight,bias}
      blocks.N.norm2.{weight,bias}
      blocks.N.mlp.{fc1,fc2}.{weight,bias}
      norm.{weight,bias}
    """

    def __init__(self, config: Optional[ViTConfig] = None):
        super().__init__()
        if config is None:
            config = ViTConfig()
        self.config = config

        self.patch_embed = TubeletEmbedding(config)
        # Positional embedding: add lazily based on actual token count
        self.blocks = [ViTBlock(config) for _ in range(config.depth)]
        self.norm = nn.LayerNorm(config.hidden_dim)

        # Positional embedding (max tokens = (256/16)^2 * 64/2 + 1 CLS = 2049)
        max_tokens = 2048 + 1
        self.pos_embed = mx.zeros((1, max_tokens, config.hidden_dim))
        self.cls_token = mx.zeros((1, 1, config.hidden_dim))

    def __call__(
        self,
        x: mx.array,
        extract_layers: Optional[List[int]] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        """
        Parameters
        ----------
        x : mx.array (B, T, H, W, C) float32
        extract_layers : list of layer indices to return; defaults to config.extract_layers

        Returns
        -------
        (final_hidden, [hidden at each extract_layer])
        Each hidden: (B, n_tokens, hidden_dim)
        """
        if extract_layers is None:
            extract_layers = self.config.extract_layers

        B = x.shape[0]
        tokens = self.patch_embed(x)          # (B, n_tokens, hidden_dim)
        n_tokens = tokens.shape[1]

        cls = mx.broadcast_to(self.cls_token, (B, 1, self.config.hidden_dim))
        tokens = mx.concatenate([cls, tokens], axis=1)  # (B, n_tokens+1, D)

        # Add positional embedding (truncate/pad to match actual token count)
        pos = self.pos_embed[:, : n_tokens + 1, :]
        tokens = tokens + pos

        hidden_states: List[mx.array] = []
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            if i in extract_layers:
                hidden_states.append(tokens)

        final = self.norm(tokens)
        return final, hidden_states


def vjepa2_vitg_config() -> ViTConfig:
    """
    Return the actual V-JEPA2 ViT-g/22 configuration.

    Verified from facebook/vjepa2-vitg-fpc64-256 config.json and safetensors:
      hidden_size=1408, num_hidden_layers=40, num_attention_heads=22,
      mlp_dim=6144 (ratio ~4.364), patch_size=16, tubelet_size=2, image_size=256
    """
    return ViTConfig(
        hidden_dim=1408,
        depth=40,
        num_heads=22,          # actual value from config (not 16 as initially assumed)
        mlp_ratio=4,           # ignored because _mlp_dim_override is set
        patch_size=16,
        temporal_patch_size=2,
        image_size=256,
        extract_layers=[30, 39],  # depth 0.75 and 1.0
        _mlp_dim_override=6144,   # verified from fc1.weight shape (6144, 1408)
    )
