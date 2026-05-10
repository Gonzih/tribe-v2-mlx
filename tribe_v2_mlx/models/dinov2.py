"""
MLX implementation of DINOv2-Large for image feature extraction.

Architecture: ViT-Large (1024 hidden, 24 layers, 16 heads, patch_size=14).
DINOv2 adds register tokens after the CLS token (4 register tokens by default).
TRIBE extracts features at depth 2/3 = layer index 16.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from tribe_v2_mlx.models.vjepa2 import ViTAttention, ViTMLP, ViTBlock, ViTConfig


@dataclass
class DINOv2Config:
    hidden_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: int = 4
    patch_size: int = 14
    image_size: int = 224
    num_register_tokens: int = 4
    qkv_bias: bool = True
    extract_layers: List[int] = field(default_factory=lambda: [16])  # depth 2/3

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def mlp_dim(self) -> int:
        return self.hidden_dim * self.mlp_ratio

    def to_vit_config(self) -> ViTConfig:
        return ViTConfig(
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            patch_size=self.patch_size,
            image_size=self.image_size,
            qkv_bias=self.qkv_bias,
            extract_layers=self.extract_layers,
        )


class DINOv2PatchEmbedding(nn.Module):
    """Patch embedding for DINOv2 (patch_size=14, no temporal dim)."""

    def __init__(self, config: DINOv2Config):
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


class MLXDINOv2Large(nn.Module):
    """
    MLX DINOv2-Large encoder.

    Processes (B, H, W, C) images (float32, ImageNet-normalised).
    Returns hidden states at `config.extract_layers`.

    Weight layout for conversion from HuggingFace Dinov2Model:
      embeddings.patch_embeddings.projection.{weight,bias}
      embeddings.cls_token
      embeddings.register_tokens
      embeddings.position_embeddings
      encoder.layer.N.norm1.{weight,bias}
      encoder.layer.N.attention.attention.{query,key,value}.{weight,bias}
      encoder.layer.N.attention.output.dense.{weight,bias}
      encoder.layer.N.norm2.{weight,bias}
      encoder.layer.N.intermediate.dense.{weight,bias}
      encoder.layer.N.output.dense.{weight,bias}
      layernorm.{weight,bias}
    """

    def __init__(self, config: Optional[DINOv2Config] = None):
        super().__init__()
        if config is None:
            config = DINOv2Config()
        self.config = config
        vit_cfg = config.to_vit_config()

        self.patch_embed = DINOv2PatchEmbedding(config)
        self.blocks = [ViTBlock(vit_cfg) for _ in range(config.depth)]
        self.norm = nn.LayerNorm(config.hidden_dim)

        # Special tokens
        self.cls_token = mx.zeros((1, 1, config.hidden_dim))
        self.register_tokens = mx.zeros((1, config.num_register_tokens, config.hidden_dim))

        # Positional embedding for patches + CLS + register tokens
        n_patches = (config.image_size // config.patch_size) ** 2
        n_special = 1 + config.num_register_tokens
        self.pos_embed = mx.zeros((1, n_patches + n_special, config.hidden_dim))

    def __call__(
        self,
        x: mx.array,
        extract_layers: Optional[List[int]] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        """
        Parameters
        ----------
        x : mx.array (B, H, W, C) float32
        extract_layers : which layer indices to capture

        Returns
        -------
        (final_cls_feature, [hidden at each extract_layer])
        final_cls_feature: (B, hidden_dim) — the CLS token
        hidden list: each (B, n_tokens, hidden_dim)
        """
        if extract_layers is None:
            extract_layers = self.config.extract_layers

        B = x.shape[0]
        patches = self.patch_embed(x)              # (B, n_patches, D)

        cls = mx.broadcast_to(self.cls_token, (B, 1, self.config.hidden_dim))
        if self.config.num_register_tokens > 0:
            regs = mx.broadcast_to(
                self.register_tokens, (B, self.config.num_register_tokens, self.config.hidden_dim)
            )
            tokens = mx.concatenate([cls, regs, patches], axis=1)  # (B, 1+R+N, D)
        else:
            tokens = mx.concatenate([cls, patches], axis=1)        # (B, 1+N, D)

        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]

        hidden_states: List[mx.array] = []
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            if i in extract_layers:
                hidden_states.append(tokens)

        final = self.norm(tokens)
        cls_out = final[:, 0, :]  # (B, D) — CLS token
        return cls_out, hidden_states
