"""
MLX implementation of Wav2Vec-BERT 2.0 audio encoder.

Architecture:
  - Convolutional feature extractor: 7 strided Conv1d layers → 512-dim features at 50 Hz
  - Feature projection: Linear(512 → 1024) + LayerNorm
  - Conformer encoder: 24 ConformerBlocks (dim=1024, heads=16, conv_kernel=31)

TRIBE extracts features at layers 18 and 24 (depth 0.75 and 1.0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class W2VBertConfig:
    hidden_dim: int = 1024
    conv_out_dim: int = 160  # facebook/w2v-bert-2.0 uses 160-dim mel features
    depth: int = 24
    num_heads: int = 16
    conv_kernel_size: int = 31
    ff_expansion: int = 4
    pos_conv_kernel: int = 128
    extract_layers: List[int] = field(default_factory=lambda: [18, 23])  # 0.75 and 1.0
    qkv_bias: bool = True


class ConvFeatureExtractor(nn.Module):
    """
    Waveform → frame features via strided Conv1d stack.
    Mirrors the HF Wav2Vec2BertFeatureExtractor: 7 layers with strides [5,2,2,2,2,2,2].
    Kept in fp16 (not quantized) because the channels are too small for int8.
    """

    CHANNEL_SIZES = [512, 512, 512, 512, 512, 512, 512]
    STRIDES = [5, 2, 2, 2, 2, 2, 2]
    KERNEL_SIZES = [10, 3, 3, 3, 3, 2, 2]

    def __init__(self):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch, k, s in zip(self.CHANNEL_SIZES, self.KERNEL_SIZES, self.STRIDES):
            layers.append(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s)
            )
            in_ch = out_ch
        self.conv_layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        """
        x: (B, T_wave) raw waveform
        returns: (B, T_feat, 512)
        """
        x = x[:, :, None]  # (B, T, 1) — MLX Conv1d expects (B, T, C)
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.gelu(x)
        return x  # (B, T_feat, 512)


class FeatureProjection(nn.Module):
    def __init__(self, config: W2VBertConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_out_dim)
        self.projection = nn.Linear(config.conv_out_dim, config.hidden_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.projection(self.layer_norm(x))


class ConformerAttention(nn.Module):
    """Standard multi-head attention for conformer blocks."""

    def __init__(self, config: W2VBertConfig):
        super().__init__()
        dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.query_proj = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.key_proj = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.value_proj = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.query_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)
        k = self.key_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)
        v = self.value_proj(x).reshape(B, N, H, D).transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module:
      LayerNorm → pointwise_conv1 (GLU gate) → depthwise_conv → BatchNorm → SiLU → pointwise_conv2
    """

    def __init__(self, config: W2VBertConfig):
        super().__init__()
        dim = config.hidden_dim
        k = config.conv_kernel_size

        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Linear(dim, 2 * dim, bias=True)  # GLU doubles dim
        # Depthwise Conv1d: groups=dim kept in fp16
        self.depthwise_conv = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=k,
            padding=k // 2, groups=dim
        )
        self.batch_norm = nn.LayerNorm(dim)  # use LayerNorm as BN substitute in MLX
        self.pointwise_conv2 = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, dim)"""
        residual = x
        x = self.layer_norm(x)

        # GLU gate
        x = self.pointwise_conv1(x)  # (B, T, 2*dim)
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        x = x1 * mx.sigmoid(x2)  # GLU

        # Depthwise conv: (B, T, dim) expected by Conv1d
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = nn.silu(x)

        x = self.pointwise_conv2(x)
        return x


class ConformerFeedForward(nn.Module):
    def __init__(self, config: W2VBertConfig):
        super().__init__()
        dim = config.hidden_dim
        ff_dim = dim * config.ff_expansion

        self.layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ff_dim, bias=True)
        self.fc2 = nn.Linear(ff_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.silu(self.fc1(self.layer_norm(x))))


class ConformerBlock(nn.Module):
    """
    Conformer block structure:
      0.5 × FF → Attention → Conv module → 0.5 × FF → LayerNorm
    """

    def __init__(self, config: W2VBertConfig):
        super().__init__()
        self.ff1 = ConformerFeedForward(config)
        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        self.attn = ConformerAttention(config)
        self.conv_module = ConformerConvModule(config)
        self.ff2 = ConformerFeedForward(config)
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(self.attn_norm(x))
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class MLXWav2VecBert(nn.Module):
    """
    MLX Wav2Vec-BERT 2.0 encoder.

    Processes raw waveforms (B, T_wave) at 16 kHz.
    Returns hidden states at `config.extract_layers`.

    Weight layout for conversion from HuggingFace Wav2Vec2BertModel:
      feature_extractor.conv_layers.N.conv.{weight,bias}
      feature_projection.{layer_norm,projection}.{weight,bias}
      encoder.pos_conv_embed.conv.{weight,bias}
      encoder.layers.N.{self_attn_layer_norm,self_attn}.{...}
      encoder.layers.N.{conv_module}.{...}
      encoder.layers.N.{feed_forward,final_layer_norm}.{...}
      encoder.layer_norm.{weight,bias}
    """

    def __init__(self, config: Optional[W2VBertConfig] = None):
        super().__init__()
        if config is None:
            config = W2VBertConfig()
        self.config = config

        self.feature_extractor = ConvFeatureExtractor()
        self.feature_projection = FeatureProjection(config)

        # Positional conv embed (from HF W2V-BERT)
        self.pos_conv_embed = nn.Conv1d(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            kernel_size=config.pos_conv_kernel,
            padding=config.pos_conv_kernel // 2,
            groups=config.hidden_dim,
        )
        self.pos_norm = nn.LayerNorm(config.hidden_dim)

        self.encoder_layers = [ConformerBlock(config) for _ in range(config.depth)]
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def __call__(
        self,
        x: mx.array,
        extract_layers: Optional[List[int]] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        """
        Parameters
        ----------
        x : mx.array
            Either (B, T_wave) raw waveform or (B, T, conv_out_dim) mel features.
            When 2D, the ConvFeatureExtractor is applied first.
            When 3D, mel features are passed directly to FeatureProjection.
        extract_layers : layer indices to capture

        Returns
        -------
        (final_hidden, [hidden at each extract_layer])
        Each: (B, T_feat, hidden_dim)
        """
        if extract_layers is None:
            extract_layers = self.config.extract_layers

        if x.ndim == 2:
            # Raw waveform path: ConvFeatureExtractor → FeatureProjection
            x = self.feature_extractor(x)          # (B, T_feat, conv_out_dim)
        # else: x is already (B, T, conv_out_dim) mel features
        x = self.feature_projection(x)             # (B, T_feat, hidden_dim)

        # Positional conv embedding (grouped conv, keep in fp16)
        pos = self.pos_conv_embed(x)
        pos = pos[:, : x.shape[1], :]              # trim any extra from padding
        x = self.pos_norm(x + pos)

        hidden_states: List[mx.array] = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i in extract_layers:
                hidden_states.append(x)

        final = self.layer_norm(x)
        return final, hidden_states
