"""
MLX implementation of the TRIBE v2 FmriEncoderModel transformer.

Two implementations are provided:

1. TribeTransformer — synthetic/test model that matches the original
   MLX design (LayerNorm, combined attn+FF blocks, SubjectLayers).
   Used by all unit tests with small random-weight models.

2. TribeRealModel — faithful MLX implementation of the real best.ckpt.
   Created by TribeTransformer.from_checkpoint() when a real checkpoint
   is loaded.  Architecture reverse-engineered from checkpoint inspection:
     - ScalarRMSNorm (single g scalar, no bias)
     - 16 alternating layers: attn(RoPE) / FF(GELU), each with scaled residual
     - Modality projectors: Linear(feature_dim, 384) for text/audio/video
     - low_rank_head + predictor (batch-matmul, 1 subject)
     - time_pos_embed (1, 1024, 1152) — positional lookup table
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic / test model  (kept for backward compatibility with tests)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TribeConfig:
    # Modality input dimensions after layer-concat
    text_in_dim: int = 18432
    video_in_dim: int = 2816
    image_in_dim: int = 1024
    audio_in_dim: int = 2048

    mod_dim: int = 288      # 1152 / 4
    hidden: int = 1152
    num_heads: int = 16
    depth: int = 8
    mlp_ratio: int = 4

    low_rank_dim: int = 2048
    n_outputs: int = 20484
    n_subjects: int = 8
    max_seq_len: int = 1024


class ModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(self.norm(x))


class SubjectLayers(nn.Module):
    def __init__(self, in_dim: int, n_outputs: int, n_subjects: int):
        super().__init__()
        self.n_subjects = n_subjects
        self.in_dim = in_dim
        self.n_outputs = n_outputs
        self.layers = [nn.Linear(in_dim, n_outputs, bias=False) for _ in range(n_subjects)]

    def __call__(self, x: mx.array, subject_id: int = 0) -> mx.array:
        return self.layers[subject_id % self.n_subjects](x)


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

        nx = self.norm1(x)
        q = self.q_proj(nx).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(nx).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(nx).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        sa = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        x = x + self.out_proj(sa)
        x = x + self.ff2(nn.gelu(self.ff1(self.norm2(x))))
        return x


class TribeTransformer(nn.Module):
    """
    Synthetic TRIBE v2 transformer for testing (random-weight compatible).

    Call TribeTransformer.from_checkpoint(path) to load a real best.ckpt —
    that returns a TribeRealModel with the actual checkpoint architecture.
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

        x = mx.concatenate(parts, axis=-1)

        n_mods = len(parts)
        if n_mods < 4:
            pad_dim = self.config.hidden - x.shape[-1]
            if pad_dim > 0:
                x = mx.concatenate(
                    [x, mx.zeros((*x.shape[:2], pad_dim), dtype=x.dtype)], axis=-1
                )

        T = x.shape[1]
        positions = mx.arange(T)
        x = x + self.time_pos_embed(positions)

        for block in self.transformer:
            x = block(x)
        x = self.norm(x)
        x = self.low_rank_head(x)
        x = self.subject_layers(x, subject_id)
        return x

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        subject_id: int = 0,
    ) -> "TribeRealModel":
        """
        Load from a TRIBE v2 PyTorch best.ckpt.

        Returns a TribeRealModel (faithful to the actual checkpoint architecture),
        not a TribeTransformer.  The returned model is callable with the same
        interface: model(features_dict, subject_id=0) → (B, T, 20484).
        """
        return TribeRealModel.from_checkpoint(ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# Real checkpoint model  (faithful to best.ckpt architecture)
# ─────────────────────────────────────────────────────────────────────────────

def _build_rotary_cache(
    inv_freq: mx.array, seq_len: int
) -> tuple[mx.array, mx.array]:
    """Build RoPE cos/sin tables for sequence length seq_len."""
    t = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(t, inv_freq)              # (T, head_dim/2)
    emb = mx.concatenate([freqs, freqs], axis=-1)  # (T, head_dim)
    return mx.cos(emb), mx.sin(emb)


def _apply_rotary(
    x: mx.array, cos: mx.array, sin: mx.array
) -> mx.array:
    """
    Apply RoPE to x of shape (B, T, H, D).
    cos/sin: (T, D).
    """
    D = x.shape[-1]
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    x_rot = mx.concatenate([-x2, x1], axis=-1)
    cos_ = cos[None, :, None, :]  # (1, T, 1, D)
    sin_ = sin[None, :, None, :]
    return x * cos_ + x_rot * sin_


class _ScalarRMSNorm(nn.Module):
    """RMSNorm with a single scalar gain (as in the TRIBE checkpoint)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.g = mx.ones((1,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * rms * self.g


class _TribeRealAttn(nn.Module):
    """Self-attention with RoPE (no bias on projections)."""

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(hidden, hidden, bias=False)
        self.to_k = nn.Linear(hidden, hidden, bias=False)
        self.to_v = nn.Linear(hidden, hidden, bias=False)
        self.to_out = nn.Linear(hidden, hidden, bias=False)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.to_q(x).reshape(B, T, H, D)
        k = self.to_k(x).reshape(B, T, H, D)
        v = self.to_v(x).reshape(B, T, H, D)

        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        q = q.transpose(0, 2, 1, 3)  # (B, H, T, D)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.to_out(out)


class _TribeRealAttnBlock(nn.Module):
    """Even-indexed transformer layer: ScalarRMSNorm → Attention → scaled residual."""

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.norm = _ScalarRMSNorm()
        self.attn = _TribeRealAttn(hidden, n_heads)
        self.residual_scale = mx.ones((hidden,))

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        return x * self.residual_scale + self.attn(self.norm(x), cos, sin)


class _TribeRealFFBlock(nn.Module):
    """Odd-indexed transformer layer: ScalarRMSNorm → FF(GELU) → scaled residual."""

    def __init__(self, hidden: int, ff_dim: int):
        super().__init__()
        self.norm = _ScalarRMSNorm()
        self.fc1 = nn.Linear(hidden, ff_dim, bias=True)
        self.fc2 = nn.Linear(ff_dim, hidden, bias=True)
        self.residual_scale = mx.ones((hidden,))

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        h = self.fc2(nn.gelu(self.fc1(self.norm(x))))
        return x * self.residual_scale + h


class TribeRealModel(nn.Module):
    """
    Faithful MLX implementation of the real TRIBE v2 best.ckpt.

    Architecture (reverse-engineered from checkpoint key inspection):
      - Linear projectors: feature_dim → 384 per modality (text/audio/video)
      - Concat (B, T, 1152) for 3 modalities
      - + time_pos_embed[:, :T, :] from (1, 1024, 1152) table
      - 16 alternating transformer layers with scaled residuals:
          even: ScalarRMSNorm → Attention(RoPE) → residual * residual_scale
          odd:  ScalarRMSNorm → FF(GELU)         → residual * residual_scale
      - final ScalarRMSNorm
      - low_rank_head: Linear(1152, 2048, bias=False)
      - predictor: (1, 2048, 20484) @ (B, T, 2048).T + bias(1, 20484)
    """

    # Hardcoded from checkpoint inspection
    HIDDEN = 1152
    N_HEADS = 16
    MOD_DIM = 384
    FF_DIM = 4608
    N_ATTN = 8       # 8 attention layers (even indices 0,2,...14)
    N_FF = 8         # 8 FF layers (odd indices 1,3,...15)
    LOW_RANK = 2048
    N_OUTPUTS = 20484
    MAX_SEQ = 1024

    def __init__(
        self,
        text_in_dim: int = 6144,
        audio_in_dim: int = 2048,
        video_in_dim: int = 2816,
        n_outputs: int = 20484,
        n_subjects: int = 1,
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.n_subjects = n_subjects

        # Config for pipeline introspection
        self.config = _TribeRealConfig(
            text_in_dim=text_in_dim,
            audio_in_dim=audio_in_dim,
            video_in_dim=video_in_dim,
            n_outputs=n_outputs,
        )

        # Modality projectors (just Linear, no LayerNorm — matches checkpoint)
        self.text_proj = nn.Linear(text_in_dim, self.MOD_DIM, bias=True)
        self.audio_proj = nn.Linear(audio_in_dim, self.MOD_DIM, bias=True)
        self.video_proj = nn.Linear(video_in_dim, self.MOD_DIM, bias=True)

        # Time positional embedding
        self.time_pos_embed = mx.zeros((1, self.MAX_SEQ, self.HIDDEN))

        # RoPE inverse frequencies (head_dim/2 = 36)
        self.rotary_inv_freq = mx.zeros((self.HIDDEN // self.N_HEADS // 2,))

        # Transformer layers (alternating attn / FF)
        self.attn_layers = [
            _TribeRealAttnBlock(self.HIDDEN, self.N_HEADS)
            for _ in range(self.N_ATTN)
        ]
        self.ff_layers = [
            _TribeRealFFBlock(self.HIDDEN, self.FF_DIM)
            for _ in range(self.N_FF)
        ]

        # Final norm
        self.final_norm = _ScalarRMSNorm()

        # Head: Linear(1152, 2048, no bias)
        self.low_rank_head = nn.Linear(self.HIDDEN, self.LOW_RANK, bias=False)

        # Predictor: (n_subjects, 2048, 20484)
        self.predictor_weights = mx.zeros((n_subjects, self.LOW_RANK, n_outputs))
        self.predictor_bias = mx.zeros((n_subjects, n_outputs))

    def __call__(
        self, features: Dict[str, mx.array], subject_id: int = 0
    ) -> mx.array:
        """
        features: dict with keys 'text', 'audio', 'video' (any subset).
          Each value: (B, T, feature_dim).
        Returns: (B, T, n_outputs).
        """
        parts: List[mx.array] = []
        if "text" in features:
            parts.append(self.text_proj(features["text"]))
        if "audio" in features:
            parts.append(self.audio_proj(features["audio"]))
        if "video" in features:
            parts.append(self.video_proj(features["video"]))
        # Also accept "image" mapped to video projector for compatibility
        if "image" in features and "video" not in features:
            parts.append(self.video_proj(features["image"]))

        if not parts:
            raise ValueError("At least one modality (text/audio/video/image) must be provided.")

        # Pad missing modalities with zeros to reach HIDDEN dim
        x = mx.concatenate(parts, axis=-1)  # (B, T, n_mod*MOD_DIM)
        pad_dim = self.HIDDEN - x.shape[-1]
        if pad_dim > 0:
            x = mx.concatenate(
                [x, mx.zeros((*x.shape[:2], pad_dim), dtype=x.dtype)], axis=-1
            )

        # Time positional embedding
        T = x.shape[1]
        x = x + self.time_pos_embed[:, :T, :]

        # RoPE cache
        cos, sin = _build_rotary_cache(self.rotary_inv_freq, T)

        # 16 alternating transformer layers
        attn_i, ff_i = 0, 0
        for layer_idx in range(self.N_ATTN + self.N_FF):
            if layer_idx % 2 == 0:
                x = self.attn_layers[attn_i](x, cos, sin)
                attn_i += 1
            else:
                x = self.ff_layers[ff_i](x, cos, sin)
                ff_i += 1

        x = self.final_norm(x)                          # (B, T, 1152)
        x = self.low_rank_head(x)                       # (B, T, 2048)

        # Subject-specific predictor: (n_subjects, 2048, 20484)
        s = subject_id % self.n_subjects
        w = self.predictor_weights[s]  # (2048, 20484)
        b = self.predictor_bias[s]     # (20484,)
        x = x @ w + b                 # (B, T, 20484)
        return x

    @classmethod
    def from_checkpoint(cls, ckpt_path: str) -> "TribeRealModel":
        """Load from TRIBE v2 best.ckpt PyTorch file."""
        import torch

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

        build_args = ckpt.get("model_build_args", {})
        n_outputs = build_args.get("n_outputs", 20484)
        feature_dims_raw = build_args.get("feature_dims", {})

        # feature_dims values can be tuples (n_layers, layer_dim) or plain ints
        def _flatten_dim(v):
            if isinstance(v, (tuple, list)) and len(v) == 2:
                return v[0] * v[1]
            return int(v)

        text_in = _flatten_dim(feature_dims_raw.get("text", (2, 3072)))
        audio_in = _flatten_dim(feature_dims_raw.get("audio", (2, 1024)))
        video_in = _flatten_dim(feature_dims_raw.get("video", (2, 1408)))

        # Inspect predictor to get n_subjects
        state = ckpt.get("state_dict", ckpt)
        pred_key = next((k for k in state if "predictor.weights" in k), None)
        if pred_key is not None:
            n_subjects = state[pred_key].shape[0]
        else:
            n_subjects = 1

        model = cls(
            text_in_dim=text_in,
            audio_in_dim=audio_in,
            video_in_dim=video_in,
            n_outputs=n_outputs,
            n_subjects=n_subjects,
        )

        # Build MLX weight dict from checkpoint
        mlx_weights: dict[str, mx.array] = {}

        def _t(tensor) -> mx.array:
            return mx.array(tensor.float().numpy())

        for ckpt_key, tensor in state.items():
            # Strip "model." prefix
            k = ckpt_key.removeprefix("model.")

            # Modality projectors
            if k == "projectors.text.weight":
                mlx_weights["text_proj.weight"] = _t(tensor)
            elif k == "projectors.text.bias":
                mlx_weights["text_proj.bias"] = _t(tensor)
            elif k == "projectors.audio.weight":
                mlx_weights["audio_proj.weight"] = _t(tensor)
            elif k == "projectors.audio.bias":
                mlx_weights["audio_proj.bias"] = _t(tensor)
            elif k == "projectors.video.weight":
                mlx_weights["video_proj.weight"] = _t(tensor)
            elif k == "projectors.video.bias":
                mlx_weights["video_proj.bias"] = _t(tensor)

            # Time positional embedding (1, 1024, 1152)
            elif k == "time_pos_embed":
                mlx_weights["time_pos_embed"] = _t(tensor)

            # RoPE
            elif k == "encoder.rotary_pos_emb.inv_freq":
                mlx_weights["rotary_inv_freq"] = _t(tensor)

            # Final norm
            elif k == "encoder.final_norm.g":
                mlx_weights["final_norm.g"] = _t(tensor)

            # Low rank head
            elif k == "low_rank_head.weight":
                mlx_weights["low_rank_head.weight"] = _t(tensor)

            # Predictor
            elif k == "predictor.weights":
                mlx_weights["predictor_weights"] = _t(tensor)
            elif k == "predictor.bias":
                mlx_weights["predictor_bias"] = _t(tensor)

            # Transformer layers (alternating even=attn, odd=ff)
            else:
                import re
                m = re.match(r"encoder\.layers\.(\d+)\.(.*)", k)
                if not m:
                    continue
                layer_idx = int(m.group(1))
                rest = m.group(2)
                sub = layer_idx // 2  # index into attn_layers or ff_layers

                if layer_idx % 2 == 0:
                    # Attention layer
                    pfx = f"attn_layers.{sub}"
                    mapping = {
                        "0.0.g": f"{pfx}.norm.g",
                        "1.to_q.weight": f"{pfx}.attn.to_q.weight",
                        "1.to_k.weight": f"{pfx}.attn.to_k.weight",
                        "1.to_v.weight": f"{pfx}.attn.to_v.weight",
                        "1.to_out.weight": f"{pfx}.attn.to_out.weight",
                        "2.residual_scale": f"{pfx}.residual_scale",
                    }
                else:
                    # FF layer
                    pfx = f"ff_layers.{sub}"
                    mapping = {
                        "0.0.g": f"{pfx}.norm.g",
                        "1.ff.0.0.weight": f"{pfx}.fc1.weight",
                        "1.ff.0.0.bias": f"{pfx}.fc1.bias",
                        "1.ff.2.weight": f"{pfx}.fc2.weight",
                        "1.ff.2.bias": f"{pfx}.fc2.bias",
                        "2.residual_scale": f"{pfx}.residual_scale",
                    }

                if rest in mapping:
                    mlx_weights[mapping[rest]] = _t(tensor)

        model.load_weights(list(mlx_weights.items()), strict=False)
        mx.eval(model.parameters())
        print(f"Loaded {len(mlx_weights)} weights into TribeRealModel.")
        return model


@dataclass
class _TribeRealConfig:
    """Minimal config for TribeRealModel — exposes dims needed by pipeline."""
    text_in_dim: int = 6144
    audio_in_dim: int = 2048
    video_in_dim: int = 2816
    n_outputs: int = 20484

    # Pipeline compatibility: image maps to video projector
    @property
    def image_in_dim(self) -> int:
        return self.video_in_dim

    # extract_layers (pipeline uses these to know how many hidden states to concat)
    # Real TRIBE uses 2 layers × layer_dim for each modality
    @property
    def extract_layers(self) -> list[int]:
        return [0, 1]
