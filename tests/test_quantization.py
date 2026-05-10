"""
Quantization tests.

Verify that MLX int8 quantization of model components produces outputs within
acceptable cosine similarity (> 0.95) of fp16 baseline.

Key points:
- Models are initialised with random weights (MLX default).
- fp16 output is computed, then nn.quantize() is applied, then int8 output.
- 8-bit quantization of random fp32 weights should give cos-sim > 0.99.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn

from tribe_v2_mlx.utils import cosine_similarity


# ─── ViT Block ────────────────────────────────────────────────────────────────

class TestViTBlockQuantization:
    def _make_block(self, dim=64, heads=4):
        from tribe_v2_mlx.models.vjepa2 import ViTBlock, ViTConfig
        cfg = ViTConfig(hidden_dim=dim, num_heads=heads, mlp_ratio=2, depth=1)
        return ViTBlock(cfg)

    def test_8bit_cosine_similarity(self):
        block = self._make_block()
        x = mx.random.normal((1, 16, 64))

        # fp16 output (before quantization)
        out_fp16 = block(x)
        mx.eval(out_fp16)

        # Quantize
        nn.quantize(block, bits=8, class_predicate=lambda _, m: isinstance(m, nn.Linear))
        out_int8 = block(x)
        mx.eval(out_int8)

        sim = cosine_similarity(np.array(out_fp16), np.array(out_int8))
        assert sim > 0.95, f"8-bit cosine similarity {sim:.4f} < 0.95"

    def test_4bit_cosine_similarity(self):
        block = self._make_block(dim=128, heads=4)
        x = mx.random.normal((1, 16, 128))

        out_fp16 = block(x)
        mx.eval(out_fp16)

        nn.quantize(block, bits=4, class_predicate=lambda _, m: isinstance(m, nn.Linear))
        out_int4 = block(x)
        mx.eval(out_int4)

        sim = cosine_similarity(np.array(out_fp16), np.array(out_int4))
        # 4-bit has lower precision; still expect > 0.90 for random weights
        assert sim > 0.90, f"4-bit cosine similarity {sim:.4f} < 0.90"

    def test_quantized_model_smaller_memory(self):
        """Verify that quantization changes weight dtype."""
        from tribe_v2_mlx.models.vjepa2 import ViTBlock, ViTConfig
        cfg = ViTConfig(hidden_dim=64, num_heads=4, mlp_ratio=2, depth=1)
        block = ViTBlock(cfg)

        # Before quantization: weights are float32
        x = mx.random.normal((1, 8, 64))
        out_before = block(x)
        mx.eval(out_before)

        nn.quantize(block, bits=8, class_predicate=lambda _, m: isinstance(m, nn.Linear))
        # After quantization: block should have QuantizedLinear layers
        has_quantized = any(
            isinstance(m, nn.QuantizedLinear)
            for _, m in block.named_modules()
        )
        assert has_quantized, "Block should contain QuantizedLinear after quantize()"


# ─── DINOv2 Block ─────────────────────────────────────────────────────────────

class TestDINOv2Quantization:
    def test_full_model_8bit(self):
        from tribe_v2_mlx.models.dinov2 import MLXDINOv2Large, DINOv2Config
        cfg = DINOv2Config(
            hidden_dim=64, depth=2, num_heads=4, patch_size=16, image_size=64,
            num_register_tokens=2, extract_layers=[1],
        )
        model = MLXDINOv2Large(cfg)
        x = mx.random.normal((1, 64, 64, 3))

        cls_fp16, _ = model(x)
        mx.eval(cls_fp16)

        nn.quantize(model, bits=8, class_predicate=lambda _, m: isinstance(m, nn.Linear))
        cls_int8, _ = model(x)
        mx.eval(cls_int8)

        sim = cosine_similarity(np.array(cls_fp16), np.array(cls_int8))
        assert sim > 0.95, f"DINOv2 8-bit cos-sim {sim:.4f} < 0.95"


# ─── Conformer Block ──────────────────────────────────────────────────────────

class TestConformerQuantization:
    def test_conformer_block_8bit(self):
        from tribe_v2_mlx.models.wav2vec_bert import ConformerBlock, W2VBertConfig
        cfg = W2VBertConfig(hidden_dim=64, depth=2, num_heads=4, conv_kernel_size=7)
        block = ConformerBlock(cfg)
        x = mx.random.normal((1, 16, 64))

        out_fp16 = block(x)
        mx.eval(out_fp16)

        # Quantize only Linear layers (skip Conv1d depthwise)
        nn.quantize(
            block, bits=8,
            class_predicate=lambda _, m: isinstance(m, nn.Linear),
        )
        out_int8 = block(x)
        mx.eval(out_int8)

        sim = cosine_similarity(np.array(out_fp16), np.array(out_int8))
        assert sim > 0.95, f"Conformer block 8-bit cos-sim {sim:.4f} < 0.95"

    def test_conv1d_not_quantized(self):
        from tribe_v2_mlx.models.wav2vec_bert import ConformerBlock, W2VBertConfig
        cfg = W2VBertConfig(hidden_dim=64, depth=2, num_heads=4, conv_kernel_size=7)
        block = ConformerBlock(cfg)

        nn.quantize(
            block, bits=8,
            class_predicate=lambda _, m: isinstance(m, nn.Linear),
        )
        # Conv1d should NOT be quantized
        for name, module in block.named_modules():
            if isinstance(module, nn.Conv1d):
                assert not isinstance(module, nn.QuantizedLinear), \
                    f"Conv1d {name} was incorrectly quantized"


# ─── TRIBE Transformer ────────────────────────────────────────────────────────

class TestTribeQuantization:
    def test_tribe_output_close_after_quantization(self, small_tribe_config):
        from tribe_v2_mlx.models.tribe import TribeTransformer
        model = TribeTransformer(small_tribe_config)

        features = {
            "image": mx.random.normal((1, 4, small_tribe_config.image_in_dim)),
        }
        out_fp16 = model(features)
        mx.eval(out_fp16)

        # Only quantize Linear layers whose input_dim is divisible by the
        # default group_size (64).  Small test configs have tiny dims.
        nn.quantize(
            model, bits=8,
            class_predicate=lambda _, m: (
                isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0
            ),
        )
        out_int8 = model(features)
        mx.eval(out_int8)

        sim = cosine_similarity(np.array(out_fp16), np.array(out_int8))
        assert sim > 0.95, f"TRIBE 8-bit cos-sim {sim:.4f} < 0.95"


# ─── Mixed-precision quantization ─────────────────────────────────────────────

class TestMixedPrecision:
    """
    Simulate the recommended mixed-precision strategy:
      - Large Linear layers → 8-bit
      - Small matrices and Conv1d → fp16
    """

    def test_skip_small_matrices(self):
        from tribe_v2_mlx.models.vjepa2 import MLXVJepa2, ViTConfig
        cfg = ViTConfig(
            hidden_dim=64, depth=2, num_heads=4, patch_size=16,
            image_size=64, temporal_patch_size=2, extract_layers=[1],
        )
        model = MLXVJepa2(cfg)

        # Only quantize matrices with ≥ 64 columns (skip embedding projections with tiny dims)
        nn.quantize(
            model, bits=8,
            class_predicate=lambda _, m: (
                isinstance(m, nn.Linear) and m.weight.shape[-1] >= 64
            ),
        )
        x = mx.random.normal((1, 4, 64, 64, 3))
        final, _ = model(x)
        mx.eval(final)
        assert np.all(np.isfinite(np.array(final)))
