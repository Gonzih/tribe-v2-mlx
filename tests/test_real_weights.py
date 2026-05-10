"""
Integration tests that require real downloaded weights.

Run with:
    pytest tests/test_real_weights.py -v -m requires_weights

These tests are skipped automatically when weights are not present.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# Default weights directory (relative to repo root)
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
MLX_DIR = WEIGHTS_DIR / "mlx"
TRIBE_CKPT = WEIGHTS_DIR / "tribe" / "best.ckpt"

pytestmark = pytest.mark.requires_weights


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mlx_dir():
    if not MLX_DIR.exists():
        pytest.skip(f"MLX weights not found at {MLX_DIR} — run scripts/convert_to_mlx.py first")
    return MLX_DIR


@pytest.fixture(scope="module")
def tribe_ckpt_path():
    if not TRIBE_CKPT.exists():
        pytest.skip(f"TRIBE checkpoint not found at {TRIBE_CKPT} — run scripts/download_weights.py first")
    return TRIBE_CKPT


@pytest.fixture(scope="module")
def dinov2_mlx_path(mlx_dir):
    p = mlx_dir / "dinov2-large-8bit.safetensors"
    if not p.exists():
        pytest.skip(f"DINOv2 MLX weights not found at {p}")
    return p


@pytest.fixture(scope="module")
def wav2vec_bert_mlx_path(mlx_dir):
    p = mlx_dir / "wav2vec-bert-8bit.safetensors"
    if not p.exists():
        pytest.skip(f"Wav2Vec-BERT MLX weights not found at {p}")
    return p


@pytest.fixture(scope="module")
def vjepa2_mlx_path(mlx_dir):
    p = mlx_dir / "vjepa2-vitg-8bit.safetensors"
    if not p.exists():
        pytest.skip(f"V-JEPA2 MLX weights not found at {p}")
    return p


# ── TRIBE checkpoint inspection ────────────────────────────────────────────────

class TestTribeCheckpoint:
    def test_checkpoint_loads(self, tribe_ckpt_path):
        """Verify TRIBE best.ckpt is a valid PyTorch checkpoint."""
        import torch
        ckpt = torch.load(str(tribe_ckpt_path), map_location="cpu", weights_only=True)
        assert isinstance(ckpt, dict), "Checkpoint must be a dict"

    def test_checkpoint_has_expected_keys(self, tribe_ckpt_path):
        import torch
        ckpt = torch.load(str(tribe_ckpt_path), map_location="cpu", weights_only=True)
        # Expected top-level keys
        ckpt_keys = set(ckpt.keys())
        # At minimum, one of these groupings must be present
        has_state = "state_dict" in ckpt_keys or any(
            k.startswith("model.") for k in ckpt_keys
        )
        assert has_state, f"No state_dict or model.* keys found. Got: {sorted(ckpt_keys)}"

    def test_checkpoint_model_build_args(self, tribe_ckpt_path):
        import torch
        ckpt = torch.load(str(tribe_ckpt_path), map_location="cpu", weights_only=True)
        if "model_build_args" in ckpt:
            args = ckpt["model_build_args"]
            print(f"\nmodel_build_args: {args}")
            if "n_outputs" in args:
                assert args["n_outputs"] > 0, "n_outputs must be positive"

    def test_tribe_loads_into_mlx(self, tribe_ckpt_path):
        """Load checkpoint into MLX TribeTransformer."""
        from tribe_v2_mlx.models.tribe import TribeTransformer
        model = TribeTransformer.from_checkpoint(str(tribe_ckpt_path))
        assert model is not None
        assert model.config.n_outputs > 0

    def test_tribe_output_shape(self, tribe_ckpt_path):
        """Verify TRIBE produces correct output shape with real weights."""
        import mlx.core as mx
        from tribe_v2_mlx.models.tribe import TribeTransformer

        model = TribeTransformer.from_checkpoint(str(tribe_ckpt_path))
        n_outputs = model.config.n_outputs
        image_in_dim = model.config.image_in_dim

        # Feed a single segment of image features
        features = {"image": mx.random.normal((1, 1, image_in_dim))}
        out = model(features, subject_id=0)
        mx.eval(out)

        assert out.shape == (1, 1, n_outputs), f"Unexpected shape: {out.shape}"
        assert np.all(np.isfinite(np.array(out))), "Output contains non-finite values"

    def test_tribe_n_outputs_is_20484(self, tribe_ckpt_path):
        """TRIBE should predict on fsaverage5 (20484 vertices)."""
        from tribe_v2_mlx.models.tribe import TribeTransformer
        model = TribeTransformer.from_checkpoint(str(tribe_ckpt_path))
        assert model.config.n_outputs == 20484, \
            f"Expected 20484 outputs (fsaverage5), got {model.config.n_outputs}"


# Shared quantize predicate (must match the conversion scripts)
_QUANTIZE_PREDICATE = lambda _, m: (
    isinstance(m, __import__("mlx.nn", fromlist=["Linear"]).Linear)
    and m.weight.shape[-1] % 64 == 0
    and m.weight.shape[-1] >= 64
)


def _quantize_and_load(model, path, bits=8):
    """Quantize model structure then load quantized weights."""
    import mlx.core as mx
    import mlx.nn as nn

    nn.quantize(
        model,
        bits=bits,
        class_predicate=lambda _, m: (
            isinstance(m, nn.Linear)
            and m.weight.shape[-1] % 64 == 0
            and m.weight.shape[-1] >= 64
        ),
    )
    weights = mx.load(str(path))
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


# ── DINOv2 MLX weights ─────────────────────────────────────────────────────────

class TestDINOv2RealWeights:
    def test_dinov2_mlx_loads(self, dinov2_mlx_path):
        """Load real DINOv2 MLX weights and run a forward pass."""
        import mlx.core as mx
        from tribe_v2_mlx.models.dinov2 import MLXDINOv2Large, DINOv2Config

        # Real facebook/dinov2-large: image_size=518, no register tokens
        cfg = DINOv2Config(image_size=518, num_register_tokens=0)
        model = MLXDINOv2Large(cfg)
        _quantize_and_load(model, dinov2_mlx_path)

        # Use 224x224 for speed (pos_embed is sliced to token count)
        x = mx.random.normal((1, 224, 224, 3))
        cls_out, _ = model(x)
        mx.eval(cls_out)

        assert cls_out.shape == (1, cfg.hidden_dim), \
            f"DINOv2 cls output shape: {cls_out.shape}"
        assert np.all(np.isfinite(np.array(cls_out))), "DINOv2 output has non-finite values"

    def test_dinov2_feature_dim(self, dinov2_mlx_path):
        """DINOv2-Large should output 1024-dim features."""
        import mlx.core as mx
        from tribe_v2_mlx.models.dinov2 import MLXDINOv2Large, DINOv2Config

        cfg = DINOv2Config(image_size=518, num_register_tokens=0)
        assert cfg.hidden_dim == 1024, f"DINOv2-Large expected 1024 dim, got {cfg.hidden_dim}"

        model = MLXDINOv2Large(cfg)
        _quantize_and_load(model, dinov2_mlx_path)

        x = mx.random.normal((1, 224, 224, 3))
        cls_out, hidden = model(x, extract_layers=cfg.extract_layers)
        mx.eval(cls_out, *hidden)
        assert cls_out.shape[-1] == 1024


# ── Wav2Vec-BERT MLX weights ──────────────────────────────────────────────────

class TestWav2VecBertRealWeights:
    def test_wav2vec_bert_mlx_loads(self, wav2vec_bert_mlx_path):
        """Load real Wav2Vec-BERT MLX weights and run a forward pass."""
        import mlx.core as mx
        from tribe_v2_mlx.models.wav2vec_bert import MLXWav2VecBert, W2VBertConfig

        # facebook/w2v-bert-2.0 takes 160-dim mel features (no conv extractor in weights)
        cfg = W2VBertConfig()  # conv_out_dim=160 by default
        model = MLXWav2VecBert(cfg)
        _quantize_and_load(model, wav2vec_bert_mlx_path)

        # Pass mel features (B, T, 160) directly — skip ConvFeatureExtractor
        mel_features = mx.random.normal((1, 50, cfg.conv_out_dim))
        final, hidden = model(mel_features)
        mx.eval(final, *hidden)

        assert final.ndim == 3, f"Expected 3D output, got {final.ndim}D"
        assert final.shape[-1] == cfg.hidden_dim
        assert np.all(np.isfinite(np.array(final))), "Wav2Vec output has non-finite values"


# ── V-JEPA2 MLX weights ────────────────────────────────────────────────────────

class TestVJepa2RealWeights:
    def test_vjepa2_mlx_loads(self, vjepa2_mlx_path):
        """Load real V-JEPA2 MLX weights and run a forward pass."""
        import mlx.core as mx
        from tribe_v2_mlx.models.vjepa2 import MLXVJepa2, vjepa2_vitg_config

        model = MLXVJepa2(vjepa2_vitg_config())
        _quantize_and_load(model, vjepa2_mlx_path)

        # 1 clip, 4 frames, 256x256, 3 channels
        x = mx.random.normal((1, 4, 256, 256, 3))
        final, _ = model(x)
        mx.eval(final)

        assert final.ndim == 3, f"Expected 3D output, got {final.ndim}D"
        assert np.all(np.isfinite(np.array(final))), "V-JEPA2 output has non-finite values"


# ── Full pipeline output shape ─────────────────────────────────────────────────

class TestPipelineOutputShape:
    def test_tribe_output_is_20484_vertices(self, tribe_ckpt_path):
        """
        Core assertion: TRIBE produces (n_segments, 20484) predictions.
        This test runs with only TRIBE (DINOv2 with random features).
        """
        import mlx.core as mx
        from tribe_v2_mlx.models.tribe import TribeTransformer

        model = TribeTransformer.from_checkpoint(str(tribe_ckpt_path))
        n_segments = 3
        image_in_dim = model.config.image_in_dim

        features = {"image": mx.random.normal((1, n_segments, image_in_dim))}
        out = model(features, subject_id=0)
        mx.eval(out)

        preds = np.array(out[0])  # (n_segments, n_outputs)
        assert preds.shape == (n_segments, 20484), \
            f"Expected ({n_segments}, 20484), got {preds.shape}"
        assert np.all(np.isfinite(preds)), "TRIBE predictions contain non-finite values"
        print(f"\nTRIBE output: shape={preds.shape}, range=[{preds.min():.4f}, {preds.max():.4f}]")
