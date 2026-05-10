"""
Integration tests for the TribeV2MLXPipeline.

All encoder models are replaced with small random-weight instances so no
actual weights need to be downloaded.  A synthetic 5-second video is
created by the `synthetic_video_path` fixture in conftest.py.
"""
from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx


def _make_mock_pipeline(n_outputs: int = 128) -> "TribeV2MLXPipeline":
    """Build a pipeline with tiny random-weight models."""
    from tribe_v2_mlx.pipeline import TribeV2MLXPipeline
    from tribe_v2_mlx.models.vjepa2 import MLXVJepa2, ViTConfig
    from tribe_v2_mlx.models.dinov2 import MLXDINOv2Large, DINOv2Config
    from tribe_v2_mlx.models.wav2vec_bert import MLXWav2VecBert, W2VBertConfig
    from tribe_v2_mlx.models.tribe import TribeTransformer, TribeConfig

    vit_cfg = ViTConfig(
        hidden_dim=64, depth=2, num_heads=4, patch_size=16, image_size=64,
        temporal_patch_size=2, extract_layers=[1],
    )
    dino_cfg = DINOv2Config(
        hidden_dim=64, depth=2, num_heads=4, patch_size=16, image_size=64,
        num_register_tokens=2, extract_layers=[1],
    )
    w2v_cfg = W2VBertConfig(
        hidden_dim=64, conv_out_dim=512, depth=2, num_heads=4,
        conv_kernel_size=7, pos_conv_kernel=16, extract_layers=[1],
    )
    tribe_cfg = TribeConfig(
        text_in_dim=64, video_in_dim=64, image_in_dim=64, audio_in_dim=64,
        mod_dim=16, hidden=64, num_heads=4, depth=2, mlp_ratio=2,
        low_rank_dim=32, n_outputs=n_outputs, n_subjects=1,
    )

    return TribeV2MLXPipeline(
        tribe_model=TribeTransformer(tribe_cfg),
        vjepa2_model=MLXVJepa2(vit_cfg),
        dinov2_model=MLXDINOv2Large(dino_cfg),
        w2v_bert_model=MLXWav2VecBert(w2v_cfg),
        llama_extractor=None,   # skip LLaMA to avoid weight download
    )


class TestPipelineInit:
    def test_pipeline_instantiation(self):
        pipeline = _make_mock_pipeline()
        assert pipeline is not None
        assert pipeline.tribe is not None

    def test_pipeline_no_encoders(self):
        from tribe_v2_mlx.pipeline import TribeV2MLXPipeline
        from tribe_v2_mlx.models.tribe import TribeTransformer, TribeConfig
        cfg = TribeConfig(n_outputs=64, n_subjects=1)
        pipeline = TribeV2MLXPipeline(tribe_model=TribeTransformer(cfg))
        assert pipeline.vjepa2 is None
        assert pipeline.dinov2 is None


class TestPredictImage:
    def test_predict_image_output_shape(self, synthetic_image_path):
        pipeline = _make_mock_pipeline(n_outputs=128)
        result = pipeline.predict_image(str(synthetic_image_path))
        assert isinstance(result, np.ndarray), "Output must be numpy array"
        assert result.ndim == 2, f"Expected 2D output, got {result.ndim}D"
        assert result.shape[1] == 128, f"Expected 128 outputs, got {result.shape[1]}"

    def test_predict_image_finite_values(self, synthetic_image_path):
        pipeline = _make_mock_pipeline(n_outputs=64)
        result = pipeline.predict_image(str(synthetic_image_path))
        assert np.all(np.isfinite(result)), "Output contains non-finite values"


class TestPredictVideo:
    def test_predict_video_output_shape(self, synthetic_video_path):
        pipeline = _make_mock_pipeline(n_outputs=128)
        result = pipeline.predict(str(synthetic_video_path))
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 128

    def test_predict_video_multiple_segments(self, synthetic_video_path):
        pipeline = _make_mock_pipeline(n_outputs=64)
        result = pipeline.predict(str(synthetic_video_path))
        # 5-second video at 4fps in 4s segments → 1-2 segments
        assert result.shape[0] >= 1

    def test_predict_video_with_caption(self, synthetic_video_path):
        pipeline = _make_mock_pipeline(n_outputs=64)
        result = pipeline.predict(str(synthetic_video_path), caption="A test video.")
        assert result.shape[1] == 64

    def test_predict_video_finite_values(self, synthetic_video_path):
        pipeline = _make_mock_pipeline(n_outputs=32)
        result = pipeline.predict(str(synthetic_video_path))
        assert np.all(np.isfinite(result))


class TestRunTribe:
    """Test the _run_tribe internal method directly."""

    def test_run_tribe_with_mock_features(self):
        from tribe_v2_mlx.pipeline import TribeV2MLXPipeline
        from tribe_v2_mlx.models.tribe import TribeTransformer, TribeConfig

        cfg = TribeConfig(
            text_in_dim=32, video_in_dim=32, image_in_dim=32, audio_in_dim=32,
            mod_dim=8, hidden=32, num_heads=4, depth=2, mlp_ratio=2,
            low_rank_dim=16, n_outputs=64, n_subjects=1,
        )
        pipeline = TribeV2MLXPipeline(tribe_model=TribeTransformer(cfg))

        features = {
            "image": mx.random.normal((1, 3, 32)),
        }
        result = pipeline._run_tribe(features)
        assert result.shape == (3, 64)

    def test_run_tribe_no_features_returns_zeros(self):
        from tribe_v2_mlx.pipeline import TribeV2MLXPipeline
        from tribe_v2_mlx.models.tribe import TribeTransformer, TribeConfig

        cfg = TribeConfig(n_outputs=20484, n_subjects=1)
        pipeline = TribeV2MLXPipeline(tribe_model=TribeTransformer(cfg))

        result = pipeline._run_tribe({})
        assert result.shape == (1, 20484)
        assert np.all(result == 0)
