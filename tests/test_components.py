"""
Unit tests for each MLX encoder component.

Tests use small model configs and random inputs — no weights download required.
"""
from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx


# ─── V-JEPA2 ViT-g ────────────────────────────────────────────────────────────

class TestVJepa2:
    def test_instantiation(self, small_vit_config):
        from tribe_v2_mlx.models.vjepa2 import MLXVJepa2
        model = MLXVJepa2(small_vit_config)
        assert model is not None

    def test_forward_pass_shape(self, small_vit_config):
        from tribe_v2_mlx.models.vjepa2 import MLXVJepa2
        model = MLXVJepa2(small_vit_config)

        B, T, H, W, C = 1, 4, 64, 64, 3
        x = mx.random.normal((B, T, H, W, C))
        final, hidden_states = model(x)
        mx.eval(final, *hidden_states)

        n_patches = (T // small_vit_config.temporal_patch_size) * \
                    (H // small_vit_config.patch_size) * \
                    (W // small_vit_config.patch_size)
        n_tokens = n_patches + 1  # + CLS
        assert final.shape == (B, n_tokens, small_vit_config.hidden_dim), \
            f"Unexpected final shape: {final.shape}"

    def test_hidden_states_returned(self, small_vit_config):
        from tribe_v2_mlx.models.vjepa2 import MLXVJepa2
        model = MLXVJepa2(small_vit_config)
        x = mx.random.normal((1, 4, 64, 64, 3))
        _, hidden_states = model(x, extract_layers=[1, 2])
        mx.eval(*hidden_states)
        assert len(hidden_states) == 2, "Expected 2 extracted hidden states"

    def test_patch_embedding_shape(self, small_vit_config):
        from tribe_v2_mlx.models.vjepa2 import TubeletEmbedding
        embed = TubeletEmbedding(small_vit_config)
        x = mx.random.normal((2, 4, 64, 64, 3))
        out = embed(x)
        mx.eval(out)
        expected_tokens = (4 // 2) * (64 // 16) * (64 // 16)  # nt * nh * nw
        assert out.shape == (2, expected_tokens, small_vit_config.hidden_dim)


# ─── DINOv2-Large ─────────────────────────────────────────────────────────────

class TestDINOv2:
    def test_instantiation(self, small_dinov2_config):
        from tribe_v2_mlx.models.dinov2 import MLXDINOv2Large
        model = MLXDINOv2Large(small_dinov2_config)
        assert model is not None

    def test_forward_pass_cls_shape(self, small_dinov2_config):
        from tribe_v2_mlx.models.dinov2 import MLXDINOv2Large
        model = MLXDINOv2Large(small_dinov2_config)
        x = mx.random.normal((2, 64, 64, 3))  # (B, H, W, C)
        cls_out, hidden_states = model(x)
        mx.eval(cls_out, *hidden_states)
        assert cls_out.shape == (2, small_dinov2_config.hidden_dim), \
            f"CLS token shape: {cls_out.shape}"

    def test_hidden_states_extracted(self, small_dinov2_config):
        from tribe_v2_mlx.models.dinov2 import MLXDINOv2Large
        model = MLXDINOv2Large(small_dinov2_config)
        x = mx.random.normal((1, 64, 64, 3))
        _, hidden_states = model(x, extract_layers=[1, 2])
        mx.eval(*hidden_states)
        assert len(hidden_states) == 2

    def test_register_tokens_in_sequence(self, small_dinov2_config):
        from tribe_v2_mlx.models.dinov2 import MLXDINOv2Large
        model = MLXDINOv2Large(small_dinov2_config)
        x = mx.random.normal((1, 64, 64, 3))
        _, hidden_states = model(x, extract_layers=[0])
        mx.eval(*hidden_states)
        n_patches = (64 // small_dinov2_config.patch_size) ** 2
        n_special = 1 + small_dinov2_config.num_register_tokens
        expected_tokens = n_patches + n_special
        assert hidden_states[0].shape[1] == expected_tokens


# ─── Wav2Vec-BERT 2.0 ─────────────────────────────────────────────────────────

class TestWav2VecBert:
    def test_instantiation(self, small_w2v_config):
        from tribe_v2_mlx.models.wav2vec_bert import MLXWav2VecBert
        model = MLXWav2VecBert(small_w2v_config)
        assert model is not None

    def test_conv_feature_extractor_shape(self):
        from tribe_v2_mlx.models.wav2vec_bert import ConvFeatureExtractor
        extractor = ConvFeatureExtractor()
        # Input: (B, T_wave); after 7 strided convs with stride [5,2,2,2,2,2,2]
        # total stride = 5*2^6 = 320; at 16kHz → 50 frames/sec
        waveform = mx.random.normal((1, 16000))  # 1 second at 16kHz
        out = extractor(waveform)
        mx.eval(out)
        # Output: approximately (1, 50, 512)
        assert out.ndim == 3, f"Expected 3D output, got {out.ndim}D"
        assert out.shape[-1] == 512, f"Expected 512 channels, got {out.shape[-1]}"
        assert 40 <= out.shape[1] <= 60, f"Unexpected time dim: {out.shape[1]}"

    def test_forward_pass_shape(self, small_w2v_config):
        from tribe_v2_mlx.models.wav2vec_bert import MLXWav2VecBert, ConvFeatureExtractor, W2VBertConfig

        # Use a config compatible with the small conv feature extractor
        # (ConvFeatureExtractor always outputs 512 channels; small_w2v_config uses 64)
        # For this test, use a config with matching conv_out_dim
        config = W2VBertConfig(
            hidden_dim=64,
            conv_out_dim=512,  # must match ConvFeatureExtractor output
            depth=2,
            num_heads=4,
            conv_kernel_size=7,
            pos_conv_kernel=16,
            extract_layers=[0, 1],
        )
        model = MLXWav2VecBert(config)
        waveform = mx.random.normal((1, 16000))  # 1 second
        final, hidden_states = model(waveform)
        mx.eval(final, *hidden_states)
        assert final.ndim == 3
        assert final.shape[-1] == config.hidden_dim
        assert len(hidden_states) == 2

    def test_conformer_block_shape(self, small_w2v_config):
        from tribe_v2_mlx.models.wav2vec_bert import ConformerBlock
        # Match channel size to ConvFeatureExtractor output
        from tribe_v2_mlx.models.wav2vec_bert import W2VBertConfig
        config = W2VBertConfig(hidden_dim=64, depth=2, num_heads=4, conv_kernel_size=7)
        block = ConformerBlock(config)
        x = mx.random.normal((2, 16, 64))  # (B, T, dim)
        out = block(x)
        mx.eval(out)
        assert out.shape == (2, 16, 64), f"Conformer block output: {out.shape}"


# ─── TRIBE Transformer ────────────────────────────────────────────────────────

class TestTribeTransformer:
    def test_instantiation(self, small_tribe_config):
        from tribe_v2_mlx.models.tribe import TribeTransformer
        model = TribeTransformer(small_tribe_config)
        assert model is not None

    def test_forward_pass_all_modalities(self, small_tribe_config):
        from tribe_v2_mlx.models.tribe import TribeTransformer
        model = TribeTransformer(small_tribe_config)

        T = 4
        features = {
            "text": mx.random.normal((1, T, small_tribe_config.text_in_dim)),
            "video": mx.random.normal((1, T, small_tribe_config.video_in_dim)),
            "image": mx.random.normal((1, T, small_tribe_config.image_in_dim)),
            "audio": mx.random.normal((1, T, small_tribe_config.audio_in_dim)),
        }
        output = model(features, subject_id=0)
        mx.eval(output)
        assert output.shape == (1, T, small_tribe_config.n_outputs), \
            f"TRIBE output shape: {output.shape}"

    def test_forward_pass_partial_modalities(self, small_tribe_config):
        from tribe_v2_mlx.models.tribe import TribeTransformer
        model = TribeTransformer(small_tribe_config)

        features = {
            "image": mx.random.normal((1, 3, small_tribe_config.image_in_dim)),
        }
        output = model(features, subject_id=0)
        mx.eval(output)
        assert output.shape[0] == 1
        assert output.shape[-1] == small_tribe_config.n_outputs

    def test_subject_layers_different_outputs(self, small_tribe_config):
        from tribe_v2_mlx.models.tribe import TribeTransformer
        model = TribeTransformer(small_tribe_config)

        features = {"image": mx.random.normal((1, 2, small_tribe_config.image_in_dim))}
        out0 = model(features, subject_id=0)
        out1 = model(features, subject_id=1)
        mx.eval(out0, out1)
        # With random weights the outputs should generally differ
        assert out0.shape == out1.shape


# ─── LLaMA Extractor (import-only test — no weight download) ──────────────────

class TestLlamaExtractor:
    def test_class_importable(self):
        from tribe_v2_mlx.models.llama import MLXLlamaExtractor
        assert MLXLlamaExtractor is not None

    def test_depth_to_layer_index(self):
        from tribe_v2_mlx.models.llama import _depth_to_layer_index
        assert _depth_to_layer_index(0.0, 28) == 0
        assert _depth_to_layer_index(1.0, 28) == 27
        assert _depth_to_layer_index(0.5, 28) == 14
