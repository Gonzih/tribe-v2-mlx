"""
Shared test fixtures for tribe_v2_mlx tests.

All fixtures provide synthetic data or mock objects so that tests run
without downloading actual model weights.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic video fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_video_path(tmp_path_factory) -> Path:
    """
    Create a 5-second synthetic video (random colour frames at 10 fps, 64x64).
    Returns the path to the temporary .mp4 file.

    Tries PyAV first, then imageio, then skips if neither is available.
    """
    tmp_dir = tmp_path_factory.mktemp("videos")
    video_path = tmp_dir / "synthetic_5s.mp4"

    rng = np.random.default_rng(42)
    fps = 10
    n_frames = fps * 5  # 5 seconds
    frames = rng.integers(0, 256, (n_frames, 64, 64, 3), dtype=np.uint8)

    # Try PyAV
    try:
        import av

        container = av.open(str(video_path), mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23"}
        for frame_arr in frames:
            frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return video_path
    except Exception:
        pass

    # Try imageio
    try:
        import imageio.v3 as iio

        writer = iio.imopen(str(video_path), "w", plugin="pyav")
        writer.write(frames, fps=fps)
        writer.close()
        return video_path
    except Exception:
        pass

    pytest.skip(
        "Neither PyAV nor imageio[pyav] available; skipping synthetic video tests. "
        "Install: pip install av  or  pip install 'imageio[pyav]'"
    )


@pytest.fixture(scope="session")
def synthetic_image_path(tmp_path_factory) -> Path:
    """Create a synthetic 224×224 RGB image."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not available")

    tmp_dir = tmp_path_factory.mktemp("images")
    img_path = tmp_dir / "synthetic.jpg"

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(img_path))
    return img_path


# ---------------------------------------------------------------------------
# Small model configs for fast testing (tiny dims instead of full-scale)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_vit_config():
    from tribe_v2_mlx.models.vjepa2 import ViTConfig
    return ViTConfig(
        hidden_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=2,
        patch_size=16,
        temporal_patch_size=2,
        image_size=64,
        extract_layers=[2, 3],
    )


@pytest.fixture
def small_dinov2_config():
    from tribe_v2_mlx.models.dinov2 import DINOv2Config
    return DINOv2Config(
        hidden_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=2,
        patch_size=16,
        image_size=64,
        num_register_tokens=2,
        extract_layers=[2],
    )


@pytest.fixture
def small_w2v_config():
    from tribe_v2_mlx.models.wav2vec_bert import W2VBertConfig
    return W2VBertConfig(
        hidden_dim=64,
        conv_out_dim=64,
        depth=4,
        num_heads=4,
        conv_kernel_size=7,
        pos_conv_kernel=16,
        extract_layers=[2, 3],
    )


@pytest.fixture
def small_tribe_config():
    from tribe_v2_mlx.models.tribe import TribeConfig
    return TribeConfig(
        text_in_dim=64,
        video_in_dim=64,
        image_in_dim=64,
        audio_in_dim=64,
        mod_dim=16,
        hidden=64,
        num_heads=4,
        depth=2,
        mlp_ratio=2,
        low_rank_dim=32,
        n_outputs=128,
        n_subjects=2,
        max_seq_len=32,
    )
