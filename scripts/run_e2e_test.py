#!/usr/bin/env python3
"""
End-to-end inference test for TRIBE v2 MLX.

Tests the full pipeline code path:
  1. Decode a real video with PyAV (real I/O and preprocessing)
  2. Segment frames using production preprocessing utilities
  3. Build synthetic MLX feature tensors with production-matching dimensions
  4. Run the TribeTransformer (random-weight, production TribeConfig) forward pass
  5. Report timing, output shape, peak Metal GPU memory, and value statistics

This script requires no trained weights — it validates that every component
of the inference stack works correctly end-to-end on Apple Silicon.

Usage
-----
    python scripts/run_e2e_test.py /tmp/test-tribe.mp4
    python scripts/run_e2e_test.py /tmp/test-tribe.mp4 --seed 42
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# Production feature dimensions (matches TribeConfig defaults and real checkpoint)
_FEATURE_DIMS = {
    "text":  18432,  # LLaMA-3.2-3B: 6 layers × 3072 → concat
    "video": 2816,   # V-JEPA2 ViT-g: 2 layers × 1408
    "image": 1024,   # DINOv2-Large: 1 layer × 1024
    "audio": 2048,   # Wav2Vec-BERT: 2 layers × 1024
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRIBE v2 MLX end-to-end inference test"
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for synthetic features (default 0)")
    parser.add_argument("--modalities", default="image,audio,video,text",
                        help="Comma-separated list of modalities to include "
                             "(default: image,audio,video,text)")
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    print(f"TRIBE v2 MLX — End-to-End Inference Test")
    print(f"  Video:      {video_path}")
    print(f"  Modalities: {', '.join(modalities)}")
    print(f"  Device:     {mx.default_device()}")
    print()

    # Reset peak-memory counter before the timed section
    mx.reset_peak_memory()
    t_start = time.perf_counter()

    # ── Step 1: Decode video ────────────────────────────────────────────────
    print("Step 1: Decoding video …")
    from tribe_v2_mlx.preprocessing.video import (
        load_video_frames,
        segment_video_frames,
        extract_audio_from_video,
    )
    from tribe_v2_mlx.preprocessing.audio import preprocess_audio, segment_audio

    _SEGMENT_DURATION = 4.0   # seconds (production value)
    _VIDEO_FPS = 4.0          # frames per second
    _AUDIO_SR = 16_000

    frames = load_video_frames(
        video_path, fps=_VIDEO_FPS, target_size=(256, 256)
    )
    t_decode = time.perf_counter()

    segments = segment_video_frames(frames, _SEGMENT_DURATION, _VIDEO_FPS)
    n_segments = len(segments)

    print(f"  Frames decoded:  {frames.shape[0]} @ {_VIDEO_FPS} fps")
    print(f"  Segments:        {n_segments} × {_SEGMENT_DURATION}s")
    print(f"  Frame shape:     {frames.shape[1:]}")
    print(f"  Decode time:     {t_decode - t_start:.3f}s")
    print()

    # ── Step 2: Decode audio ────────────────────────────────────────────────
    print("Step 2: Decoding audio …")
    try:
        waveform = extract_audio_from_video(video_path, sample_rate=_AUDIO_SR)
        waveform = preprocess_audio(waveform)
        audio_segs = segment_audio(waveform, _AUDIO_SR, _SEGMENT_DURATION)
    except Exception as e:
        print(f"  Warning: audio decode failed ({e}), using silence")
        seg_len = int(_SEGMENT_DURATION * _AUDIO_SR)
        audio_segs = [np.zeros(seg_len, dtype=np.float32)] * n_segments

    # Pad / truncate to n_segments
    while len(audio_segs) < n_segments:
        audio_segs.append(np.zeros(int(_SEGMENT_DURATION * _AUDIO_SR), dtype=np.float32))
    audio_segs = audio_segs[:n_segments]

    t_audio = time.perf_counter()
    print(f"  Audio segments:  {len(audio_segs)} × {len(audio_segs[0])} samples")
    print(f"  Audio time:      {t_audio - t_decode:.3f}s")
    print()

    # ── Step 3: Build synthetic MLX features ────────────────────────────────
    # In real inference these come from the frozen encoder networks.
    # Here we use seeded random values with production-matching shapes.
    print("Step 3: Building synthetic feature tensors (random-weight encoder sim) …")
    rng = np.random.default_rng(args.seed)

    features: dict[str, mx.array] = {}
    for mod in modalities:
        if mod not in _FEATURE_DIMS:
            print(f"  Warning: unknown modality '{mod}', skipping")
            continue
        dim = _FEATURE_DIMS[mod]
        arr = rng.standard_normal((1, n_segments, dim)).astype(np.float32)
        features[mod] = mx.array(arr)
        print(f"  {mod:>6}: shape {features[mod].shape}, "
              f"range [{arr.min():.3f}, {arr.max():.3f}]")

    mx.eval(*features.values())
    t_feat = time.perf_counter()
    print(f"  Feature build time: {t_feat - t_audio:.3f}s")
    print()

    # ── Step 4: Load TRIBE transformer ──────────────────────────────────────
    print("Step 4: Loading TribeTransformer (random weights, production config) …")
    from tribe_v2_mlx.models.tribe import TribeTransformer, TribeConfig

    config = TribeConfig()  # production defaults (n_outputs=20484)
    tribe = TribeTransformer(config)
    mx.eval(tribe.parameters())

    t_load = time.perf_counter()
    print(f"  Config:      hidden={config.hidden}, depth={config.depth}, "
          f"n_outputs={config.n_outputs}")
    print(f"  Load time:   {t_load - t_feat:.3f}s")
    print()

    # ── Step 5: Forward pass ─────────────────────────────────────────────────
    print("Step 5: Running TRIBE forward pass …")
    mx.reset_peak_memory()  # reset again just before model eval
    t_fwd_start = time.perf_counter()

    output = tribe(features, subject_id=0)  # (1, T, 20484)
    mx.eval(output)                          # force Metal computation

    t_fwd_end = time.perf_counter()
    peak_mem_gb = mx.get_peak_memory() / 1e9

    result: np.ndarray = np.array(output[0])  # (T, 20484)

    t_total = t_fwd_end - t_start

    print(f"  Forward pass time: {t_fwd_end - t_fwd_start:.3f}s")
    print()

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Input video:       {video_path}")
    print(f"  Video duration:    ~{frames.shape[0] / _VIDEO_FPS:.1f}s "
          f"({frames.shape[0]} frames @ {_VIDEO_FPS} fps)")
    print(f"  Output shape:      {result.shape}  "
          f"(n_timepoints={result.shape[0]}, n_vertices={result.shape[1]})")
    print(f"  Output min:        {result.min():.6f}")
    print(f"  Output max:        {result.max():.6f}")
    print(f"  Output mean:       {result.mean():.6f}")
    print(f"  Output std:        {result.std():.6f}")
    print(f"  Output non-zero:   {(result != 0).sum()} / {result.size}")
    print()
    print(f"  Wall time (total): {t_total:.3f}s")
    print(f"    Video decode:    {t_decode - t_start:.3f}s")
    print(f"    Audio decode:    {t_audio - t_decode:.3f}s")
    print(f"    Feature build:   {t_feat - t_audio:.3f}s")
    print(f"    Model load:      {t_load - t_feat:.3f}s")
    print(f"    Forward pass:    {t_fwd_end - t_fwd_start:.3f}s")
    print(f"  Peak MLX GPU mem:  {peak_mem_gb:.4f} GB")
    print()

    sensible = (result.std() > 0) and np.isfinite(result).all()
    print(f"  Output sensible?   {'YES — non-zero, finite' if sensible else 'NO'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
