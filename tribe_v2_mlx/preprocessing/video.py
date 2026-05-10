"""Video preprocessing utilities."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ImageNet normalization (used by V-JEPA2 and DINOv2)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_video_frames(
    video_path: str,
    fps: float = 4.0,
    target_size: Tuple[int, int] = (256, 256),
    normalize: bool = True,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Decode a video file and return sampled frames.

    Returns
    -------
    np.ndarray
        Shape (T, H, W, 3), float32, optionally normalised with ImageNet stats.
    """
    try:
        import av
    except ImportError as e:
        raise ImportError("PyAV is required for video decoding: pip install av") from e

    from PIL import Image

    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    # Determine frame indices to sample at `fps`
    native_fps = float(video_stream.average_rate)
    total_frames = video_stream.frames
    if native_fps <= 0 or total_frames <= 0:
        native_fps = 25.0
        total_frames = int(container.duration * native_fps / av.time_base) if container.duration else 250

    step = max(1, int(native_fps / fps))
    frame_indices = set(range(0, total_frames, step))
    if max_frames is not None:
        frame_indices = set(list(sorted(frame_indices))[:max_frames])

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in frame_indices:
            img = frame.to_image().resize(target_size, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            arr = arr[:, :, :3]  # drop alpha if any
            if normalize:
                arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
            frames.append(arr)
        if max_frames is not None and len(frames) >= max_frames:
            break

    container.close()

    if not frames:
        # Return a single black frame if nothing decoded
        dummy = np.zeros((1, *target_size, 3), dtype=np.float32)
        return dummy

    return np.stack(frames, axis=0)  # (T, H, W, 3)


def extract_audio_from_video(
    video_path: str,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Extract mono audio waveform from a video file.

    Returns
    -------
    np.ndarray
        Shape (n_samples,), float32, resampled to `sample_rate`.
    """
    try:
        import av
    except ImportError as e:
        raise ImportError("PyAV is required: pip install av") from e

    container = av.open(str(video_path))

    if not container.streams.audio:
        # No audio track — return silence matching video duration
        duration = float(container.duration or 0) / av.time_base
        n_samples = int(duration * sample_rate)
        return np.zeros(n_samples, dtype=np.float32)

    audio_stream = container.streams.audio[0]
    native_sr = audio_stream.rate

    pcm_chunks = []
    resampler = av.AudioResampler(
        format="fltp",
        layout="mono",
        rate=sample_rate,
    )
    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            resampled = resampler.resample(frame)
            for rf in resampled:
                chunk = rf.to_ndarray()  # (1, n_samples) float32
                pcm_chunks.append(chunk[0])

    container.close()

    if not pcm_chunks:
        return np.zeros(sample_rate, dtype=np.float32)

    return np.concatenate(pcm_chunks).astype(np.float32)


def segment_video_frames(
    frames: np.ndarray,
    segment_duration: float = 4.0,
    fps: float = 4.0,
) -> list[np.ndarray]:
    """
    Split a (T, H, W, C) frame array into fixed-length segments.

    Returns a list of (n_frames_per_segment, H, W, C) arrays.
    Pads the last segment with zeros if needed.
    """
    frames_per_segment = int(segment_duration * fps)
    T = frames.shape[0]
    segments = []
    for start in range(0, max(1, T), frames_per_segment):
        chunk = frames[start : start + frames_per_segment]
        if len(chunk) < frames_per_segment:
            pad = np.zeros(
                (frames_per_segment - len(chunk), *frames.shape[1:]),
                dtype=frames.dtype,
            )
            chunk = np.concatenate([chunk, pad], axis=0)
        segments.append(chunk)
    return segments
