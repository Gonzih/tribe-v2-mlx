"""Audio preprocessing for Wav2Vec-BERT 2.0."""
from __future__ import annotations

from typing import Optional

import numpy as np


_TARGET_SR = 16_000
# Wav2Vec-BERT 2.0 feature-extractor normalisation
_MEAN = 0.0
_STD = 0.5


def load_audio_waveform(path: str, sample_rate: int = _TARGET_SR) -> np.ndarray:
    """
    Load an audio file as a mono float32 waveform.

    Returns
    -------
    np.ndarray
        Shape (n_samples,), float32 at `sample_rate` Hz.
    """
    try:
        import av
    except ImportError as e:
        raise ImportError("PyAV is required: pip install av") from e

    container = av.open(str(path))
    if not container.streams.audio:
        return np.zeros(_TARGET_SR, dtype=np.float32)

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
    chunks = []
    for packet in container.demux(container.streams.audio[0]):
        for frame in packet.decode():
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray()[0])
    container.close()

    if not chunks:
        return np.zeros(_TARGET_SR, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def preprocess_audio(
    waveform: np.ndarray,
    sample_rate: int = _TARGET_SR,
    normalize: bool = True,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Normalise a waveform for Wav2Vec-BERT 2.0 input.

    Returns
    -------
    np.ndarray
        Shape (n_samples,), float32.
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)

    if max_length is not None:
        waveform = waveform[:max_length]

    if normalize:
        # Zero-mean / unit-variance normalisation (matches HF feature extractor)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)

    return waveform.astype(np.float32)


def segment_audio(
    waveform: np.ndarray,
    sample_rate: int = _TARGET_SR,
    segment_duration: float = 4.0,
) -> list[np.ndarray]:
    """
    Split a waveform into fixed-duration segments; pads the last.

    Returns
    -------
    list of np.ndarray
        Each element has shape (segment_length,).
    """
    seg_len = int(segment_duration * sample_rate)
    n = len(waveform)
    segments = []
    for start in range(0, max(1, n), seg_len):
        chunk = waveform[start : start + seg_len]
        if len(chunk) < seg_len:
            chunk = np.concatenate(
                [chunk, np.zeros(seg_len - len(chunk), dtype=waveform.dtype)]
            )
        segments.append(chunk)
    return segments
