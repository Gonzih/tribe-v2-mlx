"""
TRIBE v2 MLX inference pipeline.

Usage
-----
    from tribe_v2_mlx import TribeV2MLXPipeline

    pipeline = TribeV2MLXPipeline.from_weights("./weights/mlx")
    preds = pipeline.predict("video.mp4")   # (n_segments, 20484)
    preds = pipeline.predict_image("frame.jpg")  # (1, 20484)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


_SEGMENT_DURATION = 4.0   # seconds — V-JEPA2 clip duration
_VIDEO_FPS = 4.0          # frames per second for feature extraction
_AUDIO_SR = 16_000        # Wav2Vec-BERT 2.0 sample rate
_DEFAULT_SUBJECT_ID = 0


class TribeV2MLXPipeline:
    """
    End-to-end fMRI prediction pipeline using MLX on Apple Silicon.

    Architecture
    ------------
    1. Video  → frames  → V-JEPA2 ViT-g (MLX, 8-bit)  → video features
    2. Video  → frames  → DINOv2-Large  (MLX, 8-bit)  → image features
    3. Video  → audio   → Wav2Vec-BERT  (MLX, 8-bit)  → audio features
    4. Captions / dummy text → LLaMA 3.2-3B (MLX, 4-bit) → text features
    5. All features → TRIBE transformer (MLX, fp16) → (n_segments, 20484)

    Parameters
    ----------
    vjepa2_model : MLXVJepa2 or None
    dinov2_model : MLXDINOv2Large or None
    w2v_bert_model : MLXWav2VecBert or None
    llama_extractor : MLXLlamaExtractor or None
    tribe_model : TribeTransformer
    subject_id : which subject's output projection to use (default 0)
    """

    def __init__(
        self,
        tribe_model,
        vjepa2_model=None,
        dinov2_model=None,
        w2v_bert_model=None,
        llama_extractor=None,
        subject_id: int = _DEFAULT_SUBJECT_ID,
    ):
        self.tribe = tribe_model
        self.vjepa2 = vjepa2_model
        self.dinov2 = dinov2_model
        self.w2v_bert = w2v_bert_model
        self.llama = llama_extractor
        self.subject_id = subject_id

    @classmethod
    def from_weights(
        cls,
        weights_dir: str = "./weights/mlx",
        tribe_ckpt: Optional[str] = None,
        subject_id: int = _DEFAULT_SUBJECT_ID,
    ) -> "TribeV2MLXPipeline":
        """
        Load all MLX models from a weights directory.

        Expected files (produced by scripts/convert_to_mlx.py):
          weights_dir/vjepa2-vitg-8bit.safetensors
          weights_dir/dinov2-large-8bit.safetensors
          weights_dir/wav2vec-bert-8bit.safetensors
          weights_dir/llama-3.2-3b-4bit/  (directory)
          tribe_ckpt  or  weights_dir/../tribe/best.ckpt
        """
        from tribe_v2_mlx.models.vjepa2 import MLXVJepa2, vjepa2_vitg_config
        from tribe_v2_mlx.models.dinov2 import MLXDINOv2Large, DINOv2Config
        from tribe_v2_mlx.models.wav2vec_bert import MLXWav2VecBert, W2VBertConfig
        from tribe_v2_mlx.models.tribe import TribeTransformer, TribeConfig

        wdir = Path(weights_dir)

        # --- V-JEPA2 ---
        vjepa2_model = None
        vjepa2_path = wdir / "vjepa2-vitg-8bit.safetensors"
        if vjepa2_path.exists():
            print(f"Loading V-JEPA2 from {vjepa2_path} …")
            vjepa2_model = MLXVJepa2(vjepa2_vitg_config())
            weights = mx.load(str(vjepa2_path))
            vjepa2_model.load_weights(list(weights.items()), strict=False)
            mx.eval(vjepa2_model.parameters())

        # --- DINOv2 ---
        dinov2_model = None
        dinov2_path = wdir / "dinov2-large-8bit.safetensors"
        if dinov2_path.exists():
            print(f"Loading DINOv2 from {dinov2_path} …")
            dinov2_model = MLXDINOv2Large(DINOv2Config())
            weights = mx.load(str(dinov2_path))
            dinov2_model.load_weights(list(weights.items()), strict=False)
            mx.eval(dinov2_model.parameters())

        # --- Wav2Vec-BERT ---
        w2v_model = None
        w2v_path = wdir / "wav2vec-bert-8bit.safetensors"
        if w2v_path.exists():
            print(f"Loading Wav2Vec-BERT from {w2v_path} …")
            w2v_model = MLXWav2VecBert(W2VBertConfig())
            weights = mx.load(str(w2v_path))
            w2v_model.load_weights(list(weights.items()), strict=False)
            mx.eval(w2v_model.parameters())

        # --- LLaMA ---
        llama_model = None
        llama_path = wdir / "llama-3.2-3b-4bit"
        if llama_path.exists():
            print(f"Loading LLaMA from {llama_path} …")
            from tribe_v2_mlx.models.llama import MLXLlamaExtractor
            llama_model = MLXLlamaExtractor.load(str(llama_path))

        # --- TRIBE transformer ---
        tribe_model_obj = None
        if tribe_ckpt is None:
            tribe_ckpt_path = wdir.parent / "tribe" / "best.ckpt"
            if not tribe_ckpt_path.exists():
                tribe_ckpt_path = wdir.parent / "best.ckpt"
        else:
            tribe_ckpt_path = Path(tribe_ckpt)

        if tribe_ckpt_path.exists():
            print(f"Loading TRIBE transformer from {tribe_ckpt_path} …")
            tribe_model_obj = TribeTransformer.from_checkpoint(str(tribe_ckpt_path))
        else:
            print("TRIBE checkpoint not found — using random-weight model (for testing).")
            tribe_model_obj = TribeTransformer(TribeConfig())

        return cls(
            tribe_model=tribe_model_obj,
            vjepa2_model=vjepa2_model,
            dinov2_model=dinov2_model,
            w2v_bert_model=w2v_model,
            llama_extractor=llama_model,
            subject_id=subject_id,
        )

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_video_features(
        self, frames: np.ndarray
    ) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """
        Extract V-JEPA2 and DINOv2 features from a (T, H, W, C) frame array.

        Returns
        -------
        (video_feat, image_feat)
          video_feat : (1, T', n_layers*video_dim) — temporal-pooled V-JEPA2 features
          image_feat : (1, T,  n_layers*image_dim) — per-frame DINOv2 features
        """
        from tribe_v2_mlx.preprocessing.video import segment_video_frames

        segments = segment_video_frames(frames, _SEGMENT_DURATION, _VIDEO_FPS)
        n_segments = len(segments)

        video_feats = []
        image_feats = []

        for seg in segments:
            # seg: (n_frames_per_seg, H, W, C)
            if self.vjepa2 is not None:
                x = mx.array(seg[None], dtype=mx.float32)  # (1, T_seg, H, W, C)
                _, hidden = self.vjepa2(x, extract_layers=self.vjepa2.config.extract_layers)
                # Pool over spatial tokens, keep temporal dim
                # hidden[i]: (1, n_spatial_tokens, D) — no temporal dim for simple forward
                # Spatial mean → (1, D) per layer
                pooled = mx.concatenate(
                    [h.mean(axis=1) for h in hidden], axis=-1
                )  # (1, n_layers*D)
                video_feats.append(pooled)

            if self.dinov2 is not None:
                # Use centre frame from each segment
                centre_frame = seg[len(seg) // 2]  # (H, W, C)
                # Resize to model's expected image size
                dino_size = self.dinov2.config.image_size
                from PIL import Image
                img = Image.fromarray(
                    np.clip((centre_frame * 0.225 + 0.45) * 255, 0, 255).astype(np.uint8)
                ).resize((dino_size, dino_size))
                frame_arr = np.array(img, dtype=np.float32) / 255.0
                # Re-normalise
                _mean = np.array([0.485, 0.456, 0.406])
                _std = np.array([0.229, 0.224, 0.225])
                frame_arr = (frame_arr - _mean) / _std
                x = mx.array(frame_arr[None], dtype=mx.float32)  # (1, H, W, C)
                _, hidden = self.dinov2(x, extract_layers=self.dinov2.config.extract_layers)
                pooled = mx.concatenate(
                    [h.mean(axis=1) for h in hidden], axis=-1
                )  # (1, n_layers*D)
                image_feats.append(pooled)

        # Stack across segments → (1, n_segments, features)
        def stack_or_none(lst):
            if not lst:
                return None
            stacked = mx.concatenate([f[:, None, :] for f in lst], axis=1)  # (1, T, D)
            return stacked

        return stack_or_none(video_feats), stack_or_none(image_feats)

    def _extract_audio_features(
        self, waveform: np.ndarray, n_segments: int
    ) -> Optional[mx.array]:
        """
        Extract Wav2Vec-BERT features from waveform.

        Returns (1, n_segments, n_layers*audio_dim) or None.
        """
        if self.w2v_bert is None:
            return None

        from tribe_v2_mlx.preprocessing.audio import segment_audio

        segments = segment_audio(waveform, _AUDIO_SR, _SEGMENT_DURATION)
        # Pad or truncate to match n_segments
        while len(segments) < n_segments:
            segments.append(np.zeros(int(_SEGMENT_DURATION * _AUDIO_SR), dtype=np.float32))
        segments = segments[:n_segments]

        audio_feats = []
        for seg in segments:
            x = mx.array(seg[None], dtype=mx.float32)  # (1, T_wave)
            _, hidden = self.w2v_bert(x, extract_layers=self.w2v_bert.config.extract_layers)
            # Pool over time → (1, D) per layer
            pooled = mx.concatenate([h.mean(axis=1) for h in hidden], axis=-1)
            audio_feats.append(pooled)

        if not audio_feats:
            return None
        return mx.concatenate([f[:, None, :] for f in audio_feats], axis=1)  # (1, T, D)

    def _extract_text_features(
        self, text: str, n_segments: int
    ) -> Optional[mx.array]:
        """
        Extract LLaMA hidden-state features for a text description.

        Returns (1, n_segments, n_layers*text_dim) or None.
        """
        if self.llama is None:
            return None

        layer_indices = self.llama.layer_indices
        hidden = self.llama.extract_text_features(text, layer_indices=layer_indices)

        if not hidden:
            return None

        # Each value: (1, n_tokens, hidden_dim); concatenate layers
        layer_vecs = [mx.array(hidden[li]) for li in sorted(hidden.keys())]
        # Pool over tokens → (1, D) per layer, then concat layers
        pooled = mx.concatenate([lv.mean(axis=1) for lv in layer_vecs], axis=-1)  # (1, n_layers*D)

        # Broadcast to all segments
        pooled_expanded = mx.broadcast_to(
            pooled[:, None, :], (1, n_segments, pooled.shape[-1])
        )
        return pooled_expanded  # (1, n_segments, n_layers*D)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        video_path: str,
        caption: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict fMRI responses for a video.

        Parameters
        ----------
        video_path : path to a video file (.mp4, .avi, etc.)
        caption : optional text description; if None, a dummy caption is used

        Returns
        -------
        np.ndarray
            Shape (n_segments, 20484), float32 — predicted fMRI responses on
            the fsaverage5 cortical mesh (5-second hemodynamic offset baked in).
        """
        from tribe_v2_mlx.preprocessing.video import load_video_frames
        from tribe_v2_mlx.preprocessing.audio import preprocess_audio
        from tribe_v2_mlx.preprocessing.video import extract_audio_from_video

        print(f"Loading video: {video_path}")
        # Use V-JEPA2 model's configured image size if available, else 256
        vjepa2_size = self.vjepa2.config.image_size if self.vjepa2 is not None else 256
        frames = load_video_frames(
            video_path, fps=_VIDEO_FPS, target_size=(vjepa2_size, vjepa2_size)
        )
        from tribe_v2_mlx.preprocessing.video import segment_video_frames
        n_segments = len(segment_video_frames(frames, _SEGMENT_DURATION, _VIDEO_FPS))

        # Extract features per modality
        video_feat, image_feat = self._extract_video_features(frames)

        # Audio
        try:
            waveform = extract_audio_from_video(video_path, sample_rate=_AUDIO_SR)
            waveform = preprocess_audio(waveform)
        except Exception:
            waveform = np.zeros(int(n_segments * _SEGMENT_DURATION * _AUDIO_SR), dtype=np.float32)
        audio_feat = self._extract_audio_features(waveform, n_segments)

        # Text
        if caption is None:
            caption = "A video clip."
        text_feat = self._extract_text_features(caption, n_segments)

        # Build feature dict for TRIBE
        features: Dict[str, mx.array] = {}
        if text_feat is not None:
            features["text"] = text_feat
        if video_feat is not None:
            features["video"] = video_feat
        if image_feat is not None:
            features["image"] = image_feat
        if audio_feat is not None:
            features["audio"] = audio_feat

        return self._run_tribe(features)

    def predict_image(
        self,
        image_path: str,
        caption: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict fMRI response for a static image.

        Returns
        -------
        np.ndarray
            Shape (1, 20484), float32.
        """
        from PIL import Image

        # Resize to the model's expected image size (default 224 for production models)
        target_size = self.dinov2.config.image_size if self.dinov2 is not None else 224
        img = Image.open(image_path).convert("RGB").resize((target_size, target_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        _mean = np.array([0.485, 0.456, 0.406])
        _std = np.array([0.229, 0.224, 0.225])
        arr = (arr - _mean) / _std  # (H, W, C)

        features: Dict[str, mx.array] = {}
        n_segments = 1

        if self.dinov2 is not None:
            x = mx.array(arr[None], dtype=mx.float32)  # (1, H, W, C)
            _, hidden = self.dinov2(x, extract_layers=self.dinov2.config.extract_layers)
            pooled = mx.concatenate([h.mean(axis=1) for h in hidden], axis=-1)
            features["image"] = pooled[:, None, :]  # (1, 1, D)

        if caption is not None and self.llama is not None:
            text_feat = self._extract_text_features(caption, n_segments=1)
            if text_feat is not None:
                features["text"] = text_feat

        return self._run_tribe(features)

    def _run_tribe(self, features: Dict[str, mx.array]) -> np.ndarray:
        """Run the TRIBE transformer and return predictions as numpy."""
        if not features:
            # Return zeros if no features available
            return np.zeros((1, self.tribe.config.n_outputs), dtype=np.float32)

        output = self.tribe(features, subject_id=self.subject_id)  # (1, T, n_outputs)
        mx.eval(output)
        result = np.array(output[0])  # (T, n_outputs)
        return result
