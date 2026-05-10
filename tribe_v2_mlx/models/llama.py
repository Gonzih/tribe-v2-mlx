"""
MLX LLaMA 3.2-3B wrapper for hidden state extraction.

Uses mlx_lm to load the model (4-bit quantised from mlx-community), then
accesses intermediate layer outputs by running the transformer blocks
manually rather than through the normal generate path.

TRIBE extracts 6 layers at relative depths [0, 0.2, 0.4, 0.6, 0.8, 1.0]
= layer indices [0, 6, 12, 18, 24, 28] for a 28-layer LLaMA 3.2-3B.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# LLaMA 3.2-3B has 28 transformer layers
_LLAMA_3B_DEPTH = 28
_DEFAULT_EXTRACT_DEPTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _depth_to_layer_index(depth: float, total_layers: int) -> int:
    """Convert a fractional depth (0.0–1.0) to a layer index."""
    return min(int(round(depth * (total_layers - 1))), total_layers - 1)


class MLXLlamaExtractor:
    """
    Wraps an mlx_lm LLaMA model for hidden state extraction.

    Usage
    -----
    extractor = MLXLlamaExtractor.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    hidden_states = extractor.extract(token_ids, layer_indices=[0, 6, 12, 18, 24, 27])
    # hidden_states: dict[int, np.ndarray] of shape (B, T, 3072)
    """

    def __init__(self, model, tokenizer, layer_indices: Optional[List[int]] = None):
        self._model = model
        self.tokenizer = tokenizer

        # Try to infer model depth
        try:
            self.num_layers = len(model.model.layers)
        except AttributeError:
            self.num_layers = _LLAMA_3B_DEPTH

        if layer_indices is None:
            layer_indices = [
                _depth_to_layer_index(d, self.num_layers)
                for d in _DEFAULT_EXTRACT_DEPTHS
            ]
        self.layer_indices = sorted(set(layer_indices))

    @classmethod
    def load(
        cls,
        model_path: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        layer_indices: Optional[List[int]] = None,
    ) -> "MLXLlamaExtractor":
        """Load model from HuggingFace or local path via mlx_lm."""
        import mlx_lm

        model, tokenizer = mlx_lm.load(model_path)
        return cls(model, tokenizer, layer_indices)

    def tokenize(self, text: str) -> mx.array:
        """Encode text to token ids, returns (1, T) int32 array."""
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        return mx.array(ids, dtype=mx.int32)[None, :]  # (1, T)

    def extract(
        self,
        token_ids: mx.array,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[int, mx.array]:
        """
        Run a forward pass and collect hidden states at specified layers.

        Parameters
        ----------
        token_ids : mx.array (B, T) int32
        layer_indices : which layers to extract; defaults to self.layer_indices

        Returns
        -------
        dict[layer_index -> mx.array (B, T, hidden_dim)]
        """
        if layer_indices is None:
            layer_indices = self.layer_indices
        layer_set = set(layer_indices)

        llama = self._model.model  # inner transformer model

        x = llama.embed_tokens(token_ids)  # (B, T, hidden_dim)

        hidden: Dict[int, mx.array] = {}
        cache = [None] * len(llama.layers)

        for i, layer in enumerate(llama.layers):
            # Handle different mlx_lm layer call signatures
            try:
                x = layer(x, mask=None, cache=cache[i])
                # Some versions return (output, new_cache)
                if isinstance(x, tuple):
                    x = x[0]
            except TypeError:
                x = layer(x)

            if i in layer_set:
                hidden[i] = x

        x = llama.norm(x)
        # Capture output after final norm as layer index = num_layers
        if len(llama.layers) in layer_set or (len(llama.layers) - 1) in layer_set:
            hidden[len(llama.layers) - 1] = x

        return hidden

    def extract_text_features(
        self,
        text: str,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Convenience: tokenise, extract hidden states, return as numpy.

        Returns
        -------
        dict[layer_index -> np.ndarray (1, T, hidden_dim)]
        """
        if layer_indices is None:
            layer_indices = self.layer_indices

        ids = self.tokenize(text)
        hidden = self.extract(ids, layer_indices=layer_indices)
        mx.eval(*hidden.values())
        return {k: np.array(v) for k, v in hidden.items()}
