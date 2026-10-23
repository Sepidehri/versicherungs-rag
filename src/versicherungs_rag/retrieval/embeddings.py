"""Embedding models behind a common interface.

``FastEmbedEmbedder`` runs open-source sentence-embedding models locally via ONNX
(no PyTorch, no GPU needed). ``HashingEmbedder`` is a dependency-free fallback used in
tests and CI so the pipeline can run without downloading model weights.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

# E5 models were trained with these prefixes; leaving them out noticeably hurts retrieval.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


class Embedder(ABC):
    name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class FastEmbedEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)
        is_e5 = "e5" in model_name.lower()
        self._query_prefix = _E5_QUERY_PREFIX if is_e5 else ""
        self._passage_prefix = _E5_PASSAGE_PREFIX if is_e5 else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        inputs = [self._passage_prefix + t for t in texts]
        return _normalize(np.array(list(self._model.embed(inputs)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.array(next(iter(self._model.embed([self._query_prefix + text]))), dtype=np.float32)
        return _normalize(vector)


class HashingEmbedder(Embedder):
    """Bag of hashed character trigrams. Weak, but deterministic and instant."""

    name = "hashing"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                digest = hashlib.md5(padded[i : i + 3].encode()).digest()
                vector[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        return _normalize(vector)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


def build_embedder(model_name: str) -> Embedder:
    if model_name == HashingEmbedder.name:
        return HashingEmbedder()
    return FastEmbedEmbedder(model_name)
