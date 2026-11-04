"""Embeddings. HashingEmbedder is deterministic and needs no key (used in tests and CI).

OpenAICompatEmbedder calls any /embeddings endpoint that speaks the OpenAI format (Mistral, vLLM, OpenAI).
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Any, Protocol

import httpx
import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Word and word-pair hashing into a fixed number of buckets, L2-normalised.

    crc32 is used rather than Python's hash(), which changes between processes.
    """

    name = "hashing-512"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            words = _WORD.findall(text.lower())
            grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
            for gram in grams:
                bucket = zlib.crc32(gram.encode("utf-8")) % self.dim
                sign = 1.0 if zlib.crc32(("s" + gram).encode("utf-8")) % 2 == 0 else -1.0
                out[row, bucket] += sign
            norm = math.sqrt(float(np.dot(out[row], out[row])))
            if norm > 0:
                out[row] /= norm
        return out


class OpenAICompatEmbedder:
    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str, transport: httpx.BaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
            transport=transport,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.post("/embeddings", json={"model": self.model, "input": texts})
        response.raise_for_status()
        data: list[dict[str, Any]] = response.json()["data"]
        vectors = np.array([item["embedding"] for item in data], dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)
