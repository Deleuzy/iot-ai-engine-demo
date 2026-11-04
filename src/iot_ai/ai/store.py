"""A small in-memory vector store: cosine similarity over a numpy matrix.

It is deliberately simple. The corpus is a few dozen passages, so an exact search is correct and fast.
A production system would use a managed vector database; this one is not used for scale claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Hit:
    id: str
    score: float


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._matrix = np.zeros((0, dim), dtype=np.float64)

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if vectors.shape != (len(ids), self.dim):
            raise ValueError(f"expected {(len(ids), self.dim)}, got {vectors.shape}")
        if set(ids) & set(self._ids):
            raise ValueError("duplicate ids")
        self._ids.extend(ids)
        self._matrix = np.vstack([self._matrix, vectors])

    def search(self, query: np.ndarray, k: int) -> list[Hit]:
        if len(self._ids) == 0:
            return []
        scores = self._matrix @ query.reshape(-1)
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self._ids[i], float(scores[i])) for i in order]
