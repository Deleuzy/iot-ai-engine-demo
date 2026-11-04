from __future__ import annotations

import numpy as np

from iot_ai.ai.embed import HashingEmbedder
from iot_ai.ai.store import VectorStore


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    e = HashingEmbedder()
    a = e.embed(["battery voltage falling"])
    b = e.embed(["battery voltage falling"])
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-9


def test_similar_text_scores_higher() -> None:
    e = HashingEmbedder()
    store = VectorStore(e.dim)
    store.add(["battery", "clock"], e.embed(["battery voltage falling fast", "timestamps jump backwards"]))
    top = store.search(e.embed(["voltage of the battery is falling"])[0], 2)
    assert top[0].id == "battery"


def test_store_rejects_duplicates() -> None:
    e = HashingEmbedder()
    store = VectorStore(e.dim)
    store.add(["a"], e.embed(["x"]))
    try:
        store.add(["a"], e.embed(["y"]))
    except ValueError:
        return
    raise AssertionError("duplicate id accepted")
