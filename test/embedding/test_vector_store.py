from datetime import date

import numpy as np
import pytest

from autograph_rag.embedding.vector_store import (
    InMemoryVectorStore,
    PersistentVectorStore,
    RemoteVectorStore,
)
from autograph_rag.types import Chunk, Metadata, Origin, Source


def _chunks(n, texts=None):
    s = Source(id="doc1", name="doc.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return [
        Chunk(
            id=f"c{i}",
            text=(texts[i] if texts else f"Testo {i}"),
            metadata=Metadata(source=s, title="S"),
        )
        for i in range(n)
    ]


@pytest.fixture(params=["memory", "persistent", "remote"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryVectorStore()
    if request.param == "persistent":
        return PersistentVectorStore(path=str(tmp_path / "chroma"))
    return RemoteVectorStore()


def test_search_raises_before_add(store):
    with pytest.raises(RuntimeError):
        store.search(np.ones((1, 4), dtype=np.float32), top_k=1)


def test_returns_chunks_not_indices(store):
    chunks = _chunks(2)
    store.add(chunks, np.eye(2, dtype=np.float32))
    results = store.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=1)
    assert results[0].chunk.id == "c0"
    assert results[0].chunk.text == "Testo 0"


def test_scores_in_zero_one(store):
    store.add(_chunks(10), np.random.default_rng(0).random((10, 8)).astype(np.float32))
    results = store.search(np.random.default_rng(1).random((1, 8)).astype(np.float32), top_k=5)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_results_sorted_descending(store):
    store.add(_chunks(5), np.eye(5, dtype=np.float32))
    query = np.array([[1.0, 0.9, 0.0, 0.0, 0.0]], dtype=np.float32)
    results = store.search(query, top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_add_is_idempotent_on_repeated_ids(store):
    chunks = _chunks(3)
    emb = np.eye(3, dtype=np.float32)
    store.add(chunks, emb)
    store.add(chunks, emb)  # same ids again
    results = store.search(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), top_k=10)
    assert len({r.chunk.id for r in results}) == 3


def test_length_mismatch_raises(store):
    with pytest.raises(ValueError, match="mismatch"):
        store.add(_chunks(2), np.eye(3, dtype=np.float32))
