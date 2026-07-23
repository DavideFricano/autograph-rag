from datetime import date

import numpy as np
import pytest

from autograph_rag.embedding.embedder import BaseEmbedder
from autograph_rag.store.vector_store import (
    InMemoryVectorStore,
    PersistentVectorStore,
    RemoteVectorStore,
)
from autograph_rag.types import Chunk, Metadata, Origin, Source


class _FakeEmbedder(BaseEmbedder):
    """Maps exact text -> preset vector (zeros for unknown), so tests control geometry."""

    def __init__(self, table: dict[str, list[float]], dim: int) -> None:
        self._table = {t: np.asarray(v, dtype=np.float32) for t, v in table.items()}
        self._dim = dim

    def _vec(self, text: str) -> np.ndarray:
        return self._table.get(text, np.zeros(self._dim, dtype=np.float32))

    def embed_chunks(self, chunks):
        return np.array([self._vec(t) for t in chunks], dtype=np.float32)

    def embed_query(self, query):
        return np.array([self._vec(query)], dtype=np.float32)


def _chunks(n: int) -> list[Chunk]:
    s = Source(id="doc1", name="doc.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return [Chunk(id=f"c{i}", text=f"Testo {i}", metadata=Metadata(source=s, title="S")) for i in range(n)]


@pytest.fixture(params=["memory", "persistent", "remote"])
def store_factory(request, tmp_path):
    def make(embedder: BaseEmbedder):
        if request.param == "memory":
            return InMemoryVectorStore(embedder)
        if request.param == "persistent":
            return PersistentVectorStore(embedder, path=str(tmp_path / "chroma"))
        return RemoteVectorStore(embedder)

    return make


def test_memory_search_raises_before_add():
    # In-memory persists nothing: searching before any add is misuse, so fail loud.
    store = InMemoryVectorStore(_FakeEmbedder({}, dim=4))
    with pytest.raises(RuntimeError):
        store.search("anything", top_k=1)


def test_persistent_search_before_add_returns_empty(tmp_path):
    store = PersistentVectorStore(_FakeEmbedder({}, dim=4), path=str(tmp_path / "chroma"))
    assert store.search("anything", top_k=1) == []


def test_remote_search_before_add_returns_empty():
    assert RemoteVectorStore(_FakeEmbedder({}, dim=4)).search("anything", top_k=1) == []


def test_persistent_query_only_reads_existing(tmp_path):
    # A query-only process (the API) must read what a separate writer (the ingestion
    # worker) persisted, without ever calling add itself.
    path = str(tmp_path / "chroma")
    table = {"Testo 0": [1.0, 0.0], "q": [1.0, 0.0]}
    PersistentVectorStore(_FakeEmbedder(table, dim=2), path=path).add(_chunks(1))
    reader = PersistentVectorStore(_FakeEmbedder(table, dim=2), path=path)
    results = reader.search("q", top_k=1)
    assert results[0].chunk.id == "c0"


def test_returns_chunks_not_indices(store_factory):
    chunks = _chunks(2)
    store = store_factory(_FakeEmbedder({"Testo 0": [1.0, 0.0], "Testo 1": [0.0, 1.0], "q": [1.0, 0.0]}, dim=2))
    store.add(chunks)
    results = store.search("q", top_k=1)
    assert results[0].chunk.id == "c0"
    assert results[0].chunk.text == "Testo 0"


def test_scores_in_zero_one(store_factory):
    rng = np.random.default_rng(0)
    chunks = _chunks(10)
    table = {c.text: rng.random(8) for c in chunks}
    table["q"] = rng.random(8)
    store = store_factory(_FakeEmbedder(table, dim=8))
    store.add(chunks)
    results = store.search("q", top_k=5)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_results_sorted_descending(store_factory):
    chunks = _chunks(5)
    table = {f"Testo {i}": np.eye(5)[i] for i in range(5)}
    table["q"] = [1.0, 0.9, 0.0, 0.0, 0.0]
    store = store_factory(_FakeEmbedder(table, dim=5))
    store.add(chunks)
    results = store.search("q", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def _two_source_chunks() -> list[Chunk]:
    s1 = Source(id="doc1", name="a.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    s2 = Source(id="doc2", name="b.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return [
        Chunk(id="a0", text="t0", metadata=Metadata(source=s1, title="S")),
        Chunk(id="a1", text="t1", metadata=Metadata(source=s1, title="S")),
        Chunk(id="b0", text="t2", metadata=Metadata(source=s2, title="S")),
    ]


def test_delete_removes_only_that_source(store_factory):
    table = {"t0": [1.0, 0.0, 0.0], "t1": [0.0, 1.0, 0.0], "t2": [0.0, 0.0, 1.0], "q": [0.0, 0.0, 1.0]}
    store = store_factory(_FakeEmbedder(table, dim=3))
    store.add(_two_source_chunks())
    store.delete("doc1")
    results = store.search("q", top_k=10)
    assert {r.chunk.id for r in results} == {"b0"}


def test_delete_unknown_source_is_noop(store_factory):
    table = {"t0": [1.0, 0.0, 0.0], "t1": [0.0, 1.0, 0.0], "t2": [0.0, 0.0, 1.0], "q": [0.0, 0.0, 1.0]}
    store = store_factory(_FakeEmbedder(table, dim=3))
    store.add(_two_source_chunks())
    store.delete("nope")  # no such source -> unchanged
    assert len({r.chunk.id for r in store.search("q", top_k=10)}) == 3


def test_add_is_idempotent_on_repeated_ids(store_factory):
    chunks = _chunks(3)
    table = {f"Testo {i}": np.eye(3)[i] for i in range(3)}
    table["q"] = [1.0, 0.0, 0.0]
    store = store_factory(_FakeEmbedder(table, dim=3))
    store.add(chunks)
    store.add(chunks)  # same ids again
    results = store.search("q", top_k=10)
    assert len({r.chunk.id for r in results}) == 3
