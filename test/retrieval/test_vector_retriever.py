from datetime import date

import numpy as np

from autograph_rag.embedding.vector_store import InMemoryVectorStore
from autograph_rag.retrieval.vector_retriever import VectorRetriever
from autograph_rag.types import Chunk, Metadata, Origin, Source


class _FakeEmbedder:
    """Returns a unit vector along dim 0 — matches chunk c0 stored as eye(3)[0]."""

    def embed_query(self, query):
        v = np.zeros((1, 3), dtype=np.float32)
        v[0, 0] = 1.0
        return v


def _chunks():
    s = Source(id="doc1", name="doc.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return [
        Chunk(id=f"c{i}", text=f"Testo {i}", metadata=Metadata(source=s, title="S"))
        for i in range(3)
    ]


def _retriever():
    store = InMemoryVectorStore()
    store.add(_chunks(), np.eye(3, dtype=np.float32))
    return VectorRetriever(_FakeEmbedder(), store)


def test_retrieve_returns_most_similar_chunk_first():
    results = _retriever().retrieve("query", top_k=3)
    assert results[0].chunk.id == "c0"


def test_retrieve_correct_count():
    assert len(_retriever().retrieve("query", top_k=2)) == 2


def test_retrieve_scores_sorted_descending():
    results = _retriever().retrieve("query", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_chunk_text_preserved():
    results = _retriever().retrieve("query", top_k=1)
    assert results[0].chunk.text == "Testo 0"
