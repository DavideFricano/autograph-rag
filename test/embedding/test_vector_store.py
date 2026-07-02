import numpy as np
import pytest

from autograph_rag.embedding.vector_store import FaissVectorStore


def test_search_raises_before_add():
    store = FaissVectorStore()
    with pytest.raises(RuntimeError):
        store.search(np.ones((1, 4), dtype=np.float32), top_k=1)

def test_identical_vector_gets_max_score():
    store = FaissVectorStore()
    vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    store.add(vec)
    _, scores = store.search(vec, top_k=1)
    assert abs(scores[0][0] - 1.0) < 1e-5

def test_search_returns_nearest_neighbour():
    store = FaissVectorStore()
    v0 = np.array([[1.0, 0.0]], dtype=np.float32)
    v1 = np.array([[0.0, 1.0]], dtype=np.float32)
    store.add(np.vstack([v0, v1]))
    indices, _ = store.search(v0, top_k=1)
    assert indices[0][0] == 0

def test_scores_in_zero_one():
    store = FaissVectorStore()
    store.add(np.random.default_rng(0).random((10, 8)).astype(np.float32))
    _, scores = store.search(np.random.default_rng(1).random((1, 8)).astype(np.float32), top_k=5)
    assert np.all((scores >= 0) & (scores <= 1))

def test_search_results_sorted_descending():
    store = FaissVectorStore()
    store.add(np.eye(5, dtype=np.float32))
    query = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    _, scores = store.search(query, top_k=5)
    assert list(scores[0]) == sorted(scores[0], reverse=True)
