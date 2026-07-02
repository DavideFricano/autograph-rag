from datetime import date

from autograph_rag.retrieval.lexical_retriever import BM25Retriever
from autograph_rag.types import Chunk, Metadata, Source


def _chunks():
    s = Source(id="doc1", name="doc.pdf", time=date(2024, 1, 1))
    testi = [
        "Il paziente presenta febbre alta e tosse.",
        "La pressione arteriosa è nella norma.",
        "Febbre e brividi da tre giorni.",
    ]
    return [Chunk(id=f"c{i}", text=t, metadata=Metadata(source=s, title="S")) for i, t in enumerate(testi)]


def test_fever_query_returns_fever_chunks_first():
    results = BM25Retriever(_chunks()).retrieve("febbre", top_k=2)
    top_ids = {r.chunk.id for r in results}
    assert "c0" in top_ids  # "febbre alta"
    assert "c2" in top_ids  # "Febbre e brividi"

def test_pressure_query_returns_pressure_chunk_first():
    results = BM25Retriever(_chunks()).retrieve("pressione arteriosa", top_k=1)
    assert results[0].chunk.id == "c1"
    assert "pressione arteriosa" in results[0].chunk.text.lower()

def test_retrieve_sorted_by_score():
    results = BM25Retriever(_chunks()).retrieve("febbre", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

def test_tokenize_removes_stopwords():
    r = BM25Retriever(_chunks())
    tokens = r._tokenize("il paziente ha la febbre")
    assert "il" not in tokens and "la" not in tokens
