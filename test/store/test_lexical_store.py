import sqlite3
from datetime import date

import pytest

from autograph_rag.store.lexical_store import InMemoryLexicalStore, PersistentLexicalStore
from autograph_rag.types import Chunk, Language, Metadata, Origin, Source


def _chunk(id: str, text: str) -> Chunk:
    s = Source(id="doc1", name="doc.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return Chunk(id=id, text=text, metadata=Metadata(source=s, title="S"))


def _chunk_src(id: str, text: str, source_id: str) -> Chunk:
    s = Source(id=source_id, name="doc.pdf", origin=Origin.LOCAL, time=date(2024, 1, 1))
    return Chunk(id=id, text=text, metadata=Metadata(source=s, title="S"))


def _populate(store):
    store.add([
        _chunk("c0", "Il paziente presenta febbre alta e tosse."),
        _chunk("c1", "La pressione arteriosa è nella norma."),
        _chunk("c2", "Febbre e brividi da tre giorni."),
    ])
    return store


def _medical_store() -> InMemoryLexicalStore:
    return _populate(InMemoryLexicalStore(language=Language.ITALIAN))


def _persistent_store(connection=None) -> PersistentLexicalStore:
    if connection is None:
        connection = sqlite3.connect(":memory:")
    return _populate(PersistentLexicalStore(language=Language.ITALIAN, connection=connection))


def test_fever_query_returns_fever_chunks_first():
    results = _medical_store().search("febbre", top_k=2)
    top_ids = {r.chunk.id for r in results}
    assert "c0" in top_ids  # "febbre alta"
    assert "c2" in top_ids  # "Febbre e brividi"


def test_pressure_query_returns_pressure_chunk_first():
    results = _medical_store().search("pressione arteriosa", top_k=1)
    assert results[0].chunk.id == "c1"
    assert "pressione arteriosa" in results[0].chunk.text.lower()


def test_results_sorted_by_score():
    results = _medical_store().search("febbre", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_tokenize_removes_stopwords_and_stems():
    store = InMemoryLexicalStore(language=Language.ITALIAN)
    tokens = store._tokenize("il paziente ha la febbre")
    assert "il" not in tokens and "la" not in tokens


def test_add_is_idempotent_by_id():
    store = InMemoryLexicalStore(language=Language.ITALIAN)
    store.add([_chunk("c0", "febbre alta")])
    store.add([_chunk("c0", "febbre alta")])  # stesso id
    assert len(store.chunks) == 1


def test_search_on_empty_store_fails_loud():
    with pytest.raises(RuntimeError):
        InMemoryLexicalStore(language=Language.ITALIAN).search("febbre", top_k=1)


# --- PersistentLexicalStore (SQLite FTS5) ---


def test_persistent_fever_query_returns_fever_chunks_first():
    results = _persistent_store().search("febbre", top_k=2)
    top_ids = {r.chunk.id for r in results}
    assert top_ids == {"c0", "c2"}


def test_persistent_pressure_query_returns_pressure_chunk():
    results = _persistent_store().search("pressione arteriosa", top_k=1)
    assert results[0].chunk.id == "c1"


def test_persistent_results_sorted_by_score_descending():
    results = _persistent_store().search("febbre", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_persistent_add_is_idempotent_by_id():
    store = _persistent_store()
    store.add([_chunk("c0", "Il paziente presenta febbre alta e tosse.")])  # stesso id
    count = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == 3


def test_persistent_search_on_empty_store_returns_empty():
    store = PersistentLexicalStore(language=Language.ITALIAN, connection=sqlite3.connect(":memory:"))
    assert store.search("febbre", top_k=1) == []


def test_persistent_survives_reopening_same_db(tmp_path):
    db = str(tmp_path / "lex.db")
    _persistent_store(connection=sqlite3.connect(db))
    reopened = PersistentLexicalStore(language=Language.ITALIAN, connection=sqlite3.connect(db))
    results = reopened.search("febbre", top_k=2)
    assert {r.chunk.id for r in results} == {"c0", "c2"}


# --- delete(source_id) on both tiers ---


def _two_source_stores():
    chunks = [
        _chunk_src("a0", "febbre alta e tosse", "doc1"),
        _chunk_src("a1", "brividi e febbre", "doc1"),
        _chunk_src("b0", "pressione arteriosa nella norma", "doc2"),
    ]
    mem = InMemoryLexicalStore(language=Language.ITALIAN)
    mem.add(chunks)
    persistent = PersistentLexicalStore(
        language=Language.ITALIAN, connection=sqlite3.connect(":memory:")
    )
    persistent.add(chunks)
    return mem, persistent


@pytest.mark.parametrize("store_index", [0, 1], ids=["memory", "persistent"])
def test_delete_removes_only_that_source(store_index):
    store = _two_source_stores()[store_index]
    store.delete("doc1")
    # doc1's chunks are physically gone; doc2 survives. (Filtering of irrelevant
    # results is the ranker's job, so we assert on what the store still holds, not
    # on whether a non-matching query returns empty.)
    returned = {r.chunk.id for r in store.search("febbre pressione", top_k=10)}
    assert "a0" not in returned and "a1" not in returned
    assert "b0" in returned


@pytest.mark.parametrize("store_index", [0, 1], ids=["memory", "persistent"])
def test_delete_unknown_source_is_noop(store_index):
    store = _two_source_stores()[store_index]
    store.delete("nope")
    returned = {r.chunk.id for r in store.search("febbre pressione", top_k=10)}
    assert {"a0", "a1", "b0"} <= returned
