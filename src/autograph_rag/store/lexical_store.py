from __future__ import annotations

import re
import sqlite3

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from rank_bm25 import BM25Okapi

from autograph_rag.store.base_store import BaseStore
from autograph_rag.types import Chunk, Language, ScoredChunk


class BaseLexicalStore(BaseStore):
    """Sparse store: keyword search over chunk text (no embedder needed).

    Sparse-side counterpart of BaseVectorStore. Owns the language-aware tokenizer
    (stemming + stopword removal) so every tier tokenizes index and query the same
    way. Two tiers, named by deployment role: InMemoryLexicalStore (PoC, BM25 in
    memory) and PersistentLexicalStore (local pilot, durable SQLite FTS5); a
    scalable tier (e.g. Elasticsearch/OpenSearch) would implement the same interface.
    """

    def __init__(self, language: Language = Language.ENGLISH) -> None:
        self.language = language
        nltk.download("stopwords", quiet=True)
        self.stemmer = SnowballStemmer(self.language)
        self.stopwords = set(stopwords.words(self.language))

    def _tokenize(self, text: str) -> list[str]:
        return [
            self.stemmer.stem(w)
            for w in re.findall(r"\b\w+\b", text.lower())
            if w not in self.stopwords
        ]


class InMemoryLexicalStore(BaseLexicalStore):
    """In-memory BM25Okapi index with language-aware stemming and stopword removal.

    BM25Okapi has no incremental update, so add() rebuilds the index over the full
    corpus — fine for the in-memory PoC (ingestion is a batch); the persistent tier
    indexes incrementally instead. A set of seen ids makes repeated ingestion idempotent.
    """

    def __init__(self, language: Language = Language.ENGLISH) -> None:
        super().__init__(language)
        self.chunks: list[Chunk] = []
        self._ids: set[str] = set()
        self.bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        new = [chunk for chunk in chunks if chunk.id not in self._ids]
        if not new:
            return
        self._ids.update(chunk.id for chunk in new)
        self.chunks.extend(new)
        self.bm25 = BM25Okapi([self._tokenize(chunk.text) for chunk in self.chunks])

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if self.bm25 is None:
            raise RuntimeError("InMemoryLexicalStore is empty")
        scores = self.bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [ScoredChunk(chunk=self.chunks[idx], score=float(scores[idx])) for idx in top_idx]

    def delete(self, source_id: str) -> None:
        kept = [chunk for chunk in self.chunks if chunk.metadata.source.id != source_id]
        if len(kept) == len(self.chunks):
            return
        self.chunks = kept
        self._ids = {chunk.id for chunk in kept}
        # Same rebuild as add(): BM25Okapi has no incremental delete either.
        self.bm25 = BM25Okapi([self._tokenize(c.text) for c in self.chunks]) if self.chunks else None


class PersistentLexicalStore(BaseLexicalStore):
    """Durable local store backed by SQLite FTS5 with native BM25 ranking.

    Persists to disk at `path`, single process, no server required (SQLite is in the
    stdlib). Unlike the in-memory tier it indexes incrementally: each chunk is inserted
    once and survives process restarts. Pass a custom sqlite connection to override
    storage (e.g. an in-memory ``:memory:`` connection in tests).

    Two tables keep dedup and search separate: a keyed `chunks` table (idempotent
    upsert by chunk.id, holding the serialized chunk) and a contentless FTS5 index
    addressed by the same rowid. Text is stemmed with the shared tokenizer before
    indexing, so FTS5 matches on the same tokens the in-memory tier would.
    """

    def __init__(
        self,
        language: Language = Language.ENGLISH,
        path: str = "./autograph_lexical.db",
        connection: sqlite3.Connection | None = None,
    ) -> None:
        super().__init__(language)
        self.conn = connection if connection is not None else sqlite3.connect(path)
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(body, content='')"
        )
        self.conn.commit()

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        cur = self.conn.cursor()
        for chunk in chunks:
            cur.execute(
                "INSERT OR IGNORE INTO chunks (id, data) VALUES (?, ?)",
                (chunk.id, chunk.model_dump_json()),
            )
            if cur.rowcount == 0:
                continue  # id already indexed -> idempotent
            body = " ".join(self._tokenize(chunk.text))
            cur.execute("INSERT INTO chunks_fts (rowid, body) VALUES (?, ?)", (cur.lastrowid, body))
        self.conn.commit()

    def delete(self, source_id: str) -> None:
        cur = self.conn.cursor()
        matches = [
            (rowid, id_, chunk)
            for rowid, id_, data in cur.execute("SELECT rowid, id, data FROM chunks").fetchall()
            if (chunk := Chunk.model_validate_json(data)).metadata.source.id == source_id
        ]
        if not matches:
            return
        for rowid, _id, chunk in matches:
            body = " ".join(self._tokenize(chunk.text))
            cur.execute(
                "INSERT INTO chunks_fts (chunks_fts, rowid, body) VALUES ('delete', ?, ?)",
                (rowid, body),
            )
        cur.executemany("DELETE FROM chunks WHERE id = ?", [(id_,) for _, id_, _ in matches])
        self.conn.commit()

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        # Quote each token as an FTS5 string literal so operators/punctuation can't
        # leak into the query grammar; OR-join for a bag-of-words match.
        match = " OR ".join(f'"{token}"' for token in tokens)
        rows = self.conn.execute(
            "SELECT c.data, bm25(chunks_fts) AS rank "
            "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, top_k),
        ).fetchall()
        # FTS5 bm25() is more negative = more relevant; negate so higher = more relevant.
        return [
            ScoredChunk(chunk=Chunk.model_validate_json(data), score=-float(rank))
            for data, rank in rows
        ]
