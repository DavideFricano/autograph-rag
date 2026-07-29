from __future__ import annotations

from abc import ABC, abstractmethod

from autograph_rag.storing.store import BaseStore
from autograph_rag.types import Chunk, ScoredChunk


class BaseIndex(ABC):
    """A retrieval index over chunks: insert, delete, retrieve.

    Keeps only ids plus its own representation (dense vectors, sparse/BM25 terms, …)
    and resolves chunk data back from a **shared** ``Store`` on retrieve, so retrieve
    returns ``list[ScoredChunk]``. Because the store is shared (single source of truth,
    no duplication), an index **never writes it**: adding and deleting *records* is
    coordinated by the ingestion pipeline (``store.add`` / ``store.delete`` once), while
    each index manages only its own ids — ``delete`` here removes this index's ids, not
    the shared records (a cascade from one index would strand the others). The pipeline
    does the fan-out: ``delete`` on every index, then a single ``store.delete``.

    Concrete indices implement ``insert``/``delete``/``_search`` with their backend; the
    ``db`` behind them stays hidden.
    """

    def __init__(self, store: BaseStore) -> None:
        self.store = store

    @abstractmethod
    def insert(self, chunks: list[Chunk]) -> None:
        """Index the chunks (id + vector/tokens + their source id); idempotent by chunk.id.
        Does not write the store — the pipeline adds the records once."""

    @abstractmethod
    def delete(self, source_id: str) -> None:
        """Remove from THIS index every id of the given source. Does not touch the shared
        store — the pipeline deletes the records once, after every index has removed its ids."""

    @abstractmethod
    def _search(self, query: str, top_i: int) -> list[tuple[str, float]]:
        """Backend primitive: top ``top_i`` ``(chunk_id, score)``, higher = more relevant.
        Ids + score only — ``retrieve`` resolves them to chunks through the shared store.
        Not called directly: the public entry point is ``retrieve``."""

    def retrieve(self, query: str, top_i: int) -> list[ScoredChunk]:
        scored = self._search(query, top_i)
        by_id = {chunk.id: chunk for chunk in self.store.get([id_ for id_, _ in scored])}
        return [ScoredChunk(chunk=by_id[id_], score=score) for id_, score in scored if id_ in by_id]
