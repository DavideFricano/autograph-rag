from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

from qdrant_client import QdrantClient, models

from autograph_rag.storing.store import BaseStore
from autograph_rag.types import Chunk, ScoredChunk

# Fixed namespace so a chunk id always maps to the same point id (idempotent upsert).
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


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
        Ids are unique — a chunk appears at most once, or fusion would count it twice.
        Not called directly: the public entry point is ``retrieve``."""

    def retrieve(self, query: str, top_i: int) -> list[ScoredChunk]:
        scored = self._search(query, top_i)
        by_id = {chunk.id: chunk for chunk in self.store.get([id_ for id_, _ in scored])}
        return [ScoredChunk(chunk=by_id[id_], score=score) for id_, score in scored if id_ in by_id]


class QdrantIndex(BaseIndex, ABC):
    """Shared plumbing for the indexes that retrieve by vector, dense or sparse.

    The engine behind them is an implementation detail and never appears in the public
    signatures: subclasses own only the collection setup and the query value, while point
    ids, payload, deletion and the search call are shared so they cannot drift apart.
    """

    def __init__(self, store: BaseStore, db: QdrantClient, collection: str) -> None:
        super().__init__(store)
        self.db = db
        self.collection = collection

    @abstractmethod
    def _query_vector(self, query: str) -> tuple[Any, str | None]:
        """The value to search with and the named vector to use (None = the unnamed one)."""

    def _point(self, chunk: Chunk, vector: Any) -> models.PointStruct:
        """One point per chunk: the id is a uuid5 of ``chunk.id`` so re-inserting upserts
        in place, and the payload carries only what this index must act on without reading
        the store — the id it resolves chunks by, the source it deletes by."""
        return models.PointStruct(
            id=str(uuid.uuid5(_NAMESPACE, chunk.id)),
            vector=vector,
            payload={"chunk_id": chunk.id, "source_id": chunk.metadata.source.id},
        )

    def delete(self, source_id: str) -> None:
        if not self.db.collection_exists(self.collection):
            return
        self.db.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id", match=models.MatchValue(value=source_id)
                        )
                    ]
                )
            ),
        )

    def _search(self, query: str, top_i: int) -> list[tuple[str, float]]:
        if not self.db.collection_exists(self.collection):
            return []
        value, using = self._query_vector(query)
        response = self.db.query_points(
            collection_name=self.collection,
            query=value,
            using=using,
            limit=top_i,
            with_payload=True,
        )
        return [(hit.payload["chunk_id"], float(hit.score)) for hit in response.points]
