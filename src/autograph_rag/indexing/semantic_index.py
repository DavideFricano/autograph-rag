from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from autograph_rag.embedding.embedder import BaseEmbedder
from autograph_rag.indexing.index import BaseIndex
from autograph_rag.storing.store import BaseStore
from autograph_rag.types import Chunk

# Fixed namespace so a chunk id always maps to the same point id (idempotent upsert).
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


class SemanticIndex(BaseIndex):
    """Shared dense-retrieval logic backed by Qdrant (cosine).

    Not used directly: the deployment-role classes below subclass it and differ only
    in how the client (the hidden ``db``) is built — in-memory, on disk, or a server —
    so behaviour is identical across tiers. Stores only ids + dense vectors (payload =
    chunk_id + source_id); the chunks live in the shared Store, so there is no
    duplication. The point id is a uuid5 of chunk.id → re-inserting a chunk upserts the
    same point.
    """

    def __init__(
        self, store: BaseStore, embedder: BaseEmbedder, db: QdrantClient, collection: str
    ) -> None:
        super().__init__(store)
        self.embedder = embedder
        self.db = db
        self.collection = collection
        self.dim: int | None = None

    def _point_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, chunk_id))

    def _ensure_collection(self, dim: int) -> None:
        if self.dim is not None:
            return
        self.dim = dim
        if not self.db.collection_exists(self.collection):
            self.db.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def insert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.embed_chunks([chunk.text for chunk in chunks])
        self._ensure_collection(embeddings.shape[1])
        points = [
            PointStruct(
                id=self._point_id(chunk.id),
                vector=row.tolist(),
                payload={"chunk_id": chunk.id, "source_id": chunk.metadata.source.id},
            )
            for chunk, row in zip(chunks, embeddings, strict=True)
        ]
        self.db.upsert(collection_name=self.collection, points=points)

    def delete(self, source_id: str) -> None:
        if not self.db.collection_exists(self.collection):
            return
        self.db.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))])
            ),
        )

    def _search(self, query: str, top_i: int) -> list[tuple[str, float]]:
        if not self.db.collection_exists(self.collection):
            return []
        query_emb = self.embedder.embed_query(query)
        response = self.db.query_points(
            collection_name=self.collection,
            query=query_emb[0].tolist(),
            limit=top_i,
            with_payload=True,
        )
        return [(hit.payload["chunk_id"], float(hit.score)) for hit in response.points]


class VolatileSemanticIndex(SemanticIndex):
    """Dense index in an in-memory Qdrant instance. Non-durable; zero setup."""

    def __init__(self, store: BaseStore, embedder: BaseEmbedder, collection: str = "semantic") -> None:
        super().__init__(store, embedder, QdrantClient(location=":memory:"), collection)


class PersistentSemanticIndex(SemanticIndex):
    """Dense index in an embedded Qdrant persisted under `path` (single process, no server)."""

    def __init__(
        self,
        store: BaseStore,
        embedder: BaseEmbedder,
        path: str = "./data/qdrant/semantic",
        collection: str = "semantic",
    ) -> None:
        super().__init__(store, embedder, QdrantClient(path=path), collection)


class RemoteSemanticIndex(SemanticIndex):
    """Dense index on a Qdrant server at `url` (shared across processes)."""

    def __init__(
        self,
        store: BaseStore,
        embedder: BaseEmbedder,
        url: str,
        collection: str = "semantic",
        **client_kwargs,
    ) -> None:
        super().__init__(store, embedder, QdrantClient(url=url, **client_kwargs), collection)
