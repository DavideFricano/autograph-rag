from __future__ import annotations

import uuid

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
)

from autograph_rag.indexing.index import BaseIndex
from autograph_rag.storing.store import BaseStore
from autograph_rag.types import Chunk, Language

_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


class LexicalIndex(BaseIndex):
    """Shared sparse (BM25) retrieval backed by Qdrant.

    Tokenizing, stemming and stopwords come from FastEmbed's ``Qdrant/bm25`` (language
    aware); the corpus IDF is applied server-side by Qdrant's IDF modifier — so it is
    real BM25, identical across tiers. Stores only ids + the sparse vector (payload =
    chunk_id + source_id); chunks live in the shared Store. Role subclasses differ only
    in how the ``db`` client is built.
    """

    def __init__(
        self,
        store: BaseStore,
        db: QdrantClient,
        collection: str = "lexical",
        language: Language = Language.ENGLISH,
    ) -> None:
        super().__init__(store)
        self.db = db
        self.collection = collection
        self.model = SparseTextEmbedding("Qdrant/bm25", language=str(language))
        self._ready = False

    def _ensure_collection(self) -> None:
        if self._ready:
            return
        self._ready = True
        if not self.db.collection_exists(self.collection):
            self.db.create_collection(
                collection_name=self.collection,
                vectors_config={},
                sparse_vectors_config={"text": SparseVectorParams(modifier=Modifier.IDF)},
            )

    def insert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._ensure_collection()
        embeddings = self.model.embed([chunk.text for chunk in chunks])
        points = [
            PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, chunk.id)),
                vector={"text": SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())},
                payload={"chunk_id": chunk.id, "source_id": chunk.metadata.source.id},
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
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
        emb = next(iter(self.model.query_embed(query)))
        response = self.db.query_points(
            collection_name=self.collection,
            query=SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist()),
            using="text",
            limit=top_i,
            with_payload=True,
        )
        return [(hit.payload["chunk_id"], float(hit.score)) for hit in response.points]


class VolatileLexicalIndex(LexicalIndex):
    """Sparse BM25 index in an in-memory Qdrant instance. Non-durable; zero setup."""

    def __init__(
        self, store: BaseStore, collection: str = "lexical", language: Language = Language.ENGLISH
    ) -> None:
        super().__init__(store, QdrantClient(location=":memory:"), collection, language)


class PersistentLexicalIndex(LexicalIndex):
    """Sparse BM25 index in an embedded Qdrant persisted under `path` (single process)."""

    def __init__(
        self,
        store: BaseStore,
        path: str = "./data/qdrant/lexical",
        collection: str = "lexical",
        language: Language = Language.ENGLISH,
    ) -> None:
        super().__init__(store, QdrantClient(path=path), collection, language)


class RemoteLexicalIndex(LexicalIndex):
    """Sparse BM25 index on a Qdrant server at `url` (shared across processes)."""

    def __init__(
        self,
        store: BaseStore,
        url: str,
        collection: str = "lexical",
        language: Language = Language.ENGLISH,
        **client_kwargs,
    ) -> None:
        super().__init__(store, QdrantClient(url=url, **client_kwargs), collection, language)
