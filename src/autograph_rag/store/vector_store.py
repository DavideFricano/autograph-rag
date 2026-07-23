from __future__ import annotations

import uuid

import chromadb
import faiss
import numpy as np
from numpy.typing import NDArray
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
from autograph_rag.store.base_store import BaseStore
from autograph_rag.types import Chunk, ScoredChunk

# Fixed namespace so a chunk id always maps to the same point id (idempotent upsert).
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


class BaseVectorStore(BaseStore):
    """Vector store which owns an embedder and searches by cosine similarity."""

    def __init__(self, embedder: BaseEmbedder) -> None:
        self.embedder = embedder


def _normalize(emb: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return emb / norms


class InMemoryVectorStore(BaseVectorStore):
    """In-memory FAISS flat index with cosine similarity.

    Fast, zero setup, non-durable (data is lost when the process exits). Chunks
    are kept in a list aligned to the index rows; a set of seen ids makes
    repeated ingestion of identical chunks idempotent, including within an add.
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        super().__init__(embedder)
        self.index: faiss.Index | None = None
        self.dim: int | None = None
        self.chunks: list[Chunk] = []
        self._ids: set[str] = set()

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.embed_chunks([chunk.text for chunk in chunks])
        new_chunks: list[Chunk] = []
        new_rows: list[NDArray[np.float32]] = []
        for chunk, row in zip(chunks, embeddings, strict=True):
            if chunk.id in self._ids:
                continue
            self._ids.add(chunk.id)
            new_chunks.append(chunk)
            new_rows.append(row)
        if not new_chunks:
            return

        emb = _normalize(np.asarray(new_rows, dtype=np.float32))
        if self.index is None:
            self.dim = emb.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)
        self.chunks.extend(new_chunks)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if self.index is None:
            raise RuntimeError("InMemoryVectorStore index not initialized")
        query_emb = _normalize(self.embedder.embed_query(query))
        cosine_sim, indices = self.index.search(query_emb, top_k)
        results: list[ScoredChunk] = []
        for idx, sim in zip(indices[0], cosine_sim[0], strict=True):
            if idx == -1:
                continue
            score = (float(sim) + 1.0) / 2.0
            results.append(ScoredChunk(chunk=self.chunks[idx], score=score))
        return results

    def delete(self, source_id: str) -> None:
        if self.index is None:
            return
        keep = np.array([c.metadata.source.id != source_id for c in self.chunks])
        if keep.all():
            return
        # Reconstruct the stored vectors, drop the deleted rows, and rebuild the flat
        # index — FAISS remove_ids on a flat index swap-erases (breaks list alignment),
        # so a clean rebuild is simpler and correct for the in-memory PoC tier.
        vectors = self.index.reconstruct_n(0, self.index.ntotal)[keep]
        self.chunks = [c for c, k in zip(self.chunks, keep) if k]
        self._ids = {c.id for c in self.chunks}
        if len(vectors) == 0:
            self.index = None
            self.dim = None
            return
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)


class PersistentVectorStore(BaseVectorStore):
    """Durable local store backed by Chroma with cosine distance.

    Persists to disk under `path`, single process, no server required. Pass a
    custom chromadb client to override storage (e.g. an ephemeral client in
    tests). The collection is created on first add.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        path: str = "./autograph_store",
        collection: str = "autograph",
        client=None,
    ) -> None:
        super().__init__(embedder)
        if client is None:
            client = chromadb.PersistentClient(path=path)
        self.client = client
        self.collection_name = collection
        self.collection = None

    def _ensure_collection(self):
        if self.collection is None:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self.collection

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.embed_chunks([chunk.text for chunk in chunks])
        collection = self._ensure_collection()
        # Upsert keyed by chunk.id makes re-ingestion idempotent. The full chunk
        # is stored serialized because Chroma metadata must be flat scalars.
        collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings.tolist(),
            documents=[chunk.text for chunk in chunks],
            # The full chunk rides along serialized in _chunk (Chroma metadata must be
            # flat scalars, so the nested chunk can't be stored structurally).
            metadatas=[{"_chunk": chunk.model_dump_json()} for chunk in chunks],
        )

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        # Bind to the on-disk collection here too, so a query-only process (e.g. the
        # API) pointed at data a separate ingestion worker wrote can search without
        # ever calling add. An absent/empty collection yields no results, not an error.
        collection = self._ensure_collection()
        query_emb = self.embedder.embed_query(query)
        response = collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=top_k,
            include=["distances", "metadatas"],
        )
        results: list[ScoredChunk] = []
        for meta, dist in zip(response["metadatas"][0], response["distances"][0], strict=True):
            chunk = Chunk.model_validate_json(meta["_chunk"])
            score = 1.0 - float(dist) / 2.0
            results.append(ScoredChunk(chunk=chunk, score=score))
        return results

    def delete(self, source_id: str) -> None:
        # Chroma can't filter inside the serialized _chunk, so scan the stored chunks,
        # match on the nested source id, and delete the matching ids in one call.
        collection = self._ensure_collection()
        stored = collection.get(include=["metadatas"])
        ids = [
            id_
            for id_, meta in zip(stored["ids"], stored["metadatas"], strict=True)
            if Chunk.model_validate_json(meta["_chunk"]).metadata.source.id == source_id
        ]
        if ids:
            collection.delete(ids=ids)


class RemoteVectorStore(BaseVectorStore):
    """Scalable store backed by a Qdrant server (cosine distance).

    With url=None uses an in-memory instance (tests); pass url (e.g.
    "http://localhost:6333") plus any api_key for a running server. The
    collection is created lazily on the first add, using the dim.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        collection: str = "autograph",
        url: str | None = None,
        **client_kwargs,
    ) -> None:
        super().__init__(embedder)
        self.collection = collection
        if url is None:
            self.client = QdrantClient(location=":memory:", **client_kwargs)
        else:
            self.client = QdrantClient(url=url, **client_kwargs)
        self.dim: int | None = None

    def _point_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, chunk_id))

    def _ensure_collection(self, dim: int) -> None:
        if self.dim is not None:
            return
        self.dim = dim
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.embed_chunks([chunk.text for chunk in chunks])
        self._ensure_collection(embeddings.shape[1])
        points = [
            PointStruct(
                id=self._point_id(chunk.id),
                vector=row.tolist(),
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, row in zip(chunks, embeddings, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        # No dependence on a local add: a query-only process reads whatever the
        # server already holds. If the collection isn't there yet, return nothing.
        if not self.client.collection_exists(self.collection):
            return []
        query_emb = self.embedder.embed_query(query)
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_emb[0].tolist(),
            limit=top_k,
            with_payload=True,
        )
        results: list[ScoredChunk] = []
        for hit in response.points:
            chunk = Chunk.model_validate(hit.payload)
            score = (float(hit.score) + 1.0) / 2.0
            results.append(ScoredChunk(chunk=chunk, score=score))
        return results

    def delete(self, source_id: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        # The payload is the serialized chunk, so source id lives at the nested key
        # metadata.source.id; Qdrant filters directly on the dotted path.
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source.id", match=MatchValue(value=source_id)
                        )
                    ]
                )
            ),
        )
