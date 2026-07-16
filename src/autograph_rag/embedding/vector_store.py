from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

import chromadb
import faiss
import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from autograph_rag.types import Chunk, ScoredChunk

# Fixed namespace so a chunk id always maps to the same point id (idempotent upsert).
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


class BaseVectorStore(ABC):
    """Stores chunks together with their dense embeddings and searches them.

    Backends own both the vectors and the chunks, so search returns chunks
    directly. Embeddings are always computed client-side and passed in.

    Three tiers implement this interface, named by deployment role rather than
    technology: InMemoryVectorStore (PoC), PersistentVectorStore (local pilot),
    RemoteVectorStore (scalable production).
    """

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: NDArray[np.float32]) -> None:
        """Idempotent upsert keyed by chunk.id."""

    @abstractmethod
    def search(self, query_emb: NDArray[np.float32], top_k: int) -> list[ScoredChunk]:
        """Returns chunks with scores in [0, 1], higher = more similar."""


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

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.dim: int | None = None
        self.chunks: list[Chunk] = []
        self._ids: set[str] = set()

    def add(self, chunks: list[Chunk], embeddings: NDArray[np.float32]) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) length mismatch"
            )
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

    def search(self, query_emb: NDArray[np.float32], top_k: int) -> list[ScoredChunk]:
        if self.index is None:
            raise RuntimeError("InMemoryVectorStore index not initialized")
        query_emb = _normalize(query_emb)
        cosine_sim, indices = self.index.search(query_emb, top_k)
        results: list[ScoredChunk] = []
        for idx, sim in zip(indices[0], cosine_sim[0], strict=True):
            if idx == -1:
                continue
            score = (float(sim) + 1.0) / 2.0
            results.append(ScoredChunk(chunk=self.chunks[idx], score=score))
        return results


class PersistentVectorStore(BaseVectorStore):
    """Durable local store backed by Chroma with cosine distance.

    Persists to disk under `path`, single process, no server required. Embeddings
    are computed client-side. Pass a custom chromadb client to override storage
    (e.g. an ephemeral client in tests). The collection is created on first add.
    """

    def __init__(
        self,
        path: str = "./autograph_store",
        collection: str = "autograph",
        client=None,
    ) -> None:
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

    def add(self, chunks: list[Chunk], embeddings: NDArray[np.float32]) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) length mismatch"
            )
        if not chunks:
            return
        collection = self._ensure_collection()
        # Upsert keyed by chunk.id makes re-ingestion idempotent. The full chunk
        # is stored serialized because Chroma metadata must be flat scalars.
        collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings.tolist(),
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"_chunk": chunk.model_dump_json()} for chunk in chunks],
        )

    def search(self, query_emb: NDArray[np.float32], top_k: int) -> list[ScoredChunk]:
        if self.collection is None:
            raise RuntimeError("PersistentVectorStore collection not initialized")
        response = self.collection.query(
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


class RemoteVectorStore(BaseVectorStore):
    """Scalable store backed by a Qdrant server (cosine distance).

    Embeddings are computed client-side. With url=None uses an in-memory instance
    (tests); pass url (e.g. "http://localhost:6333") plus any api_key for a running
    server. The collection is created lazily on the first add, using the dim.
    """

    def __init__(
        self,
        collection: str = "autograph",
        url: str | None = None,
        **client_kwargs,
    ) -> None:
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

    def add(self, chunks: list[Chunk], embeddings: NDArray[np.float32]) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) length mismatch"
            )
        if not chunks:
            return
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

    def search(self, query_emb: NDArray[np.float32], top_k: int) -> list[ScoredChunk]:
        if self.dim is None:
            raise RuntimeError("RemoteVectorStore collection not initialized")
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
