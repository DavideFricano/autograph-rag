from __future__ import annotations

from abc import abstractmethod

from autograph_rag.embedding.embedder import BaseEmbedder
from autograph_rag.embedding.vector_store import BaseVectorStore, FaissVectorStore
from autograph_rag.retrieval.retriever import BaseRetriever
from autograph_rag.types import Chunk, ScoredChunk


class VectorRetriever(BaseRetriever):
    """Retriever that uses dense embeddings and a vector store for similarity search."""

    def __init__(self, chunks: list[Chunk], embedder: BaseEmbedder, vector_store: BaseVectorStore) -> None:
        super().__init__(chunks)
        self.embedder = embedder
        self.vector_store = vector_store

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        pass


class FaissVectorRetriever(VectorRetriever):
    """Vector retriever backed by a FaissVectorStore."""

    def __init__(self, chunks: list[Chunk], embedder: BaseEmbedder, vector_store: FaissVectorStore) -> None:
        super().__init__(chunks, embedder, vector_store)

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        query_vec = self.embedder.embed_query(query)
        indices, scores = self.vector_store.search(query_vec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(ScoredChunk(chunk=chunk, score=float(score)))
        return results
