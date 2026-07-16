from __future__ import annotations

from autograph_rag.embedding.embedder import BaseEmbedder
from autograph_rag.embedding.vector_store import BaseVectorStore
from autograph_rag.retrieval.retriever import BaseRetriever
from autograph_rag.types import ScoredChunk


class VectorRetriever(BaseRetriever):
    """Dense retriever: embeds the query and delegates search to the vector store.

    Backend-agnostic — the store owns the chunks and returns them directly,
    so this works with any BaseVectorStore (FAISS, Qdrant, ...).
    """

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        query_vec = self.embedder.embed_query(query)
        return self.vector_store.search(query_vec, top_k)
