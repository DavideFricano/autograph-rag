from __future__ import annotations

from abc import abstractmethod

from autograph_rag.retrieval.retriever import BaseRetriever
from autograph_rag.types import Chunk, ScoredChunk


class GraphRetriever(BaseRetriever):
    """Base class for graph-based retrieval strategies (e.g. knowledge graphs)."""

    def __init__(self, chunks: list[Chunk]) -> None:
        super().__init__(chunks)

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        pass
