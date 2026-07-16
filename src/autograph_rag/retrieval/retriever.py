from __future__ import annotations

from abc import ABC, abstractmethod

from autograph_rag.types import ScoredChunk


class BaseRetriever(ABC):
    """Base interface for all retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        pass
