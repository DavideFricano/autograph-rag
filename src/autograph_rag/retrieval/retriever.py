from __future__ import annotations

from abc import ABC, abstractmethod

from autograph_rag.types import Chunk, ScoredChunk


class BaseRetriever(ABC):
    """Base class for all retrieval strategies."""

    def __init__(self, chunks: list[Chunk], **kwargs) -> None:
        self.chunks = chunks

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        pass
