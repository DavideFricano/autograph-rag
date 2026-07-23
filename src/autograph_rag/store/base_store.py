from __future__ import annotations

from abc import ABC, abstractmethod

from autograph_rag.types import Chunk, ScoredChunk


class BaseStore(ABC):
    """A retrieval store for one modality: indexes chunks and searches them by text.

    Both the dense (vector) and sparse (lexical) stores implement this, so the
    pipelines can treat any mix of them uniformly as a list. A vector store owns
    its embedder, so ``search`` takes plain text just like the lexical one — there
    is no separate retriever layer.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Idempotent upsert keyed by chunk.id."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Returns the top_k most relevant chunks with scores, higher = more relevant."""

    @abstractmethod
    def delete(self, source_id: str) -> None:
        """Remove every chunk belonging to the given source document.

        Keyed by ``chunk.metadata.source.id``. Idempotent: deleting a source that
        isn't indexed is a no-op. Lets a periodic re-ingestion keep the corpus in
        sync (drop a source's chunks before re-adding) instead of only appending.
        """
