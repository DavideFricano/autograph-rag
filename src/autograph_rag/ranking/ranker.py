from __future__ import annotations

from autograph_rag.types import ScoredChunk


class BaseRanker:
    """Shared utility for ranking operations."""

    @staticmethod
    def extract_top_k(scored_list: list[ScoredChunk], top_k: int | None) -> list[ScoredChunk]:
        """Returns the top_k highest-scored chunks, or all if top_k is None."""
        scored_list.sort(key=lambda x: x.score, reverse=True)
        return scored_list if top_k is None else scored_list[:top_k]
