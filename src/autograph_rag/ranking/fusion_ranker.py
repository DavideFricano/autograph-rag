from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from autograph_rag.ranking.ranker import BaseRanker
from autograph_rag.types import Chunk, ScoredChunk


class FusionRanker(BaseRanker, ABC):
    """Combines results from multiple retrievers into a single ranked list."""

    @abstractmethod
    def fuse(self, scores: np.ndarray) -> np.ndarray:
        """Reduces a (n_chunks, n_retrievers) score matrix to a (n_chunks,) fused score."""
        pass

    def rank(self, results: list[list[ScoredChunk]], top_k: int | None = None) -> list[ScoredChunk]:
        chunks, scores = self._align(results)
        fused = self.fuse(scores)
        scored = [ScoredChunk(chunk=c, score=float(s)) for c, s in zip(chunks, fused, strict=False)]
        return self.extract_top_k(scored, top_k)

    def _align(self, results: list[list[ScoredChunk]]) -> tuple[list[Chunk], np.ndarray]:
        """Builds a (n_chunks, n_retrievers) score matrix, filling 0 for missing pairs."""
        seen: dict[str, Chunk] = {}
        for result in results:
            for sc in result:
                if sc.chunk.id not in seen:
                    seen[sc.chunk.id] = sc.chunk

        chunks = list(seen.values())
        chunk_idx = {chunk_id: i for i, chunk_id in enumerate(seen)}

        scores = np.zeros((len(chunks), len(results)))
        for col, result in enumerate(results):
            for sc in result:
                scores[chunk_idx[sc.chunk.id], col] = sc.score

        return chunks, scores


class RelativeFusionRanker(FusionRanker):
    """Relative Score Fusion (RSF)."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weight = np.array(weights)

    def fuse(self, scores: np.ndarray) -> np.ndarray:
        dim = min(scores.shape[1], len(self.weight))
        return scores[:, :dim] @ self.weight[:dim]


class ReciprocalFusionRanker(FusionRanker):
    """Reciprocal Rank Fusion (RRF)."""

    def __init__(self, k: float = 60.0):
        self.k = k

    def fuse(self, scores: np.ndarray) -> np.ndarray:
        ranks = np.argsort(np.argsort(-scores, axis=0), axis=0) + 1
        return np.sum(1.0 / (self.k + ranks), axis=1)
