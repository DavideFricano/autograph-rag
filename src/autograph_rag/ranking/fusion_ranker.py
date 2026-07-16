from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from autograph_rag.ranking.ranker import BaseRanker
from autograph_rag.types import Chunk, ScoredChunk


class FusionRanker(BaseRanker, ABC):
    """Combines results from multiple retrievers into a single ranked list."""

    @abstractmethod
    def fuse(self, scores: np.ndarray) -> np.ndarray:
        """Reduces a (n_chunks, n_retrievers) score matrix to a (n_chunks) fused score."""
        pass

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

    def rank(self, results: list[list[ScoredChunk]], top_k: int | None = None) -> list[ScoredChunk]:
        chunks, scores = self._align(results)
        fused = self.fuse(scores)
        scored = [ScoredChunk(chunk=c, score=float(s)) for c, s in zip(chunks, fused, strict=True)]
        return self.extract_top_k(scored, top_k)
    

class RelativeFusionRanker(FusionRanker):
    """Relative Score Fusion (RSF).

    Min-max normalizes each retriever's scores to [0, 1] before the weighted
    combination, so retrievers with different score scales (e.g. bounded
    cosine similarity vs. unbounded BM25) contribute comparably.
    """

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=float)

    def fuse(self, scores: np.ndarray) -> np.ndarray:
        if scores.shape[1] != len(self.weights):
            raise ValueError(f"Number of weights ({len(self.weights)}) does not match number of retrievers ({scores.shape[1]})")
        col_min = scores.min(axis=0, keepdims=True)
        col_max = scores.max(axis=0, keepdims=True)
        span = np.where(col_max == col_min, 1.0, col_max - col_min)
        scores = (scores - col_min) / span
        return scores @ self.weights


class ReciprocalFusionRanker(FusionRanker):
    """Reciprocal Rank Fusion (RRF)."""

    def __init__(self, k: float = 60.0):
        self.k = k

    def fuse(self, scores: np.ndarray) -> np.ndarray:
        ranks = np.argsort(np.argsort(-scores, axis=0), axis=0) + 1
        return np.sum(1.0 / (self.k + ranks), axis=1)
