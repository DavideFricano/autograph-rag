from datetime import date

import numpy as np
import pytest

from autograph_rag.ranking.fusion_ranker import ReciprocalFusionRanker, RelativeFusionRanker
from autograph_rag.types import Chunk, Metadata, ScoredChunk, Source


def _sc(chunk_id, score):
    s = Source(id="doc1", name="doc.pdf", time=date(2024, 1, 1))
    c = Chunk(id=chunk_id, text="T", metadata=Metadata(source=s, title="S"))
    return ScoredChunk(chunk=c, score=score)


def test_reciprocal_fusion_deduplicates():
    ranker = ReciprocalFusionRanker()
    shared = _sc("c0", 0.9)
    ranked = ranker.rank([[shared, _sc("c1", 0.5)], [shared, _sc("c2", 0.4)]])
    ids = [r.chunk.id for r in ranked]
    assert len(ids) == len(set(ids))


def test_rank_top_k():
    ranker = RelativeFusionRanker(weights=np.array([1.0]))
    ranked = ranker.rank([[_sc(f"c{i}", float(i)) for i in range(5)]], top_k=2)
    assert len(ranked) == 2


def test_relative_fusion_weights_change_winner():
    ranker = RelativeFusionRanker(weights=np.array([1.0, 2.0]))
    bm25   = [_sc("c0", 0.9), _sc("c1", 0.1)]
    vector = [_sc("c0", 0.1), _sc("c1", 0.8)]
    ranked = ranker.rank([bm25, vector])
    assert ranked[0].chunk.id == "c1"


def test_relative_fusion_equal_weights_other_winner():
    ranker = RelativeFusionRanker(weights=np.array([1.0, 1.0]))
    bm25   = [_sc("c0", 0.9), _sc("c1", 0.1)]
    vector = [_sc("c0", 0.1), _sc("c1", 0.8)]
    ranked = ranker.rank([bm25, vector])
    assert ranked[0].chunk.id == "c0"


def test_relative_fusion_normalizes_across_scales():
    ranker = RelativeFusionRanker(weights=np.array([1.0, 1.0]))
    bm25   = [_sc("c0", 12.0), _sc("c1", 3.0)]  # unbounded BM25 scale
    vector = [_sc("c0", 0.2), _sc("c1", 0.9)]   # bounded cosine scale
    ranked = ranker.rank([bm25, vector])
    # After min-max both retrievers land in [0, 1], so neither scale dominates:
    # c0 tops bm25 (norm 1) and bottoms vector (norm 0); c1 the reverse -> tie,
    # broken to c0 by stable ordering. Without normalization BM25 would decide it.
    assert {r.chunk.id for r in ranked} == {"c0", "c1"}
    assert ranked[0].score == ranked[1].score


def test_relative_fusion_weight_mismatch_raises():
    ranker = RelativeFusionRanker(weights=np.array([1.0, 1.0]))
    single_retriever = [[_sc("c0", 0.9), _sc("c1", 0.1)]]
    with pytest.raises(ValueError, match="does not match"):
        ranker.rank(single_retriever)