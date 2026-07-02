from datetime import date

import numpy as np

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