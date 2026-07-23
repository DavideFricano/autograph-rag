from __future__ import annotations

from collections.abc import Iterator

from autograph_rag.augmentation.augmenter import BaseAugmenter
from autograph_rag.generation.llm import BaseLLMClient
from autograph_rag.ingestion.chunker import BaseChunker
from autograph_rag.ingestion.loader import BaseLoader
from autograph_rag.ranking.fusion_ranker import FusionRanker
from autograph_rag.ranking.reranker import Reranker
from autograph_rag.store.base_store import BaseStore
from autograph_rag.types import Chunk, ScoredChunk


class IngestionPipeline:
    """Offline pipeline: loads documents, chunks them, and populates the stores."""

    def __init__(self, loader: BaseLoader, chunker: BaseChunker, stores: list[BaseStore]) -> None:
        self.loader = loader
        self.chunker = chunker
        self.stores = stores

    def ingest(self) -> list[Chunk]:
        """Loads and chunks documents, adding them to every store. Returns the chunks."""
        docs = self.loader.load()
        chunks = [chunk for doc in docs for chunk in self.chunker.chunk(doc)]
        for store in self.stores:
            store.add(chunks)
        return chunks


class QueryPipeline:
    """Online pipeline: searches the stores, fuses, optionally reranks, and generates."""

    def __init__(
        self,
        stores: list[BaseStore],
        ranker: FusionRanker,
        augmenter: BaseAugmenter,
        llm: BaseLLMClient,
        reranker: Reranker | None = None,
        top_k: int = 10,
    ) -> None:
        self.stores = stores
        self.ranker = ranker
        self.augmenter = augmenter
        self.reranker = reranker
        self.llm = llm
        self.top_k = top_k

    def _retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        results = [store.search(query, top_k) for store in self.stores]
        return self.ranker.rank(results, top_k)

    def _rank(self, ranked: list[ScoredChunk], query: str, top_n: int | None) -> list[ScoredChunk]:
        if self.reranker is not None:
            return self.reranker.rank([sc.chunk for sc in ranked], query, top_n)
        return ranked if top_n is None else ranked[:top_n]

    def query(self, query: str, top_n: int | None = None) -> str:
        ranked = self._retrieve(query, self.top_k)
        reranked = self._rank(ranked, query, top_n)
        messages = self.augmenter.build(query, reranked)
        return self.llm.answer(messages)

    def stream(self, query: str, top_n: int | None = None) -> Iterator[str]:
        ranked = self._retrieve(query, self.top_k)
        reranked = self._rank(ranked, query, top_n)
        messages = self.augmenter.build(query, reranked)
        yield from self.llm.stream(messages)


class RagPipeline:
    """Collects the ingestion and query pipelines behind one object (facade)"""

    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        stores: list[BaseStore],
        ranker: FusionRanker,
        augmenter: BaseAugmenter,
        llm: BaseLLMClient,
        reranker: Reranker | None = None,
        top_k: int = 10,
    ) -> None:
        self.ingest_pipeline = IngestionPipeline(loader, chunker, stores)
        self.query_pipeline = QueryPipeline(stores, ranker, augmenter, llm, reranker, top_k)
