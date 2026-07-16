from __future__ import annotations

from collections.abc import Iterator

from autograph_rag.augmentation.augmenter import BaseAugmenter
from autograph_rag.embedding.embedder import BaseEmbedder
from autograph_rag.embedding.vector_store import BaseVectorStore
from autograph_rag.generation.llm import BaseLLMClient
from autograph_rag.ingestion.chunker import BaseChunker
from autograph_rag.ingestion.loader import BaseLoader
from autograph_rag.ranking.fusion_ranker import FusionRanker
from autograph_rag.ranking.reranker import Reranker
from autograph_rag.retrieval.retriever import BaseRetriever
from autograph_rag.types import Chunk, ScoredChunk


class IngestionPipeline:
    """Offline pipeline: loads documents, chunks, embeds, and populates the vector store."""

    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(self) -> list[Chunk]:
        """Returns the full chunk list so the caller can wire up QueryPipeline retrievers."""
        docs = self.loader.load()
        chunks = [chunk for doc in docs for chunk in self.chunker.chunk(doc)]
        embeddings = self.embedder.embed_chunks([c.text for c in chunks])
        self.vector_store.add(chunks, embeddings)
        return chunks


class QueryPipeline:
    """Online pipeline: retrieves, fuses, optionally reranks, and generates an answer."""

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        ranker: FusionRanker,
        augmenter: BaseAugmenter,
        llm: BaseLLMClient,
        reranker: Reranker | None = None,
        top_k: int | None = 10,
    ) -> None:
        self.retrievers = retrievers
        self.ranker = ranker
        self.augmenter = augmenter
        self.reranker = reranker
        self.llm = llm
        self.top_k = top_k

    def _retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        results = [r.retrieve(query, top_k) for r in self.retrievers]
        ranked = self.ranker.rank(results, top_k)
        return ranked

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
