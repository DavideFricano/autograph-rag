from __future__ import annotations

from autograph_rag.augmentation.augmenter import PromptAugmenter
from autograph_rag.config import Settings
from autograph_rag.embedding.embedder import LocalEmbedder
from autograph_rag.embedding.vector_store import InMemoryVectorStore
from autograph_rag.generation.llm import OllamaClient
from autograph_rag.ingestion.chunker import HierarchicalChunker
from autograph_rag.ingestion.loader import FileSystemLoader
from autograph_rag.pipeline import IngestionPipeline, QueryPipeline
from autograph_rag.ranking.fusion_ranker import ReciprocalFusionRanker
from autograph_rag.retrieval.lexical_retriever import BM25Retriever
from autograph_rag.retrieval.vector_retriever import VectorRetriever

if __name__ == "__main__":
    settings = Settings()

    print("Ingestion...")
    loader = FileSystemLoader(input_dir=settings.data_dir, output_dir=settings.out_dir, save_output=True)
    chunker = HierarchicalChunker()
    embedder = LocalEmbedder(settings.embed_model)
    vector_store = InMemoryVectorStore()

    ingestion = IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )
    chunks = ingestion.ingest()
    print(f"Totale chunk: {len(chunks)}")

    print("Retrieval Augmented Generation...")
    vector_retriever = VectorRetriever(embedder, vector_store)
    bm25_retriever = BM25Retriever(chunks, language=settings.language)
    ranker = ReciprocalFusionRanker()
    system = settings.system_prompt_path.read_text(encoding="utf-8")
    augmenter = PromptAugmenter(system=system)
    llm = OllamaClient(model=settings.llm_model, url=settings.llm_url)

    pipeline = QueryPipeline(
        retrievers=[vector_retriever, bm25_retriever],
        ranker=ranker,
        augmenter=augmenter,
        llm=llm,
        top_k=10
    )

    while True:
        query = input("\nDomanda: ").strip()
        if not query:
            continue
        print("\n--- RISPOSTA ---\n")
        for token in pipeline.stream(query):
            print(token, end="", flush=True)
        print()
