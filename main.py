from __future__ import annotations

from autograph_rag.augmentation.augmenter import PromptAugmenter
from autograph_rag.config import Settings
from autograph_rag.embedding.embedder import LocalEmbedder
from autograph_rag.generation.llm import OllamaClient
from autograph_rag.ingestion.chunker import HierarchicalChunker
from autograph_rag.ingestion.loader import FileLoader
from autograph_rag.pipeline import RagPipeline
from autograph_rag.ranking.fusion_ranker import ReciprocalFusionRanker
from autograph_rag.store.lexical_store import InMemoryLexicalStore
from autograph_rag.store.vector_store import InMemoryVectorStore

if __name__ == "__main__":
    settings = Settings()

    rag = RagPipeline(
        loader=FileLoader(settings.in_dir, settings.out_dir, save_output=True),
        chunker=HierarchicalChunker(),
        stores=[
            InMemoryVectorStore(embedder=LocalEmbedder(settings.embed_model)),
            InMemoryLexicalStore(language=settings.language),
        ],
        ranker=ReciprocalFusionRanker(),
        augmenter=PromptAugmenter(system=settings.system_prompt_path.read_text(encoding="utf-8")),
        llm=OllamaClient(model=settings.llm_model, url=settings.llm_url),
        top_k=10,
    )

    print("Ingestion...")
    chunks = rag.ingest_pipeline.ingest()
    print(f"Totale chunk: {len(chunks)}")

    print("Query...")
    while True:
        try:
            question = input("\nDomanda: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        print("\n--- RISPOSTA ---\n")
        for token in rag.query_pipeline.stream(question):
            print(token, end="", flush=True)
        print()
