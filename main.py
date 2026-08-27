from __future__ import annotations

from autograph_rag.augmentation.augmenter import PromptAugmenter
from autograph_rag.authorization.schema import AccessSchema
from autograph_rag.config import Settings
from autograph_rag.embedding.embedder import LocalEmbedder
from autograph_rag.generation.llm import OllamaClient
from autograph_rag.indexing.similarity.lexical_index import VolatileLexicalIndex
from autograph_rag.indexing.similarity.semantic_index import VolatileSemanticIndex
from autograph_rag.ingestion.chunker import HierarchicalChunker
from autograph_rag.ingestion.labeler import PropagatingLabeler
from autograph_rag.ingestion.loader import FileLoader
from autograph_rag.pipeline import RagPipeline
from autograph_rag.ranking.fusion_ranker import ReciprocalRankFusionRanker
from autograph_rag.storing.store import VolatileStore

if __name__ == "__main__":
    settings = Settings()

    # One shared store for the chunk data; the indexes hold only ids + vectors/tokens
    # and resolve chunks back through it.
    store = VolatileStore()
    embedder = LocalEmbedder(settings.embed_model)

    # The access schema is the contract between the two pipelines: declared once here and
    # injected into everything that writes attributes or filters on them. Left unset it is
    # None, and this stays a deployment without access control.
    path = settings.access_schema_path
    schema = AccessSchema.from_file(path) if path is not None else None

    rag = RagPipeline(
        loader=FileLoader(settings.in_dir, settings.out_dir, save_output=True),
        # Validates what arrived with each document. Nothing puts attributes on a local
        # file yet, so with a schema declared every chunk is unlabeled — and therefore
        # denied. Swap in a ManifestLabeler to curate them by hand.
        labeler=PropagatingLabeler(schema) if schema is not None else None,
        chunker=HierarchicalChunker(),
        store=store,
        indexes=[
            VolatileSemanticIndex(store, embedder, schema=schema),
            VolatileLexicalIndex(store, language=settings.language, schema=schema),
        ],
        ranker=ReciprocalRankFusionRanker(),
        augmenter=PromptAugmenter(system=settings.system_prompt_path.read_text(encoding="utf-8")),
        llm=OllamaClient(model=settings.llm_model, url=settings.llm_url),
        top_i=settings.top_i,
        top_k=settings.top_k,
        top_n=settings.top_n,
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
