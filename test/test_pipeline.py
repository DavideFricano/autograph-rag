from __future__ import annotations

from pathlib import Path

from autograph_rag.embedding.embedder import LocalEmbedder
from autograph_rag.embedding.vector_store import FaissVectorStore
from autograph_rag.generation.llm import OllamaClient
from autograph_rag.ingestion.chunker import HierarchicalChunker
from autograph_rag.ingestion.loader import DoclingLoader
from autograph_rag.pipeline import IngestionPipeline, QueryPipeline
from autograph_rag.ranking.fusion_ranker import ReciprocalFusionRanker
from autograph_rag.retrieval.lexical_retriever import BM25Retriever
from autograph_rag.retrieval.vector_retriever import FaissVectorRetriever

DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/out")
EMBED_MODEL = "BAAI/bge-m3"
OLLAMA_MODEL = "qwen2.5:7b"
TOP_K = 10

SYSTEM = """Sei un assistente clinico che risponde sempre in lingua italiana.
Devi rispondere alla domanda posta SOLO basandoti sulle informazioni presenti nel contesto fornito.
Esso è composto da estratti di documenti clinici a cui fai riferimento per rispondere.

Regole:
- Se la domanda non rientra nell'ambito medico o è troppo generica, rifiuta di rispondere.
- Se la domanda non può essere soddisfatta con le informazioni fornite, rifiuta di rispondere.
- Se la domanda contiene insulti, volgarità o è offensiva, rifiuta di rispondere.
- Usa SOLO le informazioni presenti nel contesto.
- Se è presente una tabella, estrai i dati e basati su quelli.
- Non aggiungere nulla che non sia esplicitamente nel testo.
- La risposta deve essere chiara, concisa e pertinente alla domanda.
- Rispondi sempre in forma strutturata."""


if __name__ == "__main__":
    print("Ingestion...")
    loader = DoclingLoader(input_dir=DATA_DIR, output_dir=OUT_DIR)
    chunker = HierarchicalChunker()
    embedder = LocalEmbedder(EMBED_MODEL)
    vector_store = FaissVectorStore()

    ingestion = IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )
    chunks = ingestion.ingest(save_output=True)
    print(f"Totale chunk: {len(chunks)}")

    print("Retrieval Augmented Generation...")
    vector_retriever = FaissVectorRetriever(chunks, embedder, vector_store)
    bm25_retriever = BM25Retriever(chunks, language="italian")
    ranker = ReciprocalFusionRanker()
    llm = OllamaClient(model=OLLAMA_MODEL)

    pipeline = QueryPipeline(
        retrievers=[vector_retriever, bm25_retriever],
        ranker=ranker,
        llm=llm,
        system=SYSTEM,
        top_k=TOP_K,
    )

    while True:
        query = input("\nDomanda: ").strip()
        if not query:
            continue
        print("\n--- RISPOSTA ---\n")
        for token in pipeline.stream(query):
            print(token, end="", flush=True)
        print()
