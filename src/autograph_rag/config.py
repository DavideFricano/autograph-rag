from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a .env file.

    Field names map to upper-case env vars (e.g. vector_store_url -> VECTOR_STORE_URL).
    Everything has a sensible local default, so an empty environment still works.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # General (e.g. BM25 stemming and stopwords)
    language: str = "italian"

    # Generation (Ollama endpoint by default)
    llm_url: str = "http://localhost:11434/api/chat"
    llm_model: str = "qwen2.5:7b"
    system_prompt_path: Path = Path("prompts/system_prompt.md")

    # Embedding
    embed_model: str = "BAAI/bge-m3"

    # Vector store (which tier to use is chosen in the entry point, not here)
    vector_store_collection: str = "data"
    vector_store_path: Path = Path("./data/db/vector")  # used by the persistent tier
    vector_store_url: str | None = None  # used by the remote tier
    vector_store_api_key: str | None = None  # used by the remote tier

    # Ingestion paths
    data_dir: Path = Path("data/raw")
    out_dir: Path = Path("data/out")
