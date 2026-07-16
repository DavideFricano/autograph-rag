from pathlib import Path

from autograph_rag.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.llm_url == "http://localhost:11434/api/chat"
    assert s.vector_store_url is None
    assert s.data_dir == Path("data/raw")


def test_env_override(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_URL", "http://localhost:6333")
    s = Settings(_env_file=None)
    assert s.vector_store_url == "http://localhost:6333"
