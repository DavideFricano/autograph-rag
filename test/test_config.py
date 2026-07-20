from pathlib import Path

import pytest

from autograph_rag.config import Settings


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """Settings sees only code defaults: no .env in the cwd, no ambient env vars.

    A defaults test must not depend on the developer's .env — that would test the
    machine, not the code. env var names are derived from the model fields.
    """
    monkeypatch.chdir(tmp_path)  # no .env here
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def test_defaults(isolated_env):
    s = Settings()
    assert s.llm_url == "http://localhost:11434/api/chat"
    assert s.vector_store_url is None
    assert s.in_dir == Path("data/raw")


def test_env_override(isolated_env, monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_URL", "http://localhost:6333")
    s = Settings()
    assert s.vector_store_url == "http://localhost:6333"
