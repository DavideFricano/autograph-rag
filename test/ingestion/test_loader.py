import tempfile
from pathlib import Path

import pytest

from autograph_rag.ingestion.loader import BaseLoader
from autograph_rag.types import Document


class _DummyLoader(BaseLoader):
    def _convert(self, path):
        return ""


def test_raises_if_input_dir_missing():
    with pytest.raises(FileNotFoundError):
        _DummyLoader(Path("/tmp/cartella_non_esistente_xyz"), Path("/tmp/out"))


def test_raises_if_input_is_file():
    with tempfile.NamedTemporaryFile() as f, pytest.raises(NotADirectoryError):
        _DummyLoader(Path(f.name), Path("/tmp/out"))


def test_creates_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        output_dir = Path(tmpdir) / "nested" / "output"
        _DummyLoader(input_dir, output_dir)
        assert output_dir.exists()
