from pathlib import Path

import pytest

from autograph_rag.ingestion.converter import (
    BaseConverter,
    MarkdownConverter,
    media_type_for_path,
)


def test_markdown_converter_is_a_base_converter():
    assert isinstance(MarkdownConverter(), BaseConverter)


# --- convert: dispatch on media type (light paths only, no heavy parsers) ---

def test_text_payload_is_decoded_as_is():
    assert MarkdownConverter().convert_stream(b"ciao mondo", "text/plain") == "ciao mondo"


def test_markdown_payload_passes_through():
    assert MarkdownConverter().convert_stream(b"# Titolo", "text/markdown") == "# Titolo"


def test_csv_is_converted_to_a_markdown_table():
    out = MarkdownConverter().convert_stream(b"nome,eta\nMario,42", "text/csv", name="p.csv")
    assert "Mario" in out and "nome" in out


def test_unknown_media_type_fails_loud():
    with pytest.raises(ValueError):
        MarkdownConverter().convert_stream(b"x", "application/octet-stream")


# --- media_type_for_path: extension resolution (a source concern) ---

def test_resolves_media_type_from_extension_case_insensitive():
    assert media_type_for_path(Path("a/b.PDF")) == "application/pdf"
    assert media_type_for_path(Path("x.csv")) == "text/csv"


def test_unknown_extension_fails_loud():
    with pytest.raises(ValueError):
        media_type_for_path(Path("mystery.zzz"))
