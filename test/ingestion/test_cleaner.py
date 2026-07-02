from autograph_rag.ingestion.cleaner import Cleaner


def test_collapses_whitespace():
    assert Cleaner.make_plain("foo   bar") == "foo bar"


def test_strips_edges():
    assert Cleaner.make_plain("  hello  ") == "hello"


def test_newlines_and_tabs():
    assert Cleaner.make_plain("a\n\tb") == "a b"
