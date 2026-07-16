import base64
from collections.abc import Iterable

from datetime import date

from autograph_rag.ingestion.loader import ApiLoader, RemoteLoader
from autograph_rag.types import Document


def _item(**overrides):
    text = overrides.pop("text", "hello world")
    base = {
        "data": base64.b64encode(text.encode()).decode(),
        "media_type": "text/plain",
        "external_id": "doc-1",
        "title": "referto.pdf",
        "ingested_at": "2026-07-13",
    }
    base.update(overrides)
    return base


class _FakeRemoteLoader(RemoteLoader):
    """Exercises RemoteLoader's contract + mapping without any transport."""

    def __init__(self, records: list[dict]) -> None:
        super().__init__()
        self._records = records

    def _fetch(self) -> Iterable[dict]:
        return self._records


# --- RemoteLoader: contract + mapping (transport-independent) ---

def test_load_maps_remote_records_to_documents():
    loader = _FakeRemoteLoader([_item(), _item(external_id="doc-2", title="b.pdf")])

    docs = list(loader.load())

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    first = docs[0]
    assert first.text == "hello world"
    assert first.source.id == "doc-1"
    assert first.source.name == "referto.pdf"
    assert first.source.time == date(2026, 7, 13)


def test_load_decodes_base64_payload():
    docs = list(_FakeRemoteLoader([_item(text="referto del paziente")]).load())

    assert docs[0].text == "referto del paziente"


def test_load_is_tolerant_to_unknown_fields():
    docs = list(_FakeRemoteLoader([_item(extra_field="ignored", version=3)]).load())

    assert len(docs) == 1
    assert docs[0].source.id == "doc-1"


def test_load_skips_malformed_records():
    incomplete = {"external_id": "x", "title": "t", "ingested_at": "2026-07-13"}  # no data/media_type

    docs = list(_FakeRemoteLoader([incomplete, _item()]).load())

    # one bad record is skipped, the good one still comes through
    assert len(docs) == 1
    assert docs[0].source.id == "doc-1"


def test_load_skips_records_with_unsupported_media_type():
    bad = _item(external_id="doc-bad", media_type="application/octet-stream")

    docs = list(_FakeRemoteLoader([bad, _item(external_id="doc-2")]).load())

    assert {d.source.id for d in docs} == {"doc-2"}


def test_load_returns_empty_when_service_has_no_documents():
    assert list(_FakeRemoteLoader([]).load()) == []


# --- ApiLoader: the HTTP call ---

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_api_loader_calls_endpoint_and_maps(monkeypatch):
    captured = {}

    def fake_get(url, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return _FakeResponse({"items": [_item()]})

    monkeypatch.setattr("autograph_rag.ingestion.loader.requests.get", fake_get)

    loader = ApiLoader("https://gw.example/", path="/documents", headers={"Authorization": "Bearer x"})
    docs = list(loader.load())

    assert captured["url"] == "https://gw.example/documents"  # trailing slash trimmed
    assert captured["headers"] == {"Authorization": "Bearer x"}
    assert len(docs) == 1
    assert docs[0].source.id == "doc-1"


def test_api_loader_accepts_bare_list_payload(monkeypatch):
    monkeypatch.setattr(
        "autograph_rag.ingestion.loader.requests.get",
        lambda *a, **k: _FakeResponse([_item(), _item(external_id="doc-2")]),
    )

    docs = list(ApiLoader("https://gw.example").load())

    assert {d.source.id for d in docs} == {"doc-1", "doc-2"}
