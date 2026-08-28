from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

from autograph_rag.authorization.schema import AccessSchema
from autograph_rag.types import Document


class BaseLabeler(ABC):
    """Writes the access attributes of a document, at the ingestion boundary.

    **Labelling is not deciding.** It marks the data for *what it is*, never for *who may
    see it*: the rules live in the policy engine, so changing a rule must never force a
    re-ingestion. That is why the only contract shared by ingestion and query is the
    attribute schema, not a policy.

    It runs on the ``Document``, before chunking: the attributes belong to the source
    document, so they are written once on its ``Source`` and every chunk inherits them by
    carrying it — the chunker already copies ``doc.source`` into each ``Metadata``.

    Subclasses only say *where the values come from* (``_attributes``); validating them
    against the declared vocabulary and attaching them is shared, so no labeler can put an
    undeclared attribute into the store.
    """

    def __init__(self, schema: AccessSchema) -> None:
        self.schema = schema

    @abstractmethod
    def _attributes(self, document: Document) -> Mapping[str, object]:
        """The raw attributes for this document, before validation."""

    def label(self, document: Document) -> Document:
        """A copy of the document whose ``source.access`` holds the validated attributes.

        Invalid attributes raise rather than being skipped: a corrupt file is a problem
        with *that* datum and the loader is right to drop it, but an attribute the schema
        doesn't declare means the producer and the declaration disagree — a configuration
        error that concerns the whole corpus. Skipping would silently leave a hole in it.
        """
        try:
            access = self.schema.validate_access(self._attributes(document))
        except ValueError as error:
            # the schema knows the vocabulary, only this side knows which document
            raise ValueError(f"document {document.source.id!r}: {error}") from error
        return document.model_copy(
            update={"source": document.source.model_copy(update={"access": access})}
        )


class PropagatingLabeler(BaseLabeler):
    """Keeps the attributes the document already arrived with, and validates them.

    The main path: whoever produced the corpus knows the attributes — the FHIR gateway
    knows the patient and the consent, the script that prepared the folder knows the
    tenant — and the library does not try to re-infer from bytes what it cannot know. Here
    the labeler adds no values; it is the boundary where what came from outside is checked
    against the declaration before it reaches the store.
    """

    def _attributes(self, document: Document) -> Mapping[str, object]:
        return document.source.access


class ManifestLabeler(BaseLabeler):
    """Reads the attributes from a JSON manifest keyed by ``source.id``.

    For the corpus with no producer upstream — a folder of documents someone curates by
    hand. ``default`` applies to every document, ``sources`` overrides it per document, so
    nothing has to be enumerated in advance: what a new document inherits is the default,
    and if that doesn't carry the required attributes it is denied at retrieval rather than
    quietly readable.

    ``{"default": {"tenant": "acme"}, "sources": {"report.pdf": {"classification": "…"}}}``
    """

    def __init__(self, schema: AccessSchema, path: Path | str) -> None:
        super().__init__(schema)
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"the access manifest must be a JSON object: {path}")
        unknown = set(manifest) - {"default", "sources"}
        if unknown:
            raise ValueError(f"unknown key in the access manifest: {sorted(unknown)}")
        self.default: dict = manifest.get("default", {})
        self.sources: dict = manifest.get("sources", {})

    def _attributes(self, document: Document) -> Mapping[str, object]:
        return {**self.default, **self.sources.get(document.source.id, {})}


class StaticLabeler(BaseLabeler):
    """Puts the same attributes on everything this pipeline ingests.

    The degenerate case, and a common one: an ``IngestionPipeline`` has a single loader, so
    a deployment whose whole corpus shares its attributes ("everything here is this
    tenant") needs nothing more. Several sources means several pipelines, each with its own
    labeler — the pairing is done where they are composed, not looked up in a table.
    """

    def __init__(self, schema: AccessSchema, attributes: Mapping[str, object]) -> None:
        super().__init__(schema)
        self.attributes = dict(attributes)

    def _attributes(self, document: Document) -> Mapping[str, object]:
        return self.attributes
