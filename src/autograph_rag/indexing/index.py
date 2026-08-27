from __future__ import annotations

from abc import ABC, abstractmethod

from autograph_rag.authorization.filter import Filter, evaluate
from autograph_rag.authorization.schema import AccessSchema
from autograph_rag.storing.store import BaseStore
from autograph_rag.types import Chunk, ScoredChunk


class BaseIndex(ABC):
    """A retrieval index over chunks: insert, delete, retrieve.

    Keeps only ids plus its own representation (dense vectors, sparse/BM25 terms, …)
    and resolves chunk data back from a **shared** ``Store`` on retrieve, so retrieve
    returns ``list[ScoredChunk]``. Because the store is shared (single source of truth,
    no duplication), an index **never writes it**: adding and deleting *records* is
    coordinated by the ingestion pipeline (``store.add`` / ``store.delete`` once), while
    each index manages only its own ids — ``delete`` here removes this index's ids, not
    the shared records (a cascade from one index would strand the others). The pipeline
    does the fan-out: ``delete`` on every index, then a single ``store.delete``.

    Stays free of any backend import, so the similarity family and the relation family
    inherit nothing engine-specific from here.

    An optional ``schema`` says whether this deployment does ABAC at all. Without one the
    index behaves as it always has: a filter is accepted if given, and omitting it returns
    everything. With one, retrieval without a filter is refused — see ``retrieve``.
    """

    def __init__(self, store: BaseStore, schema: AccessSchema | None = None) -> None:
        self.store = store
        self.schema = schema

    @abstractmethod
    def insert(self, chunks: list[Chunk]) -> None:
        """Index the chunks (id + vector/tokens + their source id); idempotent by chunk.id.
        Does not write the store — the pipeline adds the records once."""

    @abstractmethod
    def delete(self, source_id: str) -> None:
        """Remove from THIS index every id of the given source. Does not touch the shared
        store — the pipeline deletes the records once, after every index has removed its ids."""

    @abstractmethod
    def _search(self, query: str, top_i: int) -> list[tuple[str, float]]:
        """Backend primitive: top ``top_i`` ``(chunk_id, score)``, higher = more relevant.
        Ids are unique — a chunk appears at most once, or fusion would count it twice.
        Not called directly: the public entry point is ``retrieve``."""

    def retrieve(
        self, query: str, top_i: int, filter: Filter | None = None
    ) -> list[ScoredChunk]:
        """Retrieve the authorized top ``top_i``, resolving chunks through the shared store.

        The filter is applied here, before fusion: a ranker reads positions and score
        spreads *within* each list, so leaving unauthorized chunks in would let them shift
        the rank of authorized ones. Applying it here also keeps ``top_k`` meaning
        authorized results. Subclasses may additionally push the filter into their backend
        — this check stays regardless, so an index that ignores or mistranslates it cannot
        leak, only return less.

        When the deployment declared a schema, omitting the filter is refused rather than
        read as "everything": there the safe default cannot be inferred, and forgetting an
        argument must not be spelled the same way as deciding there is no restriction —
        which is what ``Allow()`` is for. A chunk that doesn't carry the required
        attributes is dropped before the predicate runs, so an unlabeled chunk stays denied
        even under a predicate a missing attribute would otherwise satisfy (``Not``).
        """
        if self.schema is not None:
            if filter is None:
                raise ValueError(
                    "this index declares an access schema, so retrieve requires a filter; "
                    "pass Allow() to state explicitly that the call has no restriction"
                )
            self.schema.validate_filter(filter)
        scored = self._search(query, top_i)
        by_id = {chunk.id: chunk for chunk in self.store.get([id_ for id_, _ in scored])}
        results = [
            ScoredChunk(chunk=by_id[id_], score=score) for id_, score in scored if id_ in by_id
        ]
        if filter is None:
            return results
        return [
            sc
            for sc in results
            if (self.schema is None or self.schema.is_labeled(sc.chunk.metadata.source.access))
            and evaluate(filter, sc.chunk.metadata.source.access)
        ]
