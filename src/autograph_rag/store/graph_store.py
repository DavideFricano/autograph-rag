from __future__ import annotations

from autograph_rag.store.base_store import BaseStore


class BaseGraphStore(BaseStore):
    """Graph store: retrieval over a knowledge graph built from chunks."""
