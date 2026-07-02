from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray


class BaseVectorStore(ABC):
    """Stores and searches dense vector embeddings."""

    @abstractmethod
    def add(self, embeddings: NDArray[np.float32]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_emb: NDArray[np.float32],
        top_k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        pass


class FaissVectorStore(BaseVectorStore):
    """In-memory vector store backed by a FAISS flat index with cosine similarity."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.dim: int | None = None

    def _normalize(self, emb: NDArray[np.float32]) -> NDArray[np.float32]:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        return emb / norms
    
    def save(self, path: Path) -> None:
        if self.index is None:
            raise RuntimeError("Nessun indice da salvare")
        faiss.write_index(self.index, str(path))

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path))
        self.dim = self.index.d

    def add(self, embeddings: NDArray[np.float32]) -> None:
        embeddings = self._normalize(embeddings)
        if self.index is None:
            self.dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    def search(
        self,
        query_emb: NDArray[np.float32],
        top_k: int = 10
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Returns (indices, scores) with cosine scores rescaled from [-1, 1] to [0, 1]."""
        if self.index is None:
            raise RuntimeError("Indice FAISS non inizializzato")
        query_emb = self._normalize(query_emb)
        cosine_sim, indices = self.index.search(query_emb, top_k)
        scores = (cosine_sim + 1.0) / 2.0
        return indices, scores