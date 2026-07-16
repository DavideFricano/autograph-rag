from __future__ import annotations

import re
from abc import abstractmethod

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from rank_bm25 import BM25Okapi

from autograph_rag.retrieval.retriever import BaseRetriever
from autograph_rag.types import Chunk, ScoredChunk


class LexicalRetriever(BaseRetriever):
    """Retriever based on lexical matching."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        pass


class BM25Retriever(LexicalRetriever):
    """BM25Okapi retriever with language-aware stemming and stopword removal."""

    def __init__(self, chunks: list[Chunk], language: str = "english") -> None:
        super().__init__(chunks)
        self.language = language.lower()
        nltk.download("stopwords", quiet=True)
        self.stemmer = SnowballStemmer(self.language)
        self.stopwords = set(stopwords.words(self.language))
        self.bm25 = BM25Okapi(self._tokenize([chunk.text for chunk in self.chunks]))

    def _tokenize(self, text: str | list[str]) -> list[str] | list[list[str]]:
        if isinstance(text, list):
            return [self._tokenize(chunk) for chunk in text]
        return [
            self.stemmer.stem(w)
            for w in re.findall(r"\b\w+\b", text.lower())
            if w not in self.stopwords
        ]

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            ScoredChunk(chunk=self.chunks[idx], score=float(scores[idx]))
            for idx in top_idx
        ]
