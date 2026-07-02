from __future__ import annotations

from autograph_rag.types import ScoredChunk


class PromptGenerator:
    """Builds structured prompts from query, context chunks, and system instructions."""

    def _join_context_sources(self, scored_chunks: list[ScoredChunk]) -> str:
        enriched_context = []
        for idx, item in enumerate(scored_chunks):
            source = item.chunk.metadata.source.name
            title = item.chunk.metadata.title
            header = f"\n# Fonte n. {idx + 1} [Sorgente: {source}, Titolo: {title}]:\n"
            text_content = item.chunk.text
            enriched_context.append(f"{header}{text_content.strip()}")
        return "\n".join(enriched_context)

    def build_system_prompt(self, system: str) -> str:
        return f"SISTEMA:\n{system.strip()}\n"

    def build_query_prompt(self, query: str) -> str:
        return f"DOMANDA:\n{query.strip()}\n"

    def build_context_prompt(self, context: str | list[ScoredChunk]) -> str:
        if not context:
            return "CONTESTO:\nNessuna fonte trovata per la domanda.\n"
        if isinstance(context, list):
            context = self._join_context_sources(context)
        return f"CONTESTO:\n{context.strip()}\n"

    def build_user_prompt(self, query: str, context: str | list[ScoredChunk]) -> str:
        query_prompt = self.build_query_prompt(query)
        context_prompt = self.build_context_prompt(context)
        user_prompt = f"{query_prompt}\n{context_prompt}\n"
        return f"\n{user_prompt.strip()}\n"

    def build_prompt(self, system: str, query: str, context: str | list[ScoredChunk]) -> str:
        system_prompt = self.build_system_prompt(system)
        user_prompt = self.build_user_prompt(query, context)
        return f"{system_prompt}\n{user_prompt}"
