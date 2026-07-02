from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator

import requests
from openai import OpenAI

from autograph_rag.augmentation.prompter import PromptGenerator
from autograph_rag.types import ScoredChunk


class BaseLLMClient(ABC):
    """Common interface for LLM backends."""

    @abstractmethod
    def stream(self, system: str, query: str, context: str | list[ScoredChunk], **kwargs) -> Iterator[str]:
        pass

    @abstractmethod
    def answer(self, system: str, query: str, context: str | list[ScoredChunk], **kwargs) -> str:
        pass


class OllamaClient(BaseLLMClient):
    """LLM client for locally-hosted Ollama models via the /api/generate endpoint."""

    def __init__(self, model: str, url: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.url = url
        self.prompt_generator = PromptGenerator()

    def _payload(self, prompt: str, temperature: float, num_ctx: int, stream: bool) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }

    def stream(self, system: str, query: str, context: str | list[ScoredChunk], temperature: float = 0.1, num_ctx: int = 4096) -> Iterator[str]:
        prompt = self.prompt_generator.build_prompt(system, query, context)
        response = requests.post(self.url, json=self._payload(prompt, temperature, num_ctx, stream=True), stream=True, timeout=800)
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done", False):
                break

    def answer(self, system: str, query: str, context: str | list[ScoredChunk], temperature: float = 0.1, num_ctx: int = 4096) -> str:
        prompt = self.prompt_generator.build_prompt(system, query, context)
        response = requests.post(self.url, json=self._payload(prompt, temperature, num_ctx, stream=False), timeout=800)
        return response.json().get("response", "")


class OpenAIClient(BaseLLMClient):
    """LLM client for OpenAI chat models. Reads the API key from the environment."""

    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model
        self.prompt_generator = PromptGenerator()

    def _messages(self, system: str, query: str, context: str) -> list[dict]:
        return [
            {"role": "system", "content": self.prompt_generator.build_system_prompt(system)},
            {"role": "user", "content": self.prompt_generator.build_user_prompt(query, context)},
        ]

    def stream(self, system: str, query: str, context: str | list[ScoredChunk], temperature: float = 0.1) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, query, context),
            temperature=temperature,
            stream=True,
        )
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    def answer(self, system: str, query: str, context: str | list[ScoredChunk], temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, query, context),
            temperature=temperature,
            stream=False,
        )
        return response.choices[0].message.content
