"""Retrieval-augmented answering against a local Ollama generator."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from ..config import Settings, get_settings
from ..log import get_logger
from ..store.sqlite_store import Hit
from .prompt import NO_CONTEXT, build_messages
from .retrieve import Retrieval, Retriever

log = get_logger(__name__)


@dataclass
class Answer:
    question: str
    text: str
    model: str
    hits: list[Hit] = field(default_factory=list)
    context_tokens: int = 0
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    eval_count: int = 0

    @property
    def tokens_per_second(self) -> float:
        return self.eval_count / self.generation_seconds if self.generation_seconds else 0.0

    def sources(self) -> list[dict]:
        return [
            {"n": i, "title": h.title, "section": h.section, "url": h.url, "kind": h.kind}
            for i, h in enumerate(self.hits, 1)
        ]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "model": self.model,
            "sources": self.sources(),
            "stats": {
                "context_tokens": self.context_tokens,
                "retrieval_seconds": round(self.retrieval_seconds, 3),
                "generation_seconds": round(self.generation_seconds, 2),
                "output_tokens": self.eval_count,
                "tokens_per_second": round(self.tokens_per_second, 1),
            },
        }


class AnswerEngine:
    def __init__(self, settings: Settings | None = None, retriever: Retriever | None = None) -> None:
        self.s = settings or get_settings()
        self.retriever = retriever or Retriever(self.s)
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            # Generous timeout: a 7B model on CPU can take minutes per answer.
            self._client = httpx.AsyncClient(base_url=self.s.ollama_host, timeout=900.0)
        return self._client

    def _chat_payload(self, question: str, context: str, model: str, stream: bool) -> dict:
        return {
            "model": model,
            "messages": build_messages(question, context),
            "stream": stream,
            "options": {
                "temperature": self.s.gen_temperature,
                "num_ctx": self.s.gen_num_ctx,
                "num_predict": self.s.gen_max_tokens,
            },
        }

    async def retrieve(self, question: str, top_k: int | None = None) -> tuple[Retrieval, float]:
        t0 = time.perf_counter()
        res = await self.retriever.search(question, top_k)
        return res, time.perf_counter() - t0

    async def answer(self, question: str, profile: str | None = None,
                     top_k: int | None = None) -> Answer:
        model = self.s.gen_model(profile)
        res, r_secs = await self.retrieve(question, top_k)

        if not res.hits:
            return Answer(question=question, text=NO_CONTEXT, model=model,
                          retrieval_seconds=r_secs)

        t0 = time.perf_counter()
        resp = await self._http().post("/api/chat",
                                       json=self._chat_payload(question, res.context, model, False))
        resp.raise_for_status()
        data = resp.json()
        return Answer(
            question=question,
            text=data["message"]["content"].strip(),
            model=model,
            hits=res.hits,
            context_tokens=res.context_tokens,
            retrieval_seconds=r_secs,
            generation_seconds=time.perf_counter() - t0,
            eval_count=data.get("eval_count", 0),
        )

    async def stream(self, question: str, profile: str | None = None,
                     top_k: int | None = None) -> AsyncIterator[dict]:
        """Yield SSE-shaped events: sources first, then answer deltas, then stats.

        Sending sources up front matters on this hardware — the user sees where
        the answer will come from within a second, while generation grinds on.
        """
        model = self.s.gen_model(profile)
        res, r_secs = await self.retrieve(question, top_k)

        yield {"type": "sources", "sources": [
            {"n": i, "title": h.title, "section": h.section, "url": h.url, "kind": h.kind}
            for i, h in enumerate(res.hits, 1)
        ], "retrieval_seconds": round(r_secs, 3), "context_tokens": res.context_tokens}

        if not res.hits:
            yield {"type": "delta", "text": NO_CONTEXT}
            yield {"type": "done", "model": model, "output_tokens": 0}
            return

        t0 = time.perf_counter()
        eval_count = 0
        async with self._http().stream(
            "POST", "/api/chat", json=self._chat_payload(question, res.context, model, True)
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                piece = data.get("message", {}).get("content")
                if piece:
                    yield {"type": "delta", "text": piece}
                if data.get("done"):
                    eval_count = data.get("eval_count", 0)

        yield {"type": "done", "model": model, "output_tokens": eval_count,
               "generation_seconds": round(time.perf_counter() - t0, 2)}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await self.retriever.embedder.aclose()
