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
from .decompose import Decomposer
from .prompt import NO_CONTEXT, build_messages
from .retrieve import Retrieval, Retriever, _rrf

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
        self._profile: str | None = None

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

    async def _generate_once(self, prompt: str) -> str:
        """One non-streaming completion, used only for decomposition."""
        resp = await self._http().post("/api/chat", json={
            "model": self.s.gen_model(self._profile),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0,
                        "num_ctx": self.s.gen_num_ctx,
                        "num_predict": self.s.decompose_num_predict},
        })
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def retrieve(self, question: str, top_k: int | None = None,
                       decompose: bool | None = None) -> tuple[Retrieval, float]:
        t0 = time.perf_counter()
        use = self.s.decompose if decompose is None else decompose
        if use:
            res = await self._retrieve_decomposed(question, top_k)
        else:
            res = await self.retriever.search(question, top_k)
        return res, time.perf_counter() - t0

    async def _retrieve_decomposed(self, question: str,
                                   top_k: int | None) -> Retrieval:
        """Split the question, retrieve for each part, and merge by rank.

        Merging by *rank* rather than by score is the load-bearing choice. An RRF
        score is a sum of `weight / (k + rank)` over whichever channels fired for
        that query, so a sub-question that matched three channels produces
        systematically larger numbers than one that matched two, whatever the
        relevance. Taking the max would hand the answer to the broadest
        sub-question. Feeding each sub-question's ranking in as one equal-weight
        channel of the same `_rrf` used everywhere else fixes that, and has the
        property we actually want: a chunk found by two sub-questions is promoted
        for exactly that reason.
        """
        parts = await Decomposer(self._generate_once,
                                 self.s.decompose_max_parts).split(question)
        if not parts:
            return await self.retriever.search(question, top_k)

        top_k = top_k or self.s.top_k_final
        rankings: list[list[int]] = []
        merged: dict[int, Retrieval] = {}
        pool: dict[int, object] = {}
        for part in parts:
            sub = await self.retriever.search(part, self.s.top_k_bm25)
            rankings.append([h.chunk_id for h in sub.hits])
            for h in sub.hits:
                pool.setdefault(h.chunk_id, h)
        if not pool:
            return await self.retriever.search(question, top_k)

        fused = _rrf([(ids, 1.0, self.s.rrf_k) for ids in rankings if ids])
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])

        # Same diversification and packing as the single-shot path, so
        # `max_chunks_per_page` and the token budget still hold.
        per_page: dict[int, int] = {}
        kept = []
        for cid, sc in ordered:
            hit = pool.get(cid)
            if hit is None:
                continue
            if per_page.get(hit.page_id, 0) >= self.s.max_chunks_per_page:
                continue
            per_page[hit.page_id] = per_page.get(hit.page_id, 0) + 1
            hit.score = sc
            kept.append(hit)
            if len(kept) >= top_k:
                break
        context, used = self.retriever._pack_context(kept)
        return Retrieval(
            question=question, hits=kept[: len(used)], context=context,
            context_tokens=sum(h.tokens for h in used), vector_used=True,
            sub_questions=parts,
        )

    async def answer(self, question: str, profile: str | None = None,
                     top_k: int | None = None,
                     decompose: bool | None = None) -> Answer:
        model = self.s.gen_model(profile)
        self._profile = profile
        res, r_secs = await self.retrieve(question, top_k, decompose)

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
                     top_k: int | None = None,
                     decompose: bool | None = None) -> AsyncIterator[dict]:
        """Yield SSE-shaped events: sources first, then answer deltas, then stats.

        Sending sources up front matters on this hardware — the user sees where
        the answer will come from within a second, while generation grinds on.
        """
        model = self.s.gen_model(profile)
        self._profile = profile
        res, r_secs = await self.retrieve(question, top_k, decompose)

        yield {"type": "sources", "sources": [
            {"n": i, "title": h.title, "section": h.section, "url": h.url, "kind": h.kind}
            for i, h in enumerate(res.hits, 1)
        ], "chains": res.kg_chains, "sub_questions": res.sub_questions,
            "retrieval_seconds": round(r_secs, 3),
            "context_tokens": res.context_tokens}

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
