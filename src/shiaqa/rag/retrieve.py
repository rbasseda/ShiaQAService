"""Hybrid retrieval: BM25 + dense vectors, fused with Reciprocal Rank Fusion.

Hybrid matters more than usual on this corpus. Questions here turn on proper
names with many spellings ("Husayn"/"Hussain"/"al-Ḥusayn", "Ghadir Khumm"),
where lexical matching is precise and embeddings are fuzzy — but questions of
the "why did X happen" kind need the semantic side. RRF combines both without
needing calibrated scores from either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import Settings, get_settings
from ..embed.ollama_embed import Embedder
from ..log import get_logger
from ..store.sqlite_store import Hit, Store, normalise_terms

log = get_logger(__name__)

# Deliberately small: this is for FTS5 query construction, not for meaning.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "from", "by", "with", "about", "as", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "have", "has", "had", "what",
    "which", "who", "whom", "whose", "when", "where", "why", "how", "that",
    "this", "these", "those", "there", "it", "its", "can", "could", "should",
    "would", "will", "shall", "may", "might", "must", "please", "tell", "me",
    "explain", "describe", "give", "list", "some", "any", "all", "more",
}

_WORD_RE = re.compile(r"[\w'’\-]+", re.UNICODE)


def to_fts_query(question: str) -> str:
    """Turn a natural question into valid, forgiving FTS5 syntax.

    Everything is OR-ed: AND-ing a full sentence returns nothing, and BM25
    already rewards documents that match more of the rare terms.
    """
    terms = []
    for tok in _WORD_RE.findall(question.lower()):
        tok = tok.strip("'’-")
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        terms.append('"' + tok.replace('"', "") + '"')
    return " OR ".join(dict.fromkeys(terms))


@dataclass
class Retrieval:
    question: str
    hits: list[Hit] = field(default_factory=list)
    context: str = ""
    context_tokens: int = 0
    vector_used: bool = True

    @property
    def sources(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for i, h in enumerate(self.hits, 1):
            if h.url not in seen:
                seen[h.url] = {"n": i, "title": h.title, "url": h.url, "sections": []}
            seen[h.url]["sections"].append(h.section)
        return list(seen.values())


def _rrf(rankings: list[tuple[list[int], float, int]]) -> dict[int, float]:
    """Weighted Reciprocal Rank Fusion, with a per-channel RRF constant."""
    scores: dict[int, float] = {}
    for ranked, weight, k in rankings:
        for rank, cid in enumerate(ranked, 1):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
    return scores


class Retriever:
    def __init__(self, settings: Settings | None = None, store: Store | None = None,
                 embedder: Embedder | None = None) -> None:
        self.s = settings or get_settings()
        self.store = store or Store(self.s)
        self.embedder = embedder or Embedder(self.s)
        self._has_vectors = self.store.counts().get("vec_chunks", 0) > 0

    async def search(self, question: str, top_k: int | None = None) -> Retrieval:
        top_k = top_k or self.s.top_k_final
        fts_q = to_fts_query(question)

        bm25 = self.store.bm25_search(fts_q, self.s.top_k_bm25) if fts_q else []
        title = self.store.title_search(
            normalise_terms(question), fts_q, self.s.title_pages, self.s.title_chunks_per_page
        )
        vec: list[tuple[int, float]] = []
        if self._has_vectors:
            embedding = await self.embedder.embed_query(question)
            vec = self.store.vector_search(embedding, self.s.top_k_vector)

        return self._fuse(question, bm25, vec, title, top_k)

    def search_sync(self, question: str, top_k: int | None = None) -> Retrieval:
        top_k = top_k or self.s.top_k_final
        fts_q = to_fts_query(question)
        bm25 = self.store.bm25_search(fts_q, self.s.top_k_bm25) if fts_q else []
        title = self.store.title_search(
            normalise_terms(question), fts_q, self.s.title_pages, self.s.title_chunks_per_page
        )
        vec: list[tuple[int, float]] = []
        if self._has_vectors:
            vec = self.store.vector_search(self.embedder.embed_query_sync(question), self.s.top_k_vector)
        return self._fuse(question, bm25, vec, title, top_k)

    def _fuse(self, question: str, bm25: list[tuple[int, float]],
              vec: list[tuple[int, float]], title: list[int], top_k: int) -> Retrieval:
        bm_ids = [cid for cid, _ in bm25]
        vec_ids = [cid for cid, _ in vec]
        fused = _rrf([
            (ids, w, k) for ids, w, k in (
                (vec_ids, self.s.weight_vector, self.s.rrf_k),
                (bm_ids, self.s.weight_bm25, self.s.rrf_k),
                (title, self.s.weight_title, self.s.rrf_k_title),
            ) if ids
        ])

        ranks = {
            cid: (vec_ids.index(cid) + 1 if cid in vec_ids else None,
                  bm_ids.index(cid) + 1 if cid in bm_ids else None)
            for cid in fused
        }
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])
        hits = self.store.hydrate(ordered, ranks)

        # Diversify: no single article may dominate the context window.
        per_page: dict[int, int] = {}
        kept: list[Hit] = []
        for h in hits:
            if per_page.get(h.page_id, 0) >= self.s.max_chunks_per_page:
                continue
            per_page[h.page_id] = per_page.get(h.page_id, 0) + 1
            kept.append(h)
            if len(kept) >= top_k:
                break

        context, used = self._pack_context(kept)
        return Retrieval(
            question=question, hits=kept[: len(used)], context=context,
            context_tokens=sum(h.tokens for h in used), vector_used=bool(vec),
        )

    def _pack_context(self, hits: list[Hit]) -> tuple[str, list[Hit]]:
        budget = self.s.context_token_budget
        blocks: list[str] = []
        used: list[Hit] = []
        for h in hits:
            if h.tokens > budget and used:
                continue
            blocks.append(f"[{len(used) + 1}] {h.label}\n{h.text}")
            used.append(h)
            budget -= h.tokens
            if budget <= 0:
                break
        return "\n\n".join(blocks), used
