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
from ..kg.ontology import Ontology
from ..kg.store import KgStore
from ..ingest.chunk import estimate_tokens
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
    kg_facts: list[str] = field(default_factory=list)

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
                 embedder: Embedder | None = None, kg: KgStore | None = None) -> None:
        self.s = settings or get_settings()
        self.store = store or Store(self.s)
        self.embedder = embedder or Embedder(self.s)
        self._has_vectors = self.store.counts().get("vec_chunks", 0) > 0
        self._kg = kg if kg is not None else self._open_kg()
        self._onto = Ontology() if self._kg is not None else None

    def _open_kg(self) -> KgStore | None:
        """Attach `data/kg.db` when it exists and something wants to use it."""
        if not (self.s.kg_enabled or self.s.weight_graph > 0 or self.s.kg_facts_in_context):
            return None
        if not self.s.kg_db_path.exists():
            # Only worth a warning if the graph was asked for explicitly. The
            # default weight is non-zero, so an install that never ran
            # `shiaqa kg build` would otherwise be nagged on every query for a
            # channel whose absence costs it nothing.
            if self.s.kg_enabled:
                log.warning("kg is enabled but %s is missing; run `shiaqa kg build`",
                            self.s.kg_db_path)
            return None
        kg = KgStore(self.s)
        if not kg.is_populated:
            kg.close()
            return None
        return kg

    @property
    def kg_active(self) -> bool:
        return self._kg is not None

    def _graph_channel(self, seed_pages: list[int]) -> list[int]:
        """Chunks from articles one typed hop off the ones the question names.

        Returns nothing unless the graph is attached *and* weighted. That is not
        just an optimisation: a zero-weight channel still injects its chunk ids
        into the fusion map with a 0.0 score, where they can occupy tail slots
        after diversification. Skipping it outright is what makes the default
        configuration byte-identical to having no graph at all.
        """
        if self._kg is None or self.s.weight_graph <= 0 or not seed_pages:
            return []
        seeds = [n for n in (self._kg.node_for_page(p) for p in seed_pages) if n]
        neighbours = self._kg.neighbour_pages(seeds, self.s.kg_neighbour_pages)
        return self.store.page_chunks(neighbours, self.s.kg_chunks_per_page)

    def _graph_facts(self, seed_pages: list[int]) -> list[str]:
        """Structured statements about the articles the question names."""
        if self._kg is None or not self.s.kg_facts_in_context or not seed_pages:
            return []
        out: list[str] = []
        for page_id in seed_pages:
            node = self._kg.node_for_page(page_id)
            if node is None:
                continue
            for line in self._kg.facts(node, self._onto.predicates,
                                       self.s.kg_facts_max - len(out)):
                if line not in out:
                    out.append(line)
            if len(out) >= self.s.kg_facts_max:
                break
        return out[: self.s.kg_facts_max]

    async def search(self, question: str, top_k: int | None = None) -> Retrieval:
        top_k = top_k or self.s.top_k_final
        fts_q = to_fts_query(question)

        bm25 = self.store.bm25_search(fts_q, self.s.top_k_bm25) if fts_q else []
        title, seed_pages = self.store.title_candidates(
            normalise_terms(question), fts_q, self.s.title_pages, self.s.title_chunks_per_page
        )
        vec: list[tuple[int, float]] = []
        if self._has_vectors:
            embedding = await self.embedder.embed_query(question)
            vec = self.store.vector_search(embedding, self.s.top_k_vector)

        return self._fuse(question, bm25, vec, title, top_k, seed_pages)

    def search_sync(self, question: str, top_k: int | None = None) -> Retrieval:
        top_k = top_k or self.s.top_k_final
        fts_q = to_fts_query(question)
        bm25 = self.store.bm25_search(fts_q, self.s.top_k_bm25) if fts_q else []
        title, seed_pages = self.store.title_candidates(
            normalise_terms(question), fts_q, self.s.title_pages, self.s.title_chunks_per_page
        )
        vec: list[tuple[int, float]] = []
        if self._has_vectors:
            vec = self.store.vector_search(self.embedder.embed_query_sync(question), self.s.top_k_vector)
        return self._fuse(question, bm25, vec, title, top_k, seed_pages)

    def _fuse(self, question: str, bm25: list[tuple[int, float]],
              vec: list[tuple[int, float]], title: list[int], top_k: int,
              seed_pages: list[int] | None = None) -> Retrieval:
        bm_ids = [cid for cid, _ in bm25]
        vec_ids = [cid for cid, _ in vec]
        graph = self._graph_channel(seed_pages or [])
        fused = _rrf([
            (ids, w, k) for ids, w, k in (
                (vec_ids, self.s.weight_vector, self.s.rrf_k),
                (bm_ids, self.s.weight_bm25, self.s.rrf_k),
                (title, self.s.weight_title, self.s.rrf_k_title),
                (graph, self.s.weight_graph, self.s.rrf_k_graph),
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

        facts = self._graph_facts(seed_pages or [])
        context, used = self._pack_context(kept, facts)
        return Retrieval(
            question=question, hits=kept[: len(used)], context=context,
            context_tokens=sum(h.tokens for h in used), vector_used=bool(vec),
            kg_facts=facts,
        )

    def _pack_context(self, hits: list[Hit],
                      facts: list[str] | None = None) -> tuple[str, list[Hit]]:
        budget = self.s.context_token_budget
        blocks: list[str] = []
        used: list[Hit] = []

        if facts:
            # Framed as WikiShia's own structured data, never as neutral fact —
            # the system prompt requires claims to be attributed to the source.
            block = ("Structured facts recorded by WikiShia:\n"
                     + "\n".join(f"- {f}" for f in facts))
            blocks.append(f"[KG] {block}")
            budget -= estimate_tokens(block)
        for h in hits:
            if h.tokens > budget and used:
                continue
            blocks.append(f"[{len(used) + 1}] {h.label}\n{h.text}")
            used.append(h)
            budget -= h.tokens
            if budget <= 0:
                break
        return "\n\n".join(blocks), used
