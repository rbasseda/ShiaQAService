"""Raw cache -> clean text -> chunks -> SQLite (+ BM25 + vectors).

Split into two phases on purpose. Phase 1 (text + lexical index) takes about a
minute. Phase 2 (embeddings) takes hours on a CPU-only machine, so it is
*resumable*: it only ever embeds chunks that have no vector yet. Interrupt it
with Ctrl-C, re-run `shiaqa index --embed-only`, and it picks up where it left off.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass

from ..config import Settings, get_settings
from ..embed.ollama_embed import Embedder
from ..log import get_logger
from ..store.sqlite_store import Store
from ..wiki.fetch import load_aliases, read_raw
from .chunk import chunk_page
from .clean import clean_page

log = get_logger(__name__)


NAV_MARKERS = ("{{disambiguation", "{{disambig", "{{dab")


def is_navigation_page(raw: dict) -> bool:
    title = raw["title"].lower()
    if "(disambiguation)" in title or title.endswith("(disambiguation)"):
        return True
    head = raw["wikitext"][:4000].lower()
    return any(marker in head for marker in NAV_MARKERS)


@dataclass
class IndexReport:
    pages: int = 0
    skipped: int = 0
    chunks: int = 0
    embedded: int = 0
    seconds: float = 0.0


def build_text_index(settings: Settings | None = None, store: Store | None = None) -> IndexReport:
    """Phase 1: clean, chunk, and write text + BM25 index."""
    s = settings or get_settings()
    st = store or Store(s)
    st.reset()

    aliases_by_target = load_aliases(s)
    rep = IndexReport()
    t0 = time.time()

    for raw in read_raw(s):
        # Disambiguation pages are navigation, not knowledge: they are lists of
        # links with no prose, and they outrank real articles on title match.
        if is_navigation_page(raw):
            rep.skipped += 1
            continue
        page = clean_page(raw)
        aliases = aliases_by_target.get(page.title, [])
        chunks = chunk_page(page, s)
        if not chunks:
            continue
        st.add_page(page, aliases)
        st.add_chunks(chunks, aliases=" ".join(aliases))
        rep.pages += 1
        rep.chunks += len(chunks)
        if rep.pages % 500 == 0:
            st.commit()
            log.info("indexed %d pages / %d chunks", rep.pages, rep.chunks)

    st.commit()
    st.set_meta("embed_model", s.embed_model)
    st.set_meta("built_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    st.set_meta("namespaces", json.dumps(s.namespaces))
    rep.seconds = time.time() - t0
    log.info("text index complete: %d pages, %d chunks in %.1fs", rep.pages, rep.chunks, rep.seconds)
    return rep


def build_vectors(settings: Settings | None = None, store: Store | None = None,
                  limit: int | None = None) -> IndexReport:
    """Phase 2: embed every chunk that does not have a vector yet."""
    s = settings or get_settings()
    st = store or Store(s)
    st.init_schema()
    emb = Embedder(s)

    ok, msg = emb.health()
    if not ok:
        raise RuntimeError(msg)

    pending = st.chunks_missing_vectors()
    if limit:
        pending = pending[:limit]
    total = len(pending)
    log.info("embedding %d chunks with %s", total, s.embed_model)

    rep = IndexReport(chunks=total)
    t0 = time.time()
    recent: deque[tuple[float, int]] = deque(maxlen=40)  # trailing window for rate/ETA
    for i in range(0, total, s.embed_batch):
        batch = pending[i : i + s.embed_batch]
        # Embed with the context header, exactly as `Chunk.embed_text()` does.
        texts = [
            f"{r['title']} — {r['section']}\n{r['text']}" if r["section"] != "Summary"
            else f"{r['title']}\n{r['text']}"
            for r in batch
        ]
        vecs = emb.embed_documents(texts)
        st.add_vectors([(r["id"], v) for r, v in zip(batch, vecs)])
        rep.embedded += len(batch)

        recent.append((time.time(), rep.embedded))
        if (i // s.embed_batch) % 10 == 0 or rep.embedded == total:
            st.commit()
            done = rep.embedded
            # Rate over a trailing window, not since start: a laptop that slept
            # or was busy elsewhere would otherwise report a cumulative average
            # far below the real throughput, and a wildly inflated ETA.
            t_old, n_old = recent[0]
            span = max(time.time() - t_old, 1e-6)
            rate = (done - n_old) / span or done / max(time.time() - t0, 1e-6)
            eta = (total - done) / rate / 60 if rate else 0
            log.info("embedded %d/%d (%.1f/s, ETA %.0f min)", done, total, rate, eta)

    st.commit()
    emb.close()
    rep.seconds = time.time() - t0
    log.info("vectors complete: %d in %.1f min", rep.embedded, rep.seconds / 60)
    return rep


def prune(settings: Settings | None = None, store: Store | None = None) -> dict[str, int]:
    """Maintenance pass over an existing index.

    Drops navigation pages and any vectors orphaned by earlier edits, without
    discarding the embeddings — which is the point, since rebuilding from
    scratch costs hours on a CPU-only machine.
    """
    s = settings or get_settings()
    st = store or Store(s)

    nav = [
        r["page_id"] for r in st.db.execute(
            "SELECT page_id, title FROM pages WHERE lower(title) LIKE '%(disambiguation)%'"
        ).fetchall()
    ]
    removed_chunks = sum(st.delete_page(pid) for pid in nav)

    orphans = st.db.execute(
        "DELETE FROM vec_chunks WHERE chunk_id NOT IN (SELECT id FROM chunks)"
    ).rowcount
    st.commit()
    st.db.execute("VACUUM")
    log.info("pruned %d navigation pages (%d chunks), %d orphan vectors",
             len(nav), removed_chunks, max(orphans, 0))
    return {"pages": len(nav), "chunks": removed_chunks, "orphan_vectors": max(orphans, 0)}
