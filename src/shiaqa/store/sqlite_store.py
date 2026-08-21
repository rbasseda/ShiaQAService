"""One SQLite file holds everything: text, BM25 index and vectors.

No server, no daemon, ~500 MB on disk. On a CPU-only laptop that beats any
client/server vector database, and `sqlite-vec` + FTS5 in the same transaction
means the lexical and semantic views can never drift out of sync.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import sqlite_vec

from ..config import Settings, get_settings
from ..ingest.chunk import Chunk
from ..log import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    page_id    INTEGER PRIMARY KEY,
    ns         INTEGER NOT NULL,
    title      TEXT NOT NULL,
    url        TEXT NOT NULL,
    revid      INTEGER,
    timestamp  TEXT,
    categories TEXT,
    aliases    TEXT
);
CREATE INDEX IF NOT EXISTS pages_title ON pages(title);

CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY,
    page_id  INTEGER NOT NULL REFERENCES pages(page_id),
    ns       INTEGER NOT NULL,
    title    TEXT NOT NULL,
    url      TEXT NOT NULL,
    section  TEXT NOT NULL,
    ordinal  INTEGER NOT NULL,
    kind     TEXT NOT NULL,
    text     TEXT NOT NULL,
    tokens   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_page ON chunks(page_id);

-- remove_diacritics 2 is the important bit: it folds the transliteration
-- diacritics this wiki is full of, so a query for "Husayn" reaches "Ḥusayn"
-- and "Ali" reaches "ʿAlī".
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title, section, aliases, text,
    content='', tokenize="unicode61 remove_diacritics 2"
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


# An article counts as "named by the question" when the question accounts for
# at least this fraction of the article's own title (or one of its aliases).
TITLE_COVERAGE_MIN = 0.6

# Dropped when comparing a question against a title: disambiguators and
# honorific suffixes that readers never type. "Imam Ali (a)" -> {imam, ali}.
_TITLE_NOISE = {"a", "s", "as", "ra", "book", "the", "of", "al", "b", "bt", "ibn", "bin"}
_TITLE_PAREN_RE = re.compile(r"\([^)]*\)")
_TITLE_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def normalise_terms(text: str) -> set[str]:
    """Fold a title or query into comparable bare terms (diacritics removed)."""
    text = _TITLE_PAREN_RE.sub(" ", text.lower())
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    out = set()
    for tok in _TITLE_WORD_RE.findall(text):
        tok = tok.strip("'\u2019-")
        if len(tok) > 1 and tok not in _TITLE_NOISE:
            out.add(tok)
    return out


def _f32(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


@dataclass
class Hit:
    chunk_id: int
    page_id: int
    title: str
    url: str
    section: str
    kind: str
    text: str
    tokens: int
    score: float
    vec_rank: int | None = None
    bm25_rank: int | None = None

    @property
    def label(self) -> str:
        return f"{self.title} — {self.section}" if self.section != "Summary" else self.title


class Store:
    def __init__(self, settings: Settings | None = None, path: Path | None = None) -> None:
        self.s = settings or get_settings()
        self.path = path or self.s.db_path
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")

    # ---- lifecycle ---------------------------------------------------------

    def init_schema(self) -> None:
        self.db.executescript(SCHEMA)
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{self.s.embed_dim}])"
        )
        self.db.commit()

    def reset(self) -> None:
        for tbl in ("vec_chunks", "chunks_fts", "chunks", "pages", "meta"):
            self.db.execute(f"DROP TABLE IF EXISTS {tbl}")
        self.db.commit()
        self.init_schema()

    def close(self) -> None:
        self.db.close()

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
        self.db.commit()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def counts(self) -> dict[str, int]:
        out = {}
        for tbl in ("pages", "chunks", "vec_chunks"):
            try:
                out[tbl] = self.db.execute(f"SELECT count(*) c FROM {tbl}").fetchone()["c"]
            except sqlite3.Error:
                out[tbl] = 0
        return out

    # ---- writing -----------------------------------------------------------

    def add_page(self, page, aliases: list[str]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?,?,?,?)",
            (page.pageid, page.ns, page.title, page.url, page.revid, page.timestamp,
             json.dumps(page.categories, ensure_ascii=False),
             json.dumps(aliases, ensure_ascii=False)),
        )

    def add_chunks(self, chunks: Iterable[Chunk], aliases: str = "") -> list[int]:
        ids: list[int] = []
        for c in chunks:
            cur = self.db.execute(
                "INSERT INTO chunks (page_id, ns, title, url, section, ordinal, kind, text, tokens)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (c.page_id, c.ns, c.title, c.url, c.section, c.ordinal, c.kind, c.text, c.tokens),
            )
            cid = cur.lastrowid
            ids.append(cid)
            # Aliases ride along in the lexical index only: they help a query
            # for "Abu Turab" find the Imam Ali article without polluting the
            # text that gets embedded or shown to the model.
            self.db.execute(
                "INSERT INTO chunks_fts (rowid, title, section, aliases, text) VALUES (?,?,?,?,?)",
                (cid, c.title, c.section, aliases, c.text),
            )
        return ids

    def delete_page(self, page_id: int) -> int:
        """Remove a page and everything derived from it. Returns chunks removed.

        `chunks_fts` is contentless (`content=''`), which means a plain DELETE
        is rejected — FTS5 needs its rows retired with the special 'delete'
        command, replaying the exact values that were indexed.
        """
        rows = self.db.execute(
            "SELECT c.id, c.title, c.section, c.text, p.aliases FROM chunks c "
            "JOIN pages p ON p.page_id = c.page_id WHERE c.page_id = ?",
            (page_id,),
        ).fetchall()
        for r in rows:
            aliases = " ".join(json.loads(r["aliases"] or "[]"))
            self.db.execute(
                "INSERT INTO chunks_fts (chunks_fts, rowid, title, section, aliases, text) "
                "VALUES ('delete', ?, ?, ?, ?, ?)",
                (r["id"], r["title"], r["section"], aliases, r["text"]),
            )
            self.db.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (r["id"],))
        self.db.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))
        self.db.execute("DELETE FROM pages WHERE page_id = ?", (page_id,))
        return len(rows)

    def add_vectors(self, pairs: Sequence[tuple[int, Sequence[float]]]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO vec_chunks (chunk_id, embedding) VALUES (?,?)",
            [(cid, _f32(vec)) for cid, vec in pairs],
        )

    def commit(self) -> None:
        self.db.commit()

    def chunks_missing_vectors(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT c.id, c.title, c.section, c.text FROM chunks c "
            "LEFT JOIN vec_chunks v ON v.chunk_id = c.id WHERE v.chunk_id IS NULL ORDER BY c.id"
        ).fetchall()

    # ---- reading -----------------------------------------------------------

    def _rows_to_hits(self, ids: Sequence[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"SELECT id, page_id, title, url, section, kind, text, tokens FROM chunks WHERE id IN ({marks})",
            list(ids),
        ).fetchall()
        return {r["id"]: r for r in rows}

    def vector_search(self, embedding: Sequence[float], k: int) -> list[tuple[int, float]]:
        rows = self.db.execute(
            "SELECT chunk_id, distance FROM vec_chunks "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_f32(embedding), k),
        ).fetchall()
        return [(r["chunk_id"], r["distance"]) for r in rows]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        try:
            rows = self.db.execute(
                "SELECT rowid, bm25(chunks_fts, 6.0, 2.0, 4.0, 1.0) AS score FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (query, k),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS5 syntax in a user question; caller falls back.
            return []
        return [(r["rowid"], r["score"]) for r in rows]

    def title_search(self, terms: set[str], fts_query: str,
                     pages: int, per_page: int) -> list[int]:
        """Chunks from articles the question actually *names*.

        Recall comes from FTS over titles and redirect aliases — 35k redirects
        mean nearly every spelling, epithet and kunya a reader might type
        ("Abu Turab", "Amir al-Mu'minin") points at the right article.
        Precision comes from re-scoring each candidate on *coverage*: how much
        of the article's own name the question accounts for. Plain BM25 is not
        enough here, because a rare word matching one alias of an unrelated
        page ("community" -> an Islamic centre) outranks the obvious article.
        Matched articles contribute their fact box and summary first.
        """
        if not fts_query or not terms:
            return []
        try:
            rows = self.db.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts, 8.0, 0.0, 5.0, 0.0) LIMIT 600",
                (f"{{title aliases}} : ({fts_query})",),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        if not rows:
            return []

        ids = [r["rowid"] for r in rows]
        marks = ",".join("?" * len(ids))
        meta = self.db.execute(
            f"SELECT c.id, c.page_id, c.kind, c.ordinal, p.title, p.aliases "
            f"FROM chunks c JOIN pages p ON p.page_id = c.page_id WHERE c.id IN ({marks})",
            ids,
        ).fetchall()

        by_page: dict[int, list] = {}
        names: dict[int, tuple[str, str]] = {}
        for r in meta:
            by_page.setdefault(r["page_id"], []).append(r)
            names[r["page_id"]] = (r["title"], r["aliases"])

        scored: list[tuple[float, int, int, int]] = []
        for pid, (title, aliases_json) in names.items():
            alias_list = json.loads(aliases_json or "[]")
            candidates = [title] + alias_list
            best = 0.0
            best_n = 0
            for name in candidates:
                name_terms = normalise_terms(name)
                if not name_terms:
                    continue
                hit = name_terms & terms
                if not hit:
                    continue
                coverage = len(hit) / len(name_terms)
                if (coverage, len(hit)) > (best, best_n):
                    best, best_n = coverage, len(hit)
            if best >= TITLE_COVERAGE_MIN:
                # Ties are common and consequential: "Imam al-Husayn b. Ali (a)"
                # and "Al-Husayn b. al-Imam al-Kazim (a)" both cover 2 of 3
                # title terms for a question about "Imam Husayn". Redirect count
                # is a good prominence prior — the famous article accumulates
                # far more alternative spellings (102 vs 28 here) — so it breaks
                # the tie towards the article a reader almost certainly meant.
                scored.append((best, best_n, len(alias_list), pid))

        scored.sort(key=lambda t: (-t[0], -t[1], -t[2]))
        page_order = [pid for *_, pid in scored[:pages]]

        order = {"infobox": 0, "summary": 1, "body": 2}
        for pid in page_order:
            by_page[pid].sort(key=lambda r: (order.get(r["kind"], 3), r["ordinal"]))
            by_page[pid] = by_page[pid][:per_page]

        # Round-robin across matched articles so one page cannot own the list.
        out: list[int] = []
        for slot in range(per_page):
            for pid in page_order:
                if slot < len(by_page[pid]):
                    out.append(by_page[pid][slot]["id"])
        return out

    def hydrate(self, scored: Sequence[tuple[int, float]],
                ranks: dict[int, tuple[int | None, int | None]] | None = None) -> list[Hit]:
        rows = self._rows_to_hits([cid for cid, _ in scored])
        hits: list[Hit] = []
        for cid, score in scored:
            r = rows.get(cid)
            if r is None:
                continue
            vr, br = (ranks or {}).get(cid, (None, None))
            hits.append(Hit(
                chunk_id=cid, page_id=r["page_id"], title=r["title"], url=r["url"],
                section=r["section"], kind=r["kind"], text=r["text"], tokens=r["tokens"],
                score=score, vec_rank=vr, bm25_rank=br,
            ))
        return hits
