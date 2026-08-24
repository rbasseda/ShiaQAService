"""`data/kg.db` — the derived, queryable side of the knowledge graph.

Deliberately a *separate* SQLite file from `data/shiaqa.db`. `Store.reset()`
drops every table it owns and `shiaqa index` with no flags calls it, so a graph
kept in that file would be collateral damage of a routine reindex. This one is
rebuilt from `data/kg/*.jsonl` in seconds and never needs an embedding pass.

Rows reference `page_id`, never `chunk_id`: MediaWiki page ids survive a
rebuild, `chunks.id` is an autoincrement rowid and does not.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

from ..config import Settings, get_settings
from ..log import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id      TEXT PRIMARY KEY,
    kind    TEXT NOT NULL,          -- entity | class | date
    label   TEXT NOT NULL,
    page_id INTEGER,
    url     TEXT,
    data    TEXT                    -- JSON: aliases, calendar components, ...
);
CREATE INDEX IF NOT EXISTS nodes_page ON nodes(page_id);
CREATE INDEX IF NOT EXISTS nodes_label ON nodes(label);

CREATE TABLE IF NOT EXISTS node_classes (node_id TEXT NOT NULL, class TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS node_classes_node ON node_classes(node_id);
CREATE INDEX IF NOT EXISTS node_classes_class ON node_classes(class);

CREATE TABLE IF NOT EXISTS edges (
    s     TEXT NOT NULL,
    p     TEXT NOT NULL,
    o     TEXT NOT NULL,
    src   TEXT,                     -- infobox | category  (llm, later)
    field TEXT,
    raw   TEXT
);
-- Both directions are indexed, which is what lets traversal read an edge
-- backwards instead of materialising a second inverse row for every fact.
CREATE INDEX IF NOT EXISTS edges_s ON edges(s, p);
CREATE INDEX IF NOT EXISTS edges_o ON edges(o, p);

CREATE TABLE IF NOT EXISTS links (s TEXT NOT NULL, o TEXT NOT NULL, n INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS links_s ON links(s);
CREATE INDEX IF NOT EXISTS links_o ON links(o);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

# Traversed first when expanding a question's named entity. Relations a reader
# is likely to be asking *about* outrank incidental ones like `affiliation`.
PREDICATE_PRIORITY: dict[str, int] = {
    "authored_by": 0, "authored": 0, "taught_by": 0, "taught": 0,
    "companion_of": 0, "issued_by": 0, "in_sura": 0, "father": 0, "mother": 0,
    "child": 0, "spouse": 0, "sibling": 0, "commander": 0, "combatant": 0,
    "narrated_by": 1, "narrated_from": 1, "source_shia": 1, "source_sunni": 1,
    "buried_at": 1, "born_in": 1, "died_in": 1, "occurred_at": 1,
    "succeeded_by": 1, "preceded_by": 1, "relative_of": 1, "descends_from": 1,
    "about_subject": 2, "known_for": 2, "participated_in": 2, "contains": 2,
    "revealed_at": 2, "revealed_because": 2, "related_verse": 2,
}
DEFAULT_PRIORITY = 3


class KgStore:
    def __init__(self, settings: Settings | None = None, path: Path | None = None) -> None:
        self.s = settings or get_settings()
        self.path = path or self.s.kg_db_path
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")

    # ---- lifecycle ---------------------------------------------------------

    def init_schema(self) -> None:
        self.db.executescript(SCHEMA)
        self.db.commit()

    def reset(self) -> None:
        """Drop and recreate. Safe: everything here is re-derived in seconds."""
        for tbl in ("nodes", "node_classes", "edges", "links", "meta"):
            self.db.execute(f"DROP TABLE IF EXISTS {tbl}")
        self.db.commit()
        self.init_schema()

    def close(self) -> None:
        self.db.close()

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        try:
            row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        except sqlite3.Error:
            return default
        return row["value"] if row else default

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for tbl in ("nodes", "edges", "links"):
            try:
                out[tbl] = self.db.execute(f"SELECT count(*) c FROM {tbl}").fetchone()["c"]
            except sqlite3.Error:
                out[tbl] = 0
        try:
            out["typed_edges"] = self.db.execute(
                "SELECT count(*) c FROM edges WHERE p NOT IN ('instance_of','subclass_of')"
            ).fetchone()["c"]
        except sqlite3.Error:
            out["typed_edges"] = 0
        return out

    @property
    def is_populated(self) -> bool:
        return self.counts().get("edges", 0) > 0

    # ---- writing -----------------------------------------------------------

    def load(self, nodes: Iterable[dict], edges: Iterable[dict], links: Iterable[dict]) -> None:
        reserved = {"id", "kind", "label", "page_id", "url", "classes"}
        for n in nodes:
            extra = {k: v for k, v in n.items() if k not in reserved}
            self.db.execute(
                "INSERT OR REPLACE INTO nodes (id, kind, label, page_id, url, data)"
                " VALUES (?,?,?,?,?,?)",
                (n["id"], n["kind"], n["label"], n.get("page_id"), n.get("url"),
                 json.dumps(extra, ensure_ascii=False) if extra else None),
            )
            for cls in n.get("classes") or []:
                self.db.execute("INSERT INTO node_classes VALUES (?,?)", (n["id"], cls))
        self.db.executemany(
            "INSERT INTO edges (s,p,o,src,field,raw) VALUES (?,?,?,?,?,?)",
            [(e["s"], e["p"], e["o"], e.get("src"), e.get("field"), e.get("raw"))
             for e in edges],
        )
        self.db.executemany(
            "INSERT INTO links (s,o,n) VALUES (?,?,?)",
            [(l["s"], l["o"], l["n"]) for l in links],
        )
        self.db.commit()

    # ---- reading -----------------------------------------------------------

    def node(self, node_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()

    def node_for_page(self, page_id: int) -> str | None:
        row = self.db.execute(
            "SELECT id FROM nodes WHERE page_id=? AND kind='entity'", (page_id,)
        ).fetchone()
        return row["id"] if row else None

    def find_by_label(self, label: str, limit: int = 10) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM nodes WHERE kind='entity' AND label LIKE ? "
            "ORDER BY length(label) LIMIT ?", (f"%{label}%", limit),
        ).fetchall()

    def classes_of(self, node_id: str) -> list[str]:
        return [r["class"] for r in self.db.execute(
            "SELECT class FROM node_classes WHERE node_id=?", (node_id,))]

    def edges_of(self, node_id: str, include_taxonomy: bool = False) -> list[dict]:
        """Every typed edge touching this node, each tagged with its direction."""
        filt = "" if include_taxonomy else " AND p NOT IN ('instance_of','subclass_of')"
        out: list[dict] = []
        for r in self.db.execute(
            f"SELECT e.*, n.label AS o_label, n.kind AS o_kind FROM edges e "
            f"LEFT JOIN nodes n ON n.id = e.o WHERE e.s = ?{filt}", (node_id,)
        ):
            out.append({"dir": "out", "p": r["p"], "other": r["o"],
                        "label": r["o_label"], "kind": r["o_kind"], "field": r["field"]})
        for r in self.db.execute(
            f"SELECT e.*, n.label AS s_label, n.kind AS s_kind FROM edges e "
            f"LEFT JOIN nodes n ON n.id = e.s WHERE e.o = ?{filt}", (node_id,)
        ):
            out.append({"dir": "in", "p": r["p"], "other": r["s"],
                        "label": r["s_label"], "kind": r["s_kind"], "field": r["field"]})
        out.sort(key=lambda e: PREDICATE_PRIORITY.get(e["p"], DEFAULT_PRIORITY))
        return out

    def neighbour_pages(self, seed_node_ids: Sequence[str], limit: int) -> list[int]:
        """Page ids one typed hop from the seeds, best predicates first.

        Round-robin across seeds so a single well-connected article cannot own
        the whole expansion — the same reasoning as `Store.title_search`.
        """
        if not seed_node_ids:
            return []
        seeds = set(seed_node_ids)
        per_seed: list[list[int]] = []
        for node_id in seed_node_ids:
            ranked: list[int] = []
            seen: set[int] = set()
            for e in self.edges_of(node_id):
                if e["kind"] != "entity" or e["other"] in seeds:
                    continue
                row = self.node(e["other"])
                if row is None or row["page_id"] is None:
                    continue
                if row["page_id"] in seen:
                    continue
                seen.add(row["page_id"])
                ranked.append(row["page_id"])
            per_seed.append(ranked)

        out: list[int] = []
        for slot in range(max((len(r) for r in per_seed), default=0)):
            for ranked in per_seed:
                if slot < len(ranked) and ranked[slot] not in out:
                    out.append(ranked[slot])
                    if len(out) >= limit:
                        return out
        return out

    def facts(self, node_id: str, predicates: dict, limit: int) -> list[str]:
        """Render this node's edges as readable one-line statements.

        Deduplicated, because the same fact is often asserted from both ends:
        al-Kulayni's `works` field names al-Kafi, and al-Kafi's `author` field
        names al-Kulayni. Two independent edges, one sentence.
        """
        row = self.node(node_id)
        if row is None:
            return []
        subject = row["label"]
        out: list[str] = []
        seen: set[str] = set()
        for e in self.edges_of(node_id):
            if not e["label"]:
                continue
            pred = predicates.get(e["p"])
            if pred is None:
                continue
            if e["dir"] == "out":
                line = f"{subject} {pred.label} {e['label']}."
            else:
                inverse = predicates.get(pred.inverse) if pred.inverse else None
                line = (f"{subject} {inverse.label} {e['label']}." if inverse
                        else f"{e['label']} {pred.label} {subject}.")
            if line in seen:
                continue
            seen.add(line)
            out.append(line)
            if len(out) >= limit:
                break
        return out
