"""Graph -> self-contained HTML.

Two views, because the whole graph has no useful picture. 7,548 nodes and 9,445
typed edges at a max degree of 382 is a hairball in any force layout, so each
view is scoped instead: the explorer draws one neighbourhood at a time, the
schema view draws one level of abstraction.

Both embed their data. The typed graph trims to well under a megabyte, so the
output is a single file that works offline with no server and no CDN — which it
has to be, since an Artifact's CSP blocks external hosts outright.
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ..config import Settings, get_settings
from ..log import get_logger
from .ontology import Ontology
from .store import KgStore

log = get_logger(__name__)

TEMPLATES = "shiaqa.kg.templates"
UNCLASSED = "(unclassed)"

# Object of an edge whose id starts with this is a minted calendar node, not a page.
_DATE_PREFIX = "date:"


def _read_template(name: str) -> str:
    return resources.files(TEMPLATES).joinpath(name).read_text(encoding="utf-8")


def render(template: str, payload: dict, title: str) -> str:
    """Fill a template's placeholders. `</script>` in data would end the block early."""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/")
    return (_read_template(template)
            .replace("{{TITLE}}", title)
            .replace("{{DATA}}", data))


# --- explorer ----------------------------------------------------------------

def explorer_payload(kg: KgStore, onto: Ontology | None = None,
                     settings: Settings | None = None) -> dict:
    """Every typed edge, interned so the whole graph fits in one file.

    Article URLs are reconstructed in the page from the title rather than stored:
    7.5k of them is nearly 400 KB of pure redundancy.
    """
    onto = onto or Ontology()
    s = settings or get_settings()

    rows = kg.db.execute(
        "SELECT id, kind, label, page_id FROM nodes ORDER BY id"
    ).fetchall()
    index = {r["id"]: i for i, r in enumerate(rows)}
    nodes = [[r["label"], {"entity": 0, "class": 1, "date": 2}.get(r["kind"], 0)]
             for r in rows]

    # Classes per node, as indexes into a small vocabulary.
    class_names: list[str] = []
    class_index: dict[str, int] = {}
    node_classes: dict[int, list[int]] = {}
    for r in kg.db.execute("SELECT node_id, class FROM node_classes"):
        i = index.get(r["node_id"])
        if i is None:
            continue
        ci = class_index.setdefault(r["class"], len(class_names))
        if ci == len(class_names):
            class_names.append(r["class"])
        node_classes.setdefault(i, []).append(ci)

    # Typed edges only. The 107k `mentions` links are 4.9 MB and say nothing at
    # this zoom level; `instance_of` would swamp the picture with 13.5k more.
    preds: list[str] = []
    pred_index: dict[str, int] = {}
    fields: list[str] = [""]
    field_index: dict[str, int] = {"": 0}
    srcs: list[str] = []
    src_index: dict[str, int] = {}
    edges: list[list[int]] = []

    for r in kg.db.execute(
        "SELECT s, p, o, field, src FROM edges "
        "WHERE p NOT IN ('instance_of','subclass_of')"
    ):
        si, oi = index.get(r["s"]), index.get(r["o"])
        if si is None or oi is None:
            continue
        pi = pred_index.setdefault(r["p"], len(preds))
        if pi == len(preds):
            preds.append(r["p"])
        fi = field_index.setdefault(r["field"] or "", len(fields))
        if fi == len(fields):
            fields.append(r["field"] or "")
        ci = src_index.setdefault(r["src"] or "", len(srcs))
        if ci == len(srcs):
            srcs.append(r["src"] or "")
        edges.append([si, pi, oi, fi, ci])

    # Category membership, shown as text in the sidebar rather than as nodes.
    cats: dict[int, list[int]] = {}
    for r in kg.db.execute("SELECT s, o FROM edges WHERE p = 'instance_of'"):
        si, oi = index.get(r["s"]), index.get(r["o"])
        if si is not None and oi is not None:
            cats.setdefault(si, []).append(oi)

    connected = {e[0] for e in edges} | {e[2] for e in edges}
    return {
        "nodes": nodes,
        "nodeClasses": {str(k): v for k, v in node_classes.items()},
        "classNames": class_names,
        "preds": preds,
        "predLabels": [onto.predicates[p].label if p in onto.predicates else p
                       for p in preds],
        "predInverse": [
            preds.index(onto.predicates[p].inverse)
            if p in onto.predicates and onto.predicates[p].inverse in preds else -1
            for p in preds
        ],
        "fields": fields,
        "srcs": srcs,
        "edges": edges,
        "cats": {str(k): v for k, v in cats.items()},
        "connected": sorted(connected),
        "viewBase": s.wiki_view_base,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "connected": len(connected),
            "entities": sum(1 for n in nodes if n[1] == 0),
        },
    }


# --- schema ------------------------------------------------------------------

@dataclass
class SchemaTriple:
    subject: str
    predicate: str
    object: str
    count: int


def schema_payload(kg: KgStore, onto: Ontology | None = None) -> dict:
    """Class-level shape of the graph: which classes relate to which, and how often.

    `(unclassed)` is a first-class bucket here rather than something filtered
    out. 57% of entities carry no class, because most pages have no infobox, and
    a diagram that hid that would imply a tidier ontology than we actually have.
    """
    onto = onto or Ontology()

    primary: dict[str, str] = {}
    for r in kg.db.execute("SELECT node_id, class FROM node_classes"):
        primary.setdefault(r["node_id"], r["class"])

    counts: dict[tuple[str, str, str], int] = {}
    for r in kg.db.execute(
        "SELECT s, p, o FROM edges WHERE p NOT IN ('instance_of','subclass_of')"
    ):
        sc = primary.get(r["s"], UNCLASSED)
        oc = "Date" if r["o"].startswith(_DATE_PREFIX) else primary.get(r["o"], UNCLASSED)
        key = (sc, r["p"], oc)
        counts[key] = counts.get(key, 0) + 1

    triples = sorted(
        (SchemaTriple(s, p, o, n) for (s, p, o), n in counts.items()),
        key=lambda t: -t.count,
    )

    class_sizes: dict[str, int] = {}
    for r in kg.db.execute("SELECT class, count(*) c FROM node_classes GROUP BY class"):
        class_sizes[r["class"]] = r["c"]
    for t in triples:                      # buckets that are not real classes
        class_sizes.setdefault(t.subject, 0)
        class_sizes.setdefault(t.object, 0)

    # Predicates the ontology defines but the corpus never fills — the honest
    # gap between what we can express and what WikiShia actually records.
    used = {t.predicate for t in triples}
    unused = sorted(p for p in onto.predicates if p not in used)

    return {
        "classes": [{"name": k, "entities": v} for k, v in
                    sorted(class_sizes.items(), key=lambda kv: -kv[1])],
        "triples": [[t.subject, t.predicate, t.object, t.count] for t in triples],
        "predLabels": {p: onto.predicates[p].label for p in onto.predicates},
        "unusedPredicates": unused,
        "stats": {
            "classes": len(onto.classes),
            "predicates": len(onto.predicates),
            "triples": len(triples),
            "edges": sum(t.count for t in triples),
        },
    }


# --- entry point -------------------------------------------------------------

def write(out_dir: Path, kinds: tuple[str, ...] = ("explorer", "schema"),
          settings: Settings | None = None) -> dict[str, Path]:
    s = settings or get_settings()
    onto = Ontology()
    kg = KgStore(s)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if "explorer" in kinds:
        path = out_dir / "explorer.html"
        path.write_text(render("explorer.html", explorer_payload(kg, onto, s),
                               "WikiShia Graph Explorer"), encoding="utf-8")
        written["explorer"] = path
    if "schema" in kinds:
        path = out_dir / "schema.html"
        path.write_text(render("schema.html", schema_payload(kg, onto),
                               "WikiShia Graph Schema"), encoding="utf-8")
        written["schema"] = path

    kg.close()
    for kind, path in written.items():
        log.info("wrote %s (%.1f KB)", path, path.stat().st_size / 1024)
    return written
