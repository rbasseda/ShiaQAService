"""Raw wikitext cache -> typed nodes, edges and an untyped link graph.

This is a *second* pass over `data/raw/`, deliberately not sharing the ingest
path. `clean._parse_infobox` renders `[[Imam al-Mahdi (a)]]` down to the string
"Imam al-Mahdi (a)" and throws the link target away, which is exactly the thing
a graph needs. Rather than widen that return type — the text it produces is
already embedded in `data/shiaqa.db`, and changing it would silently invalidate
every stored vector — the extractor reads the templates itself and takes the
targets before anything is flattened.

Nothing here touches `shiaqa.db`, and nothing here calls a model. A full run is
file I/O plus one `mwparserfromhell` parse per page.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Sequence

from ..config import Settings, get_settings
from ..ingest.clean import _parse
from ..ingest.pipeline import is_navigation_page
from ..log import get_logger
from ..wiki.fetch import load_redirects, read_raw
from . import dates, overrides as overrides_mod
from .ontology import Ontology

log = get_logger(__name__)

# Wikilink, keeping the target and discarding any display text and anchor.
_LINK_RE = re.compile(r"\[\[\s*([^\[\]|]+?)\s*(?:\|[^\[\]]*)?\]\]")

# Namespace prefixes that are not content. `Text:` is deliberately absent —
# on WikiShia it is namespace 3000, holding primary source texts we do ingest.
_NON_CONTENT_NS = frozenset({
    "category", "file", "image", "media", "template", "help", "user", "talk",
    "special", "mediawiki", "module", "portal", "book", "draft",
    "category talk", "file talk", "template talk", "user talk", "help talk",
})

_CATEGORY_PREFIX = re.compile(r"^\s*category\s*:\s*", re.I)


def normalise_title(raw: str) -> str:
    """Apply MediaWiki's own title rules: underscores, spacing, initial capital."""
    title = raw.replace("_", " ").strip()
    title = title.split("#", 1)[0].strip()   # drop section anchors
    return title[:1].upper() + title[1:] if title else ""


def _is_content_link(target: str) -> bool:
    if ":" not in target:
        return True
    prefix = target.split(":", 1)[0].strip().lower()
    # Short prefixes are interwiki/language codes ("en:", "ar:", "fa:").
    return prefix not in _NON_CONTENT_NS and len(prefix) > 3


@dataclass
class ExtractReport:
    pages: int = 0
    skipped_nav: int = 0
    entities: int = 0
    class_nodes: int = 0
    date_nodes: int = 0
    typed_edges: int = 0
    instance_of: int = 0
    subclass_of: int = 0
    date_edges: int = 0
    links: int = 0
    unresolved_targets: int = 0
    range_rejected: int = 0
    overrides_dropped: int = 0
    overrides_added: int = 0
    overrides_stale: int = 0
    unmapped_fields: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class Extraction:
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    unmapped: list[dict] = field(default_factory=list)
    report: ExtractReport = field(default_factory=ExtractReport)
    overrides: "overrides_mod.OverrideReport" = field(
        default_factory=lambda: overrides_mod.OverrideReport())


class Extractor:
    def __init__(self, settings: Settings | None = None,
                 ontology: Ontology | None = None,
                 overrides: Sequence[overrides_mod.EdgeOverride] | None = None) -> None:
        self.s = settings or get_settings()
        self.onto = ontology or Ontology()
        self.overrides = (overrides_mod.OVERRIDES if overrides is None else overrides)
        self.redirects: dict[str, str] = {}
        self.page_id: dict[str, int] = {}      # canonical title -> page id
        self.aliases: dict[int, list[str]] = {}

    # ---- resolution --------------------------------------------------------

    def resolve(self, raw_target: str) -> str | None:
        """Wikilink target -> canonical article title, or None if it lands nowhere."""
        title = normalise_title(raw_target)
        if not title or not _is_content_link(title):
            return None
        if title in self.page_id:
            return title
        via = self.redirects.get(title)
        return via if via in self.page_id else None

    def node_for(self, title: str) -> str | None:
        pid = self.page_id.get(title)
        return f"page:{pid}" if pid is not None else None

    def resolve_node(self, title: str) -> str | None:
        """Article title -> node id, honouring redirects. Used by overrides."""
        canonical = self.resolve(title)
        return self.node_for(canonical) if canonical else None

    # ---- passes ------------------------------------------------------------

    def _index_pages(self, rep: ExtractReport) -> list[dict]:
        """First pass: build the title -> page_id map the whole run depends on."""
        entity_pages: list[dict] = []
        for raw in read_raw(self.s):
            if raw["ns"] == 14:
                continue                       # categories become class nodes, not entities
            if is_navigation_page(raw):
                rep.skipped_nav += 1
                continue                       # keep the KG's page set aligned with the index
            self.page_id[raw["title"]] = raw["pageid"]
            entity_pages.append(raw)

        by_target: dict[int, list[str]] = {}
        for alias, target in self.redirects.items():
            pid = self.page_id.get(target)
            if pid is not None:
                by_target.setdefault(pid, []).append(alias)
        self.aliases = by_target
        return entity_pages

    def _infoboxes(self, wikitext: str) -> list:
        """Top-level infobox templates, outermost first."""
        code = _parse(wikitext)
        out = []
        for node in code.filter_templates(recursive=False):
            name = str(node.name).strip().lower()
            if name.startswith("infobox") or name.startswith("جعبه"):
                out.append(node)
        return out

    def run(self) -> Extraction:
        import time
        t0 = time.time()

        self.redirects = {normalise_title(k): v for k, v in load_redirects(self.s).items()}
        ex = Extraction()
        rep = ex.report
        pages = self._index_pages(rep)
        rep.pages = len(pages)
        log.info("indexed %d content pages (%d navigation pages skipped)",
                 len(pages), rep.skipped_nav)

        class_nodes: dict[str, dict] = {}
        date_nodes: dict[str, dict] = {}
        link_counts: collections.Counter[tuple[str, str]] = collections.Counter()
        unmapped: collections.Counter[tuple[str, str]] = collections.Counter()
        unmapped_links: collections.Counter[tuple[str, str]] = collections.Counter()

        def class_node(name: str) -> str:
            nid = f"cat:{name}"
            class_nodes.setdefault(nid, {"id": nid, "kind": "class", "label": name})
            return nid

        # --- pass 1: nodes, classes, taxonomy, link graph ---
        # Infobox fields are stashed as plain strings rather than re-parsed in
        # pass 2: range checks need every page's classes known up front, and
        # parsing 4.9k pages twice would double the run for nothing.
        classes_of: dict[str, set[str]] = {}
        pending: list[tuple[str, list[tuple[str, str, str]]]] = []

        for raw in pages:
            title, pid = raw["title"], raw["pageid"]
            subject = f"page:{pid}"
            boxes = self._infoboxes(raw["wikitext"])

            classes: list[str] = []
            fields: list[tuple[str, str, str]] = []
            for box in boxes:
                tname = str(box.name).strip().lower()
                for cls in self.onto.classes_for(tname):
                    if cls not in classes:
                        classes.append(cls)
                for param in box.params:
                    fname = str(param.name).strip()
                    value = str(param.value).strip()
                    if fname and not fname.isdigit() and value:
                        fields.append((tname, fname, value))

            classes_of[subject] = set(classes)
            if fields:
                pending.append((subject, fields))

            ex.nodes.append({
                "id": subject, "kind": "entity", "label": title,
                "classes": classes, "page_id": pid, "ns": raw["ns"],
                "url": raw["url"], "aliases": sorted(self.aliases.get(pid, [])),
            })
            rep.entities += 1

            for cat in raw.get("categories") or []:
                name = _CATEGORY_PREFIX.sub("", cat).strip()
                if not name:
                    continue
                ex.edges.append({"s": subject, "p": "instance_of",
                                 "o": class_node(name), "src": "category"})
                rep.instance_of += 1

            for m in _LINK_RE.finditer(raw["wikitext"]):
                target = self.resolve(m.group(1))
                if target is None or target == title:
                    continue
                obj = self.node_for(target)
                if obj:
                    link_counts[(subject, obj)] += 1

        # --- pass 2: typed and date edges ---
        seen_edges: set[tuple[str, str, str]] = set()

        for subject, fields in pending:
            class_set = classes_of.get(subject, set())
            for tname, fname, value in fields:
                targets = [m.group(1) for m in _LINK_RE.finditer(value)]
                resolved = [t for t in (self.resolve(x) for x in targets) if t]
                rep.unresolved_targets += len(targets) - len(resolved)

                pred = self.onto.predicate_for(fname, class_set)
                if pred is None:
                    if not self.onto.is_noise(fname):
                        unmapped[(tname, fname.lower())] += 1
                        unmapped_links[(tname, fname.lower())] += len(resolved)
                    continue

                raw_value = re.sub(r"\s+", " ", value)[:300]
                if pred.range == "date":
                    dv = dates.parse_field(value, resolved)
                    if not dv:
                        continue
                    node = dv.as_node()
                    date_nodes.setdefault(node["id"], node)
                    key = (subject, pred.name, node["id"])
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    ex.edges.append({
                        "s": subject, "p": pred.name, "o": node["id"],
                        "src": "infobox", "field": fname.lower(), "raw": raw_value,
                    })
                    rep.date_edges += 1
                    continue

                for target in dict.fromkeys(resolved):
                    obj = self.node_for(target)
                    if obj is None or obj == subject:
                        continue
                    # Conservative range check: only reject an object that
                    # carries classes *and* shares none of the expected ones.
                    if pred.range_classes:
                        obj_classes = classes_of.get(obj, set())
                        if obj_classes and not (obj_classes & pred.range_classes):
                            rep.range_rejected += 1
                            continue
                    key = (subject, pred.name, obj)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    ex.edges.append({
                        "s": subject, "p": pred.name, "o": obj,
                        "src": "infobox", "field": fname.lower(), "raw": raw_value,
                    })
                    rep.typed_edges += 1

        # --- subclass_of from the Category namespace ---
        for raw in read_raw(self.s, namespaces=[14]):
            name = _CATEGORY_PREFIX.sub("", raw["title"]).strip()
            if not name:
                continue
            child = class_node(name)
            class_nodes[child]["page_id"] = raw["pageid"]
            class_nodes[child]["url"] = raw["url"]
            class_nodes[child]["defined"] = True
            for parent in raw.get("categories") or []:
                pname = _CATEGORY_PREFIX.sub("", parent).strip()
                if not pname or pname == name:
                    continue
                ex.edges.append({"s": child, "p": "subclass_of",
                                 "o": class_node(pname), "src": "category"})
                rep.subclass_of += 1

        ex.nodes.extend(class_nodes.values())
        ex.nodes.extend(date_nodes.values())
        ex.links = [{"s": s, "o": o, "n": n} for (s, o), n in link_counts.items()]
        ex.unmapped = [
            {"template": t, "field": f, "pages": n, "resolved_links": unmapped_links[(t, f)]}
            for (t, f), n in sorted(unmapped.items(),
                                    key=lambda kv: -unmapped_links[kv[0]])
        ]

        # Curated corrections, last: everything downstream — edges.jsonl, kg.db,
        # `kg show`, and the retrieval channel — then sees one corrected graph.
        ex.edges, ov = overrides_mod.apply(
            ex.edges, self.resolve_node, self.overrides,
            predicates=self.onto.predicates,
        )
        ex.overrides = ov
        rep.overrides_dropped = ov.dropped
        rep.overrides_added = ov.added
        rep.overrides_stale = len(ov.stale) + len(ov.unresolved)

        # Recount from the final list so the report can never disagree with the
        # file it describes.
        counts = collections.Counter(e["p"] for e in ex.edges)
        rep.instance_of = counts["instance_of"]
        rep.subclass_of = counts["subclass_of"]
        rep.date_edges = sum(1 for e in ex.edges if e["o"].startswith("date:"))
        rep.typed_edges = (len(ex.edges) - rep.instance_of - rep.subclass_of
                           - rep.date_edges)

        rep.class_nodes = len(class_nodes)
        rep.date_nodes = len(date_nodes)
        rep.links = len(ex.links)
        rep.unmapped_fields = len(ex.unmapped)
        rep.seconds = time.time() - t0
        log.info(
            "extracted %d entities, %d typed + %d date + %d instance_of + %d subclass_of "
            "edges, %d links in %.1fs",
            rep.entities, rep.typed_edges, rep.date_edges, rep.instance_of,
            rep.subclass_of, rep.links, rep.seconds,
        )
        return ex
