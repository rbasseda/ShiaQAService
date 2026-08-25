"""Multi-hop traversal: chains out from one entity, paths between two, aligned
comparisons of two.

Pure graph work over `KgStore`. No `Settings` dependency — every knob is a
keyword argument, the way `ontology.py` stays free of config — so this module is
testable against a six-node fixture with no environment at all.

Three measured properties of the real graph shape every decision here.

**Traversal degree is not out-degree.** `edges_of` unions outbound and inbound
rows, which is the entire point of indexing `edges` both ways. So the fan-out
that matters is *total* degree: max 382 (Medina), 92 nodes above 25. An unbeamed
two-hop expansion from four seeds reaches ~1,200 nodes — which costs only ~20 ms
in four batched queries, so the beam here is a *ranking* device, not a
performance one. 1,200 candidates for six slots is noise.

**Landing on a relation and travelling through it are different things.**
`al-Mufid born_in Baghdad` is a fact worth stating; Baghdad read backwards is
three hundred strangers. `PREDICATE_PRIORITY` in `store.py` ranks relations by
how likely a reader is asking *about* them, and it is right about `born_in` and
`buried_at` being good facts. It cannot express "never continue from here", so
`NO_TRANSIT` does, separately.

**A predicate blocklist alone is not enough.** The largest hubs are places
*and* people — Imam Ali at 249, the Prophet at 185 — and the edges reaching them
include `taught_by` and `companion_of`, which are exactly the relations
multi-hop questions turn on and so can never be blocklisted. `hub_degree_max`
is what stops a chain from routing through Imam Ali. Both guards are load-bearing;
neither substitutes for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .ontology import PERSON_CLASSES
from .store import DEFAULT_PRIORITY, PREDICATE_PRIORITY, KgStore
from . import render

# Each further hop attenuates. Chosen for one property, not for feel: with the
# tier weights below, a two-hop tier-0 chain scores 0.45, which sits *below*
# every one-hop path of tier 0, 1 or 2 and above only tier-3 one-hop noise. A
# second hop can therefore never displace a good first-hop neighbour, which is
# what keeps the regression suite still when hops go from 1 to 2. Pinned by
# `test_a_two_hop_path_never_outranks_a_strong_one_hop_path`.
HOP_DECAY = 0.45
TIER_WEIGHT = (1.0, 0.75, 0.5, 0.3)

# Fine to land on, ruinous to pass through: every one of these has a place or an
# abstraction on the object side, shared by hundreds of unrelated articles.
# Verified against the ontology — all sixteen exist and carry real edges.
NO_TRANSIT: frozenset[str] = frozenset({
    "resided_in", "affiliation", "born_in", "buried_at", "studied_in",
    "died_in", "occurred_at", "migrated_to", "revealed_at", "located_in",
    "in_era", "has_type", "revelation_type", "chain_validity",
    "about_subject", "known_for",
})

# Semantically symmetric, whatever `inverse` says. `contemporary_of`, `spouse`
# and `sibling` declare themselves their own inverse; `relative_of` and
# `companion_of` are just as symmetric and do not, so the corpus stores "A was
# related to B" and "B was related to A" as two unfoldable rows and a chain
# reports the same kinship twice. Used *only* for canonicalisation here —
# correcting `ontology.py` would change `facts()` output for ~840 existing
# edges, which is a curation decision and not this feature's to make.
SYMMETRIC: frozenset[str] = frozenset({"relative_of", "companion_of"})

# A node above this may end a path but never continue one. Catches the person
# hubs that `NO_TRANSIT` structurally cannot: the edges into Imam Ali are
# `taught_by`, `companion_of`, `father` — the good predicates.
HUB_DEGREE_MAX = 40


@dataclass(frozen=True)
class Step:
    """One traversed edge, with everything needed to rank or explain it."""

    p: str
    direction: str          # "out" | "in", relative to the node departed from
    src: str
    dst: str
    dst_label: str | None
    dst_kind: str | None    # entity | class | date
    dst_page_id: int | None
    field: str | None       # the infobox field the edge came from

    @property
    def stored(self) -> tuple[str, str, str]:
        """The edge as `kg.db` holds it, regardless of which way we walked."""
        return (self.src, self.p, self.dst) if self.direction == "out" \
            else (self.dst, self.p, self.src)

    def as_edge(self) -> dict:
        """The `edges_of` dict shape, so `render` can consume a Step directly."""
        return {"dir": self.direction, "p": self.p, "other": self.dst,
                "label": self.dst_label, "kind": self.dst_kind,
                "page_id": self.dst_page_id, "field": self.field}


@dataclass(frozen=True)
class Path:
    seed: str
    seed_label: str
    steps: tuple[Step, ...]
    score: float
    shape: str = "chain"    # chain | connect

    @property
    def hops(self) -> int:
        return len(self.steps)

    @property
    def end(self) -> str:
        return self.steps[-1].dst

    @property
    def end_label(self) -> str | None:
        return self.steps[-1].dst_label

    @property
    def end_page_id(self) -> int | None:
        return self.steps[-1].dst_page_id

    @property
    def nodes(self) -> tuple[str, ...]:
        return (self.seed, *(s.dst for s in self.steps))


@dataclass(frozen=True)
class ComparisonRow:
    p: str
    direction: str
    label: str               # the predicate's English label
    a_objects: tuple[str, ...]
    b_objects: tuple[str, ...]
    shared: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    a: str
    a_label: str
    b: str
    b_label: str
    rows: tuple[ComparisonRow, ...]


def tier(p: str) -> int:
    return PREDICATE_PRIORITY.get(p, DEFAULT_PRIORITY)


def score_steps(steps: Sequence[Step]) -> float:
    """Hop decay times the weight of every predicate travelled."""
    if not steps:
        return 0.0
    out = HOP_DECAY ** (len(steps) - 1)
    for st in steps:
        out *= TIER_WEIGHT[min(tier(st.p), len(TIER_WEIGHT) - 1)]
    return out


def canonical(stored: tuple[str, str, str], predicates: dict) -> tuple[str, str, str]:
    """Fold an edge and its reciprocal onto one key.

    The corpus states 1,278 facts from both ends — al-Mufid's `students` field
    names al-Murtada, and al-Murtada's `professors` field names al-Mufid. Two
    rows, one fact. `KgStore.facts()` gets away with deduplicating on the
    rendered sentence, but a beam has to fold them *before* admission or it
    spends two slots on one relation and halves its own width.
    """
    s, p, o = stored
    pred = predicates.get(p)
    inv = pred.inverse if pred else None
    if p in SYMMETRIC:
        return (min(s, o), p, max(s, o))
    if inv is None:
        return stored
    if inv == p:
        # spouse, sibling, contemporary_of, related_verse. Without this branch
        # one marriage yields two keys.
        return (min(s, o), p, max(s, o))
    return (o, inv, s) if inv < p else stored


def _signature(seed: str, steps: Sequence[Step], predicates: dict) -> tuple:
    return (seed, tuple(canonical(st.stored, predicates) for st in steps))


def person_nodes(kg: KgStore, node_ids: Sequence[str]) -> set[str]:
    """Which of these nodes are people — for choosing "who" over "which"."""
    return {nid for nid, classes in kg.classes_for(node_ids).items()
            if PERSON_CLASSES.intersection(classes)}


def _walk(kg: KgStore, seeds: Sequence[str], predicates: dict, *, max_hops: int,
          beam: int, no_transit: frozenset[str], hub_degree_max: int,
          stop_at: set[str] | None = None) -> list[tuple[str, tuple[Step, ...]]]:
    """Breadth-first over typed edges, one `edges_from` call per level.

    Returns every path found, as `(seed, steps)`. Shared by `expand` and
    `connect`; the beam and the transit guards are identical for both.
    """
    seed_set = set(seeds)
    frontier = [(s, ()) for s in seeds]          # (seed, steps so far)
    found: list[tuple[str, tuple[Step, ...]]] = []
    seen: set[tuple] = set()

    for hop in range(max_hops):
        if not frontier:
            break
        heads = [steps[-1].dst if steps else seed for seed, steps in frontier]
        by_node = kg.edges_from(heads)

        candidates: list[tuple[float, str, tuple[Step, ...]]] = []
        for (seed, steps), head in zip(frontier, heads):
            walked = {seed, *(st.dst for st in steps)}
            for e in by_node.get(head, []):
                dst = e["other"]
                # Never walk back onto an article the question already named,
                # and never revisit a node inside this path.
                if dst in seed_set or dst in walked:
                    continue
                step = Step(p=e["p"], direction=e["dir"], src=head, dst=dst,
                            dst_label=e["label"], dst_kind=e["kind"],
                            dst_page_id=e["page_id"], field=e["field"])
                extended = (*steps, step)
                sig = _signature(seed, extended, predicates)
                if sig in seen:
                    continue
                seen.add(sig)
                found.append((seed, extended))
                # Dates and classes may end a path and be narrated, never be
                # expanded: "died in 329 AH" is a fact, but everyone who died
                # that year is not a chain.
                if hop + 1 >= max_hops or e["kind"] != "entity":
                    continue
                if e["p"] in no_transit:
                    continue
                if stop_at is not None and dst in stop_at:
                    continue
                candidates.append((score_steps(extended), seed, extended))

        if not candidates:
            break
        # Hub check is batched, and only for nodes that got this far.
        degs = kg.degrees([ext[-1].dst for _, _, ext in candidates])
        passed = [(sc, seed, ext) for sc, seed, ext in candidates
                  if degs.get(ext[-1].dst, 0) <= hub_degree_max]
        passed.sort(key=lambda c: (-c[0], c[2][-1].dst))
        frontier = [(seed, ext) for _, seed, ext in passed[:beam]]

    return found


def expand(kg: KgStore, seed_node_ids: Sequence[str], predicates: dict, *,
           max_hops: int = 2, beam: int = 24, limit: int = 12,
           no_transit: frozenset[str] = NO_TRANSIT,
           hub_degree_max: int = HUB_DEGREE_MAX) -> list[Path]:
    """Best-scoring paths up to `max_hops` out from the seeds.

    Round-robins the final selection across seeds so one densely connected
    article cannot own the expansion — the same reasoning as
    `KgStore.neighbour_pages` and `Store.title_search`.
    """
    if not seed_node_ids or max_hops < 1:
        return []
    labels = {nid: (row["label"] if row else nid)
              for nid, row in ((n, kg.node(n)) for n in seed_node_ids)}
    found = _walk(kg, seed_node_ids, predicates, max_hops=max_hops, beam=beam,
                  no_transit=no_transit, hub_degree_max=hub_degree_max)

    # Bucketed by (seed, depth), not just by seed. Every one-hop path outranks
    # every two-hop path by construction (see HOP_DECAY) — which is exactly what
    # keeps the regression suite still, and also means a flat score-ordered
    # truncation would drop every chain for any well-connected seed. A seed with
    # eight first-hop neighbours would fill `limit` before a single composed
    # relation appeared, leaving the chain block empty precisely when it matters.
    # Round-robining across depths guarantees both are represented; consumers
    # that want strict score order (the chunk channel) re-sort.
    buckets: dict[tuple[str, int], list[Path]] = {}
    for seed, steps in found:
        buckets.setdefault((seed, len(steps)), []).append(
            Path(seed=seed, seed_label=labels.get(seed, seed), steps=steps,
                 score=score_steps(steps), shape="chain"))
    # Ties are the norm rather than the exception — every two-hop tier-0 chain
    # scores exactly 0.450 — so the tie-break decides what survives `limit`, and
    # an alphabetical one decides it by accident. For "who taught the teacher of
    # al-Sharif al-Radi" that accident dropped the chain ending at al-Shaykh
    # al-Saduq (the answer) in favour of "Abu l-Jaysh al-Balkhi" and two others,
    # purely on spelling. Prominence is the same prior `title_candidates` uses to
    # choose between same-named articles.
    prom = kg.prominence([p.end for group in buckets.values() for p in group])
    for group in buckets.values():
        group.sort(key=lambda p: (-p.score, p.hops, -prom.get(p.end, 0),
                                  p.end_label or "", p.end))

    depths = sorted({d for _, d in buckets})
    cycle = [(s, d) for d in depths for s in seed_node_ids if (s, d) in buckets]
    out: list[Path] = []
    for slot in range(max((len(v) for v in buckets.values()), default=0)):
        for key in cycle:
            group = buckets[key]
            if slot < len(group):
                out.append(group[slot])
                if len(out) >= limit:
                    return out
    return out


def connect(kg: KgStore, a: str, b: str, predicates: dict, *,
            max_hops: int = 3, beam: int = 24, limit: int = 3,
            no_transit: frozenset[str] = NO_TRANSIT,
            hub_degree_max: int = HUB_DEGREE_MAX) -> list[Path]:
    """Typed paths between two named entities, strongest first.

    Ranked rather than merely shortest, which matters more than it sounds. The
    real graph joins al-Mufid and al-Tusi at length two twice over:

        al-Mufid --taught_by--> al-Sharif al-Murtada <--taught-- al-Tusi
        al-Mufid --born_in----> Baghdad             <--studied_in-- al-Tusi

    A shortest-path search returns whichever it happens to reach first. The
    second is worthless; `NO_TRANSIT` refuses to travel through Baghdad, and
    scoring settles the rest.
    """
    if not a or not b or a == b or max_hops < 1:
        return []
    row_a, row_b = kg.node(a), kg.node(b)
    if row_a is None or row_b is None:
        return []

    # Search outward from `a` only, but stop as soon as `b` is reached: `_walk`
    # already keeps whole paths, so a hit is the path, not just a meeting point.
    found = _walk(kg, [a], predicates, max_hops=max_hops, beam=beam,
                  no_transit=no_transit, hub_degree_max=hub_degree_max,
                  stop_at={b})
    hits = [steps for _, steps in found if steps[-1].dst == b]
    if not hits:
        return []
    shortest = min(len(s) for s in hits)
    # A longer route between the same pair is a detour, not an alternative.
    hits = [s for s in hits if len(s) == shortest]
    paths = [Path(seed=a, seed_label=row_a["label"], steps=s,
                  score=score_steps(s), shape="connect") for s in hits]
    paths.sort(key=lambda p: (-p.score, tuple(st.p for st in p.steps)))
    return paths[:limit]


def compare(kg: KgStore, a: str, b: str, predicates: dict, *,
            limit: int = 6) -> Comparison | None:
    """Predicates asserted of both entities, aligned side by side.

    No traversal at all — one batched call, grouped. Returns `None` when the two
    share no populated predicate, which is data rather than a judgement call:
    there is simply nothing to line up.
    """
    if not a or not b or a == b:
        return None
    row_a, row_b = kg.node(a), kg.node(b)
    if row_a is None or row_b is None:
        return None
    by_node = kg.edges_from([a, b])

    # Grouped by the *rendered* verb phrase, read from the entity's own side.
    # Keying on `(predicate, direction)` instead looks natural and is wrong: an
    # inbound `taught_by` edge would then be labelled with its own predicate,
    # printing "al-Mufid taught al-Saduq" for a row that says the opposite.
    # Going through `render.clause` also folds the reciprocal pair for free,
    # since `taught` outbound and `taught_by` inbound render identically.
    #
    # Edges that cannot be read from this side at all — an inbound predicate with
    # no inverse, 53 of the 69 — are dropped rather than reversed. That loses
    # some comparable material and cannot state anything false.
    def grouped(node: str) -> dict[tuple[int, str], tuple[str, list[str]]]:
        out: dict[tuple[int, str], tuple[str, list[str]]] = {}
        for e in by_node.get(node, []):
            said = render.clause(e, predicates)
            if said is None:
                continue
            verb, other = said
            key = (tier(e["p"]), verb)
            out.setdefault(key, (e["p"], []))[1].append(other)
        return out

    ga, gb = grouped(a), grouped(b)
    rows: list[ComparisonRow] = []
    for key in sorted(set(ga) & set(gb)):
        _, verb = key
        objs_a = tuple(dict.fromkeys(ga[key][1]))
        objs_b = tuple(dict.fromkeys(gb[key][1]))
        rows.append(ComparisonRow(
            p=ga[key][0], direction="out", label=verb,
            a_objects=objs_a, b_objects=objs_b,
            shared=tuple(o for o in objs_a if o in set(objs_b)),
        ))
        if len(rows) >= limit:
            break
    if not rows:
        return None
    return Comparison(a=a, a_label=row_a["label"], b=b, b_label=row_b["label"],
                      rows=tuple(rows))


def narrate(path: Path, predicates: dict, persons: set[str] | None = None) -> str | None:
    """A path as one attributed English sentence, or `None` if unrenderable.

    Chained with a relative pronoun while the steps read forwards — "was taught
    by al-Mufid, who was taught by al-Saduq". A step that cannot be read from
    the anchor's side **breaks** the sentence into a new clause instead of being
    chained: 53 of the 69 predicates declare no inverse, so this is the common
    case, not a corner. Chaining one regardless would state the relation
    backwards and put a quietly false sentence in the prompt, which is worse
    than a clumsy one.
    """
    if not path.steps:
        return None
    persons = persons or set()
    parts: list[str] = []
    subject = path.seed_label
    # A relative pronoun may only be used when the previous clause *ended* on the
    # node the next step departs from. A forward clause does; a flipped one ends
    # on the old subject instead, so chaining "which" onto it would bind the
    # pronoun to the wrong noun — producing "Al-Kafi, which is recorded in
    # al-Amali" for a path where it is the *hadith*, not al-Kafi, that al-Amali
    # records. A quietly false sentence in the prompt is worse than a clumsy one.
    pronoun_ok = False
    for i, step in enumerate(path.steps):
        edge = step.as_edge()
        said = render.clause(edge, predicates)
        if said is not None:
            verb, other = said
            if i == 0:
                parts.append(f"{subject} {verb} {other}")
            elif pronoun_ok:
                pronoun = "who" if step.src in persons else "which"
                parts.append(f"{pronoun} {verb} {other}")
            else:
                parts.append(f"and {subject} {verb} {other}")
            subject = other
            pronoun_ok = True
            continue
        # Cannot run forwards from here. Start a fresh clause from the far end.
        back = render.flipped(edge, predicates)
        if back is None:
            break
        other, verb = back
        parts.append(f"and {other} {verb} {subject}" if i else
                     f"{other} {verb} {subject}")
        subject = other
        pronoun_ok = False
    if not parts:
        return None
    return ", ".join(parts) + "."


def narrate_comparison(cmp: Comparison, limit_objects: int = 3) -> list[str]:
    """A comparison as one line per aligned predicate."""
    out: list[str] = []
    for row in cmp.rows:
        a = ", ".join(row.a_objects[:limit_objects]) or "nothing recorded"
        b = ", ".join(row.b_objects[:limit_objects]) or "nothing recorded"
        line = f"{cmp.a_label} {row.label} {a}; {cmp.b_label} {row.label} {b}"
        if row.shared:
            line += f" (both: {', '.join(row.shared[:limit_objects])})"
        out.append(line + ".")
    return out
