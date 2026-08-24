"""Fact-level corrections, applied on every build.

`ontology.py` is the right dial for a whole *field* being mis-mapped. It cannot
say "this one edge is wrong", and some are — faithfully, because the wiki says
so. Tahir Khushnivis Tabrizi is a calligrapher whose `works` field lists the
books he *transcribed*, so the extractor records him as their author. The range
check cannot help: a Person authoring Books is exactly the right shape, and only
the meaning is wrong.

Corrections live here, in version control, rather than in `data/kg/`. That
directory is both generated (`build._write_jsonl` opens `"w"`, `load_db` calls
`KgStore.reset()`) and gitignored, so an edit made there is destroyed by the next
`shiaqa kg build` and was never reviewable in the first place.

Keep this list short. It is for facts that are wrong at the source and cannot be
expressed as a field mapping — if it starts filling up with one shape of error,
that shape belongs in `ontology.py` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from ..log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class EdgeOverride:
    """One correction, written the way a person reads the graph: by title."""

    action: str            # "drop" | "add"
    subject: str           # article title — page ids are stable but unreadable
    predicate: str
    object: str | None     # article title; None means "every object" (drop only)
    note: str              # why. An uncommented correction cannot be reviewed.

    def describe(self) -> str:
        obj = self.object if self.object is not None else "*"
        return f"{self.action} {self.subject} --{self.predicate}--> {obj}"


OVERRIDES: tuple[EdgeOverride, ...] = (
    EdgeOverride(
        "drop", "Tahir Khushnivis Tabrizi", "authored", None,
        "A calligrapher — his own page carries Category:Shia Calligraphers. The "
        "`works` field lists Nahj al-balagha, al-Sahifa al-Sajjadiyya and Mafatih "
        "al-jinan because he transcribed them, not because he wrote them.",
    ),
)


@dataclass
class OverrideReport:
    dropped: int = 0
    added: int = 0
    # An override that matched nothing. Loud on purpose: a correction that has
    # quietly stopped applying is worse than no correction, because the graph
    # looks curated and is not.
    stale: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def needs_attention(self) -> bool:
        return bool(self.stale or self.unresolved)


def validate(overrides: Sequence[EdgeOverride], predicates: Iterable[str]) -> None:
    """Reject a malformed list at import/build time rather than mis-applying it."""
    known = set(predicates)
    problems: list[str] = []
    for o in overrides:
        if o.action not in ("drop", "add"):
            problems.append(f"{o.describe()}: unknown action {o.action!r}")
        if o.predicate not in known:
            problems.append(f"{o.describe()}: {o.predicate!r} is not a predicate")
        if o.action == "add" and o.object is None:
            problems.append(f"{o.describe()}: 'add' needs an explicit object")
        if not o.note.strip():
            problems.append(f"{o.describe()}: missing note")
    if problems:
        raise ValueError("overrides are invalid:\n  " + "\n  ".join(problems))


def apply(edges: list[dict], resolve: Callable[[str], str | None],
          overrides: Sequence[EdgeOverride] = OVERRIDES,
          predicates: Iterable[str] | None = None) -> tuple[list[dict], OverrideReport]:
    """Apply corrections to an extracted edge list.

    `resolve` maps an article title to a node id (`page:1234`), or None. Compose
    it from `Extractor.resolve` + `Extractor.node_for` so redirects and
    MediaWiki title rules are honoured exactly as they are everywhere else.
    """
    if predicates is not None:
        validate(overrides, predicates)

    rep = OverrideReport()
    if not overrides:
        return edges, rep

    # Resolve every rule once, into two lookups: an exact (s, p, o) match and a
    # wildcard (s, p) match standing for "every object".
    exact: dict[tuple[str, str, str], int] = {}
    wildcard: dict[tuple[str, str], int] = {}
    additions: list[tuple[int, dict]] = []
    hits: dict[int, int] = {}

    for i, o in enumerate(overrides):
        subject = resolve(o.subject)
        if subject is None:
            rep.unresolved.append(f"{o.describe()}  (subject title does not resolve)")
            continue
        obj: str | None = None
        if o.object is not None:
            obj = resolve(o.object)
            if obj is None:
                rep.unresolved.append(f"{o.describe()}  (object title does not resolve)")
                continue
        hits[i] = 0
        if o.action == "drop":
            if obj is None:
                wildcard[(subject, o.predicate)] = i
            else:
                exact[(subject, o.predicate, obj)] = i
        else:
            additions.append((i, {"s": subject, "p": o.predicate, "o": obj,
                                  "src": "manual", "field": None, "raw": o.note}))

    kept: list[dict] = []
    for e in edges:
        rule = exact.get((e["s"], e["p"], e["o"]))
        if rule is None:
            rule = wildcard.get((e["s"], e["p"]))
        if rule is not None:
            hits[rule] += 1
            rep.dropped += 1
            continue
        kept.append(e)

    existing = {(e["s"], e["p"], e["o"]) for e in kept}
    for i, add in additions:
        key = (add["s"], add["p"], add["o"])
        if key in existing:
            continue                      # already asserted by the wiki; not a correction
        kept.append(add)
        existing.add(key)
        hits[i] += 1
        rep.added += 1

    for i, o in enumerate(overrides):
        if i in hits and hits[i] == 0:
            rep.stale.append(f"{o.describe()}  (matched nothing)")

    if rep.dropped or rep.added:
        log.info("overrides: dropped %d, added %d edges", rep.dropped, rep.added)
    for line in rep.stale + rep.unresolved:
        log.warning("stale override: %s", line)
    return kept, rep
