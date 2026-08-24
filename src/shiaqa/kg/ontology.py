"""The curated ontology: which infobox templates are classes, which fields are relations.

Kept as Python literals rather than a YAML file for the same reason `clean.py`
keeps its template policy inline: the project declares no YAML dependency
(PyYAML is present only transitively via `uvicorn[standard]`, and `eval/run_eval.py`
hand-parses its own fixture rather than take one). This table is source that
changes together with the extractor, not data.

Every name below was read off a frequency scan of the real corpus, so the mess is
deliberate. WikiShia has a live template typo (`infobox descedant of imam`, 16
pages) that must map to the same class as the correct spelling; three spellings
of "well known relatives"; and one field whose *name* contains a wikilink
(`presence at [[ghazwas]]`). Normalising the obvious cases in `norm_field()` is
not enough — the variants that survive normalisation are listed out by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- classes -----------------------------------------------------------------

# class name -> infobox template names that assert it.
CLASSES: dict[str, tuple[str, ...]] = {
    "Scholar": (
        "infobox shia scholar",
        "infobox shia scholar without socio-political activities",
    ),
    "Companion": ("infobox companion of imam (a)", "infobox sahaba"),
    "Imam": ("infobox imam",),
    "Prophet": ("infobox prophets", "infobox prophet muhammad (s)"),
    "Person": (
        "infobox person",
        "infobox narrator",
        "infobox ruler",
        "infobox artist",
        "infobox descendant of imam",
        # A live typo in the wiki's own template name. Same class, 16 pages.
        "infobox descedant of imam",
        "infobox lady fatimah al-zahra' (a)",
    ),
    "Book": ("infobox book",),
    "Sura": ("infobox sura",),
    "Verse": ("infobox verse",),
    "Hadith": ("infobox hadith",),
    "Supplication": ("infobox supplication and ziyara text",),
    "Ritual": ("infobox rituals",),
    "Building": ("infobox religious building",),
    "Place": ("infobox city", "infobox country"),
    "Battle": ("infobox war", "infobox event"),
    "Family": ("infobox family",),
    "Organization": ("infobox shia organization", "infobox website"),
    "Film": ("infobox film",),
}

# Every person-ish class also counts as a Person, so domain restrictions can be
# written once. Order matters only for display.
PERSON_CLASSES = frozenset({"Person", "Scholar", "Companion", "Imam", "Prophet"})


@dataclass(frozen=True)
class Predicate:
    """One relation, and the infobox fields that express it."""

    name: str
    label: str
    fields: tuple[str, ...]
    # "entity" keeps only wikilinks that resolve to a real page; "date" routes
    # the value through `dates.py` instead.
    range: str = "entity"
    inverse: str | None = None
    # When non-empty, the predicate only applies to subjects of these classes.
    # Used for the handful of field names that genuinely collide across
    # templates ("next" is the next sura, but the next ruler elsewhere).
    domain: frozenset[str] = field(default_factory=frozenset)
    # When non-empty, an object that carries classes must share one of these.
    # Deliberately permissive: an object with *no* class is always kept, since
    # most pages have no infobox and so no class at all. This only catches the
    # confidently-wrong case — WikiShia lists "Battle of Siffin" under one
    # narrator's `works`, which would otherwise read as him having written it.
    range_classes: frozenset[str] = field(default_factory=frozenset)


PREDICATES: tuple[Predicate, ...] = (
    # --- teaching and transmission ---
    Predicate("taught_by", "was taught by", ("professors", "professor", "teachers"),
              inverse="taught"),
    Predicate("taught", "taught", ("students",), inverse="taught_by"),
    Predicate("ijaza_from", "held a transmission licence from",
              ("permission for hadith transmission from", "permission for ijtihad from",
               "permission to narrate from"), inverse="ijaza_to"),
    Predicate("ijaza_to", "granted a transmission licence to",
              ("permission for hadith transmission to",), inverse="ijaza_from"),
    Predicate("narrated_by", "was narrated by",
              ("narrator", "narrators", "main narrator", "narrated by")),
    Predicate("narrated_from", "narrated from",
              ("narrated from", "narrated from infallible")),

    # --- kinship and association ---
    Predicate("companion_of", "was a companion of", ("companion of",)),
    Predicate("relative_of", "was related to",
              ("well known relatives", "well-known relatives", "wellknown relatives")),
    Predicate("father", "had as father", ("father",), inverse="child",
              range_classes=frozenset(PERSON_CLASSES)),
    Predicate("mother", "had as mother", ("mother",), inverse="child",
              range_classes=frozenset(PERSON_CLASSES)),
    Predicate("spouse", "was married to", ("spouse(s)", "spouses", "husband", "wife"),
              inverse="spouse"),
    Predicate("child", "had as child",
              ("sons", "daughters", "son(s)", "daughter(s)", "children", "descendants"),
              inverse="parent"),
    # Label-only: no infobox field states it, but `child` read backwards needs a
    # phrase, and `father`/`mother` need a target for their own inverse.
    Predicate("parent", "had as parent", (), inverse="child"),
    Predicate("sibling", "was a sibling of", ("brothers", "sisters"), inverse="sibling"),
    Predicate("descends_from", "descends from", ("lineage",)),
    Predicate("contemporary_of", "was a contemporary of",
              ("contemporary with", "contemporary rulers", "contemporary prophet"),
              inverse="contemporary_of"),

    # --- life events: places ---
    Predicate("born_in", "was born in", ("place of birth", "birthplace", "birth place")),
    Predicate("died_in", "died in", ("place of martyrdom",)),
    Predicate("buried_at", "is buried at", ("burial place", "place of burial", "graveyards")),
    Predicate("resided_in", "resided in",
              ("place of residence", "places of residence", "residence",
               "home town", "hometown")),
    Predicate("studied_in", "studied in", ("place of study", "places of study")),
    Predicate("migrated_to", "migrated to", ("migration to",)),

    # --- life events: dates ---
    Predicate("born_on", "was born on", ("birth", "born", "birthday"), range="date"),
    Predicate("died_on", "died on",
              ("death", "death/martyrdom", "martyrdom", "demise"), range="date"),
    Predicate("cause_of_death", "died because of",
              ("cause of death", "cause of death/martyrdom", "cause of martyrdom")),

    # --- roles and works ---
    Predicate("affiliation", "belonged to",
              ("religious affiliation", "religion", "muhajir/ansar")),
    Predicate("authored", "wrote", ("works",), inverse="authored_by",
              range_classes=frozenset({"Book", "Supplication", "Hadith"})),
    Predicate("known_for", "is known for", ("known for", "notable roles", "role")),
    Predicate("participated_in", "took part in",
              ("activities", "other activities", "socio-political activities",
               "scholarly activities", "presence at ghazwas", "important events",
               "important rites")),

    # --- books ---
    Predicate("authored_by", "was written by", ("author", "writer"), inverse="authored",
              range_classes=frozenset(PERSON_CLASSES)),
    Predicate("published_by", "was published by",
              ("publisher", "en publisher", "english publisher")),
    Predicate("translated_by", "was translated by", ("translator", "translated by")),
    Predicate("about_subject", "is about",
              ("subject", "topic", "about", "genre", "hadith topics")),

    # --- Qur'an ---
    Predicate("in_sura", "appears in", ("sura",), domain=frozenset({"Verse"}),
              range_classes=frozenset({"Sura"})),
    Predicate("revealed_at", "was revealed at", ("place of revelation",)),
    Predicate("revealed_because", "was revealed because of", ("cause of revelation",)),
    Predicate("related_verse", "is related to", ("related verses", "others"),
              inverse="related_verse"),
    Predicate("revelation_type", "is of revelation type", ("makki/madani",)),

    # --- hadith and supplication ---
    Predicate("issued_by", "was issued by", ("issued by",)),
    Predicate("source_shia", "is recorded in the Shia source", ("shi'a sources",)),
    Predicate("source_sunni", "is recorded in the Sunni source", ("sunni sources",)),
    Predicate("monograph", "has the monograph", ("monographs",)),
    Predicate("observed_at", "is observed at", ("time",)),

    # --- battles and events ---
    Predicate("combatant", "was fought by", ("combatant1", "combatant2")),
    Predicate("commander", "was commanded by", ("commander1", "commander2")),
    Predicate("occurred_at", "took place at", ("place", "location")),
    Predicate("occurred_on", "took place on", ("date",), range="date"),
    Predicate("caused_by", "was caused by", ("cause",)),
    Predicate("part_of", "is part of", ("part of",)),

    # --- places and buildings ---
    Predicate("located_in", "is located in", ("country", "province")),
    Predicate("founded_by", "was founded by", ("founder", "architect")),
    Predicate("founded_on", "was founded on", ("established", "year of foundation"),
              range="date"),
    Predicate("contains", "contains",
              ("shrines", "mosques", "historical places", "husayniyyas", "seminary",
               "facilities", "research centers", "institutes", "capital",
               "shi'a areas")),
    Predicate("related_event", "is connected to the event", ("related events",)),

    # --- second curation pass, driven by the top of `unmapped.jsonl` ---
    Predicate("published_on", "was published in", ("published", "pub date"), range="date"),
    Predicate("imamate_began", "began his imamate in", ("beginning of imamate",),
              range="date"),
    Predicate("converted_on", "accepted Islam in",
              ("converting to islam", "conversion to islam"), range="date"),
    Predicate("reigned", "reigned in", ("reign",), range="date"),
    Predicate("renovated_on", "was renovated in", ("renovation",), range="date"),
    Predicate("has_type", "is of type", ("type", "types", "status")),
    Predicate("chain_validity", "has chain assessed as",
              ("validity of the chain of transmission", "reliability")),
    Predicate("resulted_in", "resulted in", ("result",)),
    Predicate("in_era", "belongs to the era", ("era",)),
    Predicate("opposed_by", "was opposed by", ("enemies",)),
    Predicate("supported_by_verse", "is supported by", ("qur'anic support",)),

    # --- groups ---
    Predicate("member", "counts as a member", ("figures", "scholars", "head", "rulers")),
    Predicate("originates_from", "originates from", ("origin",)),
    Predicate("succeeded_by", "was succeeded by", ("successor", "after"),
              inverse="preceded_by"),
    Predicate("preceded_by", "was preceded by", ("predecessor", "before"),
              inverse="succeeded_by"),
)

# Fields that carry wikilinks but no relation: presentation, identifiers, and
# name variants. Listed so they never reach `unmapped.jsonl` and drown the
# genuine gaps in the curation worklist.
NOISE_FIELDS: frozenset[str] = frozenset({
    "image", "image size", "imagesize", "image flag", "alt", "caption",
    "caption image", "signature", "logo", "logo caption", "width", "emblem",
    "photo", "pic", "picture", "align", "float", "box width",
    "url", "website", "official website", "isbn", "page", "pages", "volumes",
    "name", "full name", "official name", "local name", "main title", "title",
    "title orig", "original title", "old name", "other names", "nickname",
    "epithet", "teknonym", "kunya", "well known as", "note", "en title",
    "orig lang code", "series", "language", "runtime", "background",
    "coordinates", "coordinate", "casualties1", "casualties2", "strength1",
    "strength2", "sequential number", "repeat in the qur'an", "verse count",
    "word count", "letter count", "verse number", "word number", "letter number",
    "sura number", "revelation number", "juz", "juz'", "number of hadiths",
    "population", "total population", "muslim population", "shi'a population",
    "area", "capacity", "titles", "duration of imamate", "age",
    "percentage to the country's population",
})

# `[[…]]` inside a field *name* — WikiShia really does have
# `| presence at [[ghazwas]] = …` on 27 pages.
_FIELD_LINK_RE = re.compile(r"\[\[\s*(?:[^\]|]*\|)?([^\]|]+?)\s*\]\]")
_WS_RE = re.compile(r"\s+")


def norm_field(name: str) -> str:
    """Fold an infobox field name to its comparable form."""
    name = _FIELD_LINK_RE.sub(r"\1", name)
    name = name.replace("_", " ").replace("-", " ").strip().lower()
    return _WS_RE.sub(" ", name)


def norm_template(name: str) -> str:
    return _WS_RE.sub(" ", name.replace("_", " ").strip().lower())


class Ontology:
    """Compiled lookup tables over `CLASSES` / `PREDICATES`."""

    def __init__(self) -> None:
        self.classes = CLASSES
        self.predicates = {p.name: p for p in PREDICATES}

        self._by_template: dict[str, list[str]] = {}
        for cls, templates in CLASSES.items():
            for t in templates:
                self._by_template.setdefault(norm_template(t), []).append(cls)

        # field -> predicates that claim it (more than one only when they are
        # domain-separated).
        self._by_field: dict[str, list[Predicate]] = {}
        for p in PREDICATES:
            for f in p.fields:
                self._by_field.setdefault(norm_field(f), []).append(p)

        self.noise = {norm_field(f) for f in NOISE_FIELDS}
        self._validate()

    def _validate(self) -> None:
        """Fail loudly on a self-inconsistent table rather than silently mis-typing edges."""
        problems = []
        for name, p in self.predicates.items():
            if p.range not in ("entity", "date"):
                problems.append(f"{name}: unknown range {p.range!r}")
            if p.inverse and p.inverse not in self.predicates:
                problems.append(f"{name}: inverse {p.inverse!r} is not a predicate")
            for cls in p.domain:
                if cls not in CLASSES:
                    problems.append(f"{name}: domain {cls!r} is not a class")
            for cls in p.range_classes:
                if cls not in CLASSES:
                    problems.append(f"{name}: range_classes {cls!r} is not a class")
        for f, ps in self._by_field.items():
            # Two *distinct* predicates may share a field only if domains
            # disambiguate them. One predicate listing several spellings that
            # normalise together ("well known"/"well-known") is fine.
            distinct = {p.name for p in ps}
            if len(distinct) > 1 and any(not p.domain for p in ps):
                problems.append(f"field {f!r} claimed by {sorted(distinct)} without domains")
            if f in {norm_field(n) for n in NOISE_FIELDS}:
                problems.append(f"field {f!r} is both mapped and listed as noise")
        if problems:
            raise ValueError("ontology is inconsistent:\n  " + "\n  ".join(problems))

    def classes_for(self, template: str) -> list[str]:
        return self._by_template.get(norm_template(template), [])

    def predicate_for(self, raw_field: str, subject_classes: set[str]) -> Predicate | None:
        """Resolve a field name to its predicate, honouring domain restrictions."""
        candidates = self._by_field.get(norm_field(raw_field))
        if not candidates:
            return None
        # Domain-restricted predicates win when they apply; the unrestricted one
        # is the fallback.
        for p in candidates:
            if p.domain and (p.domain & subject_classes):
                return p
        for p in candidates:
            if not p.domain:
                return p
        return None

    def is_noise(self, raw_field: str) -> bool:
        return norm_field(raw_field) in self.noise

    def snapshot(self) -> dict:
        """Serialisable description of the ontology actually applied."""
        return {
            "classes": {c: list(t) for c, t in CLASSES.items()},
            "predicates": [
                {"name": p.name, "label": p.label, "fields": list(p.fields),
                 "range": p.range, "inverse": p.inverse, "domain": sorted(p.domain),
                 "range_classes": sorted(p.range_classes)}
                for p in PREDICATES
            ],
            "noise_fields": sorted(self.noise),
        }
