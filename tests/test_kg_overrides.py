"""Fact-level corrections: applied on every build, and loud when they go stale."""
import json

import pytest

from shiaqa.config import Settings
from shiaqa.kg.extract import Extractor
from shiaqa.kg.ontology import Ontology
from shiaqa.kg.overrides import OVERRIDES, EdgeOverride, apply, validate

TITLES = {"A": "page:1", "B": "page:2", "C": "page:3"}
EDGES = [
    {"s": "page:1", "p": "authored", "o": "page:2", "src": "infobox", "field": "works"},
    {"s": "page:1", "p": "authored", "o": "page:3", "src": "infobox", "field": "works"},
    {"s": "page:1", "p": "taught", "o": "page:3", "src": "infobox", "field": "students"},
]


def _resolve(title):
    return TITLES.get(title)


# --- validation --------------------------------------------------------------

def test_the_shipped_overrides_are_valid_against_the_real_ontology():
    validate(OVERRIDES, Ontology().predicates)


@pytest.mark.parametrize("bad,reason", [
    (EdgeOverride("delete", "A", "authored", "B", "n"), "unknown action"),
    (EdgeOverride("drop", "A", "not_a_predicate", "B", "n"), "is not a predicate"),
    (EdgeOverride("add", "A", "authored", None, "n"), "needs an explicit object"),
    (EdgeOverride("drop", "A", "authored", "B", "  "), "missing note"),
])
def test_a_malformed_override_is_rejected_rather_than_mis_applied(bad, reason):
    with pytest.raises(ValueError, match=reason):
        validate([bad], Ontology().predicates)


# --- applying ----------------------------------------------------------------

def test_a_wildcard_drop_removes_every_object_for_that_predicate():
    kept, rep = apply(EDGES, _resolve,
                      (EdgeOverride("drop", "A", "authored", None, "why"),))
    assert rep.dropped == 2
    assert [e["p"] for e in kept] == ["taught"]
    assert not rep.needs_attention


def test_an_exact_drop_removes_only_the_named_edge():
    kept, rep = apply(EDGES, _resolve,
                      (EdgeOverride("drop", "A", "authored", "B", "why"),))
    assert rep.dropped == 1
    assert {(e["p"], e["o"]) for e in kept} == {("authored", "page:3"), ("taught", "page:3")}


def test_an_added_edge_is_marked_manual_so_it_stays_separable():
    kept, rep = apply(EDGES, _resolve,
                      (EdgeOverride("add", "A", "father", "C", "why"),))
    assert rep.added == 1
    added = kept[-1]
    assert (added["s"], added["p"], added["o"]) == ("page:1", "father", "page:3")
    assert added["src"] == "manual"
    assert added["raw"] == "why"           # the note travels with the edge


def test_adding_an_edge_the_wiki_already_asserts_is_a_no_op_and_flagged_stale():
    kept, rep = apply(EDGES, _resolve,
                      (EdgeOverride("add", "A", "taught", "C", "why"),))
    assert rep.added == 0
    assert len(kept) == len(EDGES)
    assert rep.needs_attention


def test_overrides_do_not_touch_edges_they_do_not_name():
    kept, _ = apply(EDGES, _resolve,
                    (EdgeOverride("drop", "A", "authored", "B", "why"),))
    assert {"s": "page:1", "p": "taught", "o": "page:3",
            "src": "infobox", "field": "students"} in kept


# --- the failure mode this exists to prevent ---------------------------------

def test_an_override_that_matches_nothing_is_reported_not_silently_ignored():
    _, rep = apply(EDGES, _resolve,
                   (EdgeOverride("drop", "A", "sibling", None, "why"),))
    assert rep.dropped == 0
    assert rep.needs_attention
    assert "matched nothing" in rep.stale[0]


def test_a_title_that_no_longer_resolves_is_reported():
    _, rep = apply(EDGES, _resolve,
                   (EdgeOverride("drop", "Renamed Page", "authored", None, "why"),))
    assert rep.needs_attention
    assert "does not resolve" in rep.unresolved[0]
    # A dangling rule must not be counted as applied.
    assert rep.dropped == 0


def test_an_empty_override_list_changes_nothing():
    kept, rep = apply(EDGES, _resolve, ())
    assert kept == EDGES
    assert not rep.needs_attention


# --- end to end through the extractor ----------------------------------------

SCHOLAR = """{{Infobox Shia scholar
| name = Test Scholar
| works = [[A Book (book)]]
| students = [[Student One]]
}}
'''Test Scholar''' wrote things.
"""


@pytest.fixture
def corpus(tmp_path):
    s = Settings(data_dir=tmp_path)
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    pages = [
        {"pageid": 1, "ns": 0, "title": "Test Scholar", "url": "http://x/1",
         "revid": 1, "timestamp": None, "categories": [], "wikitext": SCHOLAR},
        {"pageid": 2, "ns": 0, "title": "A Book (book)", "url": "http://x/2",
         "revid": 1, "timestamp": None, "categories": [], "wikitext": "A book."},
        {"pageid": 3, "ns": 0, "title": "Student One", "url": "http://x/3",
         "revid": 1, "timestamp": None, "categories": [], "wikitext": "A student."},
    ]
    (s.raw_dir / "ns0.jsonl").write_text("\n".join(json.dumps(p) for p in pages) + "\n")
    (s.raw_dir / "ns14.jsonl").write_text("")
    (s.raw_dir / "ns3000.jsonl").write_text("")
    (s.raw_dir / "redirects.json").write_text(json.dumps({"The Scholar": "Test Scholar"}))
    return s


def test_extractor_applies_overrides_so_every_consumer_sees_one_graph(corpus):
    before = Extractor(corpus, overrides=()).run()
    assert any(e["p"] == "authored" for e in before.edges)

    after = Extractor(corpus, overrides=(
        EdgeOverride("drop", "Test Scholar", "authored", None, "wrong"),)).run()
    assert not any(e["p"] == "authored" for e in after.edges)
    assert after.report.overrides_dropped == 1
    assert after.report.overrides_stale == 0


def test_the_report_counts_match_the_edges_actually_written(corpus):
    """Counters are recomputed after overrides, so they cannot drift from the file."""
    ex = Extractor(corpus, overrides=(
        EdgeOverride("drop", "Test Scholar", "authored", None, "wrong"),)).run()
    rep = ex.report
    assert (rep.typed_edges + rep.date_edges + rep.instance_of + rep.subclass_of
            == len(ex.edges))


def test_an_override_written_against_a_redirect_still_resolves(corpus):
    ex = Extractor(corpus, overrides=(
        EdgeOverride("drop", "The Scholar", "authored", None, "via redirect"),)).run()
    assert ex.report.overrides_dropped == 1
