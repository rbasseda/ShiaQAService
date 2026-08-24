"""Extraction: infobox wikitext -> typed edges, and date parsing.

Offline like the rest of the suite: the extractor is pointed at a temporary
`data/raw/` written by the fixture, so nothing reads the real 230 MB corpus.
"""
import json

import pytest

from shiaqa.config import Settings
from shiaqa.kg.dates import DateValue, parse_field, parse_one
from shiaqa.kg.extract import Extractor, normalise_title
from shiaqa.kg.ontology import Ontology, norm_field


# --- ontology ----------------------------------------------------------------

def test_ontology_is_self_consistent():
    """The table validates itself on construction; this pins that it stays valid."""
    onto = Ontology()
    assert onto.classes_for("Infobox Shia scholar") == ["Scholar"]
    # A live typo in the wiki's own template name must land on the same class.
    assert onto.classes_for("infobox descedant of imam") == \
           onto.classes_for("infobox descendant of imam")


def test_field_name_variants_collapse_to_one_predicate():
    onto = Ontology()
    names = ["well known relatives", "well-known relatives", "Wellknown_relatives"]
    assert {onto.predicate_for(n, {"Companion"}).name for n in names} == {"relative_of"}


def test_field_name_containing_a_wikilink_is_normalised():
    # WikiShia really does write `| presence at [[ghazwas]] = ...`.
    assert norm_field("presence at [[ghazwas]]") == "presence at ghazwas"
    assert Ontology().predicate_for("presence at [[ghazwas]]", {"Companion"}).name \
           == "participated_in"


def test_domain_restriction_keeps_a_colliding_field_apart():
    onto = Ontology()
    assert onto.predicate_for("sura", {"Verse"}).name == "in_sura"
    assert onto.predicate_for("sura", {"Book"}) is None


# --- dates -------------------------------------------------------------------

def test_one_date_field_spanning_two_calendars_becomes_one_node():
    """`[[Sha'ban 15]], [[329]]/[[May 15]], [[941 CE|941]]` is four links, one date."""
    dv = parse_field(
        "[[Sha'ban 15]], [[329]]/[[May 15]], [[941 CE|941]]",
        ["Sha'ban 15", "329 AH", "May 15"],
    )
    assert dv.node_id == "date:ah:0329-08-15"
    assert (dv.ah_year, dv.ah_month, dv.ah_day) == (329, 8, 15)
    # The Gregorian half is not an article on this wiki, so it can only come
    # from reading the raw value.
    assert dv.ce_year == 941
    assert "329 AH" in dv.label and "941 CE" in dv.label


@pytest.mark.parametrize("text,node_id", [
    ("1344 AH", "date:ah:1344"),
    ("4 BH", "date:bh:0004"),
    ("2019 CE", "date:ce:2019"),
    ("Muharram 10", "date:ah-md:01-10"),
    ("Dhu l-Hijja 18", "date:ah-md:12-18"),
])
def test_date_titles_parse_to_stable_ids(text, node_id):
    assert parse_one(text).node_id == node_id


def test_a_value_with_no_date_yields_nothing():
    assert not parse_one("a book about ethics")
    with pytest.raises(ValueError):
        DateValue().node_id


# --- extraction over a fixture corpus ----------------------------------------

SCHOLAR = """{{Infobox Shia scholar
| name = Test Scholar
| professors = [[Teacher One]] and [[Teacher Two]]
| students = [[Student One]]
| works = [[A Book (book)]]
| death = [[Sha'ban 15]], [[329]]/[[May 15]], [[941 CE|941]]
| burial place = [[Najaf]]
| image = something.jpg
}}
'''Test Scholar''' was a scholar who studied under [[Teacher One]] in [[Najaf]].
"""

BOOK = """{{Infobox book
| author = [[Test Scholar]]
| subject = [[Ethics]]
}}
'''A Book''' is a book by [[Test Scholar]].
"""


def _page(pid, title, wikitext, categories=(), ns=0):
    return {"pageid": pid, "ns": ns, "title": title,
            "url": f"http://x/{pid}", "revid": 1, "timestamp": None,
            "categories": list(categories), "wikitext": wikitext}


@pytest.fixture
def corpus(tmp_path):
    """A miniature `data/raw/` the extractor can be pointed at."""
    s = Settings(data_dir=tmp_path)
    s.raw_dir.mkdir(parents=True, exist_ok=True)   # only get_settings() does this
    pages = [
        _page(1, "Test Scholar", SCHOLAR, ["Category:Faqihs"]),
        _page(2, "A Book (book)", BOOK, ["Category:Bibliography"]),
        _page(3, "Teacher One", "A teacher.", ["Category:Faqihs"]),
        _page(4, "Teacher Two", "Another teacher."),
        _page(5, "Student One", "A student."),
        _page(6, "Najaf", "A city."),
        _page(7, "Ethics", "A subject."),
        _page(8, "Something (disambiguation)", "{{disambiguation}}"),
        # Calendar articles, exactly as the real corpus has them: 160 `N AH`
        # year pages, month-day pages in both calendars, and bare-year
        # redirects pointing at the AH article.
        _page(9, "329 AH", "A Hijri year."),
        _page(10, "Sha'ban 15", "A Hijri date."),
        _page(11, "May 15", "A Gregorian date."),
    ]
    (s.raw_dir / "ns0.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pages) + "\n")
    (s.raw_dir / "ns14.jsonl").write_text(json.dumps(
        _page(100, "Category:Faqihs", "", ["Category:Shia scholars"], ns=14)) + "\n")
    (s.raw_dir / "ns3000.jsonl").write_text("")
    # "Test Sch." is a redirect, so links to it must resolve to the article.
    (s.raw_dir / "redirects.json").write_text(
        json.dumps({"Test Sch.": "Test Scholar", "329": "329 AH"}))
    return s


def _edges(ex, pred):
    return {(e["s"], e["o"]) for e in ex.edges if e["p"] == pred}


def test_infobox_fields_become_typed_edges(corpus):
    ex = Extractor(corpus).run()
    assert _edges(ex, "taught_by") == {("page:1", "page:3"), ("page:1", "page:4")}
    assert _edges(ex, "taught") == {("page:1", "page:5")}
    assert _edges(ex, "buried_at") == {("page:1", "page:6")}
    assert _edges(ex, "authored_by") == {("page:2", "page:1")}
    assert _edges(ex, "about_subject") == {("page:2", "page:7")}


def test_a_date_field_produces_a_single_date_edge_and_node(corpus):
    ex = Extractor(corpus).run()
    died = [e for e in ex.edges if e["p"] == "died_on"]
    assert len(died) == 1
    assert died[0]["o"] == "date:ah:0329-08-15"
    assert died[0]["raw"].startswith("[[Sha'ban 15]]")   # provenance kept
    node = next(n for n in ex.nodes if n["id"] == "date:ah:0329-08-15")
    assert node["kind"] == "date"
    # Both calendars survive on the node: the Hijri side via resolved links
    # (`[[329]]` is a redirect to "329 AH"), the Gregorian year only from the
    # raw value, since "941 CE" is not an article on this wiki.
    assert (node["ah_year"], node["ah_month"], node["ah_day"]) == (329, 8, 15)
    assert (node["ce_year"], node["ce_month"], node["ce_day"]) == (941, 5, 15)


def test_categories_become_instance_of_and_subclass_of(corpus):
    ex = Extractor(corpus).run()
    assert ("page:1", "cat:Faqihs") in _edges(ex, "instance_of")
    assert ("cat:Faqihs", "cat:Shia scholars") in _edges(ex, "subclass_of")


def test_noise_fields_are_neither_edges_nor_curation_worklist(corpus):
    ex = Extractor(corpus).run()
    assert not any(e.get("field") == "image" for e in ex.edges)
    assert not any(u["field"] == "image" for u in ex.unmapped)


def test_navigation_pages_are_skipped_so_the_kg_matches_the_index(corpus):
    ex = Extractor(corpus).run()
    assert ex.report.skipped_nav == 1
    assert not any(n["label"] == "Something (disambiguation)" for n in ex.nodes)


def test_unresolvable_targets_are_dropped_and_counted(corpus):
    """A link to a page that does not exist must not invent a node."""
    s = corpus
    pages = [json.loads(l) for l in (s.raw_dir / "ns0.jsonl").read_text().splitlines()]
    pages[0]["wikitext"] = pages[0]["wikitext"].replace(
        "[[Student One]]", "[[Student One]] and [[No Such Person]]")
    (s.raw_dir / "ns0.jsonl").write_text("\n".join(json.dumps(p) for p in pages) + "\n")

    ex = Extractor(s).run()
    assert _edges(ex, "taught") == {("page:1", "page:5")}
    assert ex.report.unresolved_targets >= 1


def test_redirects_resolve_to_the_article_they_point_at(corpus):
    s = corpus
    pages = [json.loads(l) for l in (s.raw_dir / "ns0.jsonl").read_text().splitlines()]
    pages[1]["wikitext"] = pages[1]["wikitext"].replace(
        "author = [[Test Scholar]]", "author = [[Test Sch.]]")
    (s.raw_dir / "ns0.jsonl").write_text("\n".join(json.dumps(p) for p in pages) + "\n")

    ex = Extractor(s).run()
    assert _edges(ex, "authored_by") == {("page:2", "page:1")}


def test_the_same_fact_stated_twice_yields_one_edge(corpus):
    ex = Extractor(corpus).run()
    keys = [(e["s"], e["p"], e["o"]) for e in ex.edges]
    assert len(keys) == len(set(keys))


def test_body_wikilinks_become_a_counted_link_graph(corpus):
    ex = Extractor(corpus).run()
    links = {(l["s"], l["o"]): l["n"] for l in ex.links}
    # "Teacher One" is linked from both the infobox and the prose.
    assert links[("page:1", "page:3")] == 2
    assert ("page:1", "page:1") not in links      # no self-links


def test_title_normalisation_follows_mediawiki_rules():
    assert normalise_title("imam_ali") == "Imam ali"
    assert normalise_title("Najaf#Shrine") == "Najaf"
