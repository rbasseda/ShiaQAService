"""`kg.db`: loading, traversal, and rendering edges back into sentences."""
import pytest

from shiaqa.config import Settings
from shiaqa.kg.ontology import Ontology
from shiaqa.kg.store import KgStore

NODES = [
    {"id": "page:1", "kind": "entity", "label": "Al-Kulayni", "page_id": 1,
     "url": "http://x/1", "classes": ["Scholar"], "aliases": ["Kulayni"]},
    {"id": "page:2", "kind": "entity", "label": "Al-Kafi (book)", "page_id": 2,
     "url": "http://x/2", "classes": ["Book"]},
    {"id": "page:3", "kind": "entity", "label": "Ali b. Ibrahim", "page_id": 3,
     "url": "http://x/3", "classes": ["Scholar"]},
    {"id": "page:4", "kind": "entity", "label": "Ethics", "page_id": 4,
     "url": "http://x/4", "classes": []},
    {"id": "cat:Faqihs", "kind": "class", "label": "Faqihs"},
    {"id": "date:ah:0329", "kind": "date", "label": "329 AH", "ah_year": 329},
]
EDGES = [
    {"s": "page:1", "p": "authored", "o": "page:2", "src": "infobox", "field": "works"},
    # The same fact, asserted from the book's side as well.
    {"s": "page:2", "p": "authored_by", "o": "page:1", "src": "infobox", "field": "author"},
    {"s": "page:1", "p": "taught_by", "o": "page:3", "src": "infobox",
     "field": "professors"},
    {"s": "page:1", "p": "died_on", "o": "date:ah:0329", "src": "infobox",
     "field": "death"},
    {"s": "page:2", "p": "about_subject", "o": "page:4", "src": "infobox",
     "field": "subject"},
    {"s": "page:1", "p": "instance_of", "o": "cat:Faqihs", "src": "category"},
]
LINKS = [{"s": "page:1", "o": "page:2", "n": 3}]


@pytest.fixture
def kg(tmp_path):
    s = Settings(data_dir=tmp_path)
    store = KgStore(s, path=tmp_path / "kg.db")
    store.init_schema()
    store.load(NODES, EDGES, LINKS)
    return store


def test_counts_separate_typed_edges_from_taxonomy(kg):
    c = kg.counts()
    assert c["nodes"] == 6
    assert c["edges"] == 6
    assert c["typed_edges"] == 5       # instance_of excluded
    assert c["links"] == 1
    assert kg.is_populated


def test_a_node_is_reachable_from_its_page_id(kg):
    assert kg.node_for_page(1) == "page:1"
    assert kg.node_for_page(999) is None
    assert kg.classes_of("page:1") == ["Scholar"]


def test_extra_node_attributes_survive_the_round_trip(kg):
    import json
    row = kg.node("date:ah:0329")
    assert json.loads(row["data"])["ah_year"] == 329


def test_edges_are_readable_in_both_directions(kg):
    edges = kg.edges_of("page:2")
    assert {(e["dir"], e["p"], e["other"]) for e in edges} == {
        ("out", "authored_by", "page:1"),
        ("out", "about_subject", "page:4"),
        ("in", "authored", "page:1"),
    }


def test_taxonomy_edges_are_excluded_unless_asked_for(kg):
    assert all(e["p"] != "instance_of" for e in kg.edges_of("page:1"))
    assert any(e["p"] == "instance_of"
               for e in kg.edges_of("page:1", include_taxonomy=True))


def test_facts_read_an_inbound_edge_through_its_inverse(kg):
    facts = kg.facts("page:2", Ontology().predicates, 10)
    assert "Al-Kafi (book) was written by Al-Kulayni." in facts


def test_the_same_fact_asserted_from_both_sides_renders_once(kg):
    """al-Kulayni's `works` and al-Kafi's `author` are two edges, one sentence."""
    facts = kg.facts("page:1", Ontology().predicates, 10)
    assert facts.count("Al-Kulayni wrote Al-Kafi (book).") == 1


def test_neighbour_pages_expand_one_typed_hop_without_returning_the_seed(kg):
    assert kg.neighbour_pages(["page:1"], 10) == [2, 3]


def test_a_seed_is_never_returned_as_its_own_neighbour(kg):
    """Seeds already reach the context through the title channel."""
    assert 1 not in kg.neighbour_pages(["page:1", "page:2"], 10)
    assert 2 not in kg.neighbour_pages(["page:1", "page:2"], 10)


def test_neighbour_pages_round_robin_so_one_seed_cannot_dominate(kg):
    """Each seed gets a turn before any seed gets a second slot."""
    # page:1 -> page:3, page:2 -> page:4 (page:1 and page:2 exclude each other).
    assert kg.neighbour_pages(["page:1", "page:2"], 2) == [3, 4]


def test_neighbours_of_nothing_is_nothing(kg):
    assert kg.neighbour_pages([], 5) == []
    assert kg.facts("page:404", Ontology().predicates, 5) == []


def test_dates_are_not_offered_as_neighbour_pages(kg):
    """A date node has no page, so it can never seed a chunk lookup."""
    assert 0 not in kg.neighbour_pages(["page:1"], 10)
    assert all(isinstance(p, int) and p > 0 for p in kg.neighbour_pages(["page:1"], 10))


def test_reset_empties_the_graph(kg):
    kg.reset()
    assert not kg.is_populated
