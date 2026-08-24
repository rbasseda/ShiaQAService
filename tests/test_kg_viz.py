"""Graph -> self-contained HTML: data shaping, and the constraints the page must meet."""
import re

import pytest

from shiaqa.config import Settings
from shiaqa.kg.ontology import Ontology
from shiaqa.kg.store import KgStore
from shiaqa.kg.viz import explorer_payload, render, schema_payload, write

NODES = [
    {"id": "cat:Faqihs", "kind": "class", "label": "Faqihs"},
    {"id": "date:ah:0329", "kind": "date", "label": "329 AH", "ah_year": 329},
    {"id": "page:1", "kind": "entity", "label": "Al-Kulayni", "page_id": 1,
     "url": "http://x/1", "classes": ["Scholar"]},
    {"id": "page:2", "kind": "entity", "label": "Al-Kafi (book)", "page_id": 2,
     "url": "http://x/2", "classes": ["Book"]},
    {"id": "page:3", "kind": "entity", "label": "Lonely Page", "page_id": 3,
     "url": "http://x/3", "classes": []},
]
EDGES = [
    {"s": "page:1", "p": "authored", "o": "page:2", "src": "infobox", "field": "works"},
    {"s": "page:2", "p": "authored_by", "o": "page:1", "src": "infobox", "field": "author"},
    {"s": "page:1", "p": "died_on", "o": "date:ah:0329", "src": "infobox", "field": "death"},
    {"s": "page:1", "p": "instance_of", "o": "cat:Faqihs", "src": "category"},
    {"s": "page:3", "p": "instance_of", "o": "cat:Faqihs", "src": "category"},
]


@pytest.fixture
def kg(tmp_path):
    s = Settings(data_dir=tmp_path)
    store = KgStore(s, path=tmp_path / "kg.db")
    store.init_schema()
    store.load(NODES, EDGES, [{"s": "page:1", "o": "page:2", "n": 4}])
    return store, s


# --- explorer payload --------------------------------------------------------

def test_explorer_carries_only_typed_edges(kg):
    store, s = kg
    p = explorer_payload(store, Ontology(), s)
    assert p["stats"]["edges"] == 3            # instance_of excluded
    assert "instance_of" not in p["preds"]


def test_explorer_interns_ids_so_the_whole_graph_fits_in_one_file(kg):
    store, s = kg
    p = explorer_payload(store, Ontology(), s)
    for edge in p["edges"]:
        s_i, p_i, o_i, f_i, c_i = edge
        assert isinstance(s_i, int) and isinstance(o_i, int)
        assert p["preds"][p_i]
        assert p["fields"][f_i] in ("works", "author", "death", "")
        assert p["srcs"][c_i] == "infobox"


def test_explorer_stores_no_urls_because_they_are_derivable(kg):
    store, s = kg
    p = explorer_payload(store, Ontology(), s)
    assert p["viewBase"] == s.wiki_view_base
    assert not any("http" in str(n) for n in p["nodes"])


def test_explorer_marks_which_entities_have_relations(kg):
    """The isolated half of the corpus must be reachable, not silently dropped."""
    store, s = kg
    p = explorer_payload(store, Ontology(), s)
    labels = [n[0] for n in p["nodes"]]
    assert "Lonely Page" in labels                       # present as a node
    lonely = labels.index("Lonely Page")
    assert lonely not in p["connected"]                  # but honestly marked
    assert p["stats"]["connected"] == 3                  # kulayni, kafi, the date


def test_explorer_keeps_categories_out_of_the_edge_list(kg):
    store, s = kg
    p = explorer_payload(store, Ontology(), s)
    kulayni = [n[0] for n in p["nodes"]].index("Al-Kulayni")
    faqihs = [n[0] for n in p["nodes"]].index("Faqihs")
    assert p["cats"][str(kulayni)] == [faqihs]


def test_explorer_exposes_inverses_so_reciprocal_edges_can_be_collapsed(kg):
    store, s = kg
    p = explorer_payload(store, Ontology(), s)
    ai = p["preds"].index("authored")
    assert p["preds"][p["predInverse"][ai]] == "authored_by"


# --- schema payload ----------------------------------------------------------

def test_schema_aggregates_edges_to_class_level(kg):
    store, _ = kg
    p = schema_payload(store, Ontology())
    triples = {(a, b, c): n for a, b, c, n in p["triples"]}
    assert triples[("Scholar", "authored", "Book")] == 1
    assert triples[("Book", "authored_by", "Scholar")] == 1


def test_schema_buckets_dates_separately_from_unclassed(kg):
    store, _ = kg
    p = schema_payload(store, Ontology())
    triples = {(a, b, c) for a, b, c, _ in p["triples"]}
    assert ("Scholar", "died_on", "Date") in triples


def test_schema_reports_predicates_the_corpus_never_fills(kg):
    store, _ = kg
    p = schema_payload(store, Ontology())
    assert "taught_by" in p["unusedPredicates"]          # unused in this fixture
    assert "authored" not in p["unusedPredicates"]


# --- rendering ---------------------------------------------------------------

def test_render_escapes_a_closing_tag_hidden_in_the_data():
    """A label containing `</script>` would otherwise end the block early."""
    html = render("schema.html", {"triples": [], "classes": [],
                                  "predLabels": {}, "unusedPredicates": [],
                                  "stats": {"classes": 0, "predicates": 0,
                                            "triples": 0, "edges": 0},
                                  "evil": "</script><script>alert(1)"}, "T")
    assert "</script><script>alert(1)" not in html
    assert "<\\/script>" in html


def test_generated_pages_are_self_contained(kg, tmp_path):
    """An Artifact's CSP blocks every external host except Google Fonts."""
    store, s = kg
    store.close()
    written = write(tmp_path / "viz", ("explorer", "schema"), s)
    allowed = {"fonts.googleapis.com", "fonts.gstatic.com", "www.w3.org",
               "en.wikishia.net"}
    for path in written.values():
        hosts = set(re.findall(r"https?://([A-Za-z0-9.-]+)", path.read_text()))
        assert hosts <= allowed, f"{path.name} reaches {hosts - allowed}"
        assert "<meta charset" in path.read_text()


def test_write_produces_both_views(kg, tmp_path):
    store, s = kg
    store.close()
    written = write(tmp_path / "viz", ("explorer", "schema"), s)
    assert set(written) == {"explorer", "schema"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 1000
