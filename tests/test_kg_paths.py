"""Multi-hop traversal: what a chain may cross, what it must not, and how it reads.

The fixture graph is built around the pathologies measured on the real corpus,
not around convenience: a teaching chain three deep, a place hub big enough to
trip the degree cap, a reciprocal `taught`/`taught_by` pair (the corpus has
1,278), a symmetric predicate stored from both ends, a cycle, and an inbound
predicate with no inverse (53 of the ontology's 69 have none).
"""
import pytest

from shiaqa.config import Settings
from shiaqa.kg import paths
from shiaqa.kg.ontology import Ontology
from shiaqa.kg.store import KgStore

# Alpha -> Beta -> Gamma -> Delta is the teaching chain. Alpha and Gamma also
# share a birthplace, so they are joined at length two twice over — the shape
# that makes ranked-not-shortest the whole point of `connect`.
NODES = [
    {"id": "page:1", "kind": "entity", "label": "Alpha", "page_id": 1,
     "url": "http://x/1", "classes": ["Scholar"]},
    {"id": "page:2", "kind": "entity", "label": "Beta", "page_id": 2,
     "url": "http://x/2", "classes": ["Scholar"]},
    {"id": "page:3", "kind": "entity", "label": "Gamma", "page_id": 3,
     "url": "http://x/3", "classes": ["Scholar"]},
    {"id": "page:4", "kind": "entity", "label": "Delta", "page_id": 4,
     "url": "http://x/4", "classes": ["Scholar"]},
    {"id": "page:5", "kind": "entity", "label": "BookOne", "page_id": 5,
     "url": "http://x/5", "classes": ["Book"]},
    {"id": "page:6", "kind": "entity", "label": "Epsilon", "page_id": 6,
     "url": "http://x/6", "classes": ["Scholar"]},
    {"id": "page:9", "kind": "entity", "label": "Orphan", "page_id": None,
     "url": None, "classes": []},
    {"id": "page:10", "kind": "entity", "label": "Bigtown", "page_id": 10,
     "url": "http://x/10", "classes": ["Place"]},
    {"id": "cat:Faqihs", "kind": "class", "label": "Faqihs"},
    {"id": "date:ah:0329", "kind": "date", "label": "329 AH", "ah_year": 329},
] + [
    # Fifty residents, purely to push Bigtown over the degree cap.
    {"id": f"page:{100 + i}", "kind": "entity", "label": f"Resident {i}",
     "page_id": 100 + i, "url": f"http://x/{100 + i}", "classes": ["Person"]}
    for i in range(50)
]

EDGES = [
    {"s": "page:1", "p": "taught_by", "o": "page:2", "src": "infobox",
     "field": "professors"},
    # The same fact from Beta's side. One relation, two rows.
    {"s": "page:2", "p": "taught", "o": "page:1", "src": "infobox",
     "field": "students"},
    {"s": "page:2", "p": "taught_by", "o": "page:3", "src": "infobox",
     "field": "professors"},
    {"s": "page:3", "p": "taught_by", "o": "page:4", "src": "infobox",
     "field": "professors"},
    {"s": "page:2", "p": "authored", "o": "page:5", "src": "infobox",
     "field": "works"},
    # Shared birthplace: a length-two Alpha/Gamma route that must lose.
    {"s": "page:1", "p": "born_in", "o": "page:10", "src": "infobox",
     "field": "birth place"},
    {"s": "page:3", "p": "born_in", "o": "page:10", "src": "infobox",
     "field": "birth place"},
    # Symmetric, stored from both ends, and `relative_of` declares no inverse.
    {"s": "page:1", "p": "relative_of", "o": "page:6", "src": "infobox",
     "field": "relatives"},
    {"s": "page:6", "p": "relative_of", "o": "page:1", "src": "infobox",
     "field": "relatives"},
    # Terminal kinds.
    {"s": "page:1", "p": "died_on", "o": "date:ah:0329", "src": "infobox",
     "field": "death"},
    {"s": "page:1", "p": "instance_of", "o": "cat:Faqihs", "src": "category"},
    # Inbound with no inverse: readable only from the far end.
    {"s": "page:5", "p": "narrated_by", "o": "page:4", "src": "infobox",
     "field": "narrators"},
    # Reaches a node with no page_id at all.
    {"s": "page:4", "p": "taught", "o": "page:9", "src": "infobox",
     "field": "students"},
] + [
    {"s": f"page:{100 + i}", "p": "resided_in", "o": "page:10",
     "src": "infobox", "field": "residence"} for i in range(50)
]

LINKS = [{"s": "page:1", "o": "page:2", "n": 3}]


@pytest.fixture
def kg(tmp_path):
    store = KgStore(Settings(data_dir=tmp_path), path=tmp_path / "kg.db")
    store.init_schema()
    store.load(NODES, EDGES, LINKS)
    return store


@pytest.fixture
def preds():
    return Ontology().predicates


def _ends(found):
    return {p.end for p in found}


def _narrations(kg, found, preds):
    persons = paths.person_nodes(kg, [n for p in found for n in p.nodes])
    return [paths.narrate(p, preds, persons) for p in found]


# --- reach -------------------------------------------------------------------

def test_two_hops_reach_a_node_one_hop_cannot(kg, preds):
    one = paths.expand(kg, ["page:1"], preds, max_hops=1, limit=99)
    two = paths.expand(kg, ["page:1"], preds, max_hops=2, limit=99)
    assert "page:3" not in _ends(one)
    assert "page:3" in _ends(two)


def test_a_path_never_revisits_a_node(kg, preds):
    for p in paths.expand(kg, ["page:1"], preds, max_hops=3, limit=99):
        assert len(set(p.nodes)) == len(p.nodes)


def test_the_seed_is_never_returned_as_its_own_neighbour(kg, preds):
    for p in paths.expand(kg, ["page:1"], preds, max_hops=3, limit=99):
        assert "page:1" not in [s.dst for s in p.steps]


def test_a_named_article_is_never_walked_back_onto(kg, preds):
    """Both seeds are already in context; a path between them adds nothing."""
    for p in paths.expand(kg, ["page:1", "page:3"], preds, max_hops=2, limit=99):
        assert p.end not in {"page:1", "page:3"}


# --- deduplication -----------------------------------------------------------

def test_a_reciprocal_pair_yields_one_path_not_two(kg, preds):
    """Alpha--taught_by-->Beta and Beta--taught-->Alpha are one relation."""
    found = paths.expand(kg, ["page:1"], preds, max_hops=1, limit=99)
    to_beta = [p for p in found if p.end == "page:2"]
    assert len(to_beta) == 1


def test_a_symmetric_predicate_is_canonicalised_by_endpoint(kg, preds):
    """`relative_of` declares no inverse but is stored from both ends."""
    found = paths.expand(kg, ["page:1"], preds, max_hops=1, limit=99)
    assert len([p for p in found if p.end == "page:6"]) == 1


def test_canonical_folds_an_edge_and_its_reciprocal(preds):
    forward = paths.canonical(("page:1", "taught_by", "page:2"), preds)
    backward = paths.canonical(("page:2", "taught", "page:1"), preds)
    assert forward == backward


def test_canonical_folds_a_self_inverse_predicate_by_endpoint_order(preds):
    assert (paths.canonical(("page:2", "sibling", "page:1"), preds)
            == paths.canonical(("page:1", "sibling", "page:2"), preds))


# --- transit guards ----------------------------------------------------------

def test_a_place_may_end_a_path_but_never_continue_one(kg, preds):
    """`born_in Bigtown` is a fact; Bigtown read backwards is fifty strangers."""
    found = paths.expand(kg, ["page:1"], preds, max_hops=2, limit=99)
    assert "page:10" in _ends(found)
    residents = {f"page:{100 + i}" for i in range(50)}
    assert not (_ends(found) & residents)


def test_a_node_above_the_degree_cap_is_not_expanded(kg, preds):
    """Even with transit allowed, sheer degree stops the walk.

    This is the guard `NO_TRANSIT` structurally cannot provide: on the real
    corpus the edges into the biggest hubs are `taught_by` and `companion_of`,
    the very predicates multi-hop questions turn on.
    """
    found = paths.expand(kg, ["page:1"], preds, max_hops=2, limit=99,
                         no_transit=frozenset())
    residents = {f"page:{100 + i}" for i in range(50)}
    assert not (_ends(found) & residents)
    loose = paths.expand(kg, ["page:1"], preds, max_hops=2, limit=99,
                         no_transit=frozenset(), hub_degree_max=999)
    assert _ends(loose) & residents


def test_dates_and_classes_end_a_path_and_are_never_expanded(kg, preds):
    found = paths.expand(kg, ["page:1"], preds, max_hops=3, limit=99)
    assert "date:ah:0329" in _ends(found)
    # Taxonomy is excluded from traversal entirely, as in `edges_of`.
    assert "cat:Faqihs" not in _ends(found)
    for p in found:
        for step in p.steps[:-1]:
            assert step.dst_kind == "entity"


# --- ranking -----------------------------------------------------------------

def test_a_two_hop_path_never_outranks_a_strong_one_hop_path():
    """The property that keeps the regression suite still when hops go 1 -> 2.

    A second hop may only displace tier-3 noise, never a real first-hop
    neighbour. If HOP_DECAY or TIER_WEIGHT is ever retuned, this is what must
    still hold.
    """
    def score(*preds_):
        return paths.score_steps(tuple(
            paths.Step(p=p, direction="out", src="a", dst="b", dst_label="l",
                       dst_kind="entity", dst_page_id=1, field=None)
            for p in preds_))

    two_hop_best = score("taught_by", "taught_by")
    for one_hop in ("taught_by", "buried_at", "about_subject"):
        assert score(one_hop) > two_hop_best
    assert two_hop_best > score("resided_in")


def test_expansion_is_deterministic_across_runs(kg, preds):
    runs = [[(p.seed, tuple(s.stored for s in p.steps))
             for p in paths.expand(kg, ["page:1", "page:3"], preds,
                                   max_hops=2, limit=20)] for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_deep_paths_survive_the_limit_even_though_one_hop_scores_higher(kg, preds):
    """Otherwise the chain block is empty exactly when it matters.

    Every one-hop path outranks every two-hop path by construction, so a flat
    score-ordered truncation would drop all chains for any well-connected seed.
    """
    found = paths.expand(kg, ["page:1"], preds, max_hops=2, limit=4)
    assert any(p.hops >= 2 for p in found)


def test_round_robin_stops_one_seed_from_owning_the_expansion(kg, preds):
    found = paths.expand(kg, ["page:1", "page:3"], preds, max_hops=1, limit=2)
    assert {p.seed for p in found} == {"page:1", "page:3"}


# --- connect -----------------------------------------------------------------

def test_connect_prefers_a_teaching_chain_over_a_shared_birthplace(kg, preds):
    """The measured al-Mufid/al-Tusi case, in miniature.

    Alpha and Gamma are joined at length two both by Beta (who taught both) and
    by Bigtown (where both were born). Shortest-path alone cannot tell them
    apart; refusing to travel through a birthplace can.
    """
    found = paths.connect(kg, "page:1", "page:3", preds)
    assert found
    assert all(p.steps[0].dst == "page:2" for p in found)
    assert all("page:10" not in p.nodes for p in found)


def test_connect_returns_nothing_for_unrelated_seeds(kg, preds):
    assert paths.connect(kg, "page:1", "page:9", preds, max_hops=1) == []


def test_connect_reports_the_shortest_route_not_a_detour(kg, preds):
    found = paths.connect(kg, "page:1", "page:3", preds, max_hops=3)
    assert {p.hops for p in found} == {2}


def test_connect_refuses_a_node_paired_with_itself(kg, preds):
    assert paths.connect(kg, "page:1", "page:1", preds) == []


# --- compare -----------------------------------------------------------------

def test_compare_aligns_only_predicates_present_on_both_sides(kg, preds):
    cmp = paths.compare(kg, "page:1", "page:3", preds)
    assert cmp is not None
    labels = {r.label for r in cmp.rows}
    assert "was born in" in labels          # both have it
    assert "wrote" not in labels            # only Beta authored anything


def test_compare_never_states_a_relation_backwards(kg, preds):
    """Keying on (predicate, direction) would print "Beta taught Gamma"."""
    cmp = paths.compare(kg, "page:1", "page:2", preds)
    assert cmp is not None
    for row in cmp.rows:
        if row.label == "taught":
            # Beta taught Alpha; Alpha taught nobody in this fixture.
            assert "Gamma" not in row.a_objects


def test_compare_marks_what_the_two_share(kg, preds):
    cmp = paths.compare(kg, "page:1", "page:3", preds)
    born = next(r for r in cmp.rows if r.label == "was born in")
    assert born.shared == ("Bigtown",)


def test_compare_returns_none_when_nothing_aligns(kg, preds):
    """Bigtown's only edges are inbound with no inverse, so nothing lines up."""
    assert paths.compare(kg, "page:5", "page:10", preds) is None


# --- narration ---------------------------------------------------------------

def test_a_chain_reads_as_one_sentence_with_a_relative_pronoun(kg, preds):
    found = paths.expand(kg, ["page:1"], preds, max_hops=2, limit=99)
    chain = next(p for p in found if p.hops == 2 and p.end == "page:3")
    assert (paths.narrate(chain, preds, paths.person_nodes(kg, list(chain.nodes)))
            == "Alpha was taught by Beta, who was taught by Gamma.")


def test_a_non_person_link_reads_which_not_who(kg, preds):
    found = paths.expand(kg, ["page:3"], preds, max_hops=3, limit=99)
    chain = next(p for p in found if p.end == "page:5")
    said = paths.narrate(chain, preds, paths.person_nodes(kg, list(chain.nodes)))
    assert "which wrote" not in said
    assert said.endswith("BookOne.")


def test_a_chain_with_an_uninvertible_step_breaks_the_sentence(kg, preds):
    """`narrated_by` has no inverse, so the chain cannot run forwards through it.

    Chaining it regardless would assert the relation backwards — a quietly false
    sentence in the prompt, which is worse than a clumsy one.
    """
    found = paths.expand(kg, ["page:3"], preds, max_hops=2, limit=99)
    # Second step is the *inbound* `narrated_by`: read from Delta's side there is
    # no verb to continue with, because the relation only runs the other way.
    chain = next(p for p in found
                 if p.end == "page:5" and p.steps[-1].direction == "in")
    said = paths.narrate(chain, preds, paths.person_nodes(kg, list(chain.nodes)))
    assert said == "Gamma was taught by Delta, and BookOne was narrated by Delta."
    assert "who was narrated by" not in said


def test_an_outbound_uninvertible_step_still_chains(kg, preds):
    """The mirror case: `narrated_by` read forwards needs no inverse at all."""
    found = paths.expand(kg, ["page:2"], preds, max_hops=2, limit=99)
    chain = next(p for p in found
                 if p.end == "page:4" and "page:5" in p.nodes)
    said = paths.narrate(chain, preds, paths.person_nodes(kg, list(chain.nodes)))
    assert said == "Beta wrote BookOne, which was narrated by Delta."


def test_a_one_step_path_narrates_as_a_plain_fact(kg, preds):
    found = paths.expand(kg, ["page:1"], preds, max_hops=1, limit=99)
    chain = next(p for p in found if p.end == "page:2")
    assert paths.narrate(chain, preds) == "Alpha was taught by Beta."


def test_narrating_an_empty_path_is_none(preds):
    assert paths.narrate(
        paths.Path(seed="page:1", seed_label="Alpha", steps=(), score=0.0),
        preds) is None


# --- provenance --------------------------------------------------------------

def test_every_step_carries_the_infobox_field_it_came_from(kg, preds):
    for p in paths.expand(kg, ["page:1"], preds, max_hops=2, limit=99):
        for step in p.steps:
            assert step.field is not None


def test_a_step_carries_the_page_id_so_retrieval_needs_no_second_lookup(kg, preds):
    found = paths.expand(kg, ["page:1"], preds, max_hops=1, limit=99)
    beta = next(p for p in found if p.end == "page:2")
    assert beta.end_page_id == 2


def test_a_node_without_a_page_id_still_traverses(kg, preds):
    """Orphan has no page, so it can be narrated but never retrieved."""
    found = paths.expand(kg, ["page:4"], preds, max_hops=1, limit=99)
    orphan = next(p for p in found if p.end == "page:9")
    assert orphan.end_page_id is None


# --- degenerate input --------------------------------------------------------

def test_no_seeds_yields_no_paths(kg, preds):
    assert paths.expand(kg, [], preds) == []


def test_zero_hops_yields_no_paths(kg, preds):
    assert paths.expand(kg, ["page:1"], preds, max_hops=0) == []


def test_an_unknown_seed_yields_no_paths(kg, preds):
    assert paths.expand(kg, ["page:99999"], preds, max_hops=2) == []
