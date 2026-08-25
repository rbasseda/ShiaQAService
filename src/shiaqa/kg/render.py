"""Turning one edge back into one English clause.

Extracted from `KgStore.facts()`, which used to own this logic alone. Path
narration in `paths.py` needs exactly the same decisions — which direction the
sentence runs, whether an inverse predicate exists to run it forwards — and two
copies of that reasoning is the silent-drift failure mode this project
documents elsewhere (`Chunk.embed_text()` versus `pipeline.build_vectors()`).
One copy, two callers.

The subtlety worth preserving: an inbound edge can only be read from the
anchor's side when its predicate declares an `inverse`. `al-Kafi authored_by
al-Kulayni` read from al-Kulayni's side becomes "al-Kulayni wrote al-Kafi"
because `authored_by.inverse == "authored"`. But `narrated_by`, `source_shia`
and `combatant` have no inverse, and there the sentence simply cannot start at
the anchor — it has to start at the other end instead. Forcing it would assert
the relation backwards.
"""

from __future__ import annotations


def clause(edge: dict, predicates: dict) -> tuple[str, str] | None:
    """`(verb phrase, other label)` read from the anchor's side.

    `None` when the sentence cannot start at the anchor: an unmapped predicate,
    an unlabelled far end, or an inbound edge whose predicate has no inverse.
    Callers wanting a sentence regardless should fall back to `flipped()`.
    """
    label = edge.get("label")
    if not label:
        return None
    pred = predicates.get(edge["p"])
    if pred is None:
        return None
    if edge["dir"] == "out":
        return pred.label, label
    inverse = predicates.get(pred.inverse) if pred.inverse else None
    if inverse is None:
        return None
    return inverse.label, label


def flipped(edge: dict, predicates: dict) -> tuple[str, str] | None:
    """`(other label, verb phrase)` for an inbound edge read from the far end."""
    label = edge.get("label")
    if not label:
        return None
    pred = predicates.get(edge["p"])
    if pred is None:
        return None
    return label, pred.label


def sentence(subject: str, edge: dict, predicates: dict) -> str | None:
    """One standalone fact, ending in a full stop. `None` if unrenderable."""
    said = clause(edge, predicates)
    if said is not None:
        verb, other = said
        return f"{subject} {verb} {other}."
    back = flipped(edge, predicates)
    if back is None:
        return None
    other, verb = back
    return f"{other} {verb} {subject}."
