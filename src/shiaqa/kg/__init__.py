"""Structural knowledge graph over the WikiShia corpus.

Built from the raw wikitext cache with no model in the loop: infobox fields are
already typed relations, category pages are already a taxonomy, and 35k
redirects already resolve alternative spellings. See `ontology.py` for the
curated mapping and `extract.py` for why this is a separate pass from ingest.
"""

from .ontology import Ontology
from .store import KgStore

__all__ = ["Ontology", "KgStore"]
