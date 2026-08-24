"""Orchestration: raw cache -> `data/kg/*.jsonl` -> `data/kg.db`.

The JSONL files are canonical. They are greppable, diffable and survive any
schema change to the database, which mirrors how `data/raw/` relates to
`data/shiaqa.db`. `kg.db` is a derived index, rebuilt from them in seconds.

Reads `data/raw/` and writes `data/kg*`. It never opens `data/shiaqa.db`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from ..config import Settings, get_settings
from ..log import get_logger
from .extract import ExtractReport, Extractor
from .ontology import Ontology
from .overrides import OverrideReport
from .store import KgStore

log = get_logger(__name__)

NODES = "nodes.jsonl"
EDGES = "edges.jsonl"
LINKS = "links.jsonl"
UNMAPPED = "unmapped.jsonl"
ONTOLOGY = "ontology.json"


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


@dataclass
class BuildReport:
    extract: ExtractReport
    files: dict[str, int]
    db_counts: dict[str, int]
    seconds: float
    overrides: OverrideReport = field(default_factory=OverrideReport)


def build(settings: Settings | None = None, jsonl_only: bool = False) -> BuildReport:
    """Full rebuild. Cheap enough (~15 s) that there is no resumable phase."""
    s = settings or get_settings()
    onto = Ontology()
    t0 = time.time()

    ex = Extractor(s, onto).run()

    s.kg_dir.mkdir(parents=True, exist_ok=True)
    files = {
        NODES: _write_jsonl(s.kg_dir / NODES, ex.nodes),
        EDGES: _write_jsonl(s.kg_dir / EDGES, ex.edges),
        LINKS: _write_jsonl(s.kg_dir / LINKS, ex.links),
        UNMAPPED: _write_jsonl(s.kg_dir / UNMAPPED, ex.unmapped),
    }
    (s.kg_dir / ONTOLOGY).write_text(
        json.dumps({"ontology": onto.snapshot(), "extraction": ex.report.as_dict(),
                    "overrides": {"dropped": ex.overrides.dropped,
                                  "added": ex.overrides.added,
                                  "stale": ex.overrides.stale,
                                  "unresolved": ex.overrides.unresolved}},
                   indent=2, ensure_ascii=False)
    )
    log.info("wrote %s", ", ".join(f"{k} ({v})" for k, v in files.items()))

    db_counts: dict[str, int] = {}
    if not jsonl_only:
        db_counts = load_db(s)

    return BuildReport(extract=ex.report, files=files, db_counts=db_counts,
                       seconds=time.time() - t0, overrides=ex.overrides)


def load_db(settings: Settings | None = None) -> dict[str, int]:
    """Rebuild `data/kg.db` from the JSONL files."""
    s = settings or get_settings()
    kg = KgStore(s)
    kg.reset()
    kg.load(
        read_jsonl(s.kg_dir / NODES),
        read_jsonl(s.kg_dir / EDGES),
        read_jsonl(s.kg_dir / LINKS),
    )
    kg.set_meta("built_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    kg.set_meta("source", str(s.raw_dir))
    kg.db.commit()
    counts = kg.counts()
    kg.close()
    log.info("kg.db: %s", counts)
    return counts
