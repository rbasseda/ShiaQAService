"""Pull the corpus down to disk once, then never touch the network again.

Raw wikitext is cached as JSONL per namespace. Cleaning, chunking and
embedding all read from that cache, so you can iterate on the pipeline
without re-fetching (and without being a nuisance to the wiki).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..config import Settings, get_settings
from ..log import get_logger
from .client import WikiClient

log = get_logger(__name__)


def raw_path(settings: Settings, namespace: int) -> Path:
    return settings.raw_dir / f"ns{namespace}.jsonl"


def read_raw(settings: Settings | None = None, namespaces: list[int] | None = None) -> Iterator[dict]:
    """Stream cached pages back off disk."""
    s = settings or get_settings()
    for ns in namespaces if namespaces is not None else s.namespaces:
        path = raw_path(s, ns)
        if not path.exists():
            log.warning("no cache for namespace %s (%s); run `shiaqa fetch` first", ns, path)
            continue
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def raw_counts(settings: Settings | None = None) -> dict[int, int]:
    s = settings or get_settings()
    counts = {}
    for ns in s.namespaces:
        path = raw_path(s, ns)
        counts[ns] = sum(1 for _ in path.open()) if path.exists() else 0
    return counts


async def fetch_namespace(
    client: WikiClient, namespace: int, settings: Settings, force: bool = False
) -> int:
    out_path = raw_path(settings, namespace)
    if out_path.exists() and not force:
        n = sum(1 for _ in out_path.open())
        log.info("namespace %s already cached (%d pages); use --force to refetch", namespace, n)
        return n

    pages = await client.list_pages(namespace)
    titles = [p["title"] for p in pages]
    log.info("namespace %s: fetching wikitext for %d pages", namespace, len(titles))

    tmp = out_path.with_suffix(".jsonl.part")
    written = 0
    with tmp.open("w") as fh:
        for i in range(0, len(titles), settings.batch_titles):
            batch = titles[i : i + settings.batch_titles]
            for page in await client.fetch_batch(batch):
                fh.write(json.dumps(page, ensure_ascii=False) + "\n")
                written += 1
            done = min(i + settings.batch_titles, len(titles))
            if done % 500 < settings.batch_titles or done == len(titles):
                log.info("namespace %s: %d/%d pages", namespace, done, len(titles))
    tmp.replace(out_path)
    log.info("namespace %s: wrote %d pages to %s", namespace, written, out_path)
    return written


async def fetch_all(
    settings: Settings | None = None,
    namespaces: list[int] | None = None,
    force: bool = False,
) -> dict[int, int]:
    s = settings or get_settings()
    targets = namespaces if namespaces is not None else s.namespaces
    results: dict[int, int] = {}
    async with WikiClient(s) as client:
        for ns in targets:
            results[ns] = await fetch_namespace(client, ns, s, force=force)

        # Redirect titles are free aliases for article titles; store them
        # alongside the raw pages for use as extra retrieval signal.
        alias_path = s.raw_dir / "redirects.json"
        if force or not alias_path.exists():
            log.info("collecting redirect aliases (ns 0)")
            aliases = await client.redirect_map(0)
            alias_path.write_text(json.dumps(aliases, ensure_ascii=False, indent=0))
            log.info("stored %d redirect aliases", len(aliases))
    return results


def load_redirects(settings: Settings | None = None) -> dict[str, str]:
    """`{alias: target title}` — the file as stored.

    Retrieval wants the inverse (`load_aliases`), but resolving a wikilink to
    the article it lands on needs this direction.
    """
    s = settings or get_settings()
    path = s.raw_dir / "redirects.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_aliases(settings: Settings | None = None) -> dict[str, list[str]]:
    """`{target title: [alias, ...]}`."""
    by_target: dict[str, list[str]] = {}
    for src, dst in load_redirects(settings).items():
        by_target.setdefault(dst, []).append(src)
    return by_target
