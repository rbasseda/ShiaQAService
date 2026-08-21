"""A small, well-behaved MediaWiki API client.

WikiShia is a volunteer-run wiki on modest hosting, so this client identifies
itself, serialises its requests, respects `maxlag`, and retries with backoff
instead of hammering. The whole corpus is ~103 batched requests, so there is
no reason to be aggressive.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

import httpx

from ..config import Settings, get_settings
from ..log import get_logger

log = get_logger(__name__)


class WikiError(RuntimeError):
    pass


class WikiClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self._client: httpx.AsyncClient | None = None
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "WikiClient":
        self._client = httpx.AsyncClient(
            headers={"User-Agent": self.s.user_agent, "Accept-Encoding": "gzip"},
            timeout=httpx.Timeout(60.0, connect=15.0),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.s.request_delay:
            await asyncio.sleep(self.s.request_delay - elapsed)
        self._last_call = time.monotonic()

    async def get(self, **params: Any) -> dict[str, Any]:
        """One API call, serialised and retried. Returns the parsed JSON body."""
        if self._client is None:
            raise WikiError("WikiClient must be used as an async context manager")
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        params.setdefault("maxlag", self.s.maxlag)

        delay = 1.0
        for attempt in range(6):
            async with self._lock:
                await self._throttle()
                try:
                    resp = await self._client.get(self.s.wiki_api, params=params)
                except httpx.HTTPError as exc:  # network hiccup
                    log.warning("network error (%s), retrying in %.1fs", exc, delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue

            if resp.status_code in (429, 503):
                wait = float(resp.headers.get("Retry-After", delay))
                log.warning("HTTP %s from wiki, sleeping %.1fs", resp.status_code, wait)
                await asyncio.sleep(wait)
                delay = min(delay * 2, 30)
                continue
            resp.raise_for_status()
            data = resp.json()

            # maxlag rejections come back as a normal 200 with an error block.
            err = data.get("error")
            if err and err.get("code") == "maxlag":
                log.warning("wiki replication lag, sleeping %.1fs", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            if err:
                raise WikiError(f"{err.get('code')}: {err.get('info')}")
            return data

        raise WikiError(f"giving up after repeated failures: {params}")

    async def paged(self, **params: Any) -> AsyncIterator[dict[str, Any]]:
        """Follow MediaWiki's `continue` protocol, yielding each response."""
        cont: dict[str, Any] = {}
        while True:
            data = await self.get(**params, **cont)
            yield data
            if "continue" not in data:
                return
            cont = data["continue"]

    async def list_pages(self, namespace: int) -> list[dict[str, Any]]:
        """Every non-redirect page in a namespace: [{pageid, ns, title}, ...]."""
        out: list[dict[str, Any]] = []
        async for data in self.paged(
            action="query",
            list="allpages",
            apnamespace=namespace,
            aplimit="max",
            apfilterredir="nonredirects",
        ):
            out.extend(data.get("query", {}).get("allpages", []))
            log.info("namespace %s: enumerated %d pages", namespace, len(out))
        return out

    async def fetch_batch(self, titles: list[str]) -> list[dict[str, Any]]:
        """Wikitext + metadata for up to `batch_titles` pages in one request."""
        data = await self.get(
            action="query",
            prop="revisions|categories|info",
            rvprop="content|ids|timestamp",
            rvslots="main",
            cllimit="max",
            clshow="!hidden",
            inprop="url",
            titles="|".join(titles),
        )
        pages = data.get("query", {}).get("pages", [])
        out = []
        for p in pages:
            if p.get("missing") or not p.get("revisions"):
                continue
            rev = p["revisions"][0]
            content = rev.get("slots", {}).get("main", {}).get("content")
            if not content:
                continue
            out.append(
                {
                    "pageid": p["pageid"],
                    "ns": p["ns"],
                    "title": p["title"],
                    "url": p.get("fullurl") or self.s.wiki_view_base + p["title"].replace(" ", "_"),
                    "revid": rev.get("revid"),
                    "timestamp": rev.get("timestamp"),
                    "categories": [c["title"] for c in p.get("categories", [])],
                    "wikitext": content,
                }
            )
        return out

    async def redirect_map(self, namespace: int = 0) -> dict[str, str]:
        """`{redirect title: target title}` — these are the alternative names
        (transliterations, honorifics, epithets) readers actually type, so they
        make excellent extra search surface for each article."""
        out: dict[str, str] = {}
        # A redirect page contains exactly one wikilink: its target. Asking for
        # `prop=links` over a redirect-only generator therefore yields the whole
        # alias map in a handful of requests.
        async for data in self.paged(
            action="query",
            generator="allpages",
            gapnamespace=namespace,
            gaplimit="max",
            gapfilterredir="redirects",
            prop="links",
            pllimit="max",
            plnamespace=namespace,
        ):
            for page in data.get("query", {}).get("pages", []) or []:
                links = page.get("links") or []
                if len(links) == 1:
                    out[page["title"]] = links[0]["title"]
        return out
