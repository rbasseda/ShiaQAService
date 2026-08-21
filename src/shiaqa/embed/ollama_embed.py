"""Embeddings via a local Ollama server.

Vectors are L2-normalised on the way in, which makes sqlite-vec's Euclidean
ranking identical to cosine ranking — the metric nomic-embed-text is trained for.
"""

from __future__ import annotations

import math
from typing import Sequence

import httpx

from ..config import Settings, get_settings
from ..log import get_logger

log = get_logger(__name__)


def _normalise(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else list(vec)


class Embedder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self._sync = httpx.Client(base_url=self.s.ollama_host, timeout=300.0)
        self._async: httpx.AsyncClient | None = None

    # ---- shared -----------------------------------------------------------

    def _payload(self, texts: Sequence[str]) -> dict:
        return {"model": self.s.embed_model, "input": list(texts)}

    @staticmethod
    def _parse(data: dict) -> list[list[float]]:
        vecs = data.get("embeddings")
        if vecs is None:  # older single-input response shape
            vecs = [data["embedding"]]
        return [_normalise(v) for v in vecs]

    # ---- sync (ingestion) --------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [self.s.embed_doc_prefix + t for t in texts]
        resp = self._sync.post("/api/embed", json=self._payload(prefixed))
        resp.raise_for_status()
        return self._parse(resp.json())

    def embed_query_sync(self, text: str) -> list[float]:
        resp = self._sync.post("/api/embed", json=self._payload([self.s.embed_query_prefix + text]))
        resp.raise_for_status()
        return self._parse(resp.json())[0]

    # ---- async (serving) ---------------------------------------------------

    async def embed_query(self, text: str) -> list[float]:
        if self._async is None:
            self._async = httpx.AsyncClient(base_url=self.s.ollama_host, timeout=120.0)
        resp = await self._async.post(
            "/api/embed", json=self._payload([self.s.embed_query_prefix + text])
        )
        resp.raise_for_status()
        return self._parse(resp.json())[0]

    def close(self) -> None:
        self._sync.close()

    async def aclose(self) -> None:
        self.close()
        if self._async is not None:
            await self._async.aclose()
            self._async = None

    def health(self) -> tuple[bool, str]:
        try:
            tags = self._sync.get("/api/tags", timeout=5.0).json()
        except Exception as exc:  # server down
            return False, f"cannot reach Ollama at {self.s.ollama_host}: {exc}"
        names = {m["name"].split(":")[0] for m in tags.get("models", [])}
        if self.s.embed_model.split(":")[0] not in names:
            return False, f"model {self.s.embed_model} not pulled (`ollama pull {self.s.embed_model}`)"
        return True, "ok"
