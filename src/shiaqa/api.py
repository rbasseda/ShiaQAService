"""FastAPI surface for the QA service."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .log import get_logger
from .rag.answer import AnswerEngine

log = get_logger(__name__)
_engine: AnswerEngine | None = None


def engine() -> AnswerEngine:
    if _engine is None:
        raise HTTPException(503, "service still starting")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    s = get_settings()
    if not s.db_path.exists():
        log.warning("no index at %s — run `shiaqa fetch && shiaqa index`", s.db_path)
    _engine = AnswerEngine(s)
    log.info("index: %s", _engine.retriever.store.counts())
    yield
    await _engine.aclose()
    _engine = None


app = FastAPI(
    title="ShiaQAService",
    description="Retrieval-augmented question answering over WikiShia, running locally on Ollama.",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    profile: str | None = Field(
        default=None,
        description="'fast' (llama3.2:3b), 'quality' (qwen2.5:7b), or an explicit Ollama tag.",
    )
    top_k: int | None = Field(default=None, ge=1, le=20)
    decompose: bool | None = Field(
        default=None,
        description="Split the question into sub-questions first. Costs one "
                    "extra model call, so noticeably slower on CPU.",
    )


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    eng = engine()
    ok, msg = eng.retriever.embedder.health()
    counts = eng.retriever.store.counts()
    complete = counts["vec_chunks"] >= counts["chunks"] > 0
    return {
        # "degraded" while embeddings are still being built: BM25 and title
        # matching already work, but the semantic channel is incomplete.
        "status": "ok" if ok and complete else "degraded",
        "ollama": msg,
        "index": counts,
        "vectors_complete": complete,
        "vectors_progress": round(100 * counts["vec_chunks"] / counts["chunks"], 1) if counts["chunks"] else 0.0,
        "models": {"embed": s.embed_model, "fast": s.gen_model_fast, "quality": s.gen_model_quality},
    }


@app.post("/search")
async def search(req: AskRequest) -> dict:
    """Retrieval only — no generation. Fast, and useful for tuning."""
    res, secs = await engine().retrieve(req.question, req.top_k, req.decompose)
    return {
        "question": req.question,
        "retrieval_seconds": round(secs, 3),
        "context_tokens": res.context_tokens,
        "chains": res.kg_chains,
        "kg_facts": res.kg_facts,
        "sub_questions": res.sub_questions,
        "hits": [
            {"n": i, "title": h.title, "section": h.section, "url": h.url,
             "kind": h.kind, "score": round(h.score, 5),
             "vec_rank": h.vec_rank, "bm25_rank": h.bm25_rank, "text": h.text}
            for i, h in enumerate(res.hits, 1)
        ],
    }


@app.post("/ask")
async def ask(req: AskRequest) -> dict:
    return (await engine().answer(req.question, req.profile, req.top_k,
                                  req.decompose)).to_dict()


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    eng = engine()

    async def events():
        async for event in eng.stream(req.question, req.profile, req.top_k,
                                      req.decompose):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
