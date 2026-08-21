"""Command line interface: `shiaqa <command>`."""

from __future__ import annotations

import asyncio
import json as jsonlib

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .config import get_settings
from .log import get_logger

app = typer.Typer(add_completion=False, help="Local RAG question answering over WikiShia.")
console = Console()
log = get_logger(__name__)


@app.command()
def fetch(
    force: bool = typer.Option(False, "--force", help="Refetch even if the cache exists."),
    namespace: list[int] = typer.Option(None, "--ns", help="Namespaces to fetch (default: config)."),
) -> None:
    """Download WikiShia into the local raw cache (~50 MB, a few minutes)."""
    from .wiki.fetch import fetch_all

    counts = asyncio.run(fetch_all(namespaces=list(namespace) or None, force=force))
    for ns, n in counts.items():
        console.print(f"namespace {ns}: [bold]{n}[/] pages")


@app.command()
def index(
    text_only: bool = typer.Option(False, "--text-only", help="Build text + BM25, skip embeddings."),
    embed_only: bool = typer.Option(False, "--embed-only", help="Resume embeddings only."),
    limit: int = typer.Option(None, "--limit", help="Embed at most N chunks (for testing)."),
) -> None:
    """Build the search index. Embedding is resumable — rerun with --embed-only."""
    from .ingest.pipeline import build_text_index, build_vectors

    if not embed_only:
        rep = build_text_index()
        console.print(f"text index: [bold]{rep.pages}[/] pages, [bold]{rep.chunks}[/] chunks")
    if not text_only:
        rep = build_vectors(limit=limit)
        console.print(f"embedded [bold]{rep.embedded}[/] chunks in {rep.seconds / 60:.1f} min")


@app.command()
def prune() -> None:
    """Drop navigation pages and orphaned vectors from an existing index."""
    from .ingest.pipeline import prune as run_prune

    res = run_prune()
    console.print(f"removed [bold]{res['pages']}[/] pages, [bold]{res['chunks']}[/] chunks, "
                  f"[bold]{res['orphan_vectors']}[/] orphan vectors")


@app.command()
def status() -> None:
    """Show index and model status."""
    from .embed.ollama_embed import Embedder
    from .store.sqlite_store import Store
    from .wiki.fetch import raw_counts

    s = get_settings()
    store = Store(s)
    counts = store.counts()
    ok, msg = Embedder(s).health()

    table = Table(show_header=False, box=None)
    for ns, n in raw_counts(s).items():
        table.add_row(f"raw cache ns{ns}", str(n))
    table.add_row("pages indexed", str(counts["pages"]))
    table.add_row("chunks", str(counts["chunks"]))
    pct = 100 * counts["vec_chunks"] / counts["chunks"] if counts["chunks"] else 0
    table.add_row("vectors", f"{counts['vec_chunks']} ({pct:.1f}%)")
    table.add_row("built at", store.get_meta("built_at") or "-")
    table.add_row("ollama", msg)
    table.add_row("db", f"{s.db_path} ({s.db_path.stat().st_size / 1e6:.0f} MB)"
                  if s.db_path.exists() else "missing")
    console.print(table)


@app.command()
def search(
    question: str,
    top_k: int = typer.Option(6, "--top-k", "-k"),
    full: bool = typer.Option(False, "--full", help="Print the full chunk text."),
) -> None:
    """Retrieve passages without generating an answer (fast)."""
    from .rag.retrieve import Retriever

    res = Retriever().search_sync(question, top_k)
    if not res.hits:
        console.print("[yellow]no matches[/]")
        raise typer.Exit(1)
    for i, h in enumerate(res.hits, 1):
        console.print(f"[bold cyan][{i}][/] {h.label}  "
                      f"[dim](score {h.score:.4f}, vec {h.vec_rank}, bm25 {h.bm25_rank})[/]")
        console.print(f"    [dim]{h.url}[/]")
        console.print(f"    {h.text if full else h.text[:220] + '…'}\n")
    console.print(f"[dim]context tokens: {res.context_tokens}[/]")


@app.command()
def ask(
    question: str,
    profile: str = typer.Option(None, "--profile", "-p", help="fast | quality | <ollama tag>"),
    top_k: int = typer.Option(None, "--top-k", "-k"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Ask a question and stream a cited answer."""
    from .rag.answer import AnswerEngine

    async def run() -> None:
        engine = AnswerEngine()
        try:
            if json_out:
                ans = await engine.answer(question, profile, top_k)
                console.print_json(jsonlib.dumps(ans.to_dict(), ensure_ascii=False))
                return

            parts: list[str] = []
            sources: list[dict] = []
            with console.status("retrieving…"):
                gen = engine.stream(question, profile, top_k)
                first = await gen.__anext__()
            sources = first.get("sources", [])
            for src in sources:
                console.print(f"[dim][{src['n']}] {src['title']} — {src['section']}[/]")
            console.print()

            async for ev in gen:
                if ev["type"] == "delta":
                    parts.append(ev["text"])
                    console.print(ev["text"], end="", markup=False, highlight=False)
                elif ev["type"] == "done":
                    secs = ev.get("generation_seconds", 0)
                    tps = ev.get("output_tokens", 0) / secs if secs else 0
                    console.print(f"\n\n[dim]{ev['model']} · {ev.get('output_tokens', 0)} tokens "
                                  f"· {secs}s · {tps:.1f} tok/s[/]")
            for src in sources:
                console.print(f"[dim][{src['n']}] {src['url']}[/]")
        finally:
            await engine.aclose()

    asyncio.run(run())


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the HTTP service."""
    import uvicorn

    uvicorn.run("shiaqa.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
