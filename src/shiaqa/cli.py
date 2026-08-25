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
kg_app = typer.Typer(add_completion=False, help="Structural knowledge graph over the corpus.")
app.add_typer(kg_app, name="kg")
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

    if s.kg_db_path.exists():
        from .kg.store import KgStore
        kg = KgStore(s)
        kc = kg.counts()
        built = kg.get_meta("built_at") or "-"
        table.add_row("kg nodes / edges", f"{kc['nodes']} / {kc['edges']} "
                                          f"({kc['typed_edges']} typed)")
        table.add_row("kg links", str(kc["links"]))
        table.add_row("kg built at", built)
        # The graph is derived from the raw cache; if that moved on, say so
        # rather than letting a stale graph look current.
        newest_raw = max((p.stat().st_mtime for p in s.raw_dir.glob("*.jsonl")), default=0)
        if newest_raw and kg.path.stat().st_mtime < newest_raw:
            table.add_row("kg freshness", "[yellow]stale — raw cache is newer[/yellow]")
        kg.close()
    else:
        table.add_row("kg", "not built (`shiaqa kg build`)")
    table.add_row("kg in retrieval",
                  f"weight_graph={s.weight_graph} hops={s.kg_max_hops} "
                  f"chains={s.kg_chains_in_context} "
                  f"facts={s.kg_facts_in_context}")
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
    if res.kg_chains:
        console.print("\n[bold]reasoning chain[/]")
        for line, path in zip(res.kg_chains, res.kg_paths):
            console.print(f"  → {line}")
            if path is not None:
                trail = " -> ".join(
                    f"{st.p}{'' if st.direction == 'out' else ' (rev)'}"
                    for st in path.steps)
                console.print(f"    [dim]via {trail}[/]")
        console.print()
    if res.kg_facts:
        console.print("[bold]graph facts[/]")
        for f in res.kg_facts:
            console.print(f"  • {f}")
        console.print()
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
    decompose: bool = typer.Option(
        False, "--decompose",
        help="Split the question first (one extra model call; slower)."),
) -> None:
    """Ask a question and stream a cited answer."""
    from .rag.answer import AnswerEngine

    async def run() -> None:
        engine = AnswerEngine()
        try:
            if json_out:
                ans = await engine.answer(question, profile, top_k, decompose)
                console.print_json(jsonlib.dumps(ans.to_dict(), ensure_ascii=False))
                return

            parts: list[str] = []
            sources: list[dict] = []
            with console.status("splitting the question…" if decompose
                                else "retrieving…"):
                gen = engine.stream(question, profile, top_k, decompose)
                first = await gen.__anext__()
            for sub in first.get("sub_questions") or []:
                console.print(f"[dim]· {sub}[/]")
            for chain in first.get("chains") or []:
                console.print(f"[dim]→ {chain}[/]")
            if first.get("sub_questions") or first.get("chains"):
                console.print()
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


# --- knowledge graph -------------------------------------------------------


@kg_app.command("build")
def kg_build(
    jsonl_only: bool = typer.Option(False, "--jsonl-only",
                                    help="Write data/kg/*.jsonl but skip data/kg.db."),
) -> None:
    """Extract the graph from the raw cache into data/kg/ and data/kg.db.

    Reads data/raw/ only — it never opens shiaqa.db, so it cannot disturb the
    index or the embeddings. A full rebuild takes about 15 seconds.
    """
    from .kg.build import build

    rep = build(get_settings(), jsonl_only=jsonl_only)
    table = Table(show_header=False, box=None)
    for k, v in rep.extract.as_dict().items():
        if k == "seconds":
            continue
        table.add_row(k.replace("_", " "), str(v))
    for name, n in rep.files.items():
        table.add_row(f"wrote {name}", str(n))
    if rep.db_counts:
        table.add_row("kg.db", ", ".join(f"{k}={v}" for k, v in rep.db_counts.items()))
    table.add_row("elapsed", f"{rep.seconds:.1f}s")
    console.print(table)

    # A correction that has quietly stopped matching is worse than none at all:
    # the graph looks curated and is not. Never let this scroll past silently.
    if rep.overrides.needs_attention:
        console.print("\n[yellow]Overrides needing attention:[/yellow]")
        for line in rep.overrides.stale + rep.overrides.unresolved:
            console.print(f"  [yellow]![/yellow] {line}")


@kg_app.command("stats")
def kg_stats(
    top: int = typer.Option(12, "--top", help="How many rows per breakdown."),
) -> None:
    """Counts by class and predicate, plus the unmapped-field worklist."""
    from .kg.build import UNMAPPED, read_jsonl
    from .kg.store import KgStore

    s = get_settings()
    if not s.kg_db_path.exists():
        console.print("[red]No graph yet. Run `shiaqa kg build`.[/red]")
        raise typer.Exit(1)

    kg = KgStore(s)
    counts = kg.counts()
    console.print(f"[bold]{counts['nodes']}[/bold] nodes, "
                  f"[bold]{counts['edges']}[/bold] edges "
                  f"([bold]{counts['typed_edges']}[/bold] typed), "
                  f"[bold]{counts['links']}[/bold] links\n")

    table = Table(title="entities by class", box=None)
    table.add_column("class"); table.add_column("n", justify="right")
    for r in kg.db.execute(
        "SELECT class, count(*) c FROM node_classes GROUP BY class ORDER BY c DESC LIMIT ?",
        (top,),
    ):
        table.add_row(r["class"], str(r["c"]))
    console.print(table)

    table = Table(title="typed edges by predicate", box=None)
    table.add_column("predicate"); table.add_column("n", justify="right")
    for r in kg.db.execute(
        "SELECT p, count(*) c FROM edges WHERE p NOT IN ('instance_of','subclass_of') "
        "GROUP BY p ORDER BY c DESC LIMIT ?", (top,),
    ):
        table.add_row(r["p"], str(r["c"]))
    console.print(table)

    worklist = list(read_jsonl(s.kg_dir / UNMAPPED))[:top]
    if worklist:
        table = Table(title="unmapped infobox fields (curation worklist)", box=None)
        table.add_column("template"); table.add_column("field")
        table.add_column("links", justify="right"); table.add_column("pages", justify="right")
        for u in worklist:
            table.add_row(u["template"], u["field"],
                          str(u["resolved_links"]), str(u["pages"]))
        console.print(table)
    kg.close()


@kg_app.command("show")
def kg_show(
    title: str,
    limit: int = typer.Option(25, "--limit", "-n"),
) -> None:
    """Print everything the graph records about one article."""
    from .kg.ontology import Ontology
    from .kg.store import KgStore

    s = get_settings()
    if not s.kg_db_path.exists():
        console.print("[red]No graph yet. Run `shiaqa kg build`.[/red]")
        raise typer.Exit(1)

    kg = KgStore(s)
    matches = kg.find_by_label(title, 5)
    if not matches:
        console.print(f"[yellow]Nothing in the graph matching {title!r}.[/yellow]")
        raise typer.Exit(1)

    node = matches[0]
    if len(matches) > 1:
        others = ", ".join(m["label"] for m in matches[1:])
        console.print(f"[dim]also matched: {others}[/dim]")

    classes = kg.classes_of(node["id"])
    console.print(f"\n[bold]{node['label']}[/bold]"
                  f"{'  (' + ', '.join(classes) + ')' if classes else ''}")
    console.print(f"[dim]{node['url'] or ''}[/dim]\n")

    for line in kg.facts(node["id"], Ontology().predicates, limit):
        console.print(f"  {line}")

    cats = [r["label"] for r in kg.db.execute(
        "SELECT n.label FROM edges e JOIN nodes n ON n.id = e.o "
        "WHERE e.s = ? AND e.p = 'instance_of' ORDER BY n.label", (node["id"],))]
    if cats:
        console.print(f"\n[dim]categories: {', '.join(cats)}[/dim]")
    kg.close()


@kg_app.command("viz")
def kg_viz(
    out: str = typer.Option(None, "--out", "-o",
                            help="Directory for the HTML (default: <data_dir>/kg/viz)."),
    kind: str = typer.Option("both", "--kind",
                             help="explorer | schema | both"),
) -> None:
    """Write self-contained HTML views of the graph.

    Each file embeds the data it needs, so it opens straight from disk with no
    server and no network — which is also what lets it be published as an
    Artifact, where external hosts are blocked outright.
    """
    from pathlib import Path

    from .kg import viz

    s = get_settings()
    if not s.kg_db_path.exists():
        console.print("[red]No graph yet. Run `shiaqa kg build`.[/red]")
        raise typer.Exit(1)

    kinds = ("explorer", "schema") if kind == "both" else (kind,)
    unknown = [k for k in kinds if k not in ("explorer", "schema")]
    if unknown:
        console.print(f"[red]Unknown --kind {unknown[0]!r}: use explorer, schema or both.[/red]")
        raise typer.Exit(1)

    written = viz.write(Path(out) if out else s.kg_dir / "viz", kinds, s)
    table = Table(show_header=False, box=None)
    for name, path in written.items():
        table.add_row(name, f"{path}  ({path.stat().st_size / 1024:.0f} KB)")
    console.print(table)


def _kg_open():
    """Open the graph, or exit with the same message every kg command uses."""
    from .kg.store import KgStore

    s = get_settings()
    if not s.kg_db_path.exists():
        console.print("[red]No graph yet. Run `shiaqa kg build`.[/red]")
        raise typer.Exit(1)
    return KgStore(s), s


def _kg_resolve(kg, title: str):
    """One entity node from a title, or exit."""
    matches = kg.find_by_label(title, 5)
    if not matches:
        console.print(f"[yellow]Nothing in the graph matching {title!r}.[/yellow]")
        raise typer.Exit(1)
    if len(matches) > 1:
        others = ", ".join(m["label"] for m in matches[1:])
        console.print(f"[dim]{title!r} -> {matches[0]['label']}; "
                      f"also matched: {others}[/dim]")
    return matches[0]


@kg_app.command("chain")
def kg_chain(
    title: str,
    hops: int = typer.Option(2, "--hops", "-h", help="How many typed hops to walk."),
    limit: int = typer.Option(12, "--limit", "-n", help="How many chains to print."),
    deep_only: bool = typer.Option(False, "--deep-only",
                                   help="Only paths of two or more hops."),
) -> None:
    """Walk outward from one article and narrate what the graph connects it to."""
    from .kg import paths
    from .kg.ontology import Ontology

    kg, s = _kg_open()
    node = _kg_resolve(kg, title)
    preds = Ontology().predicates
    found = paths.expand(kg, [node["id"]], preds, max_hops=hops,
                         beam=s.kg_beam_width, limit=max(limit * 4, limit),
                         hub_degree_max=s.kg_hub_degree_max)
    if deep_only:
        found = [p for p in found if p.hops >= 2]
    if not found:
        console.print("[yellow]No typed relations to walk from here.[/yellow]")
        raise typer.Exit(1)

    persons = paths.person_nodes(kg, [n for p in found for n in p.nodes])
    console.print(f"\n[bold]{node['label']}[/bold] — {hops} hop(s)\n")
    for p in found[:limit]:
        console.print(f"  [dim]{p.score:.3f}  h{p.hops}[/dim]  "
                      f"{paths.narrate(p, preds, persons)}")
        trail = " -> ".join(f"{st.p}{'' if st.direction == 'out' else ' (rev)'}"
                            for st in p.steps)
        fields = ", ".join(sorted({st.field for st in p.steps if st.field}))
        console.print(f"          [dim]via {trail}"
                      f"{'  |  fields: ' + fields if fields else ''}[/dim]")
    kg.close()


@kg_app.command("path")
def kg_path(
    start: str,
    end: str,
    hops: int = typer.Option(None, "--hops", "-h",
                             help="Longest route to consider."),
    limit: int = typer.Option(3, "--limit", "-n"),
) -> None:
    """Find how the graph connects two articles."""
    from .kg import paths
    from .kg.ontology import Ontology

    kg, s = _kg_open()
    a, b = _kg_resolve(kg, start), _kg_resolve(kg, end)
    preds = Ontology().predicates
    found = paths.connect(kg, a["id"], b["id"], preds,
                          max_hops=hops or s.kg_connect_max_hops,
                          beam=s.kg_beam_width, limit=limit,
                          hub_degree_max=s.kg_hub_degree_max)
    console.print(f"\n[bold]{a['label']}[/bold] -> [bold]{b['label']}[/bold]\n")
    if not found:
        console.print("[yellow]No typed route within that many hops.[/yellow]")
        console.print("[dim]Routes through places and other hubs are refused "
                      "on purpose: a shared birthplace connects hundreds of "
                      "unrelated people.[/dim]")
        kg.close()
        raise typer.Exit(1)

    persons = paths.person_nodes(kg, [n for p in found for n in p.nodes])
    for p in found:
        console.print(f"  [dim]{p.score:.3f}  h{p.hops}[/dim]  "
                      f"{paths.narrate(p, preds, persons)}")
        trail = " -> ".join(f"{st.p}{'' if st.direction == 'out' else ' (rev)'}"
                            for st in p.steps)
        console.print(f"          [dim]via {trail}[/dim]")
    kg.close()


@kg_app.command("compare")
def kg_compare(
    first: str,
    second: str,
    limit: int = typer.Option(8, "--limit", "-n"),
) -> None:
    """Line up what the graph records about two articles, predicate by predicate."""
    from .kg import paths
    from .kg.ontology import Ontology

    kg, _ = _kg_open()
    a, b = _kg_resolve(kg, first), _kg_resolve(kg, second)
    cmp = paths.compare(kg, a["id"], b["id"], Ontology().predicates, limit=limit)
    if cmp is None:
        console.print("[yellow]Nothing the graph records of both.[/yellow]")
        kg.close()
        raise typer.Exit(1)

    console.print(f"\n[bold]{cmp.a_label}[/bold] vs [bold]{cmp.b_label}[/bold]\n")
    for row in cmp.rows:
        console.print(f"  [bold]{row.label}[/bold]")
        console.print(f"    {cmp.a_label}: {', '.join(row.a_objects[:5]) or '—'}")
        console.print(f"    {cmp.b_label}: {', '.join(row.b_objects[:5]) or '—'}")
        if row.shared:
            console.print(f"    [green]both:[/green] {', '.join(row.shared[:5])}")
    kg.close()
