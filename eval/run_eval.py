"""Grid-search retrieval settings against eval/questions.yaml.

Retrieval only — no generation — so a full sweep runs in seconds. Reports
recall@k (did any expected article appear) and MRR (how high it ranked).
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shiaqa.config import Settings, get_settings          # noqa: E402
from shiaqa.embed.ollama_embed import Embedder            # noqa: E402
from shiaqa.rag.retrieve import Retriever                 # noqa: E402
from shiaqa.store.sqlite_store import Store               # noqa: E402


def load_cases(path: Path) -> list[dict]:
    """Minimal parser for this file's fixed shape (avoids a PyYAML dependency)."""
    cases, cur = [], None
    for line in path.read_text().splitlines():
        line = line.split("#")[0].rstrip()
        if not line.strip():
            continue
        if m := re.match(r"^-\s*q:\s*\"(.*)\"\s*$", line):
            cur = {"q": m.group(1), "expect": []}
            cases.append(cur)
        elif m := re.match(r"^\s+expect:\s*\[(.*)\]\s*$", line):
            cur["expect"] = [t.strip().strip('"') for t in re.findall(r'"[^"]*"', m.group(1))]
        elif m := re.match(r'^\s+answer:\s*"(.*)"\s*$', line):
            cur["answer"] = m.group(1)
    return cases


# Blocks the retriever prepends to the context that are not retrieved passages.
KG_BLOCK_TAGS = ("[KG]", "[KG-PATH]")


def excerpts_only(context: str) -> str:
    """The context minus the graph blocks.

    Without this, answer-in-context stops measuring anything once chains are
    enabled: a chain block that states "al-Radi was taught by al-Mufid, who was
    taught by al-Saduq" trivially satisfies a search for "al-Saduq" in the
    context. That is not cheating — putting the composed relation in the prompt
    is the whole feature — but it cannot be the *only* number, or a large win
    would be reported that is purely definitional. Scoring both, and reading the
    gap, separates "we retrieved the right passage" from "we asserted the fact".
    """
    return "\n\n".join(b for b in context.split("\n\n")
                        if not b.startswith(KG_BLOCK_TAGS))


def score(retriever: Retriever, cases: list[dict], k: int) -> dict:
    hits = 0
    rr = 0.0
    answerable = 0
    in_excerpts = 0
    n_answer = 0
    tokens: list[int] = []
    misses: list[str] = []
    unanswerable: list[str] = []
    for case in cases:
        res = retriever.search_sync(case["q"], k)
        tokens.append(res.context_tokens)
        titles = [h.title for h in res.hits]
        rank = next((i for i, t in enumerate(titles, 1) if t in case["expect"]), None)
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            misses.append(case["q"])
        if "answer" in case:
            n_answer += 1
            needle = case["answer"].lower()
            if needle in res.context.lower():
                answerable += 1
            else:
                unanswerable.append(f"{case['q']}  (expected \"{case['answer']}\" in context)")
            if needle in excerpts_only(res.context).lower():
                in_excerpts += 1
    n = len(cases)
    return {"recall": hits / n, "mrr": rr / n,
            "answerable": answerable / n_answer if n_answer else 0.0,
            "answerable_excerpts": in_excerpts / n_answer if n_answer else 0.0,
            "median_tokens": statistics.median(tokens) if tokens else 0,
            "misses": misses, "unanswerable": unanswerable}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--sweep", action="store_true", help="Grid-search fusion settings.")
    ap.add_argument("--kg", action="store_true",
                    help="Compare graph channel off vs on, on both question sets.")
    ap.add_argument("--hops", action="store_true",
                    help="Compare kg_max_hops=1 vs 2 across all three sets.")
    ap.add_argument("--weight-graph", type=float, default=None,
                    help="Graph-channel weight to test with --kg "
                         "(default: the configured weight_graph).")
    args = ap.parse_args()

    cases = load_cases(Path(__file__).parent / "questions.yaml")
    store, embedder = Store(), Embedder()

    if args.hops:
        run_hop_comparison(store, embedder, cases, args)
        return

    if args.kg:
        run_kg_comparison(store, embedder, cases, args)
        return

    print(f"{len(cases)} questions, top_k={args.k}\n")

    if not args.sweep:
        r = score(Retriever(get_settings(), store, embedder), cases, args.k)
        print(f"recall@{args.k}={r['recall']:.2f}  MRR={r['mrr']:.3f}  "
              f"answer-in-context={r['answerable']:.2f}")
        for m in r["misses"]:
            print("  MISS   :", m)
        for m in r["unanswerable"]:
            print("  NOANSWER:", m)
        return

    grid = itertools.product(
        [1.0, 1.4, 2.0],        # weight_title
        [60, 20, 10],           # rrf_k_title
        [2, 3, 4],              # title_pages
    )
    results = []
    for w_title, k_title, pages in grid:
        s = Settings(weight_title=w_title, rrf_k_title=k_title, title_pages=pages)
        r = score(Retriever(s, store, embedder), cases, args.k)
        results.append((r, w_title, k_title, pages))
        print(f"w_title={w_title:<4} rrf_k_title={k_title:<3} pages={pages}  "
              f"recall={r['recall']:.2f} MRR={r['mrr']:.3f} answerable={r['answerable']:.2f}")

    print("\nbest by (answer-in-context, recall, MRR):")
    for r, w, kt, p in sorted(results, key=lambda x: (-x[0]["answerable"], -x[0]["recall"], -x[0]["mrr"]))[:5]:
        print(f"  answerable={r['answerable']:.2f} recall={r['recall']:.2f} MRR={r['mrr']:.3f}  "
              f"weight_title={w} rrf_k_title={kt} title_pages={p}")
        for m in r["misses"] + r["unanswerable"]:
            print("      ", m)


def run_hop_comparison(store: Store, embedder: Embedder, cases: list[dict],
                       args) -> None:
    """One hop vs two, across the regression, relational and multi-hop sets.

    Reports answer-in-context twice: over the whole prompt context, and over the
    retrieved passages alone. The first is what the model sees; the second is
    what retrieval actually found. When they diverge, the chain block is carrying
    the answer and the graph channel is unmoved — a real result, but a different
    one, and not something to tune `weight_graph` against.
    """
    kg_cases = load_cases(Path(__file__).parent / "questions_kg.yaml")
    mh_cases = load_cases(Path(__file__).parent / "questions_multihop.yaml")
    kg_db = get_settings().kg_db_path
    if not kg_db.exists():
        print(f"No graph at {kg_db}. Run `shiaqa kg build` first.")
        return

    suites = [("regression (questions.yaml)", cases),
              ("relational (questions_kg.yaml)", kg_cases),
              ("multi-hop (questions_multihop.yaml)", mh_cases)]
    configs = [
        ("1 hop ", Settings(kg_enabled=True, kg_max_hops=1)),
        ("2 hops", Settings(kg_enabled=True, kg_max_hops=2)),
    ]
    for name, suite in suites:
        print(f"\n=== {name} — {len(suite)} questions, top_k={args.k} ===")
        for label, s in configs:
            r = score(Retriever(s, store, embedder), suite, args.k)
            print(f"  {label}  recall={r['recall']:.2f}  MRR={r['mrr']:.3f}  "
                  f"answer-in-context={r['answerable']:.2f}  "
                  f"in-excerpts={r['answerable_excerpts']:.2f}  "
                  f"tokens={r['median_tokens']:.0f}")
            for m in r["misses"]:
                print("        MISS    :", m)
            for m in r["unanswerable"]:
                print("        NOANSWER:", m)


def run_kg_comparison(store: Store, embedder: Embedder, cases: list[dict], args) -> None:
    """Graph channel off vs on, over the regression set and the relational set.

    Two things matter and they are not the same thing. The 15 questions in
    questions.yaml are the gate: the graph must not move them. The relational
    set is the only place a graph win can show up at all, because those
    questions turn on an edge rather than on a passage.
    """
    kg_cases = load_cases(Path(__file__).parent / "questions_kg.yaml")
    kg_db = get_settings().kg_db_path
    if not kg_db.exists():
        print(f"No graph at {kg_db}. Run `shiaqa kg build` first.")
        return

    weight = args.weight_graph if args.weight_graph is not None \
        else get_settings().weight_graph
    suites = [("regression (questions.yaml)", cases),
              ("relational (questions_kg.yaml)", kg_cases)]
    configs = [
        ("graph off (weight=0.0)", Settings(weight_graph=0.0)),
        (f"graph on  (weight={weight})",
         Settings(weight_graph=weight, kg_enabled=True)),
    ]

    for suite_name, suite in suites:
        print(f"\n=== {suite_name} — {len(suite)} questions, top_k={args.k} ===")
        for label, settings in configs:
            r = score(Retriever(settings, store, embedder), suite, args.k)
            print(f"  {label:32s} recall={r['recall']:.2f}  MRR={r['mrr']:.3f}  "
                  f"answer-in-context={r['answerable']:.2f}")
            for m in r["misses"]:
                print("        MISS    :", m)
            for m in r["unanswerable"]:
                print("        NOANSWER:", m)


if __name__ == "__main__":
    main()
