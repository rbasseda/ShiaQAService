# ShiaQAService

Retrieval-augmented question answering over [WikiShia](https://en.wikishia.net), running
entirely on your own machine via [Ollama](https://ollama.com). No API keys, no data leaves
the laptop.

Answers are grounded in retrieved encyclopaedia passages and carry inline citations `[1]`,
`[2]` back to the source articles.

---

## What it does

```
MediaWiki API ──▶ raw cache ──▶ clean ──▶ chunk ──▶ SQLite ──▶ hybrid retrieval ──▶ Ollama ──▶ cited answer
   (~3 min)        47 MB       wikitext   section-    text + BM25    BM25 + vectors      llama3.2 / qwen2.5
                                          aware        + vectors     + title match
```

| Stage | Detail |
| --- | --- |
| **Source** | MediaWiki API at `/w/api.php`, batched 50 titles/request — no HTML scraping |
| **Corpus** | namespaces `0` (articles), `3000` (`Text:` primary sources), `14` (categories) |
| **Scale** | ~6,900 pages → 5,154 indexed pages → **39,952 chunks**, all embedded |
| **Storage** | one SQLite file: rows + FTS5 BM25 index + `sqlite-vec` vectors |
| **Retrieval** | three channels fused with weighted Reciprocal Rank Fusion |
| **Generation** | Ollama, switchable per request between a fast and a higher-quality model |

## Why it is built this way

This runs on an **Intel** MacBook Pro (i7-9750H, 16 GB). Ollama has no GPU path on Intel
Macs — Metal is Apple-Silicon only — so **everything is CPU-bound**, and that shaped
the design:

- **SQLite over a vector database.** 40k vectors do not need a server. One file, no daemon,
  and the lexical and semantic indexes update in the same transaction so they cannot drift.
- **Embedding is resumable.** Building the vectors takes ~3 hours at ~4 chunks/s. Interrupt
  it whenever; `shiaqa index --embed-only` continues from the first chunk without a vector.
- **Sources stream before the answer.** Generation runs at 5–12 tok/s, so the API emits the
  retrieved sources as its first SSE event — you see where the answer will come from within
  a second, while the model is still working.
- **Tight context budget** (~2,600 tokens). On CPU, prompt length is the dominant latency
  cost, so retrieval aims for precision over recall.

### Retrieval: three channels, not one

Questions here turn on proper names with many spellings — *Husayn / Hussain / al-Ḥusayn*,
*Ghadir Khumm*, *Nahj al-balagha*. Pure vector search is fuzzy about exactly the tokens
that matter most, so three channels are fused:

1. **BM25** over FTS5, tokenised with `remove_diacritics 2` — a query for `Husayn` reaches
   `Ḥusayn`, `Ali` reaches `ʿAlī`.
2. **Dense vectors** (`nomic-embed-text`, 768-d, cosine) for "why"/"how" questions where
   wording does not overlap.
3. **Title & alias matching.** The wiki's **35,224 redirects** are a gift: almost every
   epithet or spelling a reader might type already points at the right article. Candidates
   are re-scored on *coverage* — how much of the article's own name the question accounts
   for — because plain BM25 lets a rare word matching one alias of an unrelated page
   outrank the article being asked about. Coverage ties are broken by redirect count, a
   good prominence prior: for "the mother of Imam Husayn", the third Imam (102 redirects)
   and an unrelated namesake (28) cover the title equally well, and the famous one wins.

The title channel uses a **smaller RRF constant** (`rrf_k_title=20`) than the other two.
It is short and precision-ordered, so rank within it means much more than rank within a
30-deep noisy list. At the usual flat `k=60`, a fact box sitting at title-rank 1 scored
below any chunk appearing mid-list in two channels — which is exactly how "who was the
mother of Imam Husayn?" retrieved the right *article* but not the chunk holding the answer.

Each chunk is embedded with a context header (`Article — Section > Subsection`). Passages on
this wiki constantly say "he was martyred at Karbala" without naming *whose* biography it is;
the header restores that.

## Install

Requires Python ≥ 3.11 and a running Ollama.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

```bash
ollama pull nomic-embed-text && ollama pull llama3.2:3b && ollama pull qwen2.5:7b-instruct-q4_K_M
```

## Build the index

```bash
.venv/bin/shiaqa fetch
```

```bash
.venv/bin/shiaqa index
```

`fetch` takes ~3 minutes and caches raw wikitext under `data/raw/`; re-running is free.
`index` builds the text and BM25 index in ~20 seconds, then embeds all 40k chunks — **about
3 hours on CPU**. It is safe to interrupt and resume:

```bash
.venv/bin/shiaqa index --embed-only
```

BM25 and title matching work as soon as the text index exists, so the service is usable
(with a weaker semantic channel) while embeddings build. Check progress any time:

```bash
.venv/bin/shiaqa status
```

## Ask

```bash
.venv/bin/shiaqa ask "What happened at Ghadir Khumm?"
```

```bash
.venv/bin/shiaqa ask "Why is Ashura commemorated?" --profile quality
```

Retrieval only, no generation — instant, and the fastest way to judge index quality:

```bash
.venv/bin/shiaqa search "Who compiled al-Kafi?" --full
```

## Serve

```bash
.venv/bin/shiaqa serve --port 8000
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Ollama reachability, index counts, embedding progress |
| `POST /search` | retrieval only — hits, scores, per-channel ranks |
| `POST /ask` | complete cited answer as JSON |
| `POST /ask/stream` | SSE: `sources` event first, then `delta` tokens, then `done` |

```bash
curl -N -X POST localhost:8000/ask/stream -H 'Content-Type: application/json' \
  -d '{"question":"What is taqiyya?","profile":"fast"}'
```

`profile` selects the generator per request: `fast` (llama3.2:3b), `quality`
(qwen2.5:7b-instruct-q4_K_M), or any Ollama tag you have pulled.

Interactive docs are at `/docs`.

## Configuration

Everything tunable lives in `src/shiaqa/config.py` and can be overridden by environment
variables with a `SHIAQA_` prefix, or a `.env` file:

```bash
SHIAQA_TOP_K_FINAL=8
SHIAQA_CONTEXT_TOKEN_BUDGET=3200
SHIAQA_GEN_DEFAULT_PROFILE=quality
```

The knobs that matter most: `top_k_final` and `context_token_budget` trade answer quality
against CPU latency; `weight_vector` / `weight_bm25` / `weight_title` rebalance the three
retrieval channels.

## Maintenance

```bash
.venv/bin/shiaqa prune
```

Removes disambiguation pages and orphaned vectors from an existing index **without**
discarding embeddings. To pick up wiki edits, `shiaqa fetch --force` then `shiaqa index`
(a full re-embed).

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

171 tests, all offline — no Ollama, no network.

## Evaluation

`eval/questions.yaml` holds 15 hand-checked questions. Each names the article(s) that should
be retrieved, and five also assert a **string that must appear in the assembled context** —
the metric that actually matters, since retrieving the right article is worthless if the
chunk holding the fact never reaches the prompt.

```bash
.venv/bin/python eval/run_eval.py --k 6
```

Current: **recall@6 = 1.00, MRR = 0.933, answer-in-context = 1.00**, median retrieval
latency **109 ms**.

Two further sets exercise the graph. `eval/questions_kg.yaml` holds 10 questions that turn on
a *relation* rather than a passage; `eval/questions_multihop.yaml` holds 12 that need two
typed hops, where `expect` names only the far article — the one a single hop cannot reach.

```bash
.venv/bin/python eval/run_eval.py --k 6 --kg      # graph channel off vs on
.venv/bin/python eval/run_eval.py --k 6 --hops    # one hop vs two
```

`--hops` reports answer-in-context twice, once over the whole prompt and once over the
retrieved passages alone. The second number is the honest one: a chain block that states the
composed relation satisfies the first by construction, so without the split a large and
purely definitional win would be reported.

To re-tune the fusion after changing the corpus or models:

```bash
.venv/bin/python eval/run_eval.py --k 6 --sweep
```

Retrieval-only, so a full grid runs in seconds. Treat its output with suspicion: 15
questions is enough to catch a structural bug, not enough to justify finely-tuned weights.
The current settings change exactly one parameter from the defaults, chosen because it was
the only one whose effect held across the whole grid.

## Measured performance

On the Intel i7-9750H, nothing else competing:

| | retrieval | generation | total |
| --- | --- | --- | --- |
| `--profile fast` (llama3.2:3b) | 0.11 s | 14.2 s @ 9.7 tok/s | **~14 s** |
| `--profile quality` (qwen2.5:7b-q4) | 0.85 s | 25.9 s @ 4.9 tok/s | **~27 s** |

Indexing was a one-off 12.2 h wall clock (~4 chunks/s of actual work, plus overnight
sleep) for 39,952 embeddings. The database is 182 MB.

## Notes and limitations

- **Attribution.** WikiShia is a confessional encyclopaedia written from a Twelver Shia
  perspective, and it is CC-licensed content maintained by volunteers. The system prompt
  instructs the model to attribute doctrinal claims ("According to WikiShia…") rather than
  state them as neutral fact. Keep that framing in anything user-facing, and keep the
  citation links visible.
- **Politeness.** The fetcher identifies itself with a contact address, serialises requests
  with a delay, honours `maxlag`, and caches everything so a rebuild needs no network.
  `robots.txt` disallows `/w/` for generic crawlers (stock MediaWiki config — Wikipedia
  ships the same); this uses the sanctioned API rather than scraping article HTML, but if
  you deploy this beyond personal use, check the wiki's terms first.
- **The model is the weak link, not retrieval.** With retrieval at 1.00 answer-in-context on
  the eval set, observed wrong answers are now generation failures: the 3B model has been
  seen citing excerpt [1] about a *different* person of the same name while the correct
  excerpt sat at [2]. Use `--profile quality` when the answer matters, and `shiaqa search`
  to tell a retrieval failure from a generation failure.
- **Distinguishing same-named figures is the sharpest remaining edge.** This corpus is full
  of them (several Husayns, several Fatimas). Prominence tie-breaking handles the common
  case; an explicitly disambiguating question ("Husayn son of al-Kazim") is handled by the
  lexical channel.
- **Multi-hop is structural, not semantic.** Two typed hops of the knowledge graph now reach
  articles a single hop cannot, and the composed relation is narrated into the prompt
  ("al-Sharif al-Radi was taught by al-Shaykh al-Mufid, who was taught by al-Shaykh
  al-Saduq"). What is still missing is any sense of *which* relation a question asked about:
  for "who taught the teacher of al-Sharif al-Radi" the graph offers al-Saduq and al-Tusi at
  identical scores, and nothing prefers the one the question means. So the composed fact
  usually reaches the prompt, while the far article itself only sometimes wins a top-6 slot.
- **Question decomposition costs model calls, so it is opt-in.** `shiaqa ask --decompose`
  splits a question, retrieves for each part, and merges by rank — useful for genuine
  cross-article comparisons that no typed edge relates. It adds roughly five seconds to the
  retrieval phase; total time stays dominated by CPU generation.
- Images, tables and infobox media are dropped during cleaning; the text pipeline is
  text-only by design.

## License

Source code is licensed under the [Apache License 2.0](LICENSE).

**The code license does not cover WikiShia's content.** This project downloads
encyclopaedia text written by WikiShia's contributors, which is governed by that wiki's own
terms. No such content is distributed in this repository — the corpus and index live under
`data/`, which is gitignored, and `shiaqa fetch` downloads it to your machine. WikiShia does
not expose a machine-readable license via the MediaWiki `rightsinfo` API, so check the terms
published on the wiki before redistributing its text or deploying this publicly. See
[NOTICE](NOTICE).
