"""Central configuration. Every tunable lives here so the CPU-bound knobs
(batch sizes, context budget, model choice) can be changed without touching code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SHIAQA_", env_file=".env", extra="ignore"
    )

    # ---- data source -------------------------------------------------------
    wiki_api: str = "https://en.wikishia.net/w/api.php"
    wiki_view_base: str = "https://en.wikishia.net/view/"
    # Namespaces to ingest: 0 = articles, 3000 = Text (primary sources), 14 = Category.
    namespaces: list[int] = Field(default_factory=lambda: [0, 3000, 14])
    # Identify the client honestly; MediaWiki operators ask for a contact address.
    user_agent: str = "ShiaQAService/0.1 (https://github.com/local/shiaqa; r.basseda@gmail.com)"
    request_delay: float = 0.35  # seconds between API calls, be a polite guest
    maxlag: int = 5  # back off when the wiki's replicas fall behind
    batch_titles: int = 50  # titles per revisions request (API cap for anon users)

    # ---- storage -----------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "shiaqa.db"

    # The knowledge graph lives in its own files on purpose. `Store.reset()`
    # drops every table in `shiaqa.db`, and `shiaqa index` with no flags calls
    # it — a KG kept in there would be destroyed by a routine reindex.
    @property
    def kg_dir(self) -> Path:
        return self.data_dir / "kg"

    @property
    def kg_db_path(self) -> Path:
        return self.data_dir / "kg.db"

    # ---- models ------------------------------------------------------------
    ollama_host: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    # nomic-embed-text is asymmetric: documents and queries need different prefixes.
    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "
    embed_batch: int = 16

    gen_model_fast: str = "llama3.2:3b"
    gen_model_quality: str = "qwen2.5:7b-instruct-q4_K_M"
    gen_default_profile: str = "fast"
    gen_num_ctx: int = 8192
    gen_temperature: float = 0.2
    gen_max_tokens: int = 700

    # ---- chunking ----------------------------------------------------------
    chunk_target_tokens: int = 380
    chunk_max_tokens: int = 620
    chunk_overlap_tokens: int = 60
    chunk_min_tokens: int = 25

    # ---- retrieval ---------------------------------------------------------
    top_k_vector: int = 30
    top_k_bm25: int = 30
    top_k_final: int = 6
    rrf_k: int = 60
    # Relative trust in each retrieval channel when fusing ranks. The title /
    # alias channel is weighted highest: on a wiki, a question that names an
    # article is usually answered by that article.
    weight_vector: float = 1.0
    weight_bm25: float = 1.0
    weight_title: float = 1.4
    # The title channel is short and precision-ordered, so early ranks in it are
    # far more meaningful than early ranks in the two long, noisy channels.
    # A smaller RRF constant sharpens its head without changing the others.
    # Tuned on eval/questions.yaml: at the default 60, a fact box sitting at
    # title-rank 1 scored below any chunk appearing mid-list in two channels,
    # so "who was the mother of Imam Husayn?" retrieved the right article but
    # not the chunk holding the answer. 20 fixes that at every weight tested.
    rrf_k_title: int = 20
    title_pages: int = 4
    title_chunks_per_page: int = 3
    context_token_budget: int = 2600
    # Cap on chunks pulled from any single article, so one long page cannot
    # crowd out every other source in the context window.
    max_chunks_per_page: int = 3

    # ---- knowledge graph --------------------------------------------------
    # `weight_graph` was raised off zero only after `eval/run_eval.py --k 6 --kg`
    # showed it pays for itself: relational answer-in-context 0.80 -> 0.90 with
    # the 15-question regression set completely unmoved (recall 1.00, MRR 0.933,
    # answer-in-context 1.00). The result is a plateau, not a spike — 0.5, 0.8
    # and 1.0 score identically — which is why a mid-plateau value is safe to
    # pick from only 25 questions. It is not a free dial: at 2.0 the graph
    # channel drowns the other three and regression MRR collapses to 0.516.
    # Set to 0.0 to make retrieval byte-identical to the pre-graph system.
    kg_enabled: bool = False
    weight_graph: float = 0.8
    # Like the title channel, the graph channel is short and precision-ordered,
    # so early ranks in it mean more than early ranks in the long noisy
    # channels. Same reasoning as `rrf_k_title` above.
    rrf_k_graph: int = 20
    kg_neighbour_pages: int = 6
    kg_chunks_per_page: int = 2
    # Predicates worth traversing first when a question names an entity.
    kg_facts_in_context: bool = False
    kg_facts_max: int = 12

    def gen_model(self, profile: str | None = None) -> str:
        profile = (profile or self.gen_default_profile).lower()
        if profile in ("fast", "small", "llama"):
            return self.gen_model_fast
        if profile in ("quality", "big", "qwen"):
            return self.gen_model_quality
        # Allow passing an explicit Ollama model tag straight through.
        return profile


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    s.kg_dir.mkdir(parents=True, exist_ok=True)
    return s
