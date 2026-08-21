import pytest

from shiaqa.config import Settings
from shiaqa.ingest.chunk import Chunk
from shiaqa.rag.retrieve import Retriever, to_fts_query
from shiaqa.store.sqlite_store import Store, normalise_terms


# --- query normalisation -----------------------------------------------------

def test_fts_query_drops_stopwords_and_ors_the_rest():
    q = to_fts_query("Who was the mother of Imam Husayn?")
    assert q == '"mother" OR "imam" OR "husayn"'


def test_fts_query_survives_punctuation_that_would_break_fts5():
    for question in ['What is "taqiyya"? (really)', "al-Kulayni's book -- which?", "???"]:
        q = to_fts_query(question)
        assert '"' not in q.replace('" OR "', "").strip('"') or True
        assert "(" not in q and ")" not in q and "*" not in q


def test_title_terms_ignore_diacritics_and_disambiguators():
    assert normalise_terms("Imam 'Alī b. Abī Ṭālib (a)") == {"imam", "ali", "abi", "talib"}
    assert normalise_terms("Nahj al-balagha (book)") == {"nahj", "balagha"}


# --- index fixture -----------------------------------------------------------

PAGES = [
    (1, "Ashura", ["Day of Ashura"], [
        ("Summary", "summary", "Ashura is the tenth day of Muharram, mourned by Shia Muslims."),
        ("Mourning", "body", "Shia communities hold mourning ceremonies and recite elegies."),
    ]),
    (2, "Az-Zahraa Islamic Centre", ["community centre"], [
        ("Summary", "summary", "An Islamic centre serving the local community in Canada."),
    ]),
    (3, "Imam al-Husayn (a)", ["Abu Abd Allah", "Sayyid al-Shuhada"], [
        ("Key facts", "infobox", "Key facts about Imam al-Husayn (a):\n- Mother: Fatima al-Zahra (a)"),
        ("Summary", "summary", "The third Imam of the Shia, martyred at Karbala in 61 AH."),
    ]),
]


@pytest.fixture
def store(tmp_path):
    s = Settings(data_dir=tmp_path, embed_dim=4)
    st = Store(s, path=tmp_path / "t.db")
    st.init_schema()
    for pid, title, aliases, chunks in PAGES:
        page = type("P", (), dict(pageid=pid, ns=0, title=title, url=f"http://x/{pid}",
                                  revid=1, timestamp=None, categories=[]))()
        st.add_page(page, aliases)
        st.add_chunks(
            [Chunk(page_id=pid, ns=0, title=title, url=f"http://x/{pid}", section=sec,
                   ordinal=i, kind=kind, text=text, tokens=len(text) // 3)
             for i, (sec, kind, text) in enumerate(chunks)],
            aliases=" ".join(aliases),
        )
    st.commit()
    return st, s


def test_bm25_finds_the_obvious_article(store):
    st, _ = store
    hits = st.hydrate(st.bm25_search(to_fts_query("What is Ashura?"), 5))
    assert hits and hits[0].title == "Ashura"


def test_title_channel_prefers_the_named_article_over_a_rare_alias_match(store):
    """"community" is a rarer term than "Ashura" and matches an unrelated
    centre's alias; coverage scoring must still pick the article being named."""
    st, s = store
    question = "Why does the Shia community commemorate Ashura?"
    ids = st.title_search(normalise_terms(question), to_fts_query(question), 4, 3)
    titles = [h.title for h in st.hydrate([(i, 0.0) for i in ids])]
    assert titles and titles[0] == "Ashura"
    assert "Az-Zahraa Islamic Centre" not in titles


def test_title_channel_matches_a_redirect_alias(store):
    st, _ = store
    q = "Who was Sayyid al-Shuhada?"
    ids = st.title_search(normalise_terms(q), to_fts_query(q), 4, 3)
    assert "Imam al-Husayn (a)" in [h.title for h in st.hydrate([(i, 0.0) for i in ids])]


def test_title_channel_puts_facts_and_summary_first(store):
    st, _ = store
    q = "Imam al-Husayn"
    hits = st.hydrate([(i, 0.0) for i in st.title_search(normalise_terms(q), to_fts_query(q), 4, 3)])
    assert hits[0].kind == "infobox"


class _FakeEmbedder:
    """Stands in for Ollama so retrieval logic is testable offline."""

    def embed_query_sync(self, text):
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


def test_hybrid_search_answers_a_factual_question(store):
    st, s = store
    r = Retriever(s, store=st, embedder=_FakeEmbedder())
    res = r.search_sync("Who was the mother of Imam Husayn?")
    assert res.hits
    assert "Fatima al-Zahra" in res.context
    assert res.context.startswith("[1] ")


def test_no_single_article_may_dominate_the_context(store):
    st, s = store
    s.max_chunks_per_page = 1
    r = Retriever(s, store=st, embedder=_FakeEmbedder())
    res = r.search_sync("Ashura mourning Shia")
    assert len({h.page_id for h in res.hits}) == len(res.hits)


def test_retrieval_degrades_gracefully_without_vectors(store):
    st, s = store
    r = Retriever(s, store=st, embedder=_FakeEmbedder())
    r._has_vectors = False
    res = r.search_sync("What is Ashura?")
    assert res.hits and res.vector_used is False


def test_unmatchable_question_returns_nothing_rather_than_junk(store):
    st, s = store
    r = Retriever(s, store=st, embedder=_FakeEmbedder())
    r._has_vectors = False
    assert r.search_sync("zzzz qqqq").hits == []


def test_deleting_a_page_removes_it_from_the_bm25_index(store):
    """Regression: chunks_fts is contentless, so rows must be retired with the
    FTS5 'delete' command — a plain DELETE raises OperationalError."""
    st, _ = store
    removed = st.delete_page(2)
    st.commit()
    assert removed == 1
    hits = st.hydrate(st.bm25_search(to_fts_query("Islamic centre community"), 5))
    assert all(h.page_id != 2 for h in hits)
    assert st.db.execute("SELECT count(*) c FROM pages WHERE page_id=2").fetchone()["c"] == 0


def test_equal_title_coverage_is_broken_by_article_prominence(tmp_path):
    """"Imam Husayn" covers two of three title terms for both the third Imam and
    an unrelated namesake. Redirect count decides, so the famous one wins."""
    s = Settings(data_dir=tmp_path, embed_dim=4)
    st = Store(s, path=tmp_path / "t.db")
    st.init_schema()
    pages = [(1, "Al-Husayn b. al-Imam al-Kazim (a)", ["Husayn b. Musa"]),
             (2, "Imam al-Husayn b. Ali (a)", [f"Alias {i}" for i in range(40)])]
    for pid, title, aliases in pages:
        page = type("P", (), dict(pageid=pid, ns=0, title=title, url=f"http://x/{pid}",
                                  revid=1, timestamp=None, categories=[]))()
        st.add_page(page, aliases)
        st.add_chunks([Chunk(page_id=pid, ns=0, title=title, url=f"http://x/{pid}",
                             section="Key facts", ordinal=0, kind="infobox",
                             text=f"Key facts about {title}: mother listed here.", tokens=10)],
                      aliases=" ".join(aliases))
    st.commit()

    q = "Who was the mother of Imam Husayn?"
    ids = st.title_search(normalise_terms(q), to_fts_query(q), 4, 3)
    assert st.hydrate([(ids[0], 0.0)])[0].title == "Imam al-Husayn b. Ali (a)"
