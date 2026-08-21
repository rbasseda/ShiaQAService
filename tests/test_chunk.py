import pytest

from shiaqa.config import Settings
from shiaqa.ingest.chunk import chunk_page, estimate_tokens
from shiaqa.ingest.clean import CleanPage, Section


@pytest.fixture
def settings():
    return Settings(chunk_target_tokens=100, chunk_max_tokens=160,
                    chunk_overlap_tokens=20, chunk_min_tokens=5)


def _page(**kw) -> CleanPage:
    base = dict(pageid=1, ns=0, title="Imam al-Husayn (a)", url="http://example/H",
                revid=1, timestamp=None, summary="A short summary of the article.",
                sections=[], infobox={}, categories=[], pointers=[], citations=[])
    base.update(kw)
    return CleanPage(**base)


def test_token_estimate_reflects_transliteration_density():
    # ~2.6 chars/token, denser than the usual 4 chars/token rule of thumb.
    assert estimate_tokens("a" * 260) == 100


def test_chunks_never_span_two_sections(settings):
    page = _page(sections=[
        Section(path=["Life"], level=2, text="Life content. " * 30),
        Section(path=["Death"], level=2, text="Death content. " * 30),
    ])
    for chunk in chunk_page(page, settings):
        if chunk.kind != "body":
            continue
        assert not ("Life content" in chunk.text and "Death content" in chunk.text)


def test_every_chunk_carries_its_context_header(settings):
    page = _page(sections=[Section(path=["Martyrdom"], level=2, text="He was killed at Karbala. " * 20)])
    body = [c for c in chunk_page(page, settings) if c.kind == "body"]
    assert body
    for chunk in body:
        assert chunk.embed_text().startswith("Imam al-Husayn (a) — Martyrdom")


def test_oversized_sections_are_split_and_stay_within_the_hard_max(settings):
    page = _page(sections=[Section(path=["Long"], level=2,
                                   text="\n\n".join(f"Paragraph number {i} with some text." * 4
                                                    for i in range(40)))])
    body = [c for c in chunk_page(page, settings) if c.kind == "body"]
    assert len(body) > 1
    assert max(c.tokens for c in body) <= settings.chunk_max_tokens * 1.25


def test_infobox_and_summary_become_their_own_chunks(settings):
    page = _page(infobox={"Mother": "Fatima al-Zahra (a)", "Image": "x.jpg"})
    kinds = {c.kind: c for c in chunk_page(page, settings)}
    assert "infobox" in kinds and "summary" in kinds
    assert "Mother: Fatima al-Zahra (a)" in kinds["infobox"].text
    assert "x.jpg" not in kinds["infobox"].text      # presentational field filtered


def test_trivially_short_content_is_not_indexed(settings):
    assert chunk_page(_page(summary="Tiny."), settings) == []
