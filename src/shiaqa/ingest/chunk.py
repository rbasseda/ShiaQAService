"""Section-aware chunking.

Two decisions matter here:

1. **Chunks never cross a heading.** A section already is a topical unit, so
   respecting it keeps chunks coherent for free.
2. **Every chunk is embedded with its context header** ("<article> — <section
   path>"). Retrieval on this wiki hinges on proper names that often appear only
   in the article title, not in the paragraph body ("he was martyred at
   Karbala" never says *whose* biography it is). Prefixing the header fixes that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import Settings, get_settings
from .clean import CleanPage, infobox_as_text

# Measured against llama3.2 and qwen2.5 on this corpus: transliterated Arabic
# with diacritics tokenises at ~2.7 chars/token, far denser than typical English
# prose. Rounding down keeps us on the safe side of the context window.
CHARS_PER_TOKEN = 2.6

# Presentational infobox fields that carry no answerable content.
INFOBOX_NOISE = {
    "Image", "Image Size", "Imagesize", "Alt", "Caption", "Signature",
    "Photo", "Pic", "Picture", "Width", "Align", "Float", "Box Width",
}

_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?؟])\s+(?=[A-Z؀-ۿ])")


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


@dataclass
class Chunk:
    page_id: int
    ns: int
    title: str
    url: str
    section: str
    ordinal: int
    kind: str  # summary | infobox | body
    text: str
    tokens: int

    @property
    def context_header(self) -> str:
        if self.section and self.section != "Summary":
            return f"{self.title} — {self.section}"
        return self.title

    def embed_text(self) -> str:
        return f"{self.context_header}\n{self.text}"

    def cite_label(self) -> str:
        return self.context_header


def _split_oversized(block: str, max_chars: int) -> list[str]:
    """Break a too-long paragraph on sentence boundaries, then hard-wrap."""
    parts = _SENT_RE.split(block)
    out: list[str] = []
    cur = ""
    for part in parts:
        if len(cur) + len(part) + 1 <= max_chars:
            cur = f"{cur} {part}".strip()
            continue
        if cur:
            out.append(cur)
        while len(part) > max_chars:
            out.append(part[:max_chars])
            part = part[max_chars:]
        cur = part
    if cur:
        out.append(cur)
    return out


def _pack(text: str, s: Settings) -> list[str]:
    """Greedily pack paragraphs up to the target size, with a small overlap."""
    target = int(s.chunk_target_tokens * CHARS_PER_TOKEN)
    hard_max = int(s.chunk_max_tokens * CHARS_PER_TOKEN)
    overlap = int(s.chunk_overlap_tokens * CHARS_PER_TOKEN)

    blocks: list[str] = []
    for para in _PARA_RE.split(text):
        para = para.strip()
        if not para:
            continue
        blocks.extend(_split_oversized(para, hard_max) if len(para) > hard_max else [para])

    chunks: list[str] = []
    cur = ""
    for block in blocks:
        candidate = f"{cur}\n\n{block}".strip() if cur else block
        if len(candidate) <= target or not cur:
            cur = candidate
            continue
        chunks.append(cur)
        # Carry the tail of the previous chunk so a fact split across the
        # boundary is still answerable from either side.
        tail = cur[-overlap:] if overlap and len(cur) > overlap else ""
        if tail:
            tail = tail[tail.find(" ") + 1 :]
        cur = f"{tail}\n\n{block}".strip() if tail else block
    if cur:
        chunks.append(cur)
    return chunks


def chunk_page(page: CleanPage, settings: Settings | None = None) -> list[Chunk]:
    s = settings or get_settings()
    out: list[Chunk] = []
    ordinal = 0

    def add(section: str, kind: str, text: str) -> None:
        nonlocal ordinal
        text = text.strip()
        if estimate_tokens(text) < s.chunk_min_tokens:
            return
        out.append(
            Chunk(
                page_id=page.pageid, ns=page.ns, title=page.title, url=page.url,
                section=section, ordinal=ordinal, kind=kind,
                text=text, tokens=estimate_tokens(text),
            )
        )
        ordinal += 1

    # The infobox first: it is the densest answer source for "when/where/who" questions.
    box = {k: v for k, v in page.infobox.items() if k not in INFOBOX_NOISE}
    if box:
        add("Key facts", "infobox", infobox_as_text(page.title, box))

    for piece in _pack(page.summary, s):
        add("Summary", "summary", piece)

    for section in page.sections:
        for piece in _pack(section.text, s):
            add(section.label, "body", piece)

    return out


def chunk_stats(chunks: list[Chunk]) -> dict[str, float]:
    if not chunks:
        return {}
    toks = [c.tokens for c in chunks]
    return {
        "count": len(chunks),
        "tokens_total": sum(toks),
        "tokens_mean": sum(toks) / len(toks),
        "tokens_max": max(toks),
    }
