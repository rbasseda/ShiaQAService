"""Wikitext -> retrieval-ready plain text.

The template policy below is derived from a frequency scan of the actual
WikiShia corpus rather than from generic MediaWiki assumptions. The three
things that matter most for answer quality:

  * infoboxes are fact-dense and become `Key: value` lines,
  * quote templates carry real content (hadith, Qur'an, supplications),
  * `<ref>` tags and the References/Notes sections are pure bibliography and
    are pulled out of the body so they never eat context-window budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import mwparserfromhell as mwp

# --- template policy ---------------------------------------------------------

# Rendered as their inner text (original-script Arabic/Persian, emphasis, etc.).
PASSTHROUGH = {"ia", "iarabic", "arabic", "lang", "nowrap", "small", "big", "transl"}

# Navigation, maintenance and layout scaffolding: removed outright.
DROP = {
    "editorial box", "section", "notes", "references", "reflist", "end", "cb",
    "col-begin", "col-break", "col-end", "colbegin", "colend", "year nav",
    "calendar", "today/ad/ah", "about", "otheruses", "disambiguation", "dab",
    "navbox", "portal", "commons", "commonscat", "stub", "clear", "-", "!!",
    "toc", "tocright", "tocleft", "center", "sisterlinks", "shiaislam",
    "authority control", "hijri", "ahref", "enote", "efn", "sfn",
}

# Cross-reference pointers: not body text, but useful as related-page metadata.
POINTERS = {"main", "see also", "seealso", "further", "fulltext", "main article"}

QUOTE_TEMPLATES = {
    "quote box", "quotebox", "pull quote", "pullquote", "centered pull quote",
    "quote", "cquote", "blockquote", "poem quote",
}

INFOBOX_RE = re.compile(r"^infobox\b|^جعبه", re.I)

# Sections that are bibliography or navigation, not knowledge.
SKIP_SECTIONS = {
    "references", "notes", "note", "sources", "bibliography", "see also",
    "see-also", "external links", "external link", "further reading",
    "footnotes", "related articles", "gallery", "links",
}

HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)

# Apostrophes are everywhere in transliterated Arabic ("al-Shifa'"), which
# regularly leaves mwparserfromhell's italic/bold state machine unbalanced. When
# that happens the parser silently stops recognising *every* template in the
# block, so infoboxes leak through as raw markup. Skipping style tags avoids the
# failure entirely; the leftover quote markup is cleaned with a regex instead.
_STYLE_RE = re.compile(r"'{2,5}")


def _parse(text: str):
    return mwp.parse(text, skip_style_tags=True)


@dataclass
class Section:
    """One heading-delimited block of prose."""

    path: list[str]
    level: int
    text: str

    @property
    def label(self) -> str:
        return " > ".join(self.path) if self.path else "Summary"


@dataclass
class CleanPage:
    pageid: int
    ns: int
    title: str
    url: str
    revid: int | None
    timestamp: str | None
    summary: str
    sections: list[Section]
    infobox: dict[str, str] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)
    pointers: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    @property
    def char_len(self) -> int:
        return len(self.summary) + sum(len(s.text) for s in self.sections)


# --- low-level scrubbing -----------------------------------------------------

_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_REF_FULL_RE = re.compile(r"<ref[^>/]*>(.*?)</ref>", re.S | re.I)
_REF_SELF_RE = re.compile(r"<ref[^>]*/\s*>", re.I)
_TAG_RE = re.compile(r"</?(?:onlyinclude|noinclude|includeonly|div|span|small|big|"
                     r"center|sup|sub|u|s|font|poem|blockquote)[^>]*>", re.I)
_GALLERY_RE = re.compile(r"<gallery[^>]*>.*?</gallery>", re.S | re.I)
_TABLE_RE = re.compile(r"^\s*\{\|.*?^\s*\|\}\s*$", re.S | re.M)
_FILE_LINK_RE = re.compile(r"\[\[\s*(?:File|Image|Media)\s*:.*?\]\]", re.S | re.I)
_WS_RE = re.compile(r"[ \t]+")
_BLANKS_RE = re.compile(r"\n{3,}")


def _pull_citations(text: str) -> tuple[str, list[str]]:
    """Strip `<ref>` bodies out of the prose, returning them separately."""
    cites: list[str] = []

    def _grab(m: re.Match[str]) -> str:
        inner = _parse(m.group(1)).strip_code().strip()
        if inner:
            cites.append(_WS_RE.sub(" ", inner))
        return ""

    text = _REF_FULL_RE.sub(_grab, text)
    text = _REF_SELF_RE.sub("", text)
    return text, cites


def _template_name(node) -> str:
    return str(node.name).strip().lower()


def _param_text(value) -> str:
    return _clean_inline(str(value)).strip()


def _parse_infobox(node) -> dict[str, str]:
    out: dict[str, str] = {}
    for param in node.params:
        key = str(param.name).strip()
        if not key or key.isdigit():
            continue
        val = _param_text(param.value)
        # Infoboxes are full of blank slots and commented-out defaults.
        if not val or val in {"-", "?"}:
            continue
        out[key.replace("_", " ").strip().title()] = val
    return out


def _render_quote(node) -> str:
    params = {str(p.name).strip().lower(): p for p in node.params}
    body = params.get("quote") or params.get("text") or params.get("1")
    if body is None:
        return ""
    text = _param_text(body)
    if not text:
        return ""
    title = _param_text(params["title"].value) if "title" in params else ""
    src = ""
    for key in ("source", "author", "cite", "2"):
        if key in params:
            src = _param_text(params[key].value)
            if src:
                break
    parts = []
    if title:
        parts.append(f"{title}:")
    parts.append(f'"{text}"')
    if src:
        parts.append(f"— {src}")
    return " ".join(parts)


def _clean_inline(text: str) -> str:
    """Resolve templates and links inside a small fragment (infobox value etc.)."""
    code = _parse(text)
    _apply_template_policy(code, infoboxes=None, pointers=None)
    return _STYLE_RE.sub("", code.strip_code()).strip()


def _apply_template_policy(code, infoboxes: list | None, pointers: list | None) -> None:
    """Walk templates outermost-first, replacing each per policy."""
    for node in code.filter_templates(recursive=False):
        name = _template_name(node)
        try:
            if INFOBOX_RE.search(name):
                if infoboxes is not None:
                    infoboxes.append(_parse_infobox(node))
                code.remove(node)
            elif name in QUOTE_TEMPLATES:
                rendered = _render_quote(node)
                code.replace(node, f"\n\n{rendered}\n\n" if rendered else "")
            elif name in POINTERS:
                # {{Main|Some Article}} / {{fulltext|place=top|Text:X}} — the
                # positional arguments name related pages worth recording.
                if pointers is not None:
                    for param in node.params:
                        if not str(param.name).strip().isdigit():
                            continue
                        val = str(param.value).split("|")[0].strip()
                        if val:
                            pointers.append(val)
                code.remove(node)
            elif name in PASSTHROUGH:
                inner = node.params[-1].value if node.params else ""
                code.replace(node, str(inner))
            elif name in DROP or name.startswith("#") or name.startswith("infobox"):
                code.remove(node)
            else:
                # Unknown template: keep its longest parameter if that value is
                # long enough to be prose, otherwise drop it. Better a little
                # noise than a silently deleted sentence.
                prose = [v for v in (str(p.value).strip() for p in node.params) if len(v) > 40]
                code.replace(node, f" {max(prose, key=len)} " if prose else "")
        except ValueError:
            # Node already removed as part of an enclosing replacement.
            continue


def _clean_body(text: str, infoboxes: list, pointers: list) -> str:
    text = _COMMENT_RE.sub("", text)
    text = _GALLERY_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = _FILE_LINK_RE.sub("", text)
    text = _TABLE_RE.sub("", text)

    code = _parse(text)
    _apply_template_policy(code, infoboxes, pointers)

    # strip_code renders [[A|B]] -> B and [[A]] -> A, drops bold/italic markup.
    out = code.strip_code(normalize=True, collapse=True)
    out = _STYLE_RE.sub("", out)

    out = re.sub(r"^[*#:;]+\s*", "- ", out, flags=re.M)   # list markers -> bullets
    out = re.sub(r"^-\s*$", "", out, flags=re.M)
    out = _WS_RE.sub(" ", out)
    out = _BLANKS_RE.sub("\n\n", out)
    return out.strip()


def _split_sections(wikitext: str) -> list[tuple[list[str], int, str]]:
    """Split raw wikitext on headings into (path, level, body) triples."""
    matches = list(HEADING_RE.finditer(wikitext))
    blocks: list[tuple[list[str], int, str]] = []

    lead = wikitext[: matches[0].start()] if matches else wikitext
    blocks.append(([], 2, lead))

    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = _STYLE_RE.sub("", _parse(m.group(2)).strip_code()).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)
        body = wikitext[m.end() : end]

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        blocks.append(([t for _, t in stack], level, body))
    return blocks


def clean_page(raw: dict) -> CleanPage:
    """Turn one cached raw page into a `CleanPage`."""
    wikitext = raw["wikitext"]
    wikitext, citations = _pull_citations(wikitext)

    infoboxes: list[dict[str, str]] = []
    pointers: list[str] = []
    sections: list[Section] = []
    summary = ""

    for path, level, body in _split_sections(wikitext):
        leaf = path[-1].lower().strip() if path else ""
        if leaf in SKIP_SECTIONS:
            continue
        # A subsection of References is still References.
        if any(p.lower().strip() in SKIP_SECTIONS for p in path):
            continue

        text = _clean_body(body, infoboxes, pointers)
        if not text:
            continue
        if not path:
            summary = text
        else:
            sections.append(Section(path=path, level=level, text=text))

    infobox: dict[str, str] = {}
    for box in infoboxes:
        for k, v in box.items():
            infobox.setdefault(k, v)

    return CleanPage(
        pageid=raw["pageid"],
        ns=raw["ns"],
        title=raw["title"],
        url=raw["url"],
        revid=raw.get("revid"),
        timestamp=raw.get("timestamp"),
        summary=summary,
        sections=sections,
        infobox=infobox,
        categories=[c.removeprefix("Category:") for c in raw.get("categories", [])],
        pointers=sorted({p for p in pointers if p}),
        citations=citations,
    )


def infobox_as_text(title: str, infobox: dict[str, str]) -> str:
    """Infobox rendered as a standalone, retrievable fact block."""
    if not infobox:
        return ""
    lines = [f"Key facts about {title}:"]
    lines += [f"- {k}: {v}" for k, v in infobox.items()]
    return "\n".join(lines)
