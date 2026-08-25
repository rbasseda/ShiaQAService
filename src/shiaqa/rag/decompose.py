"""Opt-in question decomposition, for questions the graph has no edge for.

The graph path in `kg/paths.py` costs no model calls and handles anything the
infoboxes already relate. This is the fallback for the rest — "compare the
lineages of the fourth and fifth Imams", where the answer needs facts from
several articles that no single typed relation joins.

It is opt-in because it costs model calls, and on this hardware that is the
whole story: one decompose call plus one synthesis call is roughly 20 s on
llama3.2:3b and 35 s on qwen2.5:7b, against ~14 s and ~27 s for a plain answer.
Answering each sub-question separately and then summarising would be three to
four calls — well over a minute — for no extra recall, since the sub-answers all
come from the same index. So: decompose once, retrieve cheaply for each part
(~109 ms apiece, no model involved), merge, and generate once.

Every failure path lands on the same fallback — return nothing, and let the
caller run the ordinary single-shot flow. A model that times out, emits prose,
answers the question instead of splitting it, or returns one line must never be
able to break the default path.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

from ..log import get_logger

log = get_logger(__name__)

# Kept blunt on purpose. A 3B model on CPU follows a short instruction with a
# hard output shape far more reliably than a nuanced one, and everything this
# prompt gets wrong is caught by the parser below.
PROMPT = """Break this question into the smallest set of independent factual \
sub-questions needed to answer it. Rules:
- One sub-question per line, numbered.
- At most {max_parts}.
- Each must stand alone, naming its subject in full rather than saying "he" or "it".
- If the question is already a single fact, reply with exactly: SINGLE

Question: {question}"""

_NUMBERED = re.compile(r"^\s*\d+\s*[.)\]]\s*(.+)$")
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*$")
# A model that ignores the format often opens with "Sure, here are..." or
# "Sub-questions:" — never a question, so a cheap prefix check catches it.
_PREAMBLE = re.compile(r"^(sure|here|okay|ok|certainly|sub-?questions?|answer)\b",
                       re.IGNORECASE)

Chat = Callable[[str], Awaitable[str]]


def parse(raw: str, max_parts: int = 3) -> list[str]:
    """Pull sub-questions out of whatever the model actually said.

    Returns `[]` for anything unusable — including a single sub-question, which
    is not a decomposition and would cost a model call to learn nothing.
    """
    if not raw or "SINGLE" in raw.upper():
        return []
    out: list[str] = []
    for line in raw.splitlines():
        if _FENCE.match(line):
            continue
        m = _NUMBERED.match(line)
        text = (m.group(1) if m else line).strip().strip("-*• ").strip()
        if not text or _PREAMBLE.match(text):
            continue
        # A sub-question that is very short carries no subject; a very long one
        # is the model narrating rather than asking.
        if len(text.split()) < 3 or len(text) > 200:
            continue
        if text not in out:
            out.append(text)
        if len(out) >= max_parts:
            break
    return out if len(out) >= 2 else []


class Decomposer:
    """Splits a question, given any async `chat(prompt) -> str` callable.

    The callable is injected rather than reached for through `AnswerEngine`, so
    tests exercise every failure mode without an HTTP client or an Ollama.
    """

    def __init__(self, chat: Chat, max_parts: int = 3) -> None:
        self.chat = chat
        self.max_parts = max_parts

    async def split(self, question: str) -> list[str]:
        try:
            raw = await self.chat(
                PROMPT.format(question=question, max_parts=self.max_parts))
        except Exception as exc:                     # noqa: BLE001 — see docstring
            # Timeouts, connection errors, a malformed response body: all of them
            # mean the same thing here, which is "answer it the ordinary way".
            log.debug("decomposition failed, falling back to single-shot: %s", exc)
            return []
        parts = parse(raw, self.max_parts)
        if not parts:
            log.debug("decomposition produced nothing usable; single-shot")
        return parts
