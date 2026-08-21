"""Prompt construction.

Two constraints shape this prompt. First, the generator is a 3B or 7B model on
CPU — it follows short, concrete, positively-phrased instructions far better
than long rule lists. Second, WikiShia is a confessional encyclopaedia written
from a Twelver Shia perspective, so the answer should attribute its claims to
the source rather than assert them as neutral fact.
"""

from __future__ import annotations

SYSTEM = """You are a careful research assistant answering questions about Shia Islam, its history, texts and figures.

You answer only from the numbered excerpts supplied with each question. They come from WikiShia, an encyclopaedia written from a Twelver Shia perspective.

Rules:
- Use only facts stated in the excerpts. Never add outside knowledge.
- Cite the excerpt number in square brackets after each claim, like [2].
- If the excerpts do not contain the answer, say so plainly and name what is missing. Do not guess.
- Attribute contested or doctrinal claims to the source: "According to WikiShia...". Where the excerpts note that Sunni and Shia accounts differ, say so.
- Keep dates in the form the excerpts use (Hijri/Gregorian).
- Answer in the question's language, in clear prose. Be concise: a short paragraph unless the question needs more."""

USER_TEMPLATE = """Excerpts:

{context}

Question: {question}

Answer using only the excerpts above, citing them as [1], [2], and so on."""

NO_CONTEXT = """I could not find anything in the WikiShia index that addresses this question. \
Try rephrasing it, or use a name or term as it would appear in an encyclopaedia article."""


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]
