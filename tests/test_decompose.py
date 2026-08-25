"""Opt-in question decomposition: parsing, failure modes, and the merge rule.

Fully offline. The `chat` callable is injected, so every path here runs without
httpx and without Ollama — including the ones that exercise a model returning
nonsense, which is the behaviour that most needs pinning: this feature is a
fallback, and a fallback that can break the default path is worse than none.
"""
import pytest

from shiaqa.rag.decompose import Decomposer, parse


# --- parsing -----------------------------------------------------------------

def test_a_numbered_list_parses():
    raw = ("1. Who was the father of Imam al-Sajjad?\n"
           "2. Who was the father of Imam al-Baqir?")
    assert parse(raw) == ["Who was the father of Imam al-Sajjad?",
                          "Who was the father of Imam al-Baqir?"]


@pytest.mark.parametrize("bullet", ["- ", "* ", "• ", "1) ", "2] ", "  3. "])
def test_common_list_markers_are_stripped(bullet):
    raw = f"{bullet}Who compiled the book al-Kafi?\n{bullet}Where did he live?"
    assert parse(raw)[0] == "Who compiled the book al-Kafi?"


def test_a_fenced_block_is_unwrapped():
    raw = "```\n1. Who taught al-Mufid?\n2. Who taught al-Tusi?\n```"
    assert len(parse(raw)) == 2


def test_a_preamble_line_is_dropped():
    raw = ("Sure, here are the sub-questions:\n"
           "1. Who was the mother of Imam Husayn?\n"
           "2. When was Imam Husayn born?")
    assert parse(raw) == ["Who was the mother of Imam Husayn?",
                          "When was Imam Husayn born?"]


def test_duplicate_subquestions_collapse():
    raw = "1. Who taught al-Mufid?\n2. Who taught al-Mufid?\n3. Who taught al-Tusi?"
    assert parse(raw) == ["Who taught al-Mufid?", "Who taught al-Tusi?"]


def test_the_cap_is_respected():
    raw = "\n".join(f"{i}. Who was scholar number {i} of the era?" for i in range(1, 9))
    assert len(parse(raw, max_parts=3)) == 3


# --- everything that must fall back ------------------------------------------

@pytest.mark.parametrize("raw", [
    "",
    "   \n\n  ",
    "SINGLE",
    "single",
    "I don't know.",                                  # too short, one line
    "Who compiled al-Kafi?",                          # already a single fact
    '{"sub_questions": ["a", "b"]}',                  # JSON instead of lines
    "1. no\n2. ok",                                   # both under three words
    "1. " + "word " * 60,                             # narrating, not asking
])
def test_unusable_output_yields_nothing(raw):
    """One sub-question is not a decomposition; zero is the fallback signal."""
    assert parse(raw) == []


@pytest.mark.asyncio
async def test_a_raising_chat_falls_back_rather_than_propagating():
    async def boom(_prompt):
        raise TimeoutError("ollama is busy")

    assert await Decomposer(boom).split("Compare the fourth and fifth Imams") == []


@pytest.mark.asyncio
async def test_a_model_that_answers_instead_of_splitting_falls_back():
    async def chat(_prompt):
        return "The fourth Imam was Imam al-Sajjad and the fifth was al-Baqir."

    assert await Decomposer(chat).split("Compare them") == []


@pytest.mark.asyncio
async def test_the_question_is_passed_to_the_model():
    seen = {}

    async def chat(prompt):
        seen["prompt"] = prompt
        return "1. Who was the father of Imam al-Baqir?\n2. Who taught him?"

    parts = await Decomposer(chat, max_parts=2).split("Trace the fifth Imam")
    assert "Trace the fifth Imam" in seen["prompt"]
    assert "2" in seen["prompt"]            # the cap reaches the prompt
    assert len(parts) == 2


# --- the merge rule ----------------------------------------------------------

def test_rankings_merge_by_rank_not_by_raw_score():
    """The reason `_rrf` is reused instead of taking the best score per chunk.

    An RRF score is a sum over whichever channels fired, so a broad sub-question
    produces larger numbers than a narrow one regardless of relevance. Chunk 99
    below is last for a broad sub-question that scored everything highly; chunk 1
    is first for a narrow one. Max-by-score would rank 99 above 1.
    """
    from shiaqa.rag.retrieve import _rrf

    narrow = [1, 2, 3]
    broad = [4, 5, 99]
    fused = _rrf([(narrow, 1.0, 60), (broad, 1.0, 60)])
    assert fused[1] > fused[99]


def test_a_chunk_found_by_two_subquestions_outranks_one_found_by_one():
    """The signal that makes decomposition worth the model call."""
    from shiaqa.rag.retrieve import _rrf

    fused = _rrf([([7, 1], 1.0, 60), ([9, 7], 1.0, 60)])
    assert fused[7] > fused[1]
    assert fused[7] > fused[9]
