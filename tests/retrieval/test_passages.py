"""Offline tests for chunking and passage retrieval (P2.4)."""

import itertools

from auditor.retrieval.passages import (
    Passage,
    chunk_text,
    passage_fingerprint,
    rank_passages,
)

LONG = " ".join(f"word{index} filler filler" for index in range(400))  # ~5600 characters of text


def test_chunk_text_of_empty_text() -> None:
    assert chunk_text("") == []


def test_short_text_is_one_chunk_with_exact_offsets() -> None:
    text = "A short sentence."
    (chunk,) = chunk_text(text)
    assert chunk == Passage(text=text, char_start=0, char_end=len(text))


def test_chunks_break_at_whitespace_and_keep_exact_offsets() -> None:
    chunks = chunk_text(LONG, size=1200, overlap=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text == LONG[chunk.char_start : chunk.char_end]
        assert len(chunk.text) <= 1200
        assert not chunk.text.startswith(" ")
        assert not chunk.text.endswith(" ")
    # Consecutive chunks share the overlap, so no text is skipped.
    for first, second in itertools.pairwise(chunks):
        assert second.char_start > first.char_start
        assert second.char_start < first.char_end


def test_rank_passages_finds_the_relevant_chunk_first() -> None:
    text = (
        "The model reaches 92.1 exact match on the SQuAD benchmark. " * 40
        + "The training data was collected from Wikipedia articles. " * 40
    )
    (best, *rest) = rank_passages(text, "exact match on the SQuAD benchmark", top=5)
    assert "92.1 exact match" in best.text
    assert len(rest) == 4


def test_rank_passages_is_deterministic() -> None:
    first = rank_passages(LONG, "word7", top=3)
    second = rank_passages(LONG, "word7", top=3)
    assert first == second


def test_rank_passages_of_empty_text() -> None:
    assert rank_passages("", "claim") == []


def test_rank_passages_respects_top() -> None:
    assert len(rank_passages(LONG, "word3", top=2)) == 2


def test_passage_fingerprint_is_stable_and_short() -> None:
    passage = Passage(text="some text", char_start=0, char_end=9)
    assert passage_fingerprint(passage) == passage_fingerprint(
        Passage(text="some text", char_start=5, char_end=14)
    )
    assert passage_fingerprint(passage) != passage_fingerprint(
        Passage(text="other text", char_start=0, char_end=10)
    )
    assert len(passage_fingerprint(passage)) == 16
