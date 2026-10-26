"""Offline tests for the citation module evaluation (P2.9)."""

import pytest

from auditor.agents.citation.eval import (
    LookupGold,
    LookupOutcome,
    lookup_is_correct,
    recall_at_k,
    tool_call_accuracy,
)

SOURCE_TEXT = (
    # Topic A (the gold passage's topic), topic B and topic C, each long enough to fill
    # whole chunks, so an irrelevant claim ranks B and C chunks first.
    "The model reaches 92.1 exact match on the SQuAD benchmark. " * 40
    + "Training used a batch size of 256 examples on one GPU. " * 40
    + "Quantum chromodynamics describes the strong force between quarks. " * 40
)
GOLD_PASSAGE = "The model reaches 92.1 exact match on the SQuAD benchmark."


def test_recall_at_k_finds_the_gold_passage() -> None:
    assert recall_at_k(SOURCE_TEXT, "exact match on the SQuAD benchmark", GOLD_PASSAGE)
    assert recall_at_k(SOURCE_TEXT, "exact match on the SQuAD benchmark", GOLD_PASSAGE, k=1)


def test_recall_at_k_misses_when_the_passage_is_not_relevant() -> None:
    # An irrelevant claim ranks the quantum-chromodynamics chunks first; with k=2 the
    # SQuAD passage is never among them (with k=5 the top-5 covers most of this short
    # source, so the containment test passes trivially).
    assert not recall_at_k(SOURCE_TEXT, "quantum chromodynamics gauge theory", GOLD_PASSAGE, k=2)
    assert recall_at_k(SOURCE_TEXT, "exact match on the SQuAD benchmark", GOLD_PASSAGE, k=2)


def test_recall_at_k_tolerates_whitespace_differences() -> None:
    padded = "  The model   reaches 92.1 exact match\non the SQuAD benchmark.  "
    assert recall_at_k(SOURCE_TEXT, "exact match on the SQuAD benchmark", padded)


def test_recall_at_k_of_an_empty_gold_passage() -> None:
    assert not recall_at_k(SOURCE_TEXT, "claim", "   ")


def test_lookup_is_correct() -> None:
    gold = LookupGold(key="k1", gold_title="Attention Is All You Need")
    assert lookup_is_correct(LookupOutcome("k1", "Attention Is All You Need!"), gold)
    assert not lookup_is_correct(LookupOutcome("k1", "BERT Pre-training"), gold)
    assert not lookup_is_correct(LookupOutcome("k1", None), gold)
    missing = LookupGold(key="k2", gold_title=None)
    assert lookup_is_correct(LookupOutcome("k2", None), missing)
    assert not lookup_is_correct(LookupOutcome("k2", "Anything"), missing)


def test_tool_call_accuracy() -> None:
    outcomes = [
        LookupOutcome("k1", "Attention Is All You Need"),
        LookupOutcome("k2", None),
        LookupOutcome("k3", "Wrong Work Entirely"),
        LookupOutcome("unknown", "ignored"),
    ]
    golds = [
        LookupGold("k1", "Attention Is All You Need"),
        LookupGold("k2", None),
        LookupGold("k3", "The Right Work"),
    ]
    assert tool_call_accuracy(outcomes, golds) == pytest.approx(2 / 3)
    assert tool_call_accuracy([], golds) == 0.0
