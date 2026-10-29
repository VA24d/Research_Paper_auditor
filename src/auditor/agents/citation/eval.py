"""Citation module evaluation (P2.9): Recall@5, module scores and tool-call accuracy.

- `recall_at_k`: whether the gold evidence passage recorded with a set B label is among
  the top-k passages the retriever ranks for the claim. This is the retrieval number the
  support judgement rests on: a claim the right passage never reaches cannot be judged.
- `score_citation_module` (in auditor.agents.citation.set_b): precision, recall and F1 of
  the four citation classes against the gold labels, and the false-positive rate on the
  clean controls.
- `tool_call_accuracy`: the share of reference lookups that resolve to the gold work
  (decision D8's tool-call accuracy, measured on labelled fixtures).
"""

import re
from dataclasses import dataclass

from auditor.retrieval.passages import Embedder, rank_passages
from auditor.retrieval.resolve import titles_match

_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().lower()


def recall_at_k(
    source_text: str,
    claim: str,
    gold_passage: str,
    *,
    k: int = 5,
    embedder: Embedder | None = None,
) -> bool:
    """Whether the gold passage's text is inside one of the top-k passages for the claim.

    The gold passage was cut from this same source text at build time, but the chunking
    may differ, so the test is containment (after whitespace normalisation), not exact
    equality. Pass the SAME embedder the agent runs with, or the number measures a
    different ranker than the one under test.
    """
    gold = _normalise(gold_passage)
    if not gold:
        return False
    for passage in rank_passages(source_text, claim, top=k, embedder=embedder):
        candidate = _normalise(passage.text)
        if gold in candidate or candidate in gold:
            return True
    return False


@dataclass(frozen=True)
class LookupOutcome:
    """What one reference lookup made of one labelled reference."""

    key: str
    resolved_title: str | None  # None: the lookup found nothing


@dataclass(frozen=True)
class LookupGold:
    """The labelled answer for the same reference."""

    key: str
    gold_title: str | None  # None: the reference does not exist (a fabricated work)


def lookup_is_correct(outcome: LookupOutcome, gold: LookupGold) -> bool:
    """Whether the lookup's outcome is the labelled one: found and the right work, or not
    found at all. Titles are compared the way the resolver compares them."""
    if (outcome.resolved_title is None) != (gold.gold_title is None):
        return False
    if outcome.resolved_title is None or gold.gold_title is None:
        return True
    return titles_match(outcome.resolved_title, gold.gold_title)


def tool_call_accuracy(outcomes: list[LookupOutcome], golds: list[LookupGold]) -> float:
    """The share of labelled reference lookups that came out right."""
    by_key = {gold.key: gold for gold in golds}
    scored = [outcome for outcome in outcomes if outcome.key in by_key]
    if not scored:
        return 0.0
    correct = sum(1 for outcome in scored if lookup_is_correct(outcome, by_key[outcome.key]))
    return correct / len(scored)
