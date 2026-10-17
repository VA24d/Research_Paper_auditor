"""One call per paper for its mathematics: the single-call comparison for the theory
cascade (decision D21).

The whole paper goes to the model once, with a prompt that asks only for mathematical
errors and the witness for each. It is what Google's Paper Assistant Tool compares its
pipeline against, and it tells whether the cascade's stages buy anything over one good
call. The same evidence gate applies: a verbatim quote of at least six words.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.agents.theory import MAX_SUMMARY_CHARS, MIN_QUOTE_WORDS, locate
from auditor.infra.llm import Effort, StructuredLLM
from auditor.planner.theory import THEORY_KEY, paper_listing
from auditor.schemas import (
    ErrorClass,
    ParsedPaper,
    Prediction,
    SentenceAnchor,
    SentenceKind,
    SettingTarget,
    SystemName,
)

SINGLE_PROMPT = """You check the mathematics of a research paper as a careful referee. Go
through its theorems, lemmas, propositions, derivations and bounds, and work through the
arguments yourself. Report only objective errors: a statement that is false as written, a
proof step that does not follow, a wrong formula or calculation, or a result used outside
its assumptions. Do not report typos that leave the mathematics intact, notation, style,
missing references or doubts you cannot point to. Each error needs a witness: the step that
fails and why, a counterexample, or the correct value. Copy a quote of at least six words
from the paper exactly as written where the error is, and give its location (such as
"Lemma 3.2" or "Eq. (5)"). An empty list is a valid answer."""


class SingleIssue(BaseModel):
    quote: str
    location: str = ""
    explanation: str = ""
    witness: str = ""
    confidence: Literal["low", "medium", "high"] = "low"


class SingleReview(BaseModel):
    errors: list[SingleIssue]


async def check_paper(
    paper: ParsedPaper, llm: StructuredLLM, *, run_id: str, effort: Effort | None
) -> list[Prediction]:
    result = await llm.generate(
        SingleReview,
        [SystemMessage(SINGLE_PROMPT), HumanMessage(paper_listing(paper))],
        effort=effort,
        purpose="theory-single",
    )
    sentences = [s for s in paper.sentences if s.kind in (SentenceKind.ABSTRACT, SentenceKind.BODY)]
    predictions: list[Prediction] = []
    for issue in result.value.errors:
        if not issue.witness.strip() or len(issue.quote.split()) < MIN_QUOTE_WORDS:
            continue
        for sentence in sentences:
            span = locate(issue.quote, sentence.text)
            if span is None:
                continue
            summary = f"{issue.location.strip() or 'Paper'}: {issue.explanation.strip()}"
            predictions.append(
                Prediction(
                    run_id=run_id,
                    system=SystemName.B0,
                    paper_id=paper.paper_id,
                    variant=paper.variant,
                    error_class=ErrorClass.MATH_ERROR,
                    anchor=SentenceAnchor(sentence_id=sentence.id, start=span[0], end=span[1]),
                    target=SettingTarget(key=THEORY_KEY),
                    summary=summary[:MAX_SUMMARY_CHARS],
                    finding_id=f"{paper.paper_id}-single-{len(predictions)}",
                )
            )
            break
    return predictions
