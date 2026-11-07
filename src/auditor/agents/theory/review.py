"""The merge step for theory findings: a stronger model reviews what the segment checks
found (decisions D20 and D21).

Each segment check runs on a cheap model and sees one segment, so it reports errors that a
later section settles, and the same error from two segments. As in Google's Paper
Assistant Tool, a synthesis step then reviews the candidates with the whole paper in view.
Here one call to the review model (AUDITOR_LLM_REVIEW_MODEL, or the main model) per paper
judges every candidate math error: a real error stays, a repeat points at the one it
repeats, and one that is not an error is dropped with the model's reason.
"""

from collections.abc import Sequence
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from auditor.infra.context import RunContext
from auditor.planner.theory import paper_context
from auditor.schemas import ErrorClass, SentenceAnchor, VerificationState, VerifiedFinding

# How a drop by this review starts its reason; the evaluation tells these apart from the
# verifier's drops when it scores the full system without the verifier.
NO_ERROR_REASON = "The review model found no error"

SYSTEM_PROMPT = """You are the senior referee who checks the mathematics of a research
paper. Junior referees each checked some of its claims and listed candidate errors. Judge
every candidate with the whole paper in view, working through the mathematics yourself:
- "error": the paper is wrong at that point (a false statement, a step that does not
  follow, a wrong formula, a result used outside its assumptions);
- "same_as": it reports the same error as an earlier candidate; give that number;
- "not_an_error": the paper is right, the candidate misread it, a later part of the paper
  settles it, or it is only a typo, notation or style.
Give a short reason for each verdict."""


class MathVerdict(BaseModel):
    number: int
    verdict: Literal["error", "same_as", "not_an_error"]
    same_as: int | None = None
    reason: str = ""


class MathVerdicts(BaseModel):
    verdicts: list[MathVerdict]


def _candidate(number: int, decision: VerifiedFinding, texts: dict[str, str]) -> str:
    finding = decision.finding
    line = f"{number}. {finding.summary}"
    if isinstance(finding.anchor, SentenceAnchor):
        text = texts.get(finding.anchor.sentence_id, "")
        if text:
            line += f'\n   (in: "{text[:400]}")'
    return line


async def review_math(
    verified: Sequence[VerifiedFinding], ctx: RunContext
) -> list[VerifiedFinding]:
    out = list(verified)
    positions = [
        i
        for i, d in enumerate(out)
        if d.finding.error_class is ErrorClass.MATH_ERROR
        and d.state is not VerificationState.DROPPED
    ]
    if not positions or (ctx.llm is None and ctx.review_llm is None):
        return out
    texts = {s.id: s.text for s in ctx.paper.sentences}
    listing = "\n".join(_candidate(n, out[i], texts) for n, i in enumerate(positions, start=1))
    prompt = f"{SYSTEM_PROMPT}\n\nCandidate errors:\n{listing}"
    result = await ctx.require_review_llm().generate(
        MathVerdicts,
        # The paper first, as in the deep checks, so the prompt cache covers it.
        [paper_context(ctx.paper), HumanMessage(prompt)],
        effort="medium",
        purpose="theory-challenge",
        cache_prefix=True,
    )
    by_number = dict(enumerate(positions, start=1))
    for verdict in result.value.verdicts:
        index = by_number.get(verdict.number)
        if index is None:
            continue
        decision = out[index]
        reason = verdict.reason.strip() or "no reason given"
        if verdict.verdict == "not_an_error":
            out[index] = decision.model_copy(
                update={
                    "state": VerificationState.DROPPED,
                    "reason": f"{NO_ERROR_REASON}: {reason}",
                    "accepted_evidence": (),
                }
            )
        elif verdict.verdict == "same_as":
            first = by_number.get(verdict.same_as or 0)
            if first is not None and first != index:
                out[index] = decision.model_copy(
                    update={"duplicate_of": out[first].finding.finding_id}
                )
    return out
