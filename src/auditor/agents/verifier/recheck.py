"""A second reading of the flags that rest on a model's judgement (optional, paid).

The grounding check shows that the quotes and the code are real; it cannot show that they
mean what the finding says. For the classes in `MODEL_JUDGED`, a passage does not support
a claim or code does not do what a sentence says only in a model's reading. Here the
review model (AUDITOR_LLM_REVIEW_MODEL, or the main model) sees each such flag with only
its accepted evidence, in one call per paper, and answers whether that evidence alone
shows the stated problem. "not_shown" drops the flag, "unclear" sends it to human review.

It runs only with AUDITOR_VERIFIER_RECHECK=true, since it costs a model call per paper.
"""

from collections.abc import Sequence
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.agents.verifier.policy import MODEL_JUDGED
from auditor.infra.context import RunContext
from auditor.schemas import VerificationState, VerifiedFinding

SYSTEM_PROMPT = """You check the work of other reviewers of a research paper. Each item
below is a problem a reviewer reported, with the evidence they gave: quotes from the
paper, from the work it cites, or lines from its code repository. Judge each item from its
evidence alone, and be strict: the reviewers often misread.
- "shown": the evidence by itself shows the problem as stated.
- "not_shown": the evidence does not show it (the quotes agree, the code does what the
  paper says, the cited passage is about something else, or the reviewer misread).
- "unclear": the evidence is not enough to decide either way.
Give a short reason for each verdict."""

MAX_EVIDENCE_CHARS = 1500
MAX_ITEMS = 40


class Recheck(BaseModel):
    number: int
    verdict: Literal["shown", "not_shown", "unclear"]
    reason: str = ""


class Rechecks(BaseModel):
    verdicts: list[Recheck]


def _item(number: int, decision: VerifiedFinding) -> str:
    finding = decision.finding
    lines = [f"{number}. ({finding.error_class.value}) {finding.summary}"]
    for evidence in decision.accepted_evidence:
        where = evidence.source.kind.value
        if evidence.path:
            where += f" {evidence.path}:{evidence.line_start}"
        lines.append(f"   [{where}] {evidence.text[:MAX_EVIDENCE_CHARS]}")
    return "\n".join(lines)


async def recheck(verified: Sequence[VerifiedFinding], ctx: RunContext) -> list[VerifiedFinding]:
    out = list(verified)
    positions = [
        i
        for i, d in enumerate(out)
        if d.state is VerificationState.VERIFIED_FLAG and d.finding.error_class in MODEL_JUDGED
    ][:MAX_ITEMS]
    if not positions or (ctx.llm is None and ctx.review_llm is None):
        return out
    listing = "\n\n".join(_item(n, out[i]) for n, i in enumerate(positions, start=1))
    result = await ctx.require_review_llm().generate(
        Rechecks,
        [SystemMessage(SYSTEM_PROMPT), HumanMessage(listing)],
        effort="low",
        purpose="verifier-recheck",
    )
    by_number = dict(enumerate(positions, start=1))
    answered: set[int] = set()
    for verdict in result.value.verdicts:
        index = by_number.get(verdict.number)
        if index is None or index in answered:
            continue
        answered.add(index)
        decision = out[index]
        reason = verdict.reason.strip() or "no reason given"
        if verdict.verdict == "not_shown":
            out[index] = decision.model_copy(
                update={
                    "state": VerificationState.DROPPED,
                    "reason": f"A second reading found the evidence does not show it: {reason}",
                    "accepted_evidence": (),
                }
            )
        elif verdict.verdict == "unclear":
            out[index] = decision.model_copy(
                update={
                    "state": VerificationState.NEEDS_HUMAN_REVIEW,
                    "reason": f"A second reading could not settle it: {reason}",
                }
            )
    # A flag the model gave no verdict on was not confirmed by the second reading.
    for index in set(positions) - answered:
        out[index] = out[index].model_copy(
            update={
                "state": VerificationState.NEEDS_HUMAN_REVIEW,
                "reason": "The second reading gave no verdict on it.",
            }
        )
    return out
