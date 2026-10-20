"""The citation agent (Part 2): checks what a paper claims about its citations.

A citation task asks whether the cited work supports the claim the paper makes about it.
The agent resolves the citation to a work (P2.1, P2.2), retrieves the work's text at the
best level available (P2.3), ranks the passages relevant to the claim (P2.4), and asks the
model to judge the support against those passages, quoting them exactly. A number the
claim attributes to the work is compared with the number the work's text gives.

Reference tasks are checked without a model: the entry is resolved, and its metadata is
compared with the entry's fields.
"""

import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from auditor.infra.context import RunContext
from auditor.infra.llm import LLMNotConfiguredError
from auditor.retrieval.fulltext import SourceLevel, SourceText
from auditor.retrieval.passages import Passage, rank_passages
from auditor.retrieval.resolve import Resolution
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    SourceKind,
    SourceRef,
    TaskKind,
    TaskOutcome,
    TaskResult,
)

# How many passages the support judgement sees (Recall@5 is measured on this).
TOP_PASSAGES = 5
# A quote the model returns must be at least this long to be worth checking verbatim.
MIN_QUOTE_CHARS = 20


@dataclass(frozen=True)
class SupportVerdict:
    """What the model decided about one claim, with the passage it rests on."""

    verdict: str  # "supported" | "contradicted" | "insufficient"
    quote: str | None
    reason: str
    number_findings: tuple["NumberCheck", ...] = ()


class NumberCheck(BaseModel):
    """One number the claim attributes to the work, and what the work's text says."""

    claimed: str = Field(description="The number as the paper's sentence states it")
    found: str | None = Field(
        default=None,
        description="The corresponding value in the cited work's text, as written there",
    )
    quote: str = Field(
        description="The sentence of the cited work holding that value, copied exactly"
    )
    agrees: bool = Field(description="Whether the two values say the same thing")


class SupportAnswer(BaseModel):
    """The model's judgement of one claim against the cited work's passages."""

    verdict: str = Field(
        description=(
            "supported: the passages back the claim; contradicted: they say otherwise; "
            "insufficient: they do not settle it"
        )
    )
    quote: str = Field(
        description=(
            "The passage that supports or contradicts the claim, copied exactly from the "
            "passages shown; empty when the verdict is insufficient"
        )
    )
    reason: str = Field(description="Why the verdict holds, in one or two sentences")
    numbers: list[NumberCheck] = Field(
        default_factory=list,
        description="One entry per number the claim attributes to the cited work",
    )


_NUMBER = re.compile(r"(?<![\w.])(?:\d+\.\d+|\d+)(?![\w])")


def extract_numbers(text: str) -> list[str]:
    """The plain numbers a claim states, as written (percentages and units kept out)."""
    return _NUMBER.findall(text)


def _parse_number(text: str) -> float | None:
    """Normalise a number string to a float, or None when it is not one.

    Accepts the forms a claim or a source writes: `3e-4`, `0.0003`, `12.5%`, `80`,
    `1,000`. A percentage is divided by 100 so "12.5%" and "0.125" agree.
    """
    text = text.strip().rstrip(".,;")
    is_percent = text.endswith("%")
    text = text.rstrip("%").strip().replace(",", "").replace(" ", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def numbers_agree(claimed: str, found: str) -> bool:
    """Whether the claim's number and the source's number say the same thing.

    The comparison is by value, so `3e-4` and `0.0003` agree, as do `12.5%` and
    `0.125`. Two values within 1% relative difference also agree, because a paper
    and its source round differently. Text that is not a number agrees only when
    it is the same text.
    """
    claimed_value = _parse_number(claimed)
    found_value = _parse_number(found)
    if claimed_value is not None and found_value is not None:
        if claimed_value == found_value:
            return True
        scale = max(abs(claimed_value), abs(found_value))
        return scale > 0 and abs(claimed_value - found_value) / scale < 0.01
    return claimed.strip().lower() == found.strip().lower()


def _passages_block(passages: list[Passage]) -> str:
    """The passages as a numbered block the model can quote from."""
    return "\n\n".join(
        f"[{number}] {passage.text}" for number, passage in enumerate(passages, start=1)
    )


def _evidence_for(
    task: AuditTask,
    resolution: Resolution,
    source: SourceText,
    quote: str | None = None,
) -> tuple[Evidence, ...]:
    """The evidence a finding carries: how the work was resolved, and the quote."""
    work = resolution.work
    assert work is not None
    identifier = (
        f"doi:{work.doi}" if work.doi else (f"arxiv:{work.arxiv_id}" if work.arxiv_id else work.id)
    )
    lookup = Evidence(
        kind=EvidenceKind.LOOKUP,
        source=SourceRef(
            kind=SourceKind.SCHOLARLY_INDEX,
            id=identifier,
            title=work.title,
            url=work.html_url or work.pdf_url,
        ),
        text=(
            f"Resolved to '{work.title}' "
            f"(queries: {', '.join(q for q, _ in resolution.queries) or 'none'}); "
            f"source level used: {source.level}."
        ),
    )
    if not quote or len(quote) < MIN_QUOTE_CHARS:
        return (lookup,)
    char_start = source.text.find(quote)
    char_end = char_start + len(quote) if char_start >= 0 else None
    cited = Evidence(
        kind=EvidenceKind.QUOTE,
        source=SourceRef(
            kind=SourceKind.CITED_WORK,
            id=identifier,
            title=work.title,
            url=work.html_url or work.pdf_url,
        ),
        text=quote,
        char_start=char_start if char_start >= 0 else None,
        char_end=char_end,
    )
    return (lookup, cited)


def _number_check(
    task: AuditTask,
    resolution: Resolution,
    source: SourceText,
    check: NumberCheck,
) -> CandidateFinding | None:
    """A `claim_number_mismatch` finding for one number check, or None.

    The code-side value comparison decides, not the model's `agrees` hint: a model that
    says the values agree when they do not is exactly the failure this check exists for.
    A number the model could not find in the source is not confirmed either way, so it
    makes no finding here.
    """
    if not check.found:
        return None
    if not numbers_agree(check.claimed, check.found):
        return CandidateFinding.for_task(
            task,
            0,
            error_class=ErrorClass.CLAIM_NUMBER_MISMATCH,
            summary=(
                f"The claim gives '{check.claimed}' but the cited work's text gives '{check.found}'"
            ),
            explanation=check.quote,
            evidence=_evidence_for(task, resolution, source, check.quote),
        )
    return None


async def check_citation(
    task: AuditTask,
    ctx: RunContext,
    resolution: Resolution,
    source: SourceText,
) -> TaskResult:
    """Whether the cited work supports the claim (decision P2.5).

    `resolution` is what the citation resolved to (auditor.retrieval.resolve) and
    `source` the text retrieved for it (auditor.retrieval.fulltext). The model sees
    the TOP_PASSAGES passages, and its answer is checked in code: a quote must be
    verbatim in the source, and a number mismatch is confirmed by value.
    """
    if task.kind is not TaskKind.CITATION or not task.claim:
        raise ValueError(f"{task.task_id} is not a citation task with a claim")

    findings: list[CandidateFinding] = []

    # A support judgement needs a model.
    try:
        llm = ctx.require_llm()
    except LLMNotConfiguredError:
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason="no LLM is configured for the support judgement",
            findings=findings,
        )

    # The work itself could not be read: no full text and no abstract.
    if source.level is SourceLevel.NONE or not source.text:
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason=source.reason or "the cited work's text could not be retrieved",
            findings=findings,
        )

    passages = rank_passages(source.text, task.claim, top=TOP_PASSAGES)
    if not passages:
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason="the retrieved text has no passages to judge the claim against",
            findings=findings,
        )

    prompt = (
        "Given the following passages from a cited work, decide whether they support, "
        "contradict, or are insufficient to verify the claim made in a research paper.\n\n"
        f"CLAIM: {task.claim}\n\n"
        f"PASSAGES:\n{_passages_block(passages)}\n\n"
        "Answer with the verdict, an exact quote from ONE of the passages that justifies "
        "the verdict (empty if insufficient), a short reason, and for every number the "
        "claim attributes to the cited work: the value as the paper states it, the value "
        "as the cited work's text states it (None if nowhere stated), the sentence "
        "holding that value copied exactly, and whether the two agree."
    )
    answer = (await llm.generate(SupportAnswer, [HumanMessage(content=prompt)], effort="low")).value

    quote = answer.quote.strip() or None
    if answer.verdict not in ("supported", "contradicted", "insufficient"):
        answer = answer.model_copy(update={"verdict": "insufficient"})

    verdict = SupportVerdict(
        verdict=answer.verdict,
        quote=quote,
        reason=answer.reason,
        number_findings=tuple(answer.numbers),
    )

    if verdict.verdict == "contradicted" and resolution.work is not None:
        findings.append(
            CandidateFinding.for_task(
                task,
                len(findings),
                error_class=ErrorClass.CLAIM_NOT_SUPPORTED,
                summary=verdict.reason,
                explanation=f"Support verdict: {verdict.verdict}",
                evidence=_evidence_for(task, resolution, source, verdict.quote),
            )
        )

    if resolution.work is not None:
        for entry in answer.numbers:
            finding = _number_check(task, resolution, source, entry)
            if finding is not None:
                findings.append(finding)

    if not findings:
        return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=[])
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
