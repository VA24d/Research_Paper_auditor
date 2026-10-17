"""The theory checker: is the mathematics of one segment of the paper right (decision D21)?

Each task names a segment, a run of formal sections the planner picked
(`auditor.planner.theory`). The LLM sees the whole paper for context and the segment with
its sentence ids, and reports objective errors: a false statement, a proof step that does
not follow, a wrong formula, a result used outside its assumptions. As everywhere in the
auditor, a claim needs evidence: an error is kept only if it quotes the paper verbatim, at
least six words long, and the quote fixes the sentence it is anchored on.
"""

import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.infra.context import RunContext
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    ParsedPaper,
    Sentence,
    SentenceAnchor,
    SentenceKind,
    SourceKind,
    SourceRef,
    TaskKind,
    TaskOutcome,
    TaskResult,
)

# The whole paper is context; longer papers are cut here (about 40k tokens).
MAX_CONTEXT_CHARS = 160_000
MIN_QUOTE_WORDS = 6
MAX_SUMMARY_CHARS = 400

SYSTEM_PROMPT = """You check the mathematics of a research paper as a careful referee. You
get the whole paper for context and one segment to check. Report only objective errors in
that segment: a theorem, lemma or claim that is false as stated, a proof step that does not
follow, a wrong formula or derivation, or a result used outside its assumptions. Work
through each step yourself, and where you can, give a counterexample or the correct value.
Do not report typos that leave the mathematics intact, notation, style, missing references,
or doubts you cannot point to. For each error, copy a quote of at least six words from the
segment exactly as it is written, give its location (such as "Lemma 3.2" or "Eq. (5)"),
explain why it is wrong, and rate your confidence. An empty list is a valid answer."""


class MathIssue(BaseModel):
    quote: str
    location: str
    explanation: str
    confidence: Literal["low", "medium", "high"]


class MathReview(BaseModel):
    errors: list[MathIssue]


def _body(paper: ParsedPaper) -> list[Sentence]:
    return [s for s in paper.sentences if s.kind in (SentenceKind.ABSTRACT, SentenceKind.BODY)]


def _paper_text(paper: ParsedPaper) -> str:
    titles = {s.id: s.title for s in paper.sections}
    lines: list[str] = []
    current: str | None = None
    for sentence in _body(paper):
        if sentence.section_id != current:
            current = sentence.section_id
            if titles.get(current or ""):
                lines.append(f"\n## {titles[current or '']}")
        lines.append(sentence.text)
    return "\n".join(lines)[:MAX_CONTEXT_CHARS]


def _segment(paper: ParsedPaper, sections: tuple[str, ...]) -> list[Sentence]:
    wanted = set(sections)
    return [
        s for s in paper.sentences if s.kind is SentenceKind.BODY and (s.section_id or "") in wanted
    ]


def locate(quote: str, text: str) -> tuple[int, int] | None:
    """Where `quote` occurs in `text`, allowing for different spacing."""
    words = quote.split()
    if not words:
        return None
    match = re.search(r"\s+".join(re.escape(word) for word in words), text)
    return (match.start(), match.end()) if match else None


def _finding(
    task: AuditTask, number: int, issue: MathIssue, sentences: list[Sentence]
) -> CandidateFinding | None:
    if len(issue.quote.split()) < MIN_QUOTE_WORDS:
        return None
    for sentence in sentences:
        span = locate(issue.quote, sentence.text)
        if span is None:
            continue
        start, end = span
        exact = sentence.text[start:end]
        summary = f"{issue.location.strip() or 'Segment'}: {issue.explanation.strip()}"
        return CandidateFinding.for_task(
            task,
            number,
            error_class=ErrorClass.MATH_ERROR,
            anchor=SentenceAnchor(sentence_id=sentence.id, start=start, end=end),
            summary=summary[:MAX_SUMMARY_CHARS],
            explanation=f"{issue.confidence} confidence. {issue.explanation.strip()}",
            evidence=[
                Evidence(
                    kind=EvidenceKind.QUOTE,
                    source=SourceRef(kind=SourceKind.SUBMISSION, id=task.paper_id),
                    text=exact,
                    char_start=start,
                    char_end=end,
                )
            ],
            uncertain=issue.confidence != "high",
        )
    return None


async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
    if task.kind is not TaskKind.THEORY or task.sections is None:
        raise ValueError(f"{task.task_id} is not a theory task")
    if ctx.llm is None:
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason="checking the mathematics needs an LLM, and none is configured",
        )
    paper = ctx.paper
    segment = _segment(paper, task.sections)
    if not segment:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason="the segment has no body text"
        )
    focus = "\n".join(f"[{s.id}] {s.text}" for s in segment)
    prompt = (
        f"The paper:\n{_paper_text(paper)}\n\n"
        f"The segment to check, one sentence per line with its id:\n{focus}"
    )
    result = await ctx.llm.generate(
        MathReview, [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]
    )
    # A quote is looked for in the segment first, then anywhere in the paper.
    sentences = segment + [s for s in _body(paper) if s not in segment]
    findings: list[CandidateFinding] = []
    for issue in result.value.errors:
        finding = _finding(task, len(findings), issue, sentences)
        if finding is not None and finding.anchor not in [f.anchor for f in findings]:
            findings.append(finding)
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
