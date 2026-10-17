"""The integrity screen: problems in a paper's own text that need no outside source (D22).

Some papers carry their own reason for rejection: a caption that says the numbers are
"simulated drafting results ... to be replaced before submission", an author note left in
the text, or references that never resolved. One task per paper checks for these in two
ways:

- with no model, for structural markers: unresolved references ("??", "[?]") and leftover
  markers (TODO, TBD, FIXME, placeholder numbers such as "XX%", lorem ipsum). On 195 real
  papers, "XXX" and "X%" were variable names and "[]" an empty list in pseudocode, so those
  are left to the model;
- with one cheap model call over every caption, footnote and paragraph, for statements
  that the paper's own results are placeholders, simulated, made up or still to be
  replaced, and for instructions to the authors left in the text. Which wording counts is
  the model's reading, so this works in any field.

A finding must quote its sentence verbatim.
"""

__all__ = ["RULE_ID", "ScreenFindings", "check", "integrity_tasks"]

import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.agents.integrity.results import check_results
from auditor.agents.theory import anchor_quote
from auditor.infra.context import RunContext
from auditor.planner.integrity import RESULTS_RULE_ID, RULE_ID, integrity_tasks
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    ParsedPaper,
    RuleTarget,
    Sentence,
    SentenceAnchor,
    SentenceKind,
    SourceKind,
    SourceRef,
    TaskKind,
    TaskOutcome,
    TaskResult,
)

MAX_LISTING_CHARS = 200_000
MAX_SCREEN_TOKENS = 4096
MIN_QUOTE_WORDS = 4

# Structural markers: what a typesetter or an author leaves behind, in any field.
_UNRESOLVED = re.compile(r"\?\?|\[\s*\?\s*\]|\(\s*\?\s*\)|Error! Reference source not found")
# Written in capitals ("todo" can be an ordinary word).
_NOTE = re.compile(r"\b(?:TODO\w*|TBD|FIXME)\b")
# "Section X", "Figure X": a reference whose number was never filled in (capital X only).
_UNFILLED = re.compile(
    r"\b(?:Section|Sec\.|Figure|Fig\.|Table|Tab\.|Equation|Eq\.|Appendix|Theorem|Lemma"
    r"|Algorithm)\s+X\b"
)
_INSERT = re.compile(r"\[(?:insert|add|fill in|todo|tbd|placeholder)\b[^\]]{0,80}\]", re.I)
_PLACEHOLDER = re.compile(
    r"\blorem ipsum\b|\bXX+(?:\.X+)?\s?%|\bX\.X+\s?%|\[citation needed\]", re.IGNORECASE
)

SCREEN_PROMPT = """You check a research paper submission for content that is not finished.
Find sentences, including table and figure captions and footnotes, that:
- say the paper's own results, numbers, figures or tables are placeholders, simulated,
  invented, illustrative only, drafts, or to be replaced or updated before submission
  ("placeholder_results");
- are notes or instructions to the authors or editors left in the text, such as "add the
  ablation here" or "check this number" ("author_note");
- are otherwise clearly unfinished: cut off, template text never filled in
  ("unfinished_text").
Simulations that are part of the method (a simulated environment, synthetic data the paper
studies) are not findings. For each finding give the sentence id, a quote of at least four
words copied exactly from that sentence, the kind and a one-line explanation. An empty
list is a valid answer."""


class ScreenFinding(BaseModel):
    sentence_id: str
    quote: str
    kind: Literal["placeholder_results", "author_note", "unfinished_text"]
    explanation: str = ""


class ScreenFindings(BaseModel):
    findings: list[ScreenFinding]


def _readable(paper: ParsedPaper) -> list[Sentence]:
    kinds = (SentenceKind.ABSTRACT, SentenceKind.BODY, SentenceKind.CAPTION, SentenceKind.FOOTNOTE)
    return [s for s in paper.sentences if s.kind in kinds]


def _listing(sentences: list[Sentence]) -> str:
    return "\n".join(f"[{s.id}] {s.text}" for s in sentences)[:MAX_LISTING_CHARS]


def _finding(
    task: AuditTask,
    number: int,
    sentence: Sentence,
    span: tuple[int, int],
    summary: str,
    uncertain: bool = False,
) -> CandidateFinding:
    start, end = span
    return CandidateFinding.for_task(
        task,
        number,
        error_class=ErrorClass.UNFINISHED_CONTENT,
        anchor=SentenceAnchor(sentence_id=sentence.id, start=start, end=end),
        summary=summary,
        evidence=[
            Evidence(
                kind=EvidenceKind.QUOTE,
                source=SourceRef(kind=SourceKind.SUBMISSION, id=task.paper_id),
                text=sentence.text[start:end],
                char_start=start,
                char_end=end,
            )
        ],
        uncertain=uncertain,
    )


# Roman-numbered tables or figures ("TABLE IX"): there, "Table X" is the tenth, not a gap.
_ROMAN = re.compile(r"\b(?:Table|TABLE|Figure|FIGURE|Fig\.)\s+(?:I{1,3}|IV|VI{0,3}|IX)\b")


def _marker_hits(sentences: list[Sentence]) -> list[tuple[Sentence, tuple[int, int], str]]:
    hits = []
    roman = any(_ROMAN.search(s.text) for s in sentences)
    for sentence in sentences:
        unresolved = _UNRESOLVED.search(sentence.text)
        if unresolved is not None:
            hits.append(
                (sentence, unresolved.span(), "An unresolved reference is left in the text.")
            )
            continue
        leftover = (
            _NOTE.search(sentence.text)
            or _PLACEHOLDER.search(sentence.text)
            or (None if roman else _UNFILLED.search(sentence.text))
            or _INSERT.search(sentence.text)
        )
        if leftover is not None:
            hits.append(
                (sentence, leftover.span(), "A leftover note or placeholder is in the text.")
            )
    return hits


async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
    if task.kind is not TaskKind.INTEGRITY:
        raise ValueError(f"{task.task_id} is not an integrity task")
    if isinstance(task.target, RuleTarget) and task.target.rule_id == RESULTS_RULE_ID:
        return await check_results(task, ctx)
    sentences = _readable(ctx.paper)
    findings: list[CandidateFinding] = []
    seen: set[str] = set()
    for sentence, span, summary in _marker_hits(sentences):
        findings.append(_finding(task, len(findings), sentence, span, summary))
        seen.add(sentence.id)
    if ctx.llm is not None:
        result = await ctx.llm.generate(
            ScreenFindings,
            [SystemMessage(SCREEN_PROMPT), HumanMessage(_listing(sentences))],
            effort="low",
            purpose="integrity-screen",
            max_output_tokens=MAX_SCREEN_TOKENS,
        )
        for item in result.value.findings:
            if len(item.quote.split()) < MIN_QUOTE_WORDS:
                continue
            found = anchor_quote(item.quote, item.sentence_id, sentences)
            # Prose is copied, not rendered, so only an exact quote counts here.
            if found is None or found.span is None or found.sentence.id in seen:
                continue
            seen.add(found.sentence.id)
            label = item.kind.replace("_", " ")
            summary = (
                f"{label.capitalize()}: {item.explanation.strip() or found.sentence.text[:200]}"
            )
            findings.append(_finding(task, len(findings), found.sentence, found.span, summary))
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
