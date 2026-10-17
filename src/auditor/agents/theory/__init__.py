"""The theory checker: are a paper's formal claims right (decision D21)?

The later stages of the theory cascade, after the planner listed the claims
(`auditor.planner.theory`). Thinking is spent only where it is needed:

1. Screen. One cheap call with low thinking reads every claim of the task with its own
   sentences and the definitions it uses, and calls each "fine", "suspicious" or "needs
   reasoning". A fine claim costs nothing more.
2. Deep check. Suspicious claims, and risky ones that need reasoning, are checked one at a
   time with high thinking, at most `AUDITOR_THEORY_DEEP_CHECKS` per task, with the whole
   paper as context. The default answer is "no error"; an error has to come with a witness
   (the step that fails, a counterexample or the correct value).

As everywhere in the auditor, a claim needs evidence: an error is kept only if it quotes
the paper verbatim, at least six words long, and the quote fixes the sentence it is
anchored on. The review step (`auditor.agents.theory.review`) then challenges what is
left with the stronger model.
"""

import asyncio
import re
from collections.abc import Sequence
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.infra.context import RunContext
from auditor.planner.theory import paper_listing
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    ParsedPaper,
    Sentence,
    SentenceAnchor,
    SourceKind,
    SourceRef,
    TaskKind,
    TaskOutcome,
    TaskResult,
    TheoryClaim,
)

MIN_QUOTE_WORDS = 6
MAX_SUMMARY_CHARS = 400

SCREEN_PROMPT = """You screen the formal claims of a research paper before a careful check.
For each claim you get the sentences that state and argue it, and the definitions and
results it uses. Answer for every claim:
- "fine": reading the argument shows it holds;
- "suspicious": you see a likely error (a false statement, a step that does not follow, a
  wrong formula, a result used outside its assumptions);
- "needs_reasoning": it cannot be settled without working through the details.
For anything but "fine", add a note of at most 20 words on what to look at."""

DEEP_PROMPT = """You check one formal claim of a research paper as a careful referee, with
the whole paper for context. Work through the argument yourself. The default answer is that
there is no error. Report an error only if you can show it: give the step that fails and
why, a counterexample, or the correct value as the witness. Typos that leave the
mathematics intact, notation and style are not errors. For an error, copy a quote of at
least six words from the claim's sentences exactly as written, give its location (such as
"Lemma 3.2" or "Eq. (5)") and rate your confidence."""


class ScreenVerdict(BaseModel):
    claim_id: str
    verdict: Literal["fine", "suspicious", "needs_reasoning"]
    note: str = ""


class Screen(BaseModel):
    verdicts: list[ScreenVerdict]


class DeepCheck(BaseModel):
    has_error: bool
    quote: str = ""
    location: str = ""
    explanation: str = ""
    witness: str = ""
    confidence: Literal["low", "medium", "high"] = "low"


def paper_text(paper: ParsedPaper) -> str:
    """The paper as the checks see it (one `[id] sentence` line each)."""
    return paper_listing(paper)


def _lines(paper: ParsedPaper, ids: Sequence[str]) -> str:
    texts = {s.id: s.text for s in paper.sentences}
    return "\n".join(f"[{i}] {texts[i]}" for i in ids if i in texts)


def _pack(paper: ParsedPaper, claim: TheoryClaim, note: str = "") -> str:
    head = f"Claim {claim.claim_id}"
    if claim.kind:
        head += f" ({claim.kind})"
    if claim.statement:
        head += f": {claim.statement}"
    if note:
        head += f"\nScreening note: {note}"
    return f"{head}\n{_lines(paper, claim.sentence_ids)}"


# Math delimiters that a model often swaps (\[ for \(, $ for \() when it copies a formula.
_DELIMITER = re.compile(r"\\[()\[\]]|\$+")


def _comparable(text: str) -> tuple[str, list[int]]:
    """The text without spaces and math delimiters, and where each kept character was."""
    kept: list[str] = []
    where: list[int] = []
    position = 0
    for match in _DELIMITER.finditer(text):
        for i in range(position, match.start()):
            if not text[i].isspace():
                kept.append(text[i])
                where.append(i)
        position = match.end()
    for i in range(position, len(text)):
        if not text[i].isspace():
            kept.append(text[i])
            where.append(i)
    return "".join(kept), where


def locate(quote: str, text: str) -> tuple[int, int] | None:
    """Where `quote` occurs in `text`, allowing for different spacing and math delimiters."""
    wanted, _ = _comparable(quote)
    if not wanted:
        return None
    flat, where = _comparable(text)
    found = flat.find(wanted)
    if found < 0:
        return None
    return where[found], where[found + len(wanted) - 1] + 1


def _finding(
    task: AuditTask, number: int, deep: DeepCheck, sentences: list[Sentence]
) -> CandidateFinding | None:
    if not deep.has_error or not deep.witness.strip():
        return None
    if len(deep.quote.split()) < MIN_QUOTE_WORDS:
        return None
    for sentence in sentences:
        span = locate(deep.quote, sentence.text)
        if span is None:
            continue
        start, end = span
        summary = f"{deep.location.strip() or 'Claim'}: {deep.explanation.strip()}"
        return CandidateFinding.for_task(
            task,
            number,
            error_class=ErrorClass.MATH_ERROR,
            anchor=SentenceAnchor(sentence_id=sentence.id, start=start, end=end),
            summary=summary[:MAX_SUMMARY_CHARS],
            explanation=(
                f"{deep.confidence} confidence. {deep.explanation.strip()} "
                f"Witness: {deep.witness.strip()}"
            ),
            evidence=[
                Evidence(
                    kind=EvidenceKind.QUOTE,
                    source=SourceRef(kind=SourceKind.SUBMISSION, id=task.paper_id),
                    text=sentence.text[start:end],
                    char_start=start,
                    char_end=end,
                )
            ],
            uncertain=deep.confidence != "high",
        )
    return None


_RISK = {"high": 0, "medium": 1, "low": 2}


def _escalated(
    claims: Sequence[TheoryClaim], verdicts: dict[str, ScreenVerdict], limit: int
) -> list[tuple[TheoryClaim, str]]:
    """The claims that earn a deep check: suspicious ones first, then risky claims that
    need reasoning, at most `limit`."""
    picked = []
    for claim in claims:
        verdict = verdicts.get(claim.claim_id)
        if verdict is None or verdict.verdict == "fine":
            continue
        if verdict.verdict == "needs_reasoning" and not (
            claim.needs_reasoning and claim.risk != "low"
        ):
            continue
        picked.append((verdict.verdict != "suspicious", _RISK[claim.risk], claim, verdict.note))
    picked.sort(key=lambda item: (item[0], item[1]))
    return [(claim, note) for _, _, claim, note in picked[:limit]]


async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
    if task.kind is not TaskKind.THEORY or task.claims is None:
        raise ValueError(f"{task.task_id} is not a theory task")
    if ctx.llm is None:
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason="checking the mathematics needs an LLM, and none is configured",
        )
    paper = ctx.paper
    claims = task.claims
    used = list(dict.fromkeys(i for c in claims for i in c.depends_on))
    prompt = (
        f"Definitions and results the claims use:\n{_lines(paper, used) or '(none)'}\n\n"
        "Claims:\n" + "\n\n".join(_pack(paper, c) for c in claims)
    )
    screened = await ctx.llm.generate(
        Screen,
        [SystemMessage(SCREEN_PROMPT), HumanMessage(prompt)],
        effort="low",
        purpose="theory-screen",
    )
    verdicts = {v.claim_id: v for v in screened.value.verdicts}
    escalated = _escalated(claims, verdicts, ctx.settings.theory_deep_checks)
    # The paper comes first and is the same in every deep check, so a provider's prompt
    # cache can reuse it.
    listing = paper_text(paper)
    llm = ctx.llm

    async def deep(claim: TheoryClaim, note: str) -> DeepCheck:
        message = f"The paper:\n{listing}\n\nClaim to check:\n{_pack(paper, claim, note)}"
        result = await llm.generate(
            DeepCheck,
            [SystemMessage(DEEP_PROMPT), HumanMessage(message)],
            effort="high",
            purpose="theory-deep",
        )
        return result.value

    results = await asyncio.gather(*(deep(claim, note) for claim, note in escalated))
    by_id = {s.id: s for s in paper.sentences}
    findings: list[CandidateFinding] = []
    for (claim, _), outcome in zip(escalated, results, strict=True):
        # A quote is looked for in the claim's sentences, then in what it uses.
        own = [by_id[i] for i in (*claim.sentence_ids, *claim.depends_on) if i in by_id]
        finding = _finding(task, len(findings), outcome, own)
        if finding is not None and finding.anchor not in [f.anchor for f in findings]:
            findings.append(finding)
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
