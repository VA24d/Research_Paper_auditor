"""The theory checker: are a paper's formal claims right (decision D21)?

The later stages of the theory cascade, after the planner listed the claims
(`auditor.planner.theory`). Thinking is spent only where it is needed:

1. Screen. One cheap call with low thinking reads every claim of the task with its own
   sentences and the definitions it uses, and calls each "fine", "suspicious" or "needs
   reasoning". A fine claim costs nothing more.
2. Deep check. Suspicious claims, high-risk claims that need reasoning (whatever the
   screen said) and medium-risk ones the screen could not settle are checked one at a
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
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.infra.context import RunContext
from auditor.infra.sandbox import MathSandbox
from auditor.planner.theory import paper_context, paper_listing
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
mathematics intact, notation and style are not errors. For an error, give the id of the
sentence where it is, copy a quote of at least six words from that sentence exactly as
written, give its location (such as "Lemma 3.2" or "Eq. (5)") and rate your confidence."""


TEST_OFFER = """
Before answering you may ask for one computation: put Python code in test_code (SymPy,
NumPy, mpmath, NetworkX and z3 are available; there is no network; print what you need).
You will get its output and then give your final answer, in full.
Use it to evaluate a formula at small values, search for a counterexample, or check an
algebraic step."""


# The answer after a sandbox test reads the output of a check the first round already
# reasoned out. At high thinking it took about 30k output tokens on SPOT, most of a paper's
# cost; medium thinking with this cap keeps it to about half.
TESTED_MAX_OUTPUT = 16_384
DEEP_ALLOWANCE = "theory-deep-checks"
WARM_LOCK = "theory-paper-cached"


def _sections_text(paper: ParsedPaper, claim: TheoryClaim) -> str:
    """The first section and the sections holding the claim and what it uses, each with
    its neighbours: a shorter prompt than the whole paper, at the risk of missing a step
    stated further away."""
    order = [s.id for s in paper.sections]
    by_id = {s.id: s for s in paper.sentences}
    held = {by_id[i].section_id for i in (*claim.sentence_ids, *claim.depends_on) if i in by_id}
    keep: set[str | None] = set(order[:1])
    for section in held:
        if section in order:
            at = order.index(section)
            keep.update(order[max(0, at - 1) : at + 2])
        else:
            keep.add(section)
    return paper_listing(paper, [s for s in paper.sentences if s.section_id in keep])


def make_sandbox() -> MathSandbox:
    return MathSandbox()


class ScreenVerdict(BaseModel):
    claim_id: str
    verdict: Literal["fine", "suspicious", "needs_reasoning"]
    note: str = ""


class Screen(BaseModel):
    verdicts: list[ScreenVerdict]


class DeepCheck(BaseModel):
    has_error: bool
    # Python code to run in the math sandbox before the final answer (first round only).
    test_code: str = ""
    sentence_id: str = ""
    quote: str = ""
    location: str = ""
    explanation: str = ""
    witness: str = ""
    confidence: Literal["low", "medium", "high"] = "low"


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


# Minus signs and dashes a model often writes as a plain hyphen.
_DASHES = str.maketrans(dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2212", "-"))


def _comparable(text: str) -> tuple[str, list[int]]:
    """The text without spaces and math delimiters, with dashes and compatibility characters
    folded, and where each kept character was."""
    kept: list[str] = []
    where: list[int] = []

    def keep(start: int, end: int) -> None:
        for i in range(start, end):
            if not text[i].isspace():
                kept.append(unicodedata.normalize("NFKC", text[i]).translate(_DASHES))
                where.append(i)

    position = 0
    for match in _DELIMITER.finditer(text):
        keep(position, match.start())
        position = match.end()
    keep(position, len(text))
    # NFKC can turn one character into several; keep the positions in step.
    flat: list[str] = []
    places: list[int] = []
    for piece, place in zip(kept, where, strict=True):
        flat.extend(piece)
        places.extend([place] * len(piece))
    return "".join(flat), places


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


@dataclass(frozen=True)
class QuoteAnchor:
    sentence: Sentence
    # The quote's characters in the sentence, or None when only its words match.
    span: tuple[int, int] | None


_WORD = re.compile(r"[A-Za-z]{2,}|\d+")
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
# Share of a quote's words that must occur in the sentence it names.
MIN_WORD_OVERLAP = 0.7


def _words(text: str) -> list[str]:
    text = _LATEX_COMMAND.sub(" ", unicodedata.normalize("NFKC", text))
    return [word.lower() for word in _WORD.findall(text)]


def anchor_quote(quote: str, sentence_id: str, sentences: Sequence[Sentence]) -> QuoteAnchor | None:
    """Where a model's quote is: the exact characters in the sentence it names or anywhere
    else, or else the named sentence when most of the quote's words are in it. Models
    often render a formula (\\equiv as ≡, x^2 as x²) instead of copying it, and that
    must not lose a finding the model pinned to a sentence."""
    named = next((s for s in sentences if s.id == sentence_id), None) if sentence_id else None
    ordered = ([named] if named is not None else []) + [s for s in sentences if s is not named]
    for sentence in ordered:
        span = locate(quote, sentence.text)
        if span is not None:
            return QuoteAnchor(sentence, span)
    if named is not None:
        wanted = _words(quote)
        have = set(_words(named.text))
        if len(wanted) >= 4 and sum(w in have for w in wanted) / len(wanted) >= MIN_WORD_OVERLAP:
            return QuoteAnchor(named, None)
    return None


def _finding(
    task: AuditTask, number: int, deep: DeepCheck, sentences: list[Sentence]
) -> CandidateFinding | None:
    if not deep.has_error or not deep.witness.strip():
        return None
    if len(deep.quote.split()) >= MIN_QUOTE_WORDS:
        found = anchor_quote(deep.quote, deep.sentence_id, sentences)
    elif deep.quote.strip():
        return None
    else:
        # 3.8 Flash at times names the sentence and leaves the quote empty (on a retracted
        # SPOT paper, with the right error as the witness). A sentence id that exists is
        # evidence enough; the whole sentence is quoted.
        named = next((s for s in sentences if s.id == deep.sentence_id.strip()), None)
        found = QuoteAnchor(named, None) if named is not None else None
    if found is not None:
        sentence = found.sentence
        start, end = found.span if found.span is not None else (0, len(sentence.text))
        # Some answers leave the explanation empty and put it all in the witness.
        why = deep.explanation.strip() or deep.witness.strip()
        summary = f"{deep.location.strip() or 'Claim'}: {why}"
        return CandidateFinding.for_task(
            task,
            number,
            error_class=ErrorClass.MATH_ERROR,
            anchor=SentenceAnchor(
                sentence_id=sentence.id,
                start=start if found.span is not None else None,
                end=end if found.span is not None else None,
            ),
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
    """The claims that earn a deep check, at most `limit`: suspicious ones first, then
    high-risk claims that need reasoning (a quick read with little thinking cannot clear a
    delicate argument), then medium-risk ones the screen could not settle."""
    picked = []
    for claim in claims:
        verdict = verdicts.get(claim.claim_id)
        said = verdict.verdict if verdict is not None else "needs_reasoning"
        note = verdict.note if verdict is not None else ""
        hard = claim.needs_reasoning and claim.risk == "high"
        unsettled = said == "needs_reasoning" and claim.needs_reasoning and claim.risk != "low"
        if said == "suspicious" or hard or unsettled:
            picked.append((said != "suspicious", _RISK[claim.risk], claim, note))
    picked.sort(key=lambda item: (item[0], item[1]))
    if not picked and limit:
        # A quick read clearing every claim does not clear a delicate argument: on a
        # retracted SPOT paper it called all five claims fine. The claim that rests on the
        # most others, the main result, still gets a deep check.
        central = [c for c in claims if c.needs_reasoning and c.risk != "low"]
        if central:
            claim = max(central, key=lambda c: (-_RISK[c.risk], len(c.depends_on)))
            return [(claim, "")]
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
    triaged = all(c.suspicion for c in claims)
    escalated: list[tuple[TheoryClaim, str]]
    if triaged:
        # The triage step already read the whole paper; its suspects need no screen.
        escalated = [(c, c.suspicion) for c in claims]
    else:
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
    llm = ctx.require_review_llm() if ctx.settings.theory_deep_model == "review" else ctx.llm
    whole = ctx.settings.theory_deep_context == "paper"

    sandbox = make_sandbox() if ctx.settings.theory_sandbox_enabled else None
    if sandbox is not None and not await sandbox.available():
        sandbox = None
    # One system prompt for both rounds, so the second round's prompt starts with the
    # first's and the provider's cache covers the paper in it.
    instructions = DEEP_PROMPT + (TEST_OFFER if sandbox is not None else "")
    # With the whole paper, it comes first, in the message every review-model call about
    # the paper starts with; the instructions and the claim follow.
    head: list[BaseMessage] = [paper_context(paper)] if whole else [SystemMessage(instructions)]

    async def first_round(message: str, sample: int) -> DeepCheck:
        async def call() -> DeepCheck:
            result = await llm.generate(
                DeepCheck,
                [*head, HumanMessage(message)],
                effort=ctx.settings.theory_deep_effort,
                purpose="theory-deep",
                sample=sample,
            )
            return result.value

        if not whole:
            return await call()
        # A paper's first deep check runs alone: calls sent at the same moment cannot
        # reuse each other's prompt, so the others wait until the paper is in the cache.
        lock = ctx.locks.setdefault(WARM_LOCK, asyncio.Lock())
        if not ctx.allowances.get(WARM_LOCK):
            async with lock:
                if not ctx.allowances.get(WARM_LOCK):
                    answer = await call()
                    ctx.allowances[WARM_LOCK] = 1
                    return answer
        return await call()

    async def deep(claim: TheoryClaim, note: str, sample: int) -> DeepCheck:
        claim_text = f"Claim to check:\n{_pack(paper, claim, note)}"
        message = (
            f"{instructions}\n\n{claim_text}"
            if whole
            else f"The paper:\n{_sections_text(paper, claim)}\n\n{claim_text}"
        )
        first = await first_round(message, sample)
        code = first.test_code.strip()
        if sandbox is None or not code:
            return first
        ran = await sandbox.run(code)
        output = ran.stdout if ran.ok else f"{ran.stdout}\n{ran.stderr}".strip()
        follow_up = (
            f"Your test code:\n{code}\n\nIt printed:\n{output or '(nothing)'}\n\n"
            "Now give your final answer, with every field it needs. You cannot ask for "
            "another test: leave test_code empty."
        )
        final = await llm.generate(
            DeepCheck,
            [*head, HumanMessage(message), HumanMessage(follow_up)],
            effort="medium",
            purpose="theory-deep-tested",
            max_output_tokens=TESTED_MAX_OUTPUT,
            sample=sample,
        )
        return final.value

    # High-risk claims get several tries, since one try can miss what another finds.
    tries = [
        (claim, note, sample)
        for claim, note in escalated
        for sample in range(
            ctx.settings.theory_deep_samples if claim.risk == "high" and not triaged else 1
        )
    ]
    # The paper's tasks share its deep checks; a task that comes later gets what is left.
    left = ctx.allowances.setdefault(DEEP_ALLOWANCE, ctx.settings.theory_deep_per_paper)
    tries = tries[:left]
    ctx.allowances[DEEP_ALLOWANCE] = left - len(tries)
    results = await asyncio.gather(*(deep(*one) for one in tries))
    by_id = {s.id: s for s in paper.sentences}
    findings: list[CandidateFinding] = []
    done: set[str] = set()
    for (claim, _, _), outcome in zip(tries, results, strict=True):
        if claim.claim_id in done:
            continue
        # A quote is looked for in the claim's sentences, then in what it uses, then
        # anywhere in the paper (an argument can lean on a step stated elsewhere).
        first = [by_id[i] for i in (*claim.sentence_ids, *claim.depends_on) if i in by_id]
        own = first + [s for s in paper.sentences if s not in first]
        finding = _finding(task, len(findings), outcome, own)
        if finding is not None and finding.anchor not in [f.anchor for f in findings]:
            findings.append(finding)
            done.add(claim.claim_id)
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
