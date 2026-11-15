"""The citation agent (Part 2): checks what a paper claims about its citations.

A citation task asks whether the cited work supports the claim the paper makes about it.
The agent resolves the citation to a work (P2.1, P2.2), retrieves the work's text at the
best level available (P2.3), ranks the passages relevant to the claim (P2.4), and asks the
model to judge the support against those passages, quoting them exactly. A number the
claim attributes to the work is compared with the number the work's text gives.

Reference tasks are checked without a model: the entry is resolved, and its metadata is
compared with the entry's fields.
"""

import math
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from auditor.infra.context import RunContext
from auditor.infra.llm import LLMNotConfiguredError
from auditor.retrieval.embeddings import get_embedder
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


_NUMBER = re.compile(r"(?<![\w.])(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?(?![\w])")


def extract_numbers(text: str) -> list[str]:
    """The plain numbers a claim or a source states, as written.

    Scientific notation is kept whole (`3e-4`, `1.5e10`); percentages and units stay
    attached to the number (`92.1%` -> `92.1`). Sections ("section 3.2") are numbers
    too; the value comparison decides whether one matters.
    """
    return _NUMBER.findall(text)


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "sixteen": 16,
    "twenty": 20,
    "fifty": 50,
    "hundred": 100,
    "thousand": 1000,
}

# A trailing letter or word that is not part of the value itself: "8x", "10-fold",
# "50GB" — the numeric head is the value; the suffix says what it counts.
# The typographic minus PDFs and HTML use; read as "-".
MINUS_SIGN = chr(0x2212)
_NUMBER_HEAD = re.compile(r"([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)")
_NUMBER_WORD = re.compile(r"[a-z]+")


def _parse_number(text: str) -> float | None:
    """Normalise a number string to a float, or None when it is not one.

    Accepts the forms a claim or a source writes: `3e-4`, `0.0003`, `12.5%`, `80`,
    `1,000`, `8x`, `eight`. A percentage is divided by 100 so "12.5%" and "0.125" agree;
    a trailing suffix ("8x") is dropped, because it says what is counted, not the value;
    a number word ("eight") is its value."""
    text = text.strip().rstrip(".,;")
    is_percent = text.endswith("%")
    text = text.rstrip("%").strip().replace(",", "").replace(" ", "").lower()
    text = text.replace(MINUS_SIGN, "-")
    if not text:
        return None
    head = _NUMBER_HEAD.search(text)
    if head is not None:
        value = float(head.group(1))
        return value / 100.0 if is_percent else value
    word = _NUMBER_WORD.match(text)
    if word is not None and word.group(0) in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[word.group(0)])
    return None


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


def _find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Where `quote` occurs in `text`, or None when it does not.

    An exact match is preferred. A whitespace-tolerant match re-anchors a quote the model
    reformatted (line breaks became spaces), which is the same quote, not a new one."""
    start = text.find(quote)
    if start >= 0:
        return start, start + len(quote)
    pattern = re.compile(r"\s+".join(re.escape(word) for word in quote.split()))
    match = pattern.search(text)
    if match:
        return match.start(), match.end()
    return None


def _evidence_for(
    task: AuditTask,
    resolution: Resolution,
    source: SourceText,
    quote: str | None = None,
) -> tuple[Evidence, ...]:
    """The evidence a finding carries: how the work was resolved, and the quote.

    The quote carries exact character offsets when it occurs in the retrieved text (also
    when only its whitespace differs); a quote that appears nowhere is kept as evidence
    without offsets, for the verifier and the human reviewer to judge."""
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
    found = _find_quote(source.text, quote)
    char_start, char_end = found if found is not None else (None, None)
    cited = Evidence(
        kind=EvidenceKind.QUOTE,
        source=SourceRef(
            kind=SourceKind.CITED_WORK,
            id=identifier,
            title=work.title,
            url=work.html_url or work.pdf_url,
        ),
        text=quote,
        char_start=char_start,
        char_end=char_end,
    )
    return (lookup, cited)


def _scan_confident(claimed: str) -> bool:
    """Whether `claimed` is specific enough to scan the source text for.

    A specific number is one with a decimal point or at least three digits ("92.1",
    "100,000"); a bare small number ("3", "4x") is often a derived quantity ("4x fewer
    parameters") the source states in another form, so scanning for it would produce
    false mismatches.
    """
    digits = claimed.replace(",", "").replace(".", "").strip("% ")
    return "." in claimed or len(digits) >= 3


def _source_states(source_text: str, claimed: str) -> bool:
    """Whether any number the claim states appears (by value) in the source's text."""
    source_values = extract_numbers(source_text)
    return any(
        any(numbers_agree(number, stated) for stated in source_values)
        for number in extract_numbers(claimed)
    )


# A whole number as written: its sign, thousands separators, decimal part, exponent and
# percent sign belong to it, so "500" is not read out of "12,500", "2.1" out of "92.1" or
# "4" out of "3e-4". A minus sign counts only where it cannot be a range ("10-20").
_WHOLE_NUMBER = re.compile(
    r"(?<![\w.,])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s?%)?"
    r"(?![\w%]|[.,]\d)"
)
# A model's value that is a number and nothing else (else it is words or has a suffix:
# "eight", "8x", "10-fold").
_JUST_A_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?\s?%?")
_MARKER = re.compile(r"\[\s*\d+(?:\s*[,\u2013-]\s*\d+)*\s*\]")


def _values(text: str) -> set[float]:
    """The values a number may mean: "92.1%" is 0.921, or 92.1 when written without %."""
    values = {_parse_number(text), _parse_number(text.replace("%", ""))}
    return {value for value in values if value is not None}


def _bound(text: str, value: str) -> set[float] | None:
    """What `text` says `value` is, or None when `text` does not state it.

    A number must be a whole number of `text` with exactly its value ("0.0003" for
    "3e-4"); words and numbers with a suffix ("eight", "8x") must stand on their own (not
    "one" in "someone"). The values returned are those of the text as written, so a
    percentage keeps both its readings. No rounding tolerance here: that is for
    comparing two values once both are found."""
    value = value.strip().replace(MINUS_SIGN, "-")
    if not value:
        return None
    # Citation markers ("[1]", "[2, 3]") are numbers of the text but state no value.
    text = _MARKER.sub(" ", text).replace(MINUS_SIGN, "-")
    wanted = _values(value)
    if not _JUST_A_NUMBER.fullmatch(value):
        alone = re.compile(rf"(?<![\w.,]){re.escape(value)}(?![\w]|[.,]\d)", re.IGNORECASE)
        return wanted if alone.search(text) else None
    found: set[float] = set()
    for number in _WHOLE_NUMBER.findall(text):
        values = _values(number)
        if any(math.isclose(a, b, rel_tol=1e-9) for a in values for b in wanted):
            found |= values
    return found or None


def _close(a: float, b: float) -> bool:
    """Two values a paper and its source would both write: equal, or within 1% (they
    round differently)."""
    scale = max(abs(a), abs(b))
    return a == b or (scale > 0 and abs(a - b) / scale < 0.01)


def _number_check(
    task: AuditTask,
    resolution: Resolution,
    source: SourceText,
    check: NumberCheck,
    number: int,
) -> CandidateFinding | None:
    """A `claim_number_mismatch` finding for one number check, or None.

    The code-side value comparison decides, not the model's `agrees` hint: a model that
    says the values agree when they do not is exactly the failure this check exists for.
    The values themselves are the model's reading, so each is checked against its text
    before it counts: `claimed` must be in the claim, and `found` in the quoted sentence,
    which must be in the work's text. A value that is not where the model says it is
    makes no finding (a misread is not a mismatch).

    When the model gives no value (`found` empty) and the source is a complete text, the
    code scans the text itself: a specific number the claim attributes to the work that
    the work's text nowhere states is a mismatch. Against an abstract or a publisher page
    no scan is made (the number may be in the unread full text).

    `number` is the finding's number within the task, so its id is unique even when a
    task yields several findings (a contradiction plus number mismatches).
    """
    claimed = _bound(task.claim or "", check.claimed)
    if claimed is None:
        return None
    if check.found:
        found = _bound(check.quote, check.found)
        if found is None or _find_quote(source.text, check.quote) is None:
            return None
        # Both values as their texts write them; text that is not a number compares as text.
        if claimed and found:
            agree = any(_close(a, b) for a in claimed for b in found)
        else:
            agree = numbers_agree(check.claimed, check.found)
        if not agree:
            return _mismatch_finding(
                task,
                resolution,
                source,
                check,
                number,
                f"The claim gives '{check.claimed}' but the cited work's text gives "
                f"'{check.found}'",
            )
        return None
    if source.partial or source.level not in (SourceLevel.FULL_HTML, SourceLevel.FULL_PDF):
        return None
    if _scan_confident(check.claimed) and not _source_states(source.text, check.claimed):
        return _mismatch_finding(
            task,
            resolution,
            source,
            check,
            number,
            f"The claim gives '{check.claimed}' for the cited work, but the work's full "
            "text nowhere states a matching number",
        )
    return None


def _mismatch_finding(
    task: AuditTask,
    resolution: Resolution,
    source: SourceText,
    check: NumberCheck,
    number: int,
    summary: str,
) -> CandidateFinding:
    return CandidateFinding.for_task(
        task,
        number,
        error_class=ErrorClass.CLAIM_NUMBER_MISMATCH,
        summary=summary,
        explanation=check.quote,
        evidence=_evidence_for(task, resolution, source, check.quote),
    )


async def check_citation(
    task: AuditTask,
    ctx: RunContext,
    resolution: Resolution,
    source: SourceText,
) -> TaskResult:
    """Whether the cited work supports the claim (decision P2.5).

    `resolution` is what the citation resolved to (auditor.retrieval.resolve) and
    `source` the text retrieved for it (auditor.retrieval.fulltext). The model sees
    the TOP_PASSAGES passages, and its answer is checked in code: the quote it returns
    is anchored in the source when it appears there (exactly or up to whitespace), and
    a number mismatch is confirmed by value.
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

    passages = rank_passages(
        source.text,
        task.claim,
        top=TOP_PASSAGES,
        embedder=await get_embedder(ctx.settings, ctx.http),
    )
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
        "Verdicts: supported - the passages back the claim's specific assertion; "
        "contradicted - a passage states something INCOMPATIBLE with the claim's specific "
        "assertion (a different direction of effect, a different value, a different cause); "
        "insufficient - the passages do not settle the claim. Do NOT answer contradicted "
        "merely because a passage describes different conditions, a different model, or "
        "does not mention the claim's specifics: that is insufficient, not a contradiction. "
        "A contradiction needs a passage whose statement cannot both hold with the claim.\n\n"
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
            finding = _number_check(task, resolution, source, entry, len(findings))
            if finding is not None:
                findings.append(finding)

    if not findings and verdict.verdict == "insufficient":
        # Passages that neither back nor contradict the claim leave it unchecked, not
        # clean (decision D6): the report lists it for a human to look up.
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason=(
                f"The retrieved text of the cited work ({source.level}) does not settle the "
                f"claim: {verdict.reason.strip() or 'no reason given'}"
            ),
        )
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
