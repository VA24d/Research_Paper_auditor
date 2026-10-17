"""Do the paper's statements about its own results agree with its tables (decision D22)?

The core data arguments of a paper are its claims about its own numbers: a method is the
best on a benchmark, improves by so many points, reaches a stated value. The model does not
judge these claims. It only reads them: for each claim it names the sentence, the table and
the numbers the claim rests on, copied from the table as written. The arithmetic is then
done here, with each number's rounding taken into account, and a number the model gives is
used only if it really is in the table row it names. So the model's job is extraction,
which a cheap model does well, and the verdict is computed.

Tables are also checked with no model for a value that cannot be right: a negative standard
deviation.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.agents.theory import anchor_quote
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
    Table,
    TaskOutcome,
    TaskResult,
)

MAX_LISTING_CHARS = 150_000
MAX_PLAN_TOKENS = 6144
_MINUS = "\u2212"
_NUMBER = re.compile(r"(?<![\w.])[-\u2212]?\d+(?:\.\d+)?")
_NEGATIVE_STD = re.compile(r"\u00b1\s*[-\u2212]\s*\d+(?:\.\d+)?")

PLAN_PROMPT = """You extract the claims a research paper makes about its own results from its
tables, so that they can be checked with arithmetic. Read the sentences and the tables, and
list the claims that state or compare numbers in a table:
- "value": the text states a number that is in the table;
- "difference": the text states by how much one result exceeds another, in the table's own
  units (points);
- "relative_percent": the text states a relative improvement in percent;
- "best": the text says one result is the best (or the worst, with higher_is_better set
  to false for metrics where lower is better) among the table's results it compares with;
- "comparison": the text says one number is higher (relation "higher") or lower (relation
  "lower") than another, for example "0.73, significantly higher than 0.78". The numbers
  may be in a table or in the sentence itself; for numbers in the sentence, leave table_id
  empty;
- "same_value": the paper reports one result (the same method or configuration, on the same
  data and metric) in two places, such as a baseline in the main table and the matching
  variant of an ablation table, or a number in the text and in a table. Give both numbers,
  each with the table_id of the table it is in.
For each claim give the sentence id, a quote of at least four words copied exactly from the
sentence, the table id, the number the text states (for value, difference and
relative_percent), and the table numbers it rests on: the subject (the row label as written
in the table and the number exactly as written in that row), and for difference and
relative_percent the compared result, for best every result it is compared with in the same
column. Copy numbers from the table or sentence exactly; never compute them. For a claim about
several columns, give one claim per column. Include every sentence, in the abstract too,
that says one stated number is higher, lower, better or worse than another stated number:
these are "comparison" claims even without a table, and a comparison can use numbers from
the one or two sentences just before it. Leave out claims you cannot tie to specific
numbers. List at most 20 claims, the ones the paper's conclusions rest on first."""


class Operand(BaseModel):
    row_label: str
    value: str
    # The table this number is in, when it differs from the claim's table.
    table_id: str = ""


class DataClaim(BaseModel):
    sentence_id: str
    quote: str
    kind: Literal["value", "difference", "relative_percent", "best", "comparison", "same_value"]
    # Empty when the claim's numbers are in the sentence itself, not in a table.
    table_id: str = ""
    stated: str = ""
    subject: Operand
    others: list[Operand] = []
    higher_is_better: bool = True
    # For "comparison": what the text says about the subject's number against the other's.
    relation: Literal["higher", "lower"] = "higher"


class DataClaims(BaseModel):
    claims: list[DataClaim]


@dataclass(frozen=True)
class _Number:
    """A number as written, with the interval its rounding allows."""

    value: float
    half_unit: float

    @classmethod
    def parse(cls, text: str) -> "_Number | None":
        match = _NUMBER.search(text.replace(",", ""))
        if match is None:
            return None
        written = match.group().replace(_MINUS, "-")
        decimals = len(written.split(".")[1]) if "." in written else 0
        return cls(float(written), 0.5 * 10**-decimals)

    @property
    def low(self) -> float:
        return self.value - self.half_unit

    @property
    def high(self) -> float:
        return self.value + self.half_unit


def _rows(paper: ParsedPaper, table: Table) -> list[Sentence]:
    by_id = {s.id: s for s in paper.sentences}
    return [by_id[i] for i in table.row_ids if i in by_id]


def _caption(paper: ParsedPaper, table: Table) -> str:
    label = (table.label or "").strip().rstrip(".").lower()
    if not label:
        return ""
    for sentence in paper.sentences:
        if sentence.kind is SentenceKind.CAPTION and sentence.text.lower().startswith(label):
            return sentence.text
    return ""


def _finding(
    task: AuditTask,
    number: int,
    sentence: Sentence,
    span: tuple[int, int] | None,
    summary: str,
    lookup: str,
    uncertain: bool = False,
) -> CandidateFinding:
    start, end = span if span is not None else (0, len(sentence.text))
    source = SourceRef(kind=SourceKind.SUBMISSION, id=task.paper_id)
    evidence = [
        Evidence(
            kind=EvidenceKind.QUOTE,
            source=source,
            text=sentence.text[start:end],
            char_start=start,
            char_end=end,
        )
    ]
    if lookup:
        evidence.append(Evidence(kind=EvidenceKind.LOOKUP, source=source, text=lookup))
    return CandidateFinding.for_task(
        task,
        number,
        error_class=ErrorClass.RESULT_INCONSISTENCY,
        anchor=SentenceAnchor(
            sentence_id=sentence.id,
            start=start if span is not None else None,
            end=end if span is not None else None,
        ),
        summary=summary,
        evidence=evidence,
        uncertain=uncertain,
    )


def table_findings(task: AuditTask, paper: ParsedPaper, first: int = 0) -> list[CandidateFinding]:
    """Values no table can hold: a negative standard deviation ("75.5 \u00b1 -0.5").

    A percentage over 100 is not checked here: on 195 real papers, every such value was a
    relative measure (normalised speed, a gain) or a number merged into a "%" cell by the
    parser, never an error."""
    findings: list[CandidateFinding] = []
    for table in paper.tables:
        name = table.label or "A table"
        for row in _rows(paper, table):
            for match in _NEGATIVE_STD.finditer(row.text):
                summary = f"{name} gives a negative spread: {match.group()}"
                findings.append(
                    _finding(task, first + len(findings), row, match.span(), summary, "")
                )
    return findings


def _listing(paper: ParsedPaper) -> str:
    labels = [t.label.strip().rstrip(".") for t in paper.tables if t.label]
    lines = []
    for sentence in paper.sentences:
        if sentence.kind not in (SentenceKind.ABSTRACT, SentenceKind.BODY):
            continue
        # Sentences that can state a table's numbers: they hold a digit or name a table.
        if any(c.isdigit() for c in sentence.text) or any(
            label and label in sentence.text for label in labels
        ):
            lines.append(f"[{sentence.id}] {sentence.text}")
    for table in paper.tables:
        rows = _rows(paper, table)
        if not rows:
            continue
        lines.append(f"\n[{table.id}] {table.label or ''} {_caption(paper, table)}".rstrip())
        lines += [row.text for row in rows]
    return "\n".join(lines)[:MAX_LISTING_CHARS]


def _value_in_row(
    rows: Sequence[Sentence], operand: Operand, need_label: bool = True
) -> _Number | None:
    """The operand's number, if a row naming its label (when `need_label`) holds it exactly
    as written."""
    label = operand.row_label.strip().casefold()
    wanted = _Number.parse(operand.value)
    if wanted is None or (need_label and not label):
        return None
    for row in rows:
        if need_label and label not in row.text.casefold():
            continue
        written = {m.group().replace(_MINUS, "-") for m in _NUMBER.finditer(row.text)}
        if operand.value.strip().replace(_MINUS, "-") in written:
            return wanted
    return None


def _verdict(
    claim: DataClaim, rows: Sequence[Sentence], label: str, need_label: bool = True
) -> tuple[str, str] | None:
    """(summary, lookup) when the claim disagrees with its numbers, else None. A claim
    whose numbers are not where the model says (the table row, or the sentence) is not
    judged."""
    subject = _value_in_row(rows, claim.subject, need_label)
    others = [_value_in_row(rows, o, need_label) for o in claim.others]
    if subject is None or any(o is None for o in others):
        return None
    compared = [o for o in others if o is not None]
    if claim.kind == "comparison":
        if len(compared) != 1:
            return None
        other = compared[0]
        wrong = subject.high < other.low if claim.relation == "higher" else subject.low > other.high
        if not wrong:
            return None
        summary = (
            f"The text says {claim.subject.row_label} ({claim.subject.value}) is "
            f"{claim.relation} than {claim.others[0].row_label} ({claim.others[0].value}), "
            "but the numbers say the opposite."
        )
        return summary, f"{claim.subject.value} against {claim.others[0].value}"
    if claim.kind == "best":
        better = [
            (operand, number)
            for operand, number in zip(claim.others, compared, strict=True)
            if (
                number.value > subject.value
                if claim.higher_is_better
                else number.value < subject.value
            )
        ]
        if not better:
            return None
        operand, _ = better[0]
        word = "best" if claim.higher_is_better else "lowest"
        summary = (
            f"The text says {claim.subject.row_label} is the {word}, but in {label} "
            f"{operand.row_label} has {operand.value} against {claim.subject.value}."
        )
        lookup = f"{label}: " + ", ".join(
            f"{o.row_label} {o.value}" for o in [claim.subject, *claim.others]
        )
        return summary, lookup
    stated = _Number.parse(claim.stated)
    if stated is None:
        return None
    if claim.kind == "difference":
        # "-8.6 points" states a drop of 8.6: the size is what is compared.
        stated = _Number(abs(stated.value), stated.half_unit)
    if claim.kind == "value":
        if subject.low <= stated.high and stated.low <= subject.high:
            return None
        summary = (
            f"The text says {claim.stated}, but {label} gives {claim.subject.value} "
            f"for {claim.subject.row_label}."
        )
        return summary, f"{label}: {claim.subject.row_label} {claim.subject.value}"
    if len(compared) != 1:
        return None
    other = compared[0]
    if claim.kind == "difference":
        low = abs(subject.value - other.value) - subject.half_unit - other.half_unit
        high = abs(subject.value - other.value) + subject.half_unit + other.half_unit
        computed = abs(subject.value - other.value)
        unit = " points"
    else:
        corners = [
            (a - b) / b * 100
            for a in (subject.low, subject.high)
            for b in (other.low, other.high)
            if b
        ]
        if not corners:
            return None
        low, high = min(corners), max(corners)
        computed = (subject.value - other.value) / other.value * 100 if other.value else 0.0
        unit = "%"
        # "Improves by 4.8%" often means 4.8 percentage points; either reading is accepted.
        points = abs(subject.value - other.value)
        spread = subject.half_unit + other.half_unit
        if points - spread <= stated.high and stated.low <= points + spread:
            return None
    if low <= stated.high and stated.low <= high:
        return None
    decimals = max(len(claim.stated.split(".")[1]) if "." in claim.stated else 0, 1)
    summary = (
        f"The text says {claim.stated}{unit}, but {label} gives {claim.subject.row_label} "
        f"{claim.subject.value} and {claim.others[0].row_label} {claim.others[0].value}, "
        f"which is {computed:.{decimals}f}{unit}."
    )
    lookup = (
        f"{label}: {claim.subject.row_label} {claim.subject.value}, "
        f"{claim.others[0].row_label} {claim.others[0].value}"
    )
    return summary, lookup


def _same_value(
    claim: DataClaim, paper: ParsedPaper, tables: dict[str, Table]
) -> tuple[str, str] | None:
    """A disagreement between two places that report the same result, each number read
    from its own table."""
    if len(claim.others) != 1:
        return None

    def read(operand: Operand) -> tuple[_Number | None, str]:
        table = tables.get(operand.table_id or claim.table_id)
        if table is None:
            return None, ""
        return _value_in_row(_rows(paper, table), operand), table.label or table.id

    subject, first = read(claim.subject)
    other, second = read(claim.others[0])
    if subject is None or other is None:
        return None
    if subject.low <= other.high and other.low <= subject.high:
        return None
    a, b = claim.subject, claim.others[0]
    summary = (
        f"{first} gives {a.value} for {a.row_label}, but {second} gives {b.value} for "
        f"{b.row_label}, which the paper presents as the same result."
    )
    return summary, f"{first}: {a.row_label} {a.value}; {second}: {b.row_label} {b.value}"


async def check_results(task: AuditTask, ctx: RunContext) -> TaskResult:
    paper = ctx.paper
    findings = table_findings(task, paper)
    if ctx.llm is None:
        return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
    result = await ctx.llm.generate(
        DataClaims,
        [SystemMessage(PLAN_PROMPT), HumanMessage(_listing(paper))],
        effort="low",
        purpose="results-plan",
        max_output_tokens=MAX_PLAN_TOKENS,
    )
    tables = {t.id: t for t in paper.tables}
    body = [s for s in paper.sentences if s.kind in (SentenceKind.ABSTRACT, SentenceKind.BODY)]
    flagged: set[str] = set()
    by_id = {s.id: s for s in body}
    order = {s.id: i for i, s in enumerate(body)}
    for claim in result.value.claims:
        if claim.sentence_id in flagged:
            continue
        table = tables.get(claim.table_id)
        uncertain = False
        if claim.kind == "same_value":
            verdict = _same_value(claim, paper, tables)
            # Which two results are the same one is the model's reading of the paper.
            uncertain = True
        elif table is not None:
            verdict = _verdict(claim, _rows(paper, table), table.label or table.id)
        elif claim.sentence_id in by_id:
            # Numbers in the text: they must be in the sentence or the two before it.
            at = order[claim.sentence_id]
            nearby = body[max(0, at - 2) : at + 1]
            verdict = _verdict(claim, nearby, "the text", need_label=False)
        else:
            continue
        if verdict is None:
            continue
        found = anchor_quote(claim.quote, claim.sentence_id, body)
        if found is None:
            continue
        summary, lookup = verdict
        findings.append(
            _finding(task, len(findings), found.sentence, found.span, summary, lookup, uncertain)
        )
        flagged.add(found.sentence.id)
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
