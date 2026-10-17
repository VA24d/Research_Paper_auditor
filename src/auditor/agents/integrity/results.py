"""Do the paper's statements about its own results agree with its tables (decision D22)?

The core data arguments of a paper are its claims about its own numbers: a method is the
best on a benchmark, improves by so many points, reaches a stated value. The model does not
judge these claims. It only reads them: for each claim it names the sentence, the table and
the numbers the claim rests on, copied from the table as written. The arithmetic is then
done here, with each number's rounding taken into account, and a number the model gives is
used only if it really is in the table row it names. So the model's job is extraction,
which a cheap model does well, and the verdict is computed.

Two checks need no model: a table value that cannot be right (a negative standard
deviation), and a sentence that calls one stated number higher or lower than another when it
is the other way round.
"""

import asyncio
import base64
import io
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from auditor.agents.theory import anchor_quote
from auditor.infra.context import RunContext
from auditor.schemas import (
    CELL_SEPARATOR,
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

logger = logging.getLogger(__name__)

MAX_UNIT_CHARS = 60_000
MAX_PLAN_UNITS = 10
HEADER_ROWS = 3
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
relative_percent), and the table numbers it rests on (always the numbers as the table
gives them, not as the text gives them): the subject (the row label as written
in the table and the number exactly as written in that row), and in others: for difference,
relative_percent and comparison the compared result, for best every result it is compared
with in the same column. Copy numbers from the table or sentence exactly; never compute
them. For a claim about several columns or rows ("all four models gain on GSM8k", "best on
every benchmark"), give one claim per column or row. Include every sentence, in the abstract
too,
that says one stated number is higher, lower, better or worse than another stated number:
these are "comparison" claims even without a table, and a comparison can use numbers from
the one or two sentences just before it. Leave out claims you cannot tie to specific
numbers. You see one section of the paper and the tables it uses. List at most 20 claims, the ones
the paper's conclusions rest on first."""


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
    others: list[Operand] = Field(
        default=[],
        description="The compared results: required for difference, relative_percent, "
        "comparison (the number the subject is said to be higher or lower than) and best",
    )
    higher_is_better: bool = True
    # For "comparison": what the text says about the subject's number against the other's.
    relation: Literal["higher", "lower"] = "higher"


class DataClaims(BaseModel):
    claims: list[DataClaim]


# Pages whose tables the parser could not read are shown to the model as images, at most
# this many per paper.
MAX_TABLE_IMAGES = 6
MAX_IMAGE_TOKENS = 8192
IMAGE_PROMPT = """You transcribe the tables on one page of a research paper. For every table
on the page give its label as printed (such as "Table 3") and its rows, each row a list of
cells copied exactly as printed: numbers with all their digits, signs and units. Write a
spread printed small or as a subscript with its sign, as in "60.5 \u00b1 0.39" or
"74.5 (+1.0)", never run into the number. Include the header rows. Leave out figures and
body text."""

# Two decimal points in one number ("60.50.39"): a spread run into its number, which would
# be read as the wrong value.
_RUN_TOGETHER = re.compile(r"\d+\.\d+\.\d")


class ImageTable(BaseModel):
    label: str = ""
    rows: list[list[str]]


class TableImages(BaseModel):
    tables: list[ImageTable]


def render_page(pdf: Path, page: int) -> bytes:
    """One page of the PDF as a PNG image."""
    import pdfplumber

    with pdfplumber.open(pdf) as document:
        image = document.pages[page - 1].to_image(resolution=150).original
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def _read_unread_tables(paper: ParsedPaper, ctx: RunContext) -> ParsedPaper:
    """`paper` with the tables the parser could not read transcribed from page images."""
    unread = [t for t in paper.tables if t.page is not None and not t.row_ids]
    if not unread or ctx.source_pdf is None or ctx.llm is None:
        return paper
    rows: list[Sentence] = []
    tables: list[Table] = []
    for table in unread[:MAX_TABLE_IMAGES]:
        assert table.page is not None
        try:
            png = await asyncio.to_thread(render_page, ctx.source_pdf, table.page)
        except Exception as exc:  # a broken or missing PDF must not stop the check
            logger.warning("could not render page %s of %s: %s", table.page, ctx.source_pdf, exc)
            continue
        image = base64.b64encode(png).decode("ascii")
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"Page {table.page} of the paper ({table.label})."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
            ]
        )
        result = await ctx.llm.generate(
            TableImages,
            [SystemMessage(IMAGE_PROMPT), message],
            effort="low",
            purpose="table-image",
            max_output_tokens=MAX_IMAGE_TOKENS,
        )
        for k, read in enumerate(result.value.tables):
            if any(_RUN_TOGETHER.search(cell) for row in read.rows for cell in row):
                logger.warning("a table read from page %s runs numbers together", table.page)
                continue
            ids = []
            for r, row in enumerate(read.rows):
                cells = [cell.strip() for cell in row if cell.strip()]
                if not cells:
                    continue
                sentence_id = f"{table.id}-img{k}-{r}"
                ids.append(sentence_id)
                rows.append(
                    Sentence(
                        id=sentence_id, kind=SentenceKind.TABLE, text=CELL_SEPARATOR.join(cells)
                    )
                )
            if ids:
                tables.append(
                    Table(
                        id=f"{table.id}-img{k}",
                        label=read.label or table.label,
                        row_ids=tuple(ids),
                        page=table.page,
                    )
                )
    if not tables:
        return paper
    data = paper.model_dump()
    data["sentences"] = [*data["sentences"], *(s.model_dump() for s in rows)]
    data["tables"] = [*data["tables"], *(t.model_dump() for t in tables)]
    return ParsedPaper.model_validate(data)


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


def _tables(paper: ParsedPaper) -> list[Table]:
    """The tables to check: those read from the PDF's layout or page images, and the
    parser's own tables for labels those do not cover (the same table read twice would be
    checked twice)."""

    def key(table: Table) -> str:
        return (table.label or "").strip().rstrip(".:").casefold()

    read = [t for t in paper.tables if t.page is not None and t.row_ids]
    covered = {key(t) for t in read if t.label}
    parser = [t for t in paper.tables if t.page is None and not (t.label and key(t) in covered)]
    return parser + read


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
    for table in _tables(paper):
        name = table.label or "A table"
        for row in _rows(paper, table):
            for match in _NEGATIVE_STD.finditer(row.text):
                summary = f"{name} gives a negative spread: {match.group()}"
                findings.append(
                    _finding(task, first + len(findings), row, match.span(), summary, "")
                )
    return findings


_COMPARATIVE = re.compile(
    r"\b(higher|greater|larger|more|lower|smaller|less|fewer)\b([^;]*?)\bthan\b([^;]{0,80})",
    re.IGNORECASE,
)
_UP = {"higher", "greater", "larger", "more"}
# A number with a decimal point that is not a bound ("p<0.01") or a spread ("\u00b1 0.13").
_STATED = re.compile(r"(?<![<>=\u00b1\w.])[-\u2212]?\d+\.\d+")


def text_comparisons(task: AuditTask, paper: ParsedPaper, first: int = 0) -> list[CandidateFinding]:
    """A sentence that says one stated number is higher or lower than another, when the
    numbers say the opposite: "higher IoU (0.73 \u00b1 0.13) than U-Net (0.78 \u00b1 0.01)".

    Only numbers with a decimal point are compared, the first after the comparative word
    and the first after "than", so counts and bounds ("more than 10 layers") are left out.
    "Better" and "worse" are left out too, since which way is better depends on the
    metric. On 252 real papers this found one error, in three versions of one paper."""
    findings: list[CandidateFinding] = []
    for sentence in paper.sentences:
        if sentence.kind not in (SentenceKind.ABSTRACT, SentenceKind.BODY):
            continue
        for match in _COMPARATIVE.finditer(sentence.text):
            before, after = _STATED.search(match.group(2)), _STATED.search(match.group(3))
            if before is None or after is None:
                continue
            one, two = _Number.parse(before.group()), _Number.parse(after.group())
            if one is None or two is None:
                continue
            up = match.group(1).lower() in _UP
            if not (one.high < two.low if up else one.low > two.high):
                continue
            word = match.group(1).lower()
            summary = (
                f"The text says {before.group()} is {word} than {after.group()}, but it is "
                f"{'lower' if up else 'higher'}."
            )
            lookup = f"{before.group()} against {after.group()}"
            findings.append(
                _finding(task, first + len(findings), sentence, match.span(), summary, lookup)
            )
            break
    return findings


_TABLE_NUMBER = re.compile(r"\d+\.\d+|\d{3,}")


def _units(paper: ParsedPaper) -> list[str]:
    """The paper cut into listings for the plan step, one per section that can state a
    table's numbers: the section's sentences with digits, and the tables it names or
    quotes. One listing for the whole paper made cheap models skip most claims; a section
    with its own tables keeps the model's attention on them."""
    tables = [t for t in _tables(paper) if _rows(paper, t)]
    numbers = {
        t.id: {m.group() for row in _rows(paper, t) for m in _TABLE_NUMBER.finditer(row.text)}
        for t in tables
    }

    def named(table: Table, text: str) -> bool:
        # "Table 6", "Tab. 6", and "Table . 6" where the parser split a sentence after it.
        found = re.findall(r"(\d+|[IVX]+)\b", table.label or "")
        return any(
            re.search(rf"\bTab(?:le)?\s*\.?\s*{re.escape(n)}\b", text, re.IGNORECASE) for n in found
        )

    sections: dict[str, list[Sentence]] = {}
    for sentence in paper.sentences:
        if sentence.kind in (SentenceKind.ABSTRACT, SentenceKind.BODY):
            key = "abstract" if sentence.kind is SentenceKind.ABSTRACT else sentence.section_id
            sections.setdefault(key or "body", []).append(sentence)
    units: list[tuple[int, str]] = []
    rest: list[str] = []
    for sentences in sections.values():
        text = " ".join(s.text for s in sentences)
        quoted = {m.group() for s in sentences for m in _TABLE_NUMBER.finditer(s.text)}
        used = [t for t in tables if named(t, text) or quoted & numbers[t.id]]
        lines = [f"[{s.id}] {s.text}" for s in sentences if any(c.isdigit() for c in s.text)]
        compares = any(len(re.findall(r"\d\.\d", s.text)) >= 2 for s in sentences)
        if not lines:
            continue
        if not (used or compares):
            rest += lines
            continue
        units.append((len(lines), _with_tables(paper, lines, used)))
    # The sections with the most to check first, when there are more than the limit.
    units.sort(key=lambda unit: -unit[0])
    listings = [listing for _, listing in units[: MAX_PLAN_UNITS - 1]]
    if rest and tables:
        # Sections that neither name nor quote a table may still state a difference or a
        # gain ("improves by 2.7 points"): they are checked once against every table.
        listings.append(_with_tables(paper, rest, tables))
    return listings


def _with_tables(paper: ParsedPaper, lines: list[str], tables: Sequence[Table]) -> str:
    out = list(lines)
    for table in tables:
        out.append(f"\n[{table.id}] {table.label or ''} {_caption(paper, table)}".rstrip())
        out += [row.text for row in _rows(paper, table)]
    return "\n".join(out)[:MAX_UNIT_CHARS]


def _value_in_row(
    rows: Sequence[Sentence], operand: Operand, need_label: bool = True
) -> _Number | None:
    """The operand's number, if a row naming its label (when `need_label`) holds it exactly
    as written. The label may also be a column's, named in the table's header rows."""
    label = operand.row_label.strip().casefold()
    wanted = _Number.parse(operand.value)
    if wanted is None or (need_label and not label):
        return None
    in_header = need_label and any(label in row.text.casefold() for row in rows[:HEADER_ROWS])
    for row in rows:
        if need_label and not in_header and label not in row.text.casefold():
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
        if not _names_something(claim.subject.row_label):
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
        # "Drops by 2.2%" states the size of a change: compare sizes, as for differences.
        stated = _Number(abs(stated.value), stated.half_unit)
        if high < 0:
            low, high = -high, -low
        elif low < 0:
            low, high = 0.0, max(-low, high)
        # "Improves by 4.8%" often means 4.8 percentage points; either reading is accepted.
        points = abs(subject.value - other.value)
        spread = subject.half_unit + other.half_unit
        if points - spread <= stated.high and stated.low <= points + spread:
            return None
    if low <= stated.high and stated.low <= high:
        return None
    said = claim.stated.strip().removesuffix("%").strip()
    decimals = max(len(said.split(".")[1]) if "." in said else 0, 1)
    summary = (
        f"The text says {said}{unit}, but {label} gives {claim.subject.row_label} "
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


CHALLENGE_PROMPT = """You check candidate inconsistencies between a research paper's text
and its tables. For each candidate you get the sentence (with the sentences around it), the
table the numbers were read from, and the arithmetic that disagrees with the text. Decide:
- "inconsistent": the text really disagrees with the table;
- "other_reading": the text's number follows from the table read another way: another
  column or row, another baseline, a relative instead of an absolute change, rounding, or
  a quantity the text defines differently; or the text's number is about something else
  than the table cell (one part of a system, another setting, another unit);
- "misread": the numbers checked are not the ones the sentence is about.
Report "inconsistent" only when you are sure the text and the table are about the same
quantity. Give a short reason for each verdict."""


class ResultVerdict(BaseModel):
    number: int
    verdict: Literal["inconsistent", "other_reading", "misread"]
    reason: str = ""


class ResultVerdicts(BaseModel):
    verdicts: list[ResultVerdict]


@dataclass(frozen=True)
class _Candidate:
    finding: CandidateFinding
    # The sentence with the two before and after it.
    context: str
    rows: str


async def _challenge(candidates: Sequence[_Candidate], ctx: RunContext) -> list[CandidateFinding]:
    """The candidates a second model, shown each one's sentence and table, still finds
    inconsistent. The arithmetic is exact, but the numbers it uses are the first model's
    reading of the paper, which can be the wrong column or the wrong comparison."""
    if not candidates:
        return []
    listing = "\n\n".join(
        f"{n}. {c.finding.summary}\nText: {c.context}\nTable:\n{c.rows}"
        for n, c in enumerate(candidates, start=1)
    )
    result = await ctx.require_review_llm().generate(
        ResultVerdicts,
        [SystemMessage(CHALLENGE_PROMPT), HumanMessage(listing)],
        effort="medium",
        purpose="results-challenge",
        max_output_tokens=MAX_PLAN_TOKENS,
    )
    dropped = {v.number for v in result.value.verdicts if v.verdict in ("other_reading", "misread")}
    return [c.finding for n, c in enumerate(candidates, start=1) if n not in dropped]


def _names_something(label: str) -> bool:
    """A row label with a word in it. A bare number ("128", a setting's value) names a row
    too weakly to say which quantity a number in the text is about."""
    return any(c.isalpha() for c in label)


def _row_numbers(rows: Sequence[Sentence], label: str) -> list[str] | None:
    """The numbers in the cells after the first of the one row whose text names `label`,
    or None when no row or several rows name it."""
    wanted = label.strip().casefold()
    named = [row for row in rows[1:] if wanted and wanted in row.text.casefold()]
    if len(named) != 1:
        return None
    cells = named[0].text.split(CELL_SEPARATOR)[1:]
    return [m.group() for cell in cells for m in _NUMBER.finditer(cell.replace(",", ""))]


def _text_against_rows(
    claim: DataClaim, paper: ParsedPaper, context: Sequence[Sentence]
) -> tuple[str, str] | None:
    """A number the text gives for a table row that the row does not hold. The model
    sometimes copies the text's numbers into a claim instead of the table's; when the row
    it names has a single number, that number is what the text should say."""
    said = {m.group().replace(_MINUS, "-") for s in context for m in _NUMBER.finditer(s.text)}
    candidates = [t for t in _tables(paper) if t.id == claim.table_id] or _tables(paper)
    for operand in [claim.subject, *claim.others]:
        value = operand.value.strip().replace(_MINUS, "-")
        stated = _Number.parse(value)
        if stated is None or value not in said or not _names_something(operand.row_label):
            continue
        for table in candidates:
            numbers = _row_numbers(_rows(paper, table), operand.row_label)
            if numbers is None or len(numbers) != 1:
                continue
            held = _Number.parse(numbers[0])
            if held is None or (held.low <= stated.high and stated.low <= held.high):
                break
            label = table.label or table.id
            summary = (
                f"The text gives {operand.value} for {operand.row_label}, but {label} "
                f"gives {numbers[0]}."
            )
            return summary, f"{label}: {operand.row_label} {numbers[0]}"
    return None


def _table_holding(claim: DataClaim, paper: ParsedPaper) -> Table | None:
    """The table whose rows hold all of a claim's numbers under their labels, for a claim
    the model did not tie to a table id. Numbers stated in the sentence itself have row
    labels that are not in any table, so they are not matched."""
    if claim.kind == "same_value":
        return None
    for table in _tables(paper):
        rows = _rows(paper, table)
        operands = [claim.subject, *claim.others]
        if rows and all(_value_in_row(rows, operand) is not None for operand in operands):
            return table
    return None


def _around(body: Sequence[Sentence], order: dict[str, int], finding: CandidateFinding) -> str:
    """The finding's sentence with the two before and after it."""
    if not isinstance(finding.anchor, SentenceAnchor) or finding.anchor.sentence_id not in order:
        return ""
    at = order[finding.anchor.sentence_id]
    return " ".join(s.text for s in body[max(0, at - 2) : at + 3])


async def check_results(task: AuditTask, ctx: RunContext) -> TaskResult:
    paper = ctx.paper
    findings = table_findings(task, paper)
    comparisons = text_comparisons(task, paper, len(findings))
    if ctx.llm is None:
        findings += comparisons
        return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
    paper = await _read_unread_tables(paper, ctx)
    llm = ctx.llm
    plans = await asyncio.gather(
        *(
            llm.generate(
                DataClaims,
                [SystemMessage(PLAN_PROMPT), HumanMessage(unit)],
                effort="low",
                purpose="results-plan",
                max_output_tokens=MAX_PLAN_TOKENS,
            )
            for unit in _units(paper)
        )
    )
    claims = [claim for plan in plans for claim in plan.value.claims]
    tables = {t.id: t for t in paper.tables}
    body = [s for s in paper.sentences if s.kind in (SentenceKind.ABSTRACT, SentenceKind.BODY)]
    by_id = {s.id: s for s in body}
    order = {s.id: i for i, s in enumerate(body)}
    # The comparisons the rule found go to the challenge step too: the rule cannot tell a
    # result ("higher IoU (0.73) than ...") from a setting ("lower with 0.5 dropout").
    candidates = [
        _Candidate(f, _around(body, order, f), "(numbers from the text)") for f in comparisons
    ]
    flagged = {f.anchor.sentence_id for f in comparisons if isinstance(f.anchor, SentenceAnchor)}
    for claim in claims:
        if claim.sentence_id in flagged:
            continue
        table = tables.get(claim.table_id) or _table_holding(claim, paper)
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
        if verdict is None and claim.kind != "same_value" and claim.sentence_id in by_id:
            at = order[claim.sentence_id]
            verdict = _text_against_rows(claim, paper, body[max(0, at - 2) : at + 1])
        if verdict is None:
            continue
        found = anchor_quote(claim.quote, claim.sentence_id, body)
        if found is None:
            continue
        summary, lookup = verdict
        used = [tables[i] for i in {claim.table_id, claim.subject.table_id} if i in tables]
        used = used or ([table] if table is not None else [])
        rows = "\n".join(
            f"{t.label or t.id} {_caption(paper, t)}".strip()
            + "\n"
            + "\n".join(row.text for row in _rows(paper, t))
            for t in used
        )
        number = len(findings) + len(candidates)
        candidate = _finding(task, number, found.sentence, found.span, summary, lookup, uncertain)
        context = _around(body, order, candidate)
        candidates.append(_Candidate(candidate, context, rows or "(numbers from the text)"))
        flagged.add(found.sentence.id)
    if ctx.settings.results_challenge_enabled:
        kept = await _challenge(candidates, ctx)
    else:
        kept = [c.finding for c in candidates]
    findings += kept
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
