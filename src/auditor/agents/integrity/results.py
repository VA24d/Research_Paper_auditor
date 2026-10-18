"""Do the paper's statements about its own results agree with its tables (decision D22)?

The core data arguments of a paper are its claims about its own numbers: a method is the
best on a benchmark, improves by so many points, reaches a stated value. The model does not
judge these claims. It only reads them: for each claim it names the sentence, the table and
the numbers the claim rests on, copied from the table as written. The arithmetic is then
done here, with each number's rounding taken into account, and a number the model gives is
used only if it really is in the table row it names. So the model's job is extraction,
which a cheap model does well, and the verdict is computed. A second pass reads the
sentences that quote numbers from a named table or row once more, with only those tables,
and pairs each number with the cell it comes from; the comparison is again done here.

Several checks need no model: a table value that cannot be right (a negative standard
deviation), a sentence that calls one stated number higher or lower than another when it is
the other way round, and a number a sentence gives for a row of a table that no cell of the
row holds (the cell rule).
"""

import asyncio
import base64
import io
import logging
import math
import re
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from auditor.agents.theory import anchor_quote
from auditor.infra.context import RunContext
from auditor.parser.table_crops import add_crop_tables
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
# What comes before a spread: "60.5 \u00b1 0.39", "60.5 +- 0.39".
_SPREAD_SIGNS = ("\u00b1", "+-", "+/-")
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
    # A table cut from its page box beats the layout text's reading of the same label.
    cropped = {key(t) for t in read if t.id.startswith("crop-") and t.label}
    read = [t for t in read if t.id.startswith("crop-") or key(t) not in cropped]
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


_MEAN_SD = re.compile(r"(?<![\w.])(\d+\.\d+)\s*(?:\u00b1|\+-|\+/-)\s*(\d+\.\d+)")
_AVERAGE_HEADER = re.compile(r"^(?:avg|average|mean)\.?$", re.IGNORECASE)
_TOTAL_HEADER = re.compile(r"^(?:total|sum)\.?$", re.IGNORECASE)
_GRIM_MEAN = re.compile(r"\b(?:M|[Mm]ean|[Aa]verage)\s*(?:=|of|was|is)?\s*(\d+\.\d{2})(?!\d)")
_GRIM_N = re.compile(r"\b[Nn]\s*=\s*(\d+)\b")
# GRIM says something only when a mean's last digit is finer than one answer in N.
GRIM_MAX_N = 100


def repeated_spreads(task: AuditTask, paper: ParsedPaper, first: int = 0) -> list[CandidateFinding]:
    """One mean and spread ("85.31 \u00b1 1.27") given by two rows of a table in the same
    column, when the two rows agree on nothing else: two runs rarely agree to every digit
    of both, so it is often a copied cell. Rows that share other cells too (a setting
    repeated by design, a baseline listed twice) and values shared by more than two rows
    are left alone. Uncertain."""
    findings: list[CandidateFinding] = []
    for table in _tables(paper):
        name = table.label or "A table"
        rows = [(row, row.text.split(CELL_SEPARATOR)) for row in _rows(paper, table)]
        width = max((len(cells) for _, cells in rows), default=0)
        for column in range(1, width):
            holders: dict[tuple[str, str], list[tuple[Sentence, list[str]]]] = {}
            for row, cells in rows:
                if column >= len(cells):
                    continue
                match = _MEAN_SD.search(cells[column])
                if match is None:
                    continue
                mean, spread = match.groups()
                if float(spread) == 0 or len(mean.replace(".", "").lstrip("0")) < 3:
                    continue
                holders.setdefault((mean, spread), []).append((row, cells))
            for (mean, spread), pair in holders.items():
                if len(pair) != 2:
                    continue
                (a, cells_a), (b, cells_b) = pair
                shared = sum(
                    1
                    for i in range(1, min(len(cells_a), len(cells_b)))
                    if i != column
                    and cells_a[i].strip()
                    and cells_a[i].strip() == cells_b[i].strip()
                )
                if shared:
                    continue
                value = cells_b[column].strip()
                start = b.text.find(value)
                span = (start, start + len(value)) if start >= 0 else None
                label_a, label_b = cells_a[0].strip(), cells_b[0].strip()
                summary = (
                    f"{name} gives {mean} \u00b1 {spread} for both {label_a} and {label_b} "
                    "in one column, though the rows differ everywhere else: a copied cell?"
                )
                lookup = f"{name}: {a.text} / {b.text}"
                findings.append(
                    _finding(task, first + len(findings), b, span, summary, lookup, True)
                )
    return findings


def _cell_values(row: Sentence) -> list[_Number | None]:
    """The first number of each cell after the label cell, or None for a cell without."""
    cells = row.text.split(CELL_SEPARATOR)[1:]
    return [_Number.parse(cell) if _NUMBER.search(cell) else None for cell in cells]


def wrong_aggregates(task: AuditTask, paper: ParsedPaper, first: int = 0) -> list[CandidateFinding]:
    """A row whose last cell, headed "Avg", "Mean", "Total" or "Sum", is not the mean or
    the sum of the row's other cells, in a table where most rows' last cells are. A
    column that no row averages plainly (another metric, a weighted mean, a subset) is
    left alone, so only an outlier among rows that agree is reported. Uncertain."""
    findings: list[CandidateFinding] = []
    for table in _tables(paper):
        rows = _rows(paper, table)
        header = next(
            (
                row
                for row in rows[:HEADER_ROWS]
                if _AVERAGE_HEADER.match(row.text.split(CELL_SEPARATOR)[-1].strip())
                or _TOTAL_HEADER.match(row.text.split(CELL_SEPARATOR)[-1].strip())
            ),
            None,
        )
        if header is None:
            continue
        cells = header.text.split(CELL_SEPARATOR)
        total = bool(_TOTAL_HEADER.match(cells[-1].strip()))
        checked: list[tuple[Sentence, _Number, float, int]] = []
        for row in rows[rows.index(header) + 1 :]:
            values = _cell_values(row)
            if len(values) != len(cells) - 1 or any(v is None for v in values):
                continue
            *parts, stated = [v for v in values if v is not None]
            if len(parts) < 2:
                continue
            computed = sum(p.value for p in parts)
            slack = stated.half_unit + sum(p.half_unit for p in parts)
            if not total:
                computed /= len(parts)
                slack = stated.half_unit + max(p.half_unit for p in parts)
            wrong = abs(computed - stated.value) > slack + 1e-9
            checked.append((row, stated, computed, len(parts)) if wrong else (row, stated, -1.0, 0))
        wrong_rows = [c for c in checked if c[3]]
        agree = len(checked) - len(wrong_rows)
        if len(checked) < 3 or agree < 2 * len(wrong_rows) or agree < 2:
            continue
        name = table.label or "A table"
        word = "total" if total else "average"
        for row, stated, computed, count in wrong_rows:
            label = row.text.split(CELL_SEPARATOR)[0].strip()
            summary = (
                f"{name} gives {label} an {word} of {stated.value:g}, but its other "
                f"{count} cells give {computed:.{_decimals(stated)}f}, while the other rows' "
                f"{word}s agree with their cells."
            )
            findings.append(
                _finding(task, first + len(findings), row, None, summary, row.text, True)
            )
    return findings


def _decimals(number: _Number) -> int:
    return max(0, round(-math.log10(number.half_unit * 2)))


def grim_findings(task: AuditTask, paper: ParsedPaper, first: int = 0) -> list[CandidateFinding]:
    """The GRIM test: a mean of N whole-number answers is a multiple of 1/N, so a mean
    given to two decimals with N under 100 can be impossible ("M = 3.47, N = 28").
    Uncertain: a mean over several items per person, or of non-integer data, has a finer
    grain."""
    findings: list[CandidateFinding] = []
    for sentence in paper.sentences:
        if sentence.kind not in (SentenceKind.ABSTRACT, SentenceKind.BODY):
            continue
        means = list(_GRIM_MEAN.finditer(sentence.text))
        sizes = {int(m.group(1)) for m in _GRIM_N.finditer(sentence.text)}
        if len(means) != 1 or len(sizes) != 1:
            continue
        (n,) = sizes
        mean = float(means[0].group(1))
        if not 1 < n < GRIM_MAX_N:
            continue
        nearest = round(mean * n) / n
        if abs(nearest - mean) <= 0.005 + 1e-9:
            continue
        summary = (
            f"A mean of {means[0].group(1)} is not possible for {n} whole-number values "
            f"(GRIM): the nearest possible mean is {nearest:.2f}."
        )
        findings.append(
            _finding(task, first + len(findings), sentence, means[0].span(1), summary, "", True)
        )
    return findings


_COMPARATIVE = re.compile(
    r"\b(higher|greater|larger|more|lower|smaller|less|fewer)\b([^;]*?)\bthan\b([^;]{0,80})",
    re.IGNORECASE,
)
_UP = {"higher", "greater", "larger", "more"}
# How far before a comparative word its number may be ("reaches 54.81, higher than").
LEAD_CHARS = 24
_LEAD_NUMBER = re.compile(r"(?<![<>=\u00b1\w.])[-\u2212]?\d+\.\d+(?=\)?,\s*$)")
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
            if before is None:
                # "X reaches 54.81, higher than Y (56.06)": the number just before the
                # comparative word, in the same clause.
                # Only a level set off by a comma counts: "3.16% higher" is a difference.
                lead = sentence.text[max(0, match.start() - LEAD_CHARS) : match.start()]
                before = _LEAD_NUMBER.search(lead)
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


def _names_table(table: Table, text: str) -> bool:
    """Whether `text` names the table: "Table 6", "Tab. 6", and "Table . 6" where the
    parser split a sentence after it."""
    found = re.findall(r"(\d+|[IVX]+)\b", table.label or "")
    return any(
        re.search(rf"\bTab(?:le)?\s*\.?\s*{re.escape(n)}\b", text, re.IGNORECASE) for n in found
    )


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
        used = [t for t in tables if _names_table(t, text) or quoted & numbers[t.id]]
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
    rows: Sequence[Sentence], operand: Operand, need_label: bool = True, columns: bool = True
) -> _Number | None:
    """The operand's number, if a row naming its label (when `need_label`) holds it exactly
    as written. The label may also be a column's, named in the table's header rows, unless
    `columns` is false."""
    label = operand.row_label.strip().casefold()
    wanted = _Number.parse(operand.value)
    if wanted is None or (need_label and not label):
        return None
    in_header = (
        need_label and columns and any(label in row.text.casefold() for row in rows[:HEADER_ROWS])
    )
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
    # "Drops by 2.2%" states the size of a change: sizes are compared.
    stated = _Number(abs(stated.value), stated.half_unit)
    # When the sentence itself gives both numbers, it has said which column it means.
    pinned = claim.subject.value in claim.quote and claim.others[0].value in claim.quote
    if _fits(claim.kind, stated, subject, other) or (
        not pinned and _fits_another_column(claim, rows, stated)
    ):
        return None
    if claim.kind == "difference":
        computed = abs(subject.value - other.value)
        unit = " points"
    else:
        if not other.value:
            return None
        computed = (subject.value - other.value) / other.value * 100
        unit = "%"
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


def _sizes(changes: Sequence[float]) -> tuple[float, float]:
    """The interval of sizes of changes ranging over `changes`."""
    low, high = min(changes), max(changes)
    if high < 0:
        return -high, -low
    if low < 0:
        return 0.0, max(-low, high)
    return low, high


def _fits(kind: str, stated: _Number, subject: _Number, other: _Number) -> bool:
    """Whether the size of change the text states follows from the two numbers, within
    their rounding. A relative change may be taken against either number ("degrades by
    34%" from 0.0335 to 0.0514 is 34% of 0.0514), and "improves by 4.8%" often means 4.8
    percentage points; every one of these readings is accepted."""
    spread = subject.half_unit + other.half_unit
    points = abs(subject.value - other.value)
    if points - spread <= stated.high and stated.low <= points + spread:
        return True
    if kind == "difference":
        return False
    pairs = [(a, b) for a in (subject.low, subject.high) for b in (other.low, other.high)]
    for changes in (
        [(a - b) / b * 100 for a, b in pairs if b],
        [(a - b) / a * 100 for a, b in pairs if a],
    ):
        if changes:
            low, high = _sizes(changes)
            if low <= stated.high and stated.low <= high:
                return True
    return False


def _fits_another_column(claim: DataClaim, rows: Sequence[Sentence], stated: _Number) -> bool:
    """Whether the stated change follows from the two rows' numbers in some other column
    of the table: the first model often reads the right rows in the wrong column, such as
    another setting's "all" column."""
    if len(claim.others) != 1:
        return False
    mine = _cell_numbers(rows, claim.subject.row_label)
    theirs = _cell_numbers(rows, claim.others[0].row_label)
    if mine is None or theirs is None or len(mine) != len(theirs):
        return False
    return any(
        _fits(claim.kind, stated, subject, other)
        for subject, other in zip(mine, theirs, strict=True)
        if subject is not None and other is not None
    )


def _cell_numbers(rows: Sequence[Sentence], label: str) -> list[_Number | None] | None:
    """The results in the one row naming `label`, in order, after the label, or None when
    no row or several rows name it. Cell borders are not used, since extraction often
    merges cells; a spread after "±" is not a result."""
    wanted = label.strip().casefold()
    named = [row for row in rows[1:] if wanted and wanted in row.text.casefold()]
    if len(named) != 1:
        return None
    text = named[0].text
    after = text[text.casefold().index(wanted) + len(wanted) :].replace(",", "")
    return [
        _Number.parse(m.group())
        for m in _NUMBER.finditer(after)
        if not after[: m.start()].rstrip().endswith(_SPREAD_SIGNS)
    ]


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


def _completed(claim: DataClaim, sentence: Sentence) -> DataClaim:
    """A gain the model left without the compared result, completed from its sentence
    when that gives exactly the two results ("by 3.6 points (92.83 against 90.73)"): the
    subject's number and one other, then checked against the text."""
    if claim.kind not in ("difference", "relative_percent") or claim.others:
        return claim
    stated = claim.stated.strip().removesuffix("%").strip()
    numbers = [m.group() for m in _STATED.finditer(sentence.text) if m.group() != stated]
    if len(numbers) != 2 or claim.subject.value not in numbers:
        return claim
    other = next(n for n in numbers if n != claim.subject.value)
    return claim.model_copy(
        update={
            "table_id": "",
            "subject": claim.subject.model_copy(update={"table_id": ""}),
            "others": [Operand(row_label="", value=other)],
        }
    )


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


def _rule_table(paper: ParsedPaper, finding: CandidateFinding) -> str:
    """The table holding a table rule's finding, with its caption."""
    if not isinstance(finding.anchor, SentenceAnchor):
        return ""
    for table in _tables(paper):
        if finding.anchor.sentence_id in table.row_ids:
            rows = "\n".join(row.text for row in _rows(paper, table))
            return f"{table.label or table.id} {_caption(paper, table)}".strip() + "\n" + rows
    return ""


def _around(body: Sequence[Sentence], order: dict[str, int], finding: CandidateFinding) -> str:
    """The finding's sentence with the two before and after it."""
    if not isinstance(finding.anchor, SentenceAnchor) or finding.anchor.sentence_id not in order:
        return ""
    at = order[finding.anchor.sentence_id]
    return " ".join(s.text for s in body[max(0, at - 2) : at + 3])


NUMBER_PROMPT = """You pair the numbers a research paper's text quotes from its tables with
the table cells they come from. You get some sentences of the paper, each with its id, and
the tables they name or whose rows they mention. For every number with a decimal point in
the sentences, give:
- the sentence id and the number exactly as the sentence writes it;
- a quote of at least four words copied exactly from the sentence, around the number;
- when the text means the number to be read from one of the tables (a result in a cell):
  the table id, the row label exactly as the table writes it (the first cell of the row),
  the column header of the cell, and the value the table gives in that cell, copied
  exactly from the table with all its digits, even when it differs from the sentence's
  number. Never put the sentence's number there unless the cell holds it.
For a number that is not read from a table cell (a count, a setting, a difference or gain
computed from two cells, a number from another paper, a value the tables do not have),
leave table_id, row_label, column and table_value empty."""

# Sentences read together in one call of the number pass, and the tables shown for one
# sentence: those it names first, then those holding the row labels it mentions.
NUMBER_BATCH = 4
MAX_NUMBER_TABLES = 3
MAX_NUMBER_TOKENS = 8192
MAX_LABEL_CHARS = 40
_DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?![\d.])")


class TableNumber(BaseModel):
    sentence_id: str
    number: str
    quote: str = ""
    table_id: str = ""
    row_label: str = ""
    column: str = ""
    # Empty when the number is not read from a table cell.
    table_value: str = ""


class TableNumbers(BaseModel):
    numbers: list[TableNumber]


def _row_label(row: Sentence) -> str | None:
    """The label of a results row: its first cell, when the cell has a word and is short,
    and the row holds a decimal number (which leaves out header rows)."""
    cells = row.text.split(CELL_SEPARATOR)
    label = cells[0].strip()
    if len(cells) < 2 or len(label) > MAX_LABEL_CHARS or not re.search(r"[A-Za-z]{2}", label):
        return None
    if not any(_DECIMAL.search(cell) for cell in cells[1:]):
        return None
    return label


def _mentions(label: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(label)}(?!\w)", text) is not None


def number_candidates(
    paper: ParsedPaper, skip: set[str], limit: int
) -> list[tuple[Sentence, list[Table]]]:
    """The prose sentences the number pass reads, each with the tables it uses: sentences
    with a decimal number that name a table ("Table 3", "Tab. 3") or mention the label of
    a row of some table. Sentences in `skip` are left out. At most `limit`, the ones naming
    a table first, each group in reading order."""
    tables = [t for t in _tables(paper) if _rows(paper, t)]
    labels = {
        t.id: {label for row in _rows(paper, t) if (label := _row_label(row))} for t in tables
    }
    naming: list[tuple[Sentence, list[Table]]] = []
    holding: list[tuple[Sentence, list[Table]]] = []
    for sentence in paper.sentences:
        if sentence.kind not in (SentenceKind.ABSTRACT, SentenceKind.BODY):
            continue
        if sentence.id in skip or not _DECIMAL.search(sentence.text):
            continue
        named = [t for t in tables if _names_table(t, sentence.text)]
        matched = [
            t
            for t in tables
            if t not in named and any(_mentions(label, sentence.text) for label in labels[t.id])
        ]
        used = (named + matched)[:MAX_NUMBER_TABLES]
        if used:
            (naming if named else holding).append((sentence, used))
    return (naming + holding)[:limit]


def _rescaled(stated: _Number, held: _Number) -> bool:
    """Whether the text's number is the cell's in another scale or sign: 68.76% for
    0.6876, or -3.2 for a drop the table gives as 3.2."""
    return any(
        abs(held.low) * factor - 1e-9 <= abs(stated.high)
        and abs(stated.low) <= abs(held.high) * factor + 1e-9
        for factor in (1, 100, 0.01)
    )


def _number_verdict(
    item: TableNumber, sentence: Sentence, shown: Sequence[Table], paper: ParsedPaper
) -> tuple[DataClaim, Table, tuple[str, str]] | None:
    """The `value` claim a reply makes, its table and the disagreement, when the cell's
    value is really in the row the reply names and the text's number differs from it
    beyond rounding."""
    number = item.number.strip().replace(_MINUS, "-")
    written = {m.group().replace(_MINUS, "-") for m in _NUMBER.finditer(sentence.text)}
    if not item.table_value.strip() or number not in written:
        return None
    claim = DataClaim(
        sentence_id=sentence.id,
        quote=item.quote,
        kind="value",
        stated=number,
        subject=Operand(row_label=item.row_label, value=item.table_value.strip()),
    )
    for table in [t for t in shown if t.id == item.table_id] or list(shown):
        rows = _rows(paper, table)
        # The label must be the row's: a column's label would let any row's cell count.
        if _value_in_row(rows, claim.subject, columns=False) is None:
            continue
        stated, held = _Number.parse(number), _Number.parse(claim.subject.value)
        if stated is None or held is None or _rescaled(stated, held):
            return None
        # The row holds the text's number too: the text quotes another of its cells.
        if _value_in_row(rows, Operand(row_label=item.row_label, value=number)) is not None:
            return None
        claim = claim.model_copy(update={"table_id": table.id})
        verdict = _verdict(claim, rows, table.label or table.id)
        if verdict is None:
            return None
        summary, lookup = verdict
        if item.column.strip():
            summary = f"{summary.removesuffix('.')} ({item.column.strip()})."
        return claim, table, (summary, lookup)
    return None


async def _number_pass(
    paper: ParsedPaper, skip: set[str], ctx: RunContext
) -> list[tuple[DataClaim, Table, tuple[str, str]]]:
    """The second extraction pass: sentences that quote numbers from a table, read with
    only the tables they use, and each number paired with its cell. The per-section
    extraction seldom lists such a sentence as a claim; this pass asks about nothing else.
    The model only reads the cells; whether the text agrees is computed."""
    settings = ctx.settings
    if ctx.llm is None or not settings.results_number_pass_enabled:
        return []
    picked = number_candidates(paper, skip, settings.results_number_max_sentences)
    batches = [picked[i : i + NUMBER_BATCH] for i in range(0, len(picked), NUMBER_BATCH)]
    listings = []
    for batch in batches:
        lines = [f"[{s.id}] {s.text}" for s, _ in batch]
        used = list({t.id: t for _, shown in batch for t in shown}.values())
        listings.append(_with_tables(paper, lines, used))
    llm = ctx.require_review_llm() if settings.results_number_model == "review" else ctx.llm
    replies = await asyncio.gather(
        *(
            llm.generate(
                TableNumbers,
                [SystemMessage(NUMBER_PROMPT), HumanMessage(listing)],
                effort=settings.results_number_effort,
                purpose="results-numbers",
                max_output_tokens=MAX_NUMBER_TOKENS,
            )
            for listing in listings
        )
    )
    shown = {s.id: (s, tables) for s, tables in picked}
    out: list[tuple[DataClaim, Table, tuple[str, str]]] = []
    done: set[str] = set()
    for reply in replies:
        for item in reply.value.numbers:
            if item.sentence_id not in shown or item.sentence_id in done:
                continue
            sentence, tables = shown[item.sentence_id]
            found = _number_verdict(item, sentence, tables, paper)
            if found is not None:
                out.append(found)
                done.add(sentence.id)
    return out


# The cell rule: a number a sentence gives for a row of a table it names, which no cell of
# that row holds. No model reads anything; every condition below is there to keep false
# alarms out, so the rule fires only on clear cases.
#
# How far after a row label a number may be and still be that row's ("BASE performs the
# worst (68.76% in average ACC and 74.77% in average MF1)").
CELL_WINDOW = 80
# A table named in the sentence just before, in the same section, counts as named ("...,
# Table 2). The other mismatch is ChatGPT against PPO, ... (51.4% vs 61.4%)").
CELL_REACH = 1
# Labels that name no one row: they recur in tables of every kind and in plain prose.
_GENERIC_LABELS = frozenset(
    {
        "all",
        "average",
        "avg",
        "baseline",
        "baselines",
        "best",
        "default",
        "full",
        "human",
        "humans",
        "max",
        "mean",
        "median",
        "min",
        "none",
        "ours",
        "overall",
        "random",
        "std",
        "sum",
        "test",
        "total",
        "train",
        "val",
        "validation",
    }
)
# A decimal number in prose, with the sign or bound written before it.
_CELL_TEXT_NUMBER = re.compile(
    r"(?<![\w.,])([-\u2212+\u00b1~\u2248<>\u2264\u2265\u0394])?\s?(\d+\.\d+)(?![\d.]|,\d)"
)
# Words between a label and a number that make the number another row's, a change, a
# bound or a number from elsewhere.
_CELL_BETWEEN = re.compile(
    r"\b(?:than|over|vs|versus|compared|against|relative|from|whereas|while|but|unlike|"
    r"except|instead|by|outperform\w*|surpass\w*|exceed\w*|beat\w*|margin|improv\w*|"
    r"gain\w*|boost\w*|increas\w*|decreas\w*|reduc\w*|drop\w*|degrad\w*|lag\w*|behind|"
    r"gap|differ\w*|delta|up to|at least|at most|about|around|approximately|nearly|"
    r"almost|roughly|between|reported|report|reports|et al)\b|\[\d[\d,\s\u2013-]*\]|;",
    re.IGNORECASE,
)
# What a number is followed by when it is a change, a ratio, a range or a count of things.
_CELL_AFTER = re.compile(
    r"^\s*%?\s*(?:points?\b|pts?\b|pp\b|p\.p\.|percentage|absolute|relative|improvement|"
    r"gain|increase|decrease|drop|reduction|higher|lower|better|worse|more|less|fewer|"
    r"faster|slower|smaller|larger|times|fold|x\b|\u00d7|[-\u2013\u2014~]\s*\d|to\s+\d|"
    r"\u00b1|\+-|(?:[KMBGT]|k|ms|s|sec|h|hrs?|hours?|mins?|minutes?|days?|[KMGT]B|epochs?|"
    r"steps?|iterations?|samples?|tokens?|words?|layers?|params?|parameters|[GT]?FLOPs?|"
    r"Hz|kHz|fps|FPS|px|pixels?|cm|mm|m|km|kg|g|bits?|bpp)\b)",
)
# A comparison after the number, in its clause.
_CELL_THAN = re.compile(r"^[^;.]{0,60}?\bthan\b")
# What comes just before a number in a range or a sum ("71.2-73.5", "between 3.2 and").
_CELL_RANGE_BEFORE = re.compile(
    r"(?:\d\s*(?:-|\u2013|\u2014|~|to)\s*|\bbetween\s+[\d.]+\s+and\s+)$"
)
_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "and", "in", "on", "with", "to", "for", "is", "are", "was"}
    | {"were", "we", "our", "by", "at", "as", "that", "this", "it", "its"}
)


def _label_key(table: Table) -> str:
    return (table.label or table.id).strip().rstrip(".:").casefold()


def _prose_sentence(text: str) -> bool:
    """A sentence of words, not a table run into the body text by the parser: most of its
    tokens are words, and it has the small words of a sentence."""
    tokens = text.split()
    words = [w for w in tokens if re.search(r"[A-Za-z]{3}", w)]
    small = sum(1 for w in tokens if w.casefold() in _STOPWORDS)
    if len(words) < 6 or not small:
        return False
    return (small >= 2 and len(words) >= 0.5 * len(tokens)) or len(words) >= 0.6 * len(tokens)


def _label_forms(label: str) -> list[str]:
    """The ways the text may write a row's label: as the table writes it, and without a
    citation, a footnote mark or a note in brackets after it ("BERT [2]", "ChatGPT*\u2020",
    "HDN (Ours)")."""
    forms = [label.strip()]
    bare = re.sub(r"\s*\[[^\]]*\]\s*$", "", forms[0])
    bare = re.sub(r"[\s*\u2020\u2021\u00a7\u00b6#\u2217]+$", "", bare)
    forms.append(bare)
    forms.append(re.sub(r"\s*\([^)]*\)\s*$", "", bare).strip())
    out = []
    for form in forms:
        if form in out or form.casefold() in _GENERIC_LABELS or len(form) < 2:
            continue
        # A label in small letters only ("random", "train") is too often a plain word.
        if not re.search(r"[A-Z0-9+/]", form) or not re.search(r"[A-Za-z]{2}", form):
            continue
        out.append(form)
    return out


def _significant(written: str) -> bool:
    """A number written with enough digits to be a quoted result, not a setting ("0.5")
    or a small change ("2.3"): three significant digits, or two decimals."""
    digits = written.replace(".", "").lstrip("0")
    return len(digits) >= 3 or len(written.split(".")[1]) >= 2


def _same(stated: _Number, held: _Number, row: bool = True) -> bool:
    """Whether the text's number is the cell's, allowing a percentage for a fraction
    (68.76% for 0.6876) and the sign. In the row itself one unit of the coarser number is
    allowed (71.2 for 71.29, cut rather than rounded). Elsewhere in the paper the cell must
    be written at least as finely and round to the text's number: a coarse number of some
    other table ("4.2") says little about where "4.25" comes from."""
    for factor in (1.0, 100.0, 0.01):
        if row:
            slack = 2 * max(stated.half_unit, held.half_unit * factor)
        elif held.half_unit * factor > stated.half_unit + 1e-12:
            continue
        else:
            slack = stated.half_unit + held.half_unit * factor
        if abs(abs(stated.value) - abs(held.value) * factor) <= slack + 1e-9:
            return True
    return False


def _magnitude(number: _Number) -> int:
    """The number of digits before the decimal point (0 for 0.47)."""
    whole = int(abs(number.value))
    return len(str(whole)) if whole else 0


def _comparable(stated: _Number, held: _Number) -> bool:
    """A cell that could be the one the text means: the same number of whole digits and
    at least as many decimals, so a typo or a changed value rather than another quantity."""
    if held.half_unit > stated.half_unit + 1e-12:
        return False
    return _magnitude(stated) == _magnitude(held)


_RESULT_CELL = re.compile(r"[\d.,\u00b1+\-\u2212%()\s*]+")


def _row_labels(row: Sentence) -> list[tuple[str, int]]:
    """The labels of a results row with the place of their cell: the cells with words
    before its first cell of numbers ("28 | EDM | 29.0", "(w/ Ether) | InertialAR | 90.9"),
    in a row holding a decimal number (which leaves out header rows)."""
    cells = row.text.split(CELL_SEPARATOR)
    if not any(_DECIMAL.search(cell) for cell in cells[1:]):
        return []
    out = []
    for i, cell in enumerate(cells):
        cell = cell.strip()
        if _RESULT_CELL.fullmatch(cell) and _NUMBER.search(cell):
            break
        if len(cell) <= MAX_LABEL_CHARS and re.search(r"[A-Za-z]{2}", cell):
            out.append((cell, i))
    return out if len(out) < len(cells) else []


def _results(text: str, after: int = 0) -> list[_Number]:
    """The numbers of a row after its label (the cell at `after`), in order; a spread
    after "\u00b1" is left out."""
    cells = text.split(CELL_SEPARATOR)[after + 1 :]
    out: list[_Number] = []
    for cell in cells:
        cell = cell.replace(",", "")
        for m in _NUMBER.finditer(cell):
            if cell[: m.start()].rstrip().endswith(_SPREAD_SIGNS):
                continue
            number = _Number.parse(m.group())
            if number is not None:
                out.append(number)
    return out


def _mean_is(stated: _Number, numbers: Sequence[_Number]) -> bool:
    if not numbers:
        return False
    mean = sum(n.value for n in numbers) / len(numbers)
    return abs(mean - stated.value) <= stated.half_unit + max(n.half_unit for n in numbers)


def _derived(
    stated: _Number, rows: Sequence[Sentence], mine: Sequence[tuple[Sentence, int]]
) -> bool:
    """Whether the text's number follows from the table: a difference or a relative change
    between two of the row's own numbers, or between the row's number and another row's
    at the same place; the mean of the row's numbers; or, when the label has several rows
    (one per setting), the mean of its numbers at one place over those rows."""
    pairs: list[tuple[_Number, _Number]] = []
    own_rows = [_results(row.text, at) for row, at in mine]
    for (row, _), own in zip(mine, own_rows, strict=True):
        pairs += [(a, b) for i, a in enumerate(own) for b in own[i + 1 :]]
        if _mean_is(stated, own):
            return True
        for other in rows:
            theirs = _results(other.text, len(other.text.split(CELL_SEPARATOR)) - len(own) - 1)
            if other is row or len(theirs) != len(own):
                continue
            pairs += list(zip(own, theirs, strict=True))
    width = {len(own) for own in own_rows}
    if len(own_rows) > 1 and len(width) == 1:
        for place in range(width.pop()):
            if _mean_is(stated, [own[place] for own in own_rows]):
                return True
    return any(_fits("relative_percent", stated, a, b) for a, b in pairs)


@dataclass(frozen=True)
class _CellMatch:
    sentence: Sentence
    span: tuple[int, int]
    summary: str
    lookup: str
    table: Table


def _table_numbers(paper: ParsedPaper) -> list[_Number]:
    """Every number the paper gives outside its prose: in any table row, caption or
    footnote, and in body text the parser ran a table into."""
    out: list[_Number] = []
    for sentence in paper.sentences:
        if sentence.kind in (SentenceKind.ABSTRACT, SentenceKind.BODY) and _prose_sentence(
            sentence.text
        ):
            continue
        for m in _NUMBER.finditer(sentence.text.replace(",", "")):
            if "." in m.group() and (number := _Number.parse(m.group())) is not None:
                out.append(number)
    return out


def _cell_matches(
    paper: ParsedPaper,
    skip: AbstractSet[str] = frozenset(),
    need_table: bool = True,
    reach: int = CELL_REACH,
) -> list[_CellMatch]:
    """The cell rule's matches. A table counts as named when the sentence names it, or
    one of the `reach` sentences before it in its section does; with `need_table` false,
    a sentence naming no table is checked against every table."""
    tables = [t for t in _tables(paper) if _rows(paper, t)]
    labelled: dict[str, list[tuple[str, Sentence, int]]] = {}
    for table in tables:
        for row in _rows(paper, table):
            for label, at in _row_labels(row):
                for form in _label_forms(label):
                    labelled.setdefault(table.id, []).append((form, row, at))
    known: list[_Number] | None = None
    body = [s for s in paper.sentences if s.kind in (SentenceKind.ABSTRACT, SentenceKind.BODY)]
    out: list[_CellMatch] = []
    for at, sentence in enumerate(body):
        text = sentence.text
        if sentence.id in skip or not _CELL_TEXT_NUMBER.search(text):
            continue
        if not _prose_sentence(text) or re.search(r"\brespectively\b", text, re.IGNORECASE):
            continue
        if re.search(r"\b(?:Fig(?:ure)?s?\.?)\s*\d", text):
            continue
        # "Results in Table ." followed by "6 show ...": the parser split the name.
        naming = text
        if at and re.search(r"\bTab(?:le)?\s*\.?\s*$", body[at - 1].text):
            naming = body[at - 1].text + " " + text
        for earlier in body[max(0, at - reach) : at]:
            if earlier.section_id == sentence.section_id:
                naming = earlier.text + " " + naming
        named = [t for t in tables if _names_table(t, naming)]
        used = named if need_table else (named or tables)
        if not used:
            continue
        # A named table that could not be read may hold the number.
        read = {_label_key(t) for t in tables}
        if any(_label_key(t) not in read for t in paper.tables if _names_table(t, naming)):
            continue
        mentions: list[tuple[int, int, str, Table]] = []
        for table in used:
            for form, _, _ in labelled.get(table.id, []):
                for m in re.finditer(rf"(?<![\w-]){re.escape(form)}(?![\w-])", text):
                    mentions.append((m.start(), m.end(), form, table))
        # A mention inside a longer one ("MPNet" in "MPNet-300K") is the longer one's.
        mentions = [
            m
            for m in mentions
            if not any(
                o[0] <= m[0] and m[1] <= o[1] and o[1] - o[0] > m[1] - m[0] for o in mentions
            )
        ]
        if not mentions:
            continue
        for number in _CELL_TEXT_NUMBER.finditer(text):
            sign, written = number.group(1), number.group(2)
            start = number.start(2)
            if sign or not _significant(written):
                continue
            # A spread or a bound written without a space ("1336.12\u00b173.32").
            if text[:start].rstrip()[-1:] in ("\u00b1", "+", "-", "\u2212", "<", ">", "~"):
                continue
            if _CELL_AFTER.match(text[number.end() :]) or _CELL_RANGE_BEFORE.search(text[:start]):
                continue
            # "0.1253, which is slightly worse than the reports in Table 1": the text says
            # its number is not the table's.
            if _CELL_THAN.match(text[number.end() :]):
                continue
            # The row: the nearest label before the number, in the same clause.
            before = [m for m in mentions if m[1] <= start and start - m[1] <= CELL_WINDOW]
            if not before:
                continue
            end = max(m[1] for m in before)
            owners = [m for m in before if m[1] == end]
            between = text[end:start]
            if _CELL_BETWEEN.search(between):
                continue
            # "BASE (lambda = 0.05)" sets something; "Simulation = 8.93" gives the row's.
            if text[:start].rstrip().endswith(("=", ":")) and between.strip() not in ("=", ":"):
                continue
            # Another row's label between the owner and the number makes it ambiguous.
            if any(end < m[0] < start for m in mentions):
                continue
            stated = _Number.parse(written)
            assert stated is not None
            form = owners[0][2]
            rows = [
                (table, row, at)
                for table in used
                for f, row, at in labelled.get(table.id, [])
                if f == form
            ]
            held = [(table, row, n) for table, row, at in rows for n in _results(row.text, at)]
            if not held or any(_same(stated, n) for _, _, n in held):
                continue
            close = [(t, r, n) for t, r, n in held if _comparable(stated, n)]
            if not close:
                continue
            if known is None:
                known = _table_numbers(paper)
            if any(_same(stated, n, row=False) for n in known):
                continue
            by_table: dict[str, list[tuple[Sentence, int]]] = {}
            for table, row, at in rows:
                by_table.setdefault(table.id, []).append((row, at))
            if any(
                _derived(stated, _rows(paper, t), by_table[t.id]) for t in used if t.id in by_table
            ):
                continue
            table, row, _ = min(close, key=lambda c: abs(c[2].value - stated.value))
            at = next(a for t, r, a in rows if r is row)
            name = table.label or table.id
            given = ", ".join(f"{n.value:.{_decimals(n)}f}" for n in _results(row.text, at))
            summary = (
                f"The text gives {written} for {form}, but no cell of its row in {name} "
                f"holds it (the row gives {given}), and no table of the paper has {written}."
            )
            out.append(
                _CellMatch(sentence, (start, number.end()), summary, f"{name}: {row.text}", table)
            )
            break
    return out


def cell_findings(
    task: AuditTask, paper: ParsedPaper, first: int = 0, skip: set[str] | None = None
) -> list[CandidateFinding]:
    """A number a prose sentence gives for a row of a table it names (or the sentence
    before it names), when no cell of that row holds it ("Table 3 ... BASE reaches 74.77"
    where BASE's cells are 68.76 and 63.34), found with no model. The number belongs to the
    nearest row label before it, in the same clause. It is not reported when it agrees
    with a cell within rounding, as a percentage of a fraction or with its sign changed;
    when any table of the paper holds it; when it is a change, a mean, a range, a bound or
    a count; or when the row has no cell written like it. Uncertain: the text may quote
    another setting."""
    return [
        _finding(task, first + k, m.sentence, m.span, m.summary, m.lookup, True)
        for k, m in enumerate(_cell_matches(paper, skip or set()))
    ]


async def check_results(task: AuditTask, ctx: RunContext) -> TaskResult:
    paper = ctx.paper
    findings = table_findings(task, paper)
    comparisons = text_comparisons(task, paper, len(findings))
    rules: list[CandidateFinding] = []
    for rule in (repeated_spreads, wrong_aggregates):
        rules += rule(task, paper, len(findings) + len(comparisons) + len(rules))
    # GRIM is arithmetic on what the sentence states; the challenge can only guess whether
    # the data are whole numbers, and dropped 12 of 40 planted cases in set D, so its
    # findings stay uncertain without a second look.
    findings += grim_findings(task, paper, len(findings) + len(comparisons) + len(rules))
    compared = {f.anchor.sentence_id for f in comparisons if isinstance(f.anchor, SentenceAnchor)}
    cells_on = ctx.settings.results_cell_rule_enabled
    # Without a model the crops are read only for the cell rule.
    crops = ctx.settings.results_crop_tables and (ctx.llm is not None or cells_on)
    if ctx.source_pdf is not None and crops:
        paper = await asyncio.to_thread(add_crop_tables, paper, ctx.source_pdf)
    if ctx.llm is None:
        findings += comparisons + rules
        if cells_on:
            findings += cell_findings(task, paper, len(findings), compared)
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
    # The table rules' findings are uncertain readings too: the challenge sees each with
    # its table (or, for GRIM, its sentence).
    for finding in rules:
        evidence = [e.text for e in finding.evidence if e.kind is EvidenceKind.LOOKUP]
        rows_text = _rule_table(paper, finding) or "\n".join(evidence) or "(numbers from the text)"
        candidates.append(_Candidate(finding, _around(body, order, finding), rows_text))
    # The cell rule's findings are uncertain too: a sentence may quote another setting.
    flagged = set(compared)
    for m in _cell_matches(paper, compared) if cells_on else []:
        number = len(findings) + len(candidates)
        finding = _finding(task, number, m.sentence, m.span, m.summary, m.lookup, True)
        shown = f"{m.table.label or m.table.id} {_caption(paper, m.table)}".strip()
        shown += "\n" + "\n".join(row.text for row in _rows(paper, m.table))
        candidates.append(_Candidate(finding, _around(body, order, finding), shown))
        flagged.add(m.sentence.id)
    for claim in claims:
        if claim.sentence_id in flagged:
            continue
        if claim.sentence_id in by_id:
            claim = _completed(claim, by_id[claim.sentence_id])
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
    flagged |= {
        f.anchor.sentence_id
        for f in [*findings, *(c.finding for c in candidates)]
        if isinstance(f.anchor, SentenceAnchor)
    }
    for claim, table, (summary, lookup) in await _number_pass(paper, flagged, ctx):
        found = anchor_quote(claim.quote, claim.sentence_id, body)
        if found is None or found.sentence.id != claim.sentence_id:
            continue
        number = len(findings) + len(candidates)
        candidate = _finding(task, number, found.sentence, found.span, summary, lookup)
        rows = f"{table.label or table.id} {_caption(paper, table)}".strip() + "\n"
        rows += "\n".join(row.text for row in _rows(paper, table))
        candidates.append(_Candidate(candidate, _around(body, order, candidate), rows))
    if ctx.settings.results_challenge_enabled:
        kept = await _challenge(candidates, ctx)
    else:
        kept = [c.finding for c in candidates]
    findings += kept
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
