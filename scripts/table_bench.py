"""Compare ways of reading a paper's tables against the tables in its LaTeX source.

Every command reads and writes tables in one JSON shape, `{paper id: {label: rows}}`, each
row a list of cells as strings:

    # ground truth from arXiv sources (<id>/main.tex, from scripts/fetch_arxiv.py --source)
    uv run python scripts/table_bench.py truth data/sources --out data/bench/truth.json

    # deterministic readers, no model
    uv run python scripts/table_bench.py predict --strategy layout --pdfs data/pdfs \\
        --out data/bench/layout.json
    uv run python scripts/table_bench.py predict --strategy grobid --papers data/papers \\
        --out data/bench/grobid.json
    uv run python scripts/table_bench.py predict --strategy parsed --pdfs data/pdfs \\
        --papers data/papers --out data/bench/parsed.json
    uv run python scripts/table_bench.py predict --strategy crop-text --pdfs data/pdfs \\
        --out data/bench/crop-text.json

    # model strategies: write the requests, run them elsewhere, then parse the replies
    uv run python scripts/table_bench.py predict --strategy crop-image --pdfs data/pdfs \\
        --dump-requests data/bench/crop-image
    uv run python scripts/table_bench.py predict --strategy crop-image \\
        --requests data/bench/crop-image --replies replies.json --out data/bench/crop.json

    uv run python scripts/table_bench.py score data/bench/truth.json data/bench/layout.json

    # survey of table boxes, with sample crops to look at
    uv run python scripts/table_bench.py crops data/pdfs --out data/table_crops --samples 10

Scoring aligns predicted tables with true ones by label or, failing that, by the numbers
they share, then aligns rows the same way. A number counts as recalled when it is in the
aligned row. Numbers are compared by value after removing formatting: percent signs,
bold markers, LaTeX commands, thousands separators, and Unicode minus signs.

No command here calls a model. `predict_with_model` is the hook for running the model
strategies from code.
"""

import argparse
import base64
import json
import re
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from auditor.agents.integrity.results import IMAGE_PROMPT, TableImages, render_page
from auditor.parser.latex import LatexError, find_main, parse_latex
from auditor.parser.layout_tables import layout_pages, read_layout_tables
from auditor.parser.table_crops import (
    TableBox,
    box_text,
    find_tables,
    table_captions,
    table_crops,
)
from auditor.schemas import CELL_SEPARATOR, ParsedPaper

Rows = list[list[str]]
Tables = dict[str, Rows]
Predictions = dict[str, Tables]
MODEL_STRATEGIES = ("page-image", "crop-image", "crop-image+words")
STRATEGIES = ("layout", "grobid", "parsed", "crop-text", *MODEL_STRATEGIES)

# --- ground truth from LaTeX -----------------------------------------------------------

# Commands whose leading arguments are settings, followed by the cell's content:
# \multirow{2}{*}{Text}, \textcolor{red}{85.3}, \scalebox{0.8}{...}. The number of
# setting arguments to drop.
_SETTING_ARGUMENTS = {
    "multirow": 2,
    "multicolumn": 2,
    "textcolor": 1,
    "colorbox": 1,
    "scalebox": 1,
    "raisebox": 1,
    "resizebox": 2,
    "makecell": 0,
    "num": 0,
}
# Commands dropped with their argument: citations, references, spacing, colours.
_DROPPED = (
    "cite|citep|citet|citealp|ref|eqref|autoref|cref|label|hspace|vspace|footnote|"
    "footnotemark|tnote|tablefootnote|cellcolor|rowcolor|color|includegraphics|"
    "rule|addlinespace|specialrule|cmidrule|cline"
)
# The last argument may be cut off: auditor.parser.latex strips braces at the ends of a
# cell, so "Tran \cite{tran2016}" arrives as "Tran \cite{tran2016".
_DROPPED_COMMAND = re.compile(
    rf"\\(?:{_DROPPED})(?![A-Za-z])\*?\s*(?:\[[^\]]*\]\s*)?(?:\{{[^{{}}]*\}}\s*)?"
    r"(?:\([^)]*\))?(?:\{[^{}]*(?:\}|$))?"
)
_SYMBOLS = {
    r"\pm": "\u00b1",
    r"\mp": "\u00b1",
    r"\times": "\u00d7",
    r"\%": "%",
    r"\&": "&",
    r"\$": "",
    r"\#": "#",
    r"\_": " ",
    "\\\\": " ",
    r"\,": "",
    r"\;": " ",
    r"\!": "",
    r"\ ": " ",
    r"\checkmark": "\u2713",
    r"\cmark": "\u2713",
    r"\xmark": "\u2717",
    r"\textendash": "\u2013",
    r"\textminus": "-",
    "--": "\u2013",
}
_COMMAND = re.compile(r"\\[A-Za-z]+\*?")
# A length left by a rule or line break with a setting: "[0.2em]", "[-2pt]".
_LENGTH = re.compile(r"\[\s*-?\d*\.?\d+\s*(?:em|ex|pt|mm|cm|in)\s*\]")


def _drop_settings(text: str) -> str:
    for name, count in _SETTING_ARGUMENTS.items():
        pattern = re.compile(
            rf"\\{name}\*?\s*(?:\[[^\]]*\]\s*)?" + r"\{[^{}]*\}\s*" * count + r"(?=\{)"
        )
        text = pattern.sub("", text)
    return text


def clean_latex_cell(cell: str) -> str:
    """A table cell from LaTeX source as it would print, roughly: settings, citations and
    colours removed, symbols spelled out, other commands and braces dropped."""
    text = _drop_settings(cell)
    text = _DROPPED_COMMAND.sub("", text)
    for command, symbol in sorted(_SYMBOLS.items(), key=lambda item: -len(item[0])):
        text = text.replace(command, symbol)
    text = _COMMAND.sub("", text)
    text = _LENGTH.sub(" ", text)
    # Superscripts are glued to their base as in the PDF's text ("10^{-3}" -> "10-3"),
    # subscripts are not ("85.3_{\pm 0.2}" -> "85.3 ± 0.2").
    text = re.sub(r"[{}$^\\]", "", text)
    text = re.sub(r"[_~]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def truth_tables(paper: ParsedPaper) -> Tables:
    """The tables of a paper parsed from LaTeX, as label -> rows of cleaned cells. Tables
    with no rows are left out."""
    text = {sentence.id: sentence.text for sentence in paper.sentences}
    tables: Tables = {}
    for table in paper.tables:
        rows = [
            [clean_latex_cell(cell) for cell in text[row_id].split(CELL_SEPARATOR)]
            for row_id in table.row_ids
            if row_id in text
        ]
        rows = [row for row in rows if any(row)]
        if rows:
            tables[_unique(tables, table.label or table.id)] = rows
    return tables


def truth_from_sources(folder: Path) -> Predictions:
    """Ground truth for every paper folder under `folder` (`<id>/main.tex`, or the main
    file `find_main` picks)."""
    truth: Predictions = {}
    for paper_folder in sorted(p for p in folder.iterdir() if p.is_dir()):
        main = paper_folder / "main.tex"
        try:
            if not main.is_file():
                main = find_main(paper_folder)
            paper = parse_latex(main, paper_id=paper_folder.name)
        except (LatexError, OSError, ValueError) as exc:
            print(f"skip {paper_folder.name}: {exc}", file=sys.stderr)
            continue
        truth[paper_folder.name] = truth_tables(paper)
    return truth


def _unique(tables: Mapping[str, Any], label: str) -> str:
    if label not in tables:
        return label
    k = 2
    while f"{label} ({k})" in tables:
        k += 1
    return f"{label} ({k})"


# --- numbers and labels ----------------------------------------------------------------

_MINUS_SIGNS = str.maketrans({"\u2212": "-", "\u2012": "-", "\u2013": "-", "\u2010": "-"})
_SPREAD = re.compile(r"\+\s*/\s*-|\+\s*-|\\pm")
_THOUSANDS = re.compile(r"(?<=\d)[,\u2009\u202f](?=\d{3}(?!\d))")
# A number that does not follow a letter, digit, point or hyphen ("ResNet-50", "x2" and "F1"
# hold none); a unit may follow it ("6.58M"). A minus sign counts when it does not follow a
# letter or digit.
_NUMBER = re.compile(r"(?<![\w.\-])(-?)(\d+(?:\.\d+)?|\.\d+)(?:[eE]([-+]?\d+))?")


def numbers(cell: str, spreads: bool = True) -> list[float]:
    """The numbers in a cell, by value. "85.3%" and "**85.3**" give 85.3, "1,234" gives
    1234, "\u22120.5" gives -0.5, "60.5 \u00b1 0.39" gives 60.5 and 0.39 (or only 60.5
    without `spreads`)."""
    text = cell.translate(_MINUS_SIGNS).replace("*", "")
    text = _SPREAD.sub("\u00b1", text)
    text = _THOUSANDS.sub("", text)
    found = []
    for match in _NUMBER.finditer(text):
        before = text[: match.start()].rstrip(" (")
        if not spreads and before.endswith("\u00b1"):
            continue
        sign, digits, exponent = match.groups()
        value = float(f"{sign}{digits}" + (f"e{exponent}" if exponent else ""))
        found.append(round(value, 9))
    return found


def row_numbers(row: Sequence[str], spreads: bool = True) -> Counter[float]:
    return Counter(n for cell in row for n in numbers(cell, spreads))


_CITATION = re.compile(r"\[[\d,\s\u2013-]+\]")


def row_label(row: Sequence[str]) -> str:
    """The row's first cell, normalised for comparison ("ResNet-50 [12]" -> "resnet50"),
    or "" when it holds no letters."""
    if not row:
        return ""
    first = _CITATION.sub("", clean_latex_cell(row[0]) if "\\" in row[0] else row[0])
    if not any(c.isalpha() for c in first):
        return ""
    return re.sub(r"[^a-z0-9]+", "", first.lower())


# --- scoring ---------------------------------------------------------------------------


@dataclass
class TableScore:
    paper: str
    truth_label: str
    predicted_label: str | None
    truth_numbers: int = 0
    predicted_numbers: int = 0
    matched_numbers: int = 0  # in aligned rows
    bag_matched: int = 0  # anywhere in the table
    labelled_rows: int = 0
    correct_labels: int = 0

    @property
    def recall(self) -> float | None:
        return self.matched_numbers / self.truth_numbers if self.truth_numbers else None

    @property
    def precision(self) -> float | None:
        return self.matched_numbers / self.predicted_numbers if self.predicted_numbers else None

    @property
    def label_accuracy(self) -> float | None:
        return self.correct_labels / self.labelled_rows if self.labelled_rows else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "paper": self.paper,
            "truth_label": self.truth_label,
            "predicted_label": self.predicted_label,
            "truth_numbers": self.truth_numbers,
            "predicted_numbers": self.predicted_numbers,
            "matched_numbers": self.matched_numbers,
            "bag_matched": self.bag_matched,
            "recall": self.recall,
            "precision": self.precision,
            "labelled_rows": self.labelled_rows,
            "correct_labels": self.correct_labels,
            "label_accuracy": self.label_accuracy,
        }


@dataclass
class Score:
    tables: list[TableScore] = field(default_factory=list)
    unmatched_predicted_numbers: int = 0  # numbers in predicted tables matched to no truth
    truth_tables: int = 0
    predicted_tables: int = 0
    papers: int = 0
    papers_without_prediction: list[str] = field(default_factory=list)

    def overall(self) -> dict[str, Any]:
        truth = sum(t.truth_numbers for t in self.tables)
        in_matched = sum(t.predicted_numbers for t in self.tables)
        predicted = in_matched + self.unmatched_predicted_numbers
        matched = sum(t.matched_numbers for t in self.tables)
        bag = sum(t.bag_matched for t in self.tables)
        labelled = sum(t.labelled_rows for t in self.tables)
        correct = sum(t.correct_labels for t in self.tables)
        return {
            "papers": self.papers,
            "papers_without_prediction": len(self.papers_without_prediction),
            "truth_tables": self.truth_tables,
            "predicted_tables": self.predicted_tables,
            "matched_tables": sum(t.predicted_label is not None for t in self.tables),
            "truth_numbers": truth,
            "predicted_numbers": predicted,
            "matched_numbers": matched,
            "recall": matched / truth if truth else None,
            "precision": matched / predicted if predicted else None,
            # Precision within the tables matched to a true one: tables the source lacks
            # (an \input file missing from the e-print, a table drawn as an image in the
            # source) do not count against it.
            "matched_precision": matched / in_matched if in_matched else None,
            "bag_recall": bag / truth if truth else None,
            "label_accuracy": correct / labelled if labelled else None,
        }


def _overlap(a: Counter[float], b: Counter[float]) -> int:
    return sum((a & b).values())


def _pairs(
    left: Sequence[tuple[str, Counter[float]]],
    right: Sequence[tuple[str, Counter[float]]],
    min_share: float,
) -> list[tuple[int, int]]:
    """Greedy one-to-one alignment of two lists of (label, numbers): the pairs that share
    the most numbers first, with a bonus for an equal label. A pair needs an equal label
    or at least `min_share` of the left side's numbers in common."""
    scored = []
    for i, (label_a, nums_a) in enumerate(left):
        for j, (label_b, nums_b) in enumerate(right):
            shared = _overlap(nums_a, nums_b)
            same = bool(label_a) and label_a == label_b
            share = shared / max(sum(nums_a.values()), 1)
            if not same and (shared == 0 or share < min_share):
                continue
            scored.append((shared + (0.5 + 0.5 * share if same else 0.0), share, i, j))
    scored.sort(key=lambda s: (-s[0], -s[1], s[2], s[3]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs = []
    for _, _, i, j in scored:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        pairs.append((i, j))
    return pairs


def _table_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower().rstrip(":."))


def score_table(
    paper: str,
    label: str,
    truth: Rows,
    predicted: Rows | None,
    pred_label: str | None,
    spreads: bool = True,
) -> TableScore:
    result = TableScore(paper, label, pred_label)
    truth_rows = [(row_label(row), row_numbers(row, spreads)) for row in truth]
    result.truth_numbers = sum(sum(n.values()) for _, n in truth_rows)
    result.labelled_rows = sum(1 for lab, _ in truth_rows if lab)
    if predicted is None:
        return result
    pred_rows = [(row_label(row), row_numbers(row, spreads)) for row in predicted]
    result.predicted_numbers = sum(sum(n.values()) for _, n in pred_rows)
    bag_truth: Counter[float] = sum((n for _, n in truth_rows), Counter())
    bag_pred: Counter[float] = sum((n for _, n in pred_rows), Counter())
    result.bag_matched = _overlap(bag_truth, bag_pred)
    for i, j in _pairs(truth_rows, pred_rows, min_share=0.0):
        result.matched_numbers += _overlap(truth_rows[i][1], pred_rows[j][1])
        if truth_rows[i][0] and truth_rows[i][0] == pred_rows[j][0]:
            result.correct_labels += 1
    return result


def _paper_key(paper_id: str) -> str:
    return re.sub(r"v\d+$", "", paper_id)


def score(truth: Predictions, predictions: Predictions, spreads: bool = True) -> Score:
    """Scores of `predictions` against `truth`, over the papers in both. Paper ids match
    with or without an arXiv version ("1711.10703v1" and "1711.10703")."""
    by_key = {_paper_key(pid): tables for pid, tables in predictions.items()}
    result = Score()
    for paper, true_tables in sorted(truth.items()):
        predicted = predictions.get(paper, by_key.get(_paper_key(paper)))
        if predicted is None:
            result.papers_without_prediction.append(paper)
            continue
        result.papers += 1
        result.truth_tables += len(true_tables)
        result.predicted_tables += len(predicted)
        truth_items = list(true_tables.items())
        pred_items = list(predicted.items())
        left = [
            (_table_label(lab), sum((row_numbers(r, spreads) for r in rows), Counter()))
            for lab, rows in truth_items
        ]
        right = [
            (_table_label(lab), sum((row_numbers(r, spreads) for r in rows), Counter()))
            for lab, rows in pred_items
        ]
        matched = dict(_pairs(left, right, min_share=0.1))
        for i, (label, rows) in enumerate(truth_items):
            j = matched.get(i)
            pred_label, pred_rows = pred_items[j] if j is not None else (None, None)
            result.tables.append(score_table(paper, label, rows, pred_rows, pred_label, spreads))
        used = set(matched.values())
        result.unmatched_predicted_numbers += sum(
            sum(right[j][1].values()) for j in range(len(right)) if j not in used
        )
    return result


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:5.1f}%"


def report(result: Score, per_table: bool = False) -> str:
    lines = []
    if per_table:
        lines.append(
            f"{'paper':<16} {'truth':<10} {'predicted':<12} {'nums':>5} "
            f"{'recall':>7} {'prec':>7} {'labels':>7}"
        )
        for t in result.tables:
            lines.append(
                f"{t.paper:<16} {t.truth_label:<10} {t.predicted_label or '-':<12} "
                f"{t.truth_numbers:>5} {_percent(t.recall):>7} {_percent(t.precision):>7} "
                f"{_percent(t.label_accuracy):>7}"
            )
    o = result.overall()
    lines.append(
        f"papers {o['papers']} (no prediction: {o['papers_without_prediction']}), tables "
        f"{o['matched_tables']}/{o['truth_tables']} matched, {o['predicted_tables']} predicted"
    )
    lines.append(
        f"numbers: recall {_percent(o['recall'])}, precision {_percent(o['precision'])} "
        f"({_percent(o['matched_precision'])} in matched tables), "
        f"anywhere in the table {_percent(o['bag_recall'])} ({o['matched_numbers']} of "
        f"{o['truth_numbers']}, {o['predicted_numbers']} predicted)"
    )
    lines.append(f"row labels: {_percent(o['label_accuracy'])}")
    return "\n".join(lines)


# --- deterministic strategies ----------------------------------------------------------


def _split(row: str) -> list[str]:
    return [cell.strip() for cell in row.split(CELL_SEPARATOR)]


def predict_layout(pdf: Path) -> Tables:
    """Tables read from `pdftotext -layout` text (auditor.parser.layout_tables)."""
    pages = layout_pages(pdf)
    tables: Tables = {}
    for _, label, rows in read_layout_tables(pages or []):
        tables[_unique(tables, label)] = [_split(row) for row in rows]
    return tables


def predict_grobid(paper: ParsedPaper) -> Tables:
    """GROBID's tables in a ParsedPaper (those without a page; tables with a page come
    from the layout text)."""
    text = {sentence.id: sentence.text for sentence in paper.sentences}
    tables: Tables = {}
    for table in paper.tables:
        if table.page is not None or not table.row_ids:
            continue
        rows = [_split(text[i]) for i in table.row_ids if i in text]
        tables[_unique(tables, table.label or table.id)] = rows
    return tables


def predict_crop_text(pdf: Path) -> Tables:
    """The PDF's text inside each table box (auditor.parser.table_crops), one row per
    printed line, cells split at wide gaps. Its numbers show how much of each table a
    crop holds; its rows are rough, since sub- and superscripts and wrapped cells make
    lines of their own."""
    tables: Tables = {}
    boxes, _ = find_tables(pdf)
    with pdfplumber.open(pdf) as document:
        for box in boxes:
            text = box_text(document.pages[box.page - 1], box.bbox)
            rows = [[cell.strip() for cell in line.split(" | ")] for line in text.splitlines()]
            tables[_unique(tables, box.label)] = rows
    return tables


def predict_parsed(layout: Tables, grobid: Tables) -> Tables:
    """What the results check reads (D22): layout tables, and GROBID's for the labels the
    layout text did not give."""
    merged = dict(layout)
    for label, rows in grobid.items():
        if label not in merged:
            merged[label] = rows
    return merged


# --- model strategies: requests and replies, no calls ------------------------------------

CROP_PROMPT = """You transcribe one table from a research paper. The image shows the table
cut from its page, with its caption. Give the table's label as printed (such as "Table 3")
and its rows, each row a list of cells copied exactly as printed: numbers with all their
digits, signs and units. Write a spread printed small or as a subscript with its sign, as
in "60.5 \u00b1 0.39" or "74.5 (+1.0)", never run into the number. Include the header
rows. Give one cell per column, and an empty string for an empty cell. If the image holds
more than one table, give each."""

WORDS_PROMPT = (
    CROP_PROMPT
    + """

The PDF's own text inside the image is given below the image, one printed line per line,
with wide gaps shown as " | ". Its characters are exact, so copy digits from it; take the
rows and columns from the image, since the text may split or merge cells and puts
subscripts on lines of their own."""
)


@dataclass(frozen=True)
class ModelRequest:
    """One model call for a model strategy: a system prompt, a text and one image. The
    reply must follow `TableImages` (auditor.agents.integrity.results)."""

    id: str
    paper_id: str
    page: int
    label: str | None  # the crop's table, or None for a whole page
    prompt: str
    text: str
    png: bytes = field(repr=False)

    def messages(self) -> list[BaseMessage]:
        """The request as chat messages, in the form the results check sends them."""
        image = base64.b64encode(self.png).decode("ascii")
        return [
            SystemMessage(self.prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": self.text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                ]
            ),
        ]


def model_requests(
    strategy: str, pdf: Path, paper_id: str | None = None, dpi: int = 300
) -> list[ModelRequest]:
    """The requests a model strategy makes for one PDF:

    - `page-image`: each page with a table caption, rendered at 150 dpi as the results
      check does (`render_page`), with the prompt it uses (`IMAGE_PROMPT`);
    - `crop-image`: each table found by `auditor.parser.table_crops`, rendered alone at
      `dpi`;
    - `crop-image+words`: the same crops with the PDF's text inside each one.
    """
    paper = paper_id or pdf.stem
    if strategy == "page-image":
        pages: dict[int, list[str]] = {}
        for page, label in table_captions(pdf):
            pages.setdefault(page, []).append(label)
        return [
            ModelRequest(
                id=f"{paper}/p{page}",
                paper_id=paper,
                page=page,
                label=None,
                prompt=IMAGE_PROMPT,
                text=f"Page {page} of the paper ({'; '.join(labels)}).",
                png=render_page(pdf, page),
            )
            for page, labels in sorted(pages.items())
        ]
    if strategy not in ("crop-image", "crop-image+words"):
        raise ValueError(f"not a model strategy: {strategy}")
    requests = []
    for crop in table_crops(pdf, dpi=dpi):
        text = f"{crop.label} on page {crop.page} of the paper."
        prompt = CROP_PROMPT
        if strategy == "crop-image+words":
            text += "\n\nText inside the image:\n" + crop.text
            prompt = WORDS_PROMPT
        requests.append(
            ModelRequest(
                id=f"{paper}/p{crop.page}/{crop.label.replace(' ', '')}",
                paper_id=paper,
                page=crop.page,
                label=crop.label,
                prompt=prompt,
                text=text,
                png=crop.png,
            )
        )
    return requests


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_reply(request: ModelRequest, reply: TableImages | Mapping[str, Any] | str) -> Tables:
    """The tables in a model's reply to `request`, as label -> rows. A reply may be the
    parsed `TableImages`, its JSON as a dict, or JSON text (a code fence is allowed). A
    crop's single table takes the crop's label; otherwise the model's label is used, or a
    made-up one when the model gave none. Empty cells at the ends of rows and empty rows
    are dropped; nothing else is changed, so spreads run into numbers are kept and scored
    as read."""
    if isinstance(reply, str):
        fenced = _FENCE.match(reply)
        reply = TableImages.model_validate_json(fenced.group(1) if fenced else reply)
    elif not isinstance(reply, TableImages):
        reply = TableImages.model_validate(reply)
    tables: Tables = {}
    for k, table in enumerate(reply.tables):
        rows = [[cell.strip() for cell in row] for row in table.rows]
        rows = [_trim(row) for row in rows]
        rows = [row for row in rows if row]
        if not rows:
            continue
        if request.label is not None and len(reply.tables) == 1:
            label = request.label
        else:
            label = table.label.strip() or f"page {request.page} table {k + 1}"
        tables[_unique(tables, label)] = rows
    return tables


def _trim(row: list[str]) -> list[str]:
    while row and not row[-1]:
        row = row[:-1]
    while row and not row[0]:
        row = row[1:]
    return row


ModelCall = Callable[[ModelRequest], Awaitable[TableImages | Mapping[str, Any] | str]]


async def predict_with_model(
    strategy: str, pdfs: Iterable[Path], call: ModelCall, dpi: int = 300
) -> Predictions:
    """Predictions of a model strategy (`page-image`, `crop-image`, `crop-image+words`)
    for each PDF. This is the hook for running one: `call` sends a request to a model and
    returns its reply. With the project's LLM client it can be

        async def call(request: ModelRequest) -> TableImages:
            result = await llm.generate(
                TableImages, request.messages(), effort="low", purpose="table-image",
                max_output_tokens=8192,
            )
            return result.value

    Requests run one after another. A reply that does not parse is reported on stderr
    and left out."""
    predictions: Predictions = {}
    for pdf in pdfs:
        tables: Tables = {}
        for request in model_requests(strategy, pdf, dpi=dpi):
            try:
                read = parse_reply(request, await call(request))
            except (ValidationError, ValueError) as exc:
                print(f"{request.id}: reply not read: {exc}", file=sys.stderr)
                continue
            for label, rows in read.items():
                tables[_unique(tables, label)] = rows
        predictions[pdf.stem] = tables
    return predictions


def dump_requests(requests: Sequence[ModelRequest], folder: Path) -> None:
    """Writes `requests.jsonl` (id, paper, page, label, prompt, text, image path) and the
    images under `folder`, so the requests can be run outside this script."""
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    with (folder / "requests.jsonl").open("w", encoding="utf-8") as out:
        for request in requests:
            name = re.sub(r"[^\w.-]+", "_", request.id) + ".png"
            (images / name).write_bytes(request.png)
            record = {
                "id": request.id,
                "paper_id": request.paper_id,
                "page": request.page,
                "label": request.label,
                "prompt": request.prompt,
                "text": request.text,
                "image": f"images/{name}",
                "schema": TableImages.model_json_schema(),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_requests(folder: Path) -> list[ModelRequest]:
    requests = []
    for line in (folder / "requests.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        requests.append(
            ModelRequest(
                id=record["id"],
                paper_id=record["paper_id"],
                page=record["page"],
                label=record["label"],
                prompt=record["prompt"],
                text=record["text"],
                png=(folder / record["image"]).read_bytes(),
            )
        )
    return requests


def predictions_from_replies(
    requests: Sequence[ModelRequest], replies: Mapping[str, Any]
) -> Predictions:
    """Predictions from replies keyed by request id. Every paper with a request gets an
    entry, so a paper whose replies are all missing scores as read with no tables."""
    predictions: Predictions = {}
    for request in requests:
        tables = predictions.setdefault(request.paper_id, {})
        if request.id not in replies:
            continue
        try:
            read = parse_reply(request, replies[request.id])
        except (ValidationError, ValueError) as exc:
            print(f"{request.id}: reply not read: {exc}", file=sys.stderr)
            continue
        for label, rows in read.items():
            tables[_unique(tables, label)] = rows
    return predictions


# --- survey of table boxes -------------------------------------------------------------


def survey(pdfs: Sequence[Path]) -> tuple[dict[str, list[TableBox]], dict[str, list[str]]]:
    """The boxes found in each PDF, and the captions with no box."""
    boxes: dict[str, list[TableBox]] = {}
    missing: dict[str, list[str]] = {}
    for pdf in pdfs:
        found, captions = find_tables(pdf)
        boxes[pdf.stem] = found
        have = {(b.page, b.label) for b in found}
        missing[pdf.stem] = [
            f"p{page} {label}" for page, label in captions if (page, label) not in have
        ]
    return boxes, missing


def sample(boxes: Mapping[str, Sequence[TableBox]], count: int) -> list[tuple[str, TableBox]]:
    """`count` boxes spread evenly over the papers, with every method that found any."""
    every = [(paper, box) for paper in sorted(boxes) for box in boxes[paper]]
    if count <= 0 or not every:
        return []
    picked: list[tuple[str, TableBox]] = []
    for method in ("text", "image"):
        of_method = [item for item in every if item[1].method == method]
        if of_method:
            picked.append(of_method[len(of_method) // 2])
    rest = [item for item in every if item not in picked]
    need = max(count - len(picked), 0)
    step = len(rest) / need if need else 0
    picked += [rest[int(k * step)] for k in range(min(need, len(rest)))]
    return picked[:count]


def save_crops(
    pdf_folder: Path, picked: Sequence[tuple[str, TableBox]], out: Path, dpi: int
) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for paper, box in picked:
        with pdfplumber.open(pdf_folder / f"{paper}.pdf") as document:
            image = document.pages[box.page - 1].crop(box.bbox).to_image(resolution=dpi)
        path = out / f"{paper}-p{box.page}-{box.label.replace(' ', '')}-{box.method}.png"
        image.original.save(path, format="PNG")
        paths.append(path)
    return paths


# --- command line ----------------------------------------------------------------------


def _write(data: Any, path: Path | None) -> None:
    text = json.dumps(data, indent=1, ensure_ascii=False)
    if path is None:
        print(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _predict(args: argparse.Namespace) -> int:
    strategy = args.strategy
    if strategy in MODEL_STRATEGIES:
        if args.dump_requests is not None:
            if args.pdfs is None:
                print("--dump-requests needs --pdfs", file=sys.stderr)
                return 2
            requests = [
                r
                for pdf in sorted(args.pdfs.glob("*.pdf"))
                for r in model_requests(strategy, pdf, dpi=args.dpi)
            ]
            dump_requests(requests, args.dump_requests)
            print(f"wrote {len(requests)} requests to {args.dump_requests}")
            return 0
        if args.requests is None or args.replies is None:
            print(f"{strategy} needs --dump-requests, or --requests and --replies", file=sys.stderr)
            return 2
        read = predictions_from_replies(load_requests(args.requests), _read(args.replies))
        _write(read, args.out)
        return 0
    predictions: Predictions = {}
    if strategy == "crop-text":
        if args.pdfs is None:
            print("crop-text needs --pdfs", file=sys.stderr)
            return 2
        for pdf in sorted(args.pdfs.glob("*.pdf")):
            predictions[pdf.stem] = predict_crop_text(pdf)
    if strategy in ("layout", "parsed"):
        if args.pdfs is None:
            print(f"{strategy} needs --pdfs", file=sys.stderr)
            return 2
        for pdf in sorted(args.pdfs.glob("*.pdf")):
            predictions[pdf.stem] = predict_layout(pdf)
    if strategy in ("grobid", "parsed"):
        if args.papers is None:
            print(f"{strategy} needs --papers", file=sys.stderr)
            return 2
        for path in sorted(args.papers.glob("*.json")):
            grobid = predict_grobid(ParsedPaper.model_validate_json(path.read_text("utf-8")))
            if strategy == "grobid":
                predictions[path.stem] = grobid
            elif path.stem in predictions:
                predictions[path.stem] = predict_parsed(predictions[path.stem], grobid)
    _write(predictions, args.out)
    return 0


def _crops(args: argparse.Namespace) -> int:
    pdfs = sorted(args.pdfs.glob("*.pdf"))
    boxes, missing = survey(pdfs)
    methods = Counter(b.method for found in boxes.values() for b in found)
    sides = Counter(b.caption for found in boxes.values() for b in found)
    total = sum(methods.values())
    unboxed = sum(len(m) for m in missing.values())
    print(f"{len(pdfs)} PDFs, {sum(1 for f in boxes.values() if f)} with tables")
    print(f"{total} tables found: " + ", ".join(f"{k} {v}" for k, v in methods.most_common()))
    print("caption " + ", ".join(f"{k} {v}" for k, v in sides.most_common()))
    print(f"{unboxed} captions with no table found")
    if args.out is not None:
        _write(
            {
                paper: {
                    "boxes": [
                        {
                            "page": b.page,
                            "label": b.label,
                            "bbox": list(b.bbox),
                            "method": b.method,
                            "caption": b.caption,
                        }
                        for b in found
                    ],
                    "missing": missing[paper],
                }
                for paper, found in boxes.items()
            },
            args.out / "survey.json",
        )
        for path in save_crops(args.pdfs, sample(boxes, args.samples), args.out, args.dpi):
            print(path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)

    truth = commands.add_parser("truth", help="ground-truth tables from LaTeX sources")
    truth.add_argument("sources", type=Path, help="folder of <paper id>/ source folders")
    truth.add_argument("--out", type=Path)

    predict = commands.add_parser("predict", help="tables read by one strategy")
    predict.add_argument("--strategy", choices=STRATEGIES, required=True)
    predict.add_argument("--pdfs", type=Path, help="folder of <paper id>.pdf")
    predict.add_argument("--papers", type=Path, help="folder of ParsedPaper <paper id>.json")
    predict.add_argument("--dpi", type=int, default=300, help="resolution of crops")
    predict.add_argument("--dump-requests", type=Path, help="write model requests here")
    predict.add_argument("--requests", type=Path, help="folder written by --dump-requests")
    predict.add_argument("--replies", type=Path, help="JSON object: request id -> reply")
    predict.add_argument("--out", type=Path)

    scoring = commands.add_parser("score", help="score predictions against ground truth")
    scoring.add_argument("truth", type=Path)
    scoring.add_argument("predictions", type=Path)
    scoring.add_argument("--no-spreads", action="store_true", help="leave out \u00b1 spreads")
    scoring.add_argument("--tables", action="store_true", help="print a line per table")
    scoring.add_argument("--out", type=Path, help="write the scores as JSON")

    crops = commands.add_parser("crops", help="survey of table boxes, with sample crops")
    crops.add_argument("pdfs", type=Path)
    crops.add_argument("--out", type=Path, help="folder for survey.json and sample crops")
    crops.add_argument("--samples", type=int, default=10)
    crops.add_argument("--dpi", type=int, default=200)

    args = parser.parse_args(argv)
    if args.command == "truth":
        _write(truth_from_sources(args.sources), args.out)
        return 0
    if args.command == "predict":
        return _predict(args)
    if args.command == "score":
        result = score(_read(args.truth), _read(args.predictions), spreads=not args.no_spreads)
        print(report(result, per_table=args.tables))
        if args.out is not None:
            _write(
                {"overall": result.overall(), "tables": [t.as_dict() for t in result.tables]},
                args.out,
            )
        return 0
    return _crops(args)


if __name__ == "__main__":
    sys.exit(main())
