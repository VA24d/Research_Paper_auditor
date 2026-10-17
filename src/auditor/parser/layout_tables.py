"""Tables read from the PDF's layout text, next to the ones GROBID finds.

GROBID often scrambles table columns: it merges cells, splits headers from their columns, or
flattens rows into body text, and then a paper's numbers cannot be checked against its
tables. Poppler's `pdftotext -layout` keeps the columns aligned with spaces, so its lines
split cleanly into cells. On every page with a table caption:

- lines that look like table rows (three or more cells, at least two of them numbers) are
  grouped into blocks, with up to three header lines above each;
- on a two-column page, a block that sits beside the other column's prose keeps only its
  own half, and two tables side by side become two blocks;
- each block is labelled with the nearest table caption.

A page is left unread, as a table with no rows, when one of its blocks looks mis-parsed
(cells run together, rows of very different widths, a single row) or one of its captions
has no block (the table is an image). Every captioned page is left unread when `pdftotext`
is not installed. A check that needs such a table reads the page another way (an image of
the page, in `auditor.agents.integrity.results`).
"""

import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

from auditor.schemas import CELL_SEPARATOR, ParsedPaper, Sentence, SentenceKind, Table

PDFTOTEXT_TIMEOUT_S = 120
# A caption starts a line or a column, or follows text with one space when a colon or bar
# comes after the number (a table set beside running text); "in Table 2." is a reference.
_CAPTION = re.compile(
    r"(?:(?:^|\s{2})\s*(?:Table|TABLE)\s+(\d+|[IVX]+)\s*[:.|]"
    r"|\s(?:Table|TABLE)\s+(\d+|[IVX]+)\s*[:|])"
)
_CELL_GAP = re.compile(r"\s{2,}")
_COLUMN_START = re.compile(r"\S\s{2,}(?=\S)")
# Two numbers in one cell with only a space between them: columns run together.
_MERGED = re.compile(r"\d\s+[-\u2212+]?\d")
# Non-row lines allowed inside a block, and how far a caption may be from its block.
MAX_GAP_LINES = 2
MAX_CAPTION_DISTANCE = 8
CAPTION_BELOW_PENALTY = 2


def layout_pages(path: Path) -> list[str] | None:
    """The PDF's text page by page with columns kept, or None without pdftotext."""
    if shutil.which("pdftotext") is None:
        return None
    try:
        done = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
            capture_output=True,
            text=True,
            timeout=PDFTOTEXT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    return done.stdout.split("\f")


def _cells(line: str) -> list[str]:
    return [cell for cell in _CELL_GAP.split(line.strip()) if cell]


def _numeric(cell: str) -> bool:
    """A cell that is a number, perhaps with a unit or spread ("6.58M", "0.75 \u00b1 0.02",
    "1e-5"), not words that contain a digit ("GA(M2,Precision)")."""
    return any(c.isdigit() for c in cell) and sum(c.isalpha() for c in cell) <= 3


def _is_row(line: str) -> bool:
    cells = _cells(line)
    return len(cells) >= 3 and sum(map(_numeric, cells)) >= 2


def _prose_or_empty(half: str) -> bool:
    """A half line that is a piece of running text (one cell of words) or nothing."""
    cells = _cells(half)
    if not cells:
        return True
    return len(cells) == 1 and (
        not any(c.isdigit() for c in cells[0]) or len(cells[0].split()) >= 4
    )


def _gutter(lines: Sequence[str]) -> int | None:
    """Where a two-column page splits: a position in the middle of the page where nearly
    every text line that reaches past it has blank space, and many lines have text on both
    sides. Table rows are left out, since a table may span both columns."""
    prose = [line.rstrip() for line in lines if line.strip() and not _is_row(line)]
    width = max((len(line) for line in prose), default=0)
    best, most = None, 0
    for position in range(int(width * 0.3), int(width * 0.7) + 1):
        reaching = [line for line in prose if len(line) > position + 1]
        if not reaching:
            continue
        blank = [line for line in reaching if line[position - 1 : position + 1] == "  "]
        both = sum(1 for line in blank if line[:position].strip())
        if len(blank) >= 0.8 * len(reaching) and both > most:
            best, most = position, both
    return best if most >= 8 else None


@dataclass
class _Block:
    lines: list[str]
    first: int  # index of the first line on the page
    last: int
    side: str  # "full", "left" or "right"


def _blocks(lines: Sequence[str]) -> list[tuple[int, int]]:
    """Runs of row lines, as (first, last) line indices, with short gaps allowed."""
    runs: list[tuple[int, int]] = []
    start = end = None
    for index, line in enumerate(lines):
        if not _is_row(line):
            continue
        if start is not None and end is not None and index - end - 1 <= MAX_GAP_LINES:
            end = index
            continue
        if start is not None and end is not None:
            runs.append((start, end))
        start = end = index
    if start is not None and end is not None:
        runs.append((start, end))
    return runs


def _with_header(lines: Sequence[str], first: int) -> int:
    """`first` moved up over at most three header lines (lines of several cells), past
    blank lines between them."""
    top, taken, index = first, 0, first - 1
    while index >= 0 and taken < 3:
        if not lines[index].strip():
            index -= 1
            continue
        if len(_cells(lines[index])) < 2 or _CAPTION.search(lines[index]):
            break
        top, taken, index = index, taken + 1, index - 1
    return top


def _split(lines: Sequence[str], first: int, last: int, gutter: int | None) -> list[_Block]:
    """The block of rows between `first` and `last` with its header lines, cut at the
    gutter when it sits beside the other column's text or next to another table."""

    def block(side: str) -> _Block:
        if side == "left" and gutter is not None:
            view = [line[:gutter] for line in lines]
        elif side == "right" and gutter is not None:
            view = [line[gutter:] for line in lines]
        else:
            view = list(lines)
        top = _with_header(view, first)
        return _Block(view[top : last + 1], top, last, side)

    def tables(*blocks: _Block) -> list[_Block]:
        # A half that holds no rows of its own was running text beside the other half.
        return [b for b in blocks if sum(map(_is_row, b.lines)) >= 2 or b.side == "full"]

    if gutter is None:
        return [block("full")]
    rows = [line for line in lines[first : last + 1] if _is_row(line)]
    lefts = [line[:gutter] for line in rows]
    rights = [line[gutter:] for line in rows]
    half = len(rows) / 2
    if sum(map(_prose_or_empty, lefts)) >= half:
        return tables(block("right"))
    if sum(map(_prose_or_empty, rights)) >= half:
        return tables(block("left"))
    side_by_side = sum(
        _is_row(left) and _is_row(right) and not _numeric(_cells(right)[0])
        for left, right in zip(lefts, rights, strict=True)
    )
    if side_by_side >= half:
        return tables(block("left"), block("right"))
    return [block("full")]


def _misparsed(block: _Block) -> bool:
    rows = [_cells(line) for line in block.lines if _is_row(line)]
    if len(rows) < 2:
        return True
    merged = sum(any(_MERGED.search(cell) for cell in row) for row in rows)
    if merged >= 0.25 * len(rows):
        return True
    _, common = Counter(len(row) for row in rows).most_common(1)[0]
    return common < 0.5 * len(rows)


def _captions(lines: Sequence[str], gutter: int | None) -> list[tuple[int, str, str]]:
    """(line index, table number, side) of each table caption on the page."""
    found = []
    for index, line in enumerate(lines):
        for match in _CAPTION.finditer(line):
            side = "full"
            if gutter is not None:
                word = match.start() + len(match.group()) - len(match.group().lstrip())
                side = "left" if word < gutter else "right"
            found.append((index, match.group(1) or match.group(2), side))
    return found


def _reading_order(lines: Sequence[str], gutter: int | None) -> Callable[[int, str], int]:
    """A function giving a line's place among the page's non-blank lines in reading order:
    the left column top to bottom, then the right one. A full-width line counts with the
    left column."""
    if gutter is None:
        places = list(accumulate(1 if line.strip() else 0 for line in lines))
        return lambda index, side: places[index]
    left = list(accumulate(1 if line[:gutter].strip() else 0 for line in lines))
    right = list(accumulate(1 if line[gutter:].strip() else 0 for line in lines))
    return lambda index, side: left[-1] + right[index] if side == "right" else left[index]


def _page_tables(lines: Sequence[str]) -> tuple[list[tuple[str, list[str]]], list[str]] | None:
    """The page's tables as (label, rows), and the numbers of all its captions; None when
    the page has no table caption. An empty table list means the page is left unread."""
    gutter = _gutter(lines)
    captions = _captions(lines, gutter)
    if not captions:
        return None
    numbers = list(dict.fromkeys(number for _, number, _ in captions))
    blocks: list[_Block] = []
    for first, last in _blocks(lines):
        blocks += _split(lines, first, last, gutter)

    rank = _reading_order(lines, gutter)

    def caption_end(line: int) -> int:
        """The last line of a caption: it runs on over lines of text up to a blank line or
        a table row."""
        end = line
        while (
            end + 1 < len(lines)
            and end - line < 4
            and lines[end + 1].strip()
            and not _is_row(lines[end + 1])
        ):
            end += 1
        return end

    def distance(caption: tuple[int, str, str], block: _Block) -> int:
        """Lines of text between the caption and the block in reading order, with a small
        penalty for a caption below its table: tables usually have their caption above."""
        line, _, side = caption
        at, end = rank(line, side), rank(caption_end(line), side)
        first, last = rank(block.first, block.side), rank(block.last, block.side)
        if at < first:
            return max(first - end - 1, 0)
        return at - last - 1 + CAPTION_BELOW_PENALTY if at > last else 0

    # Captions and blocks are paired cheapest first. A block of several rows is preferred
    # to a single row, which near a caption is more often a figure's axis labels. Blocks no
    # caption takes (numbers in figures, mostly) are left out.
    pairs = sorted(
        (_misparsed(block), distance(caption, block), c, b)
        for c, caption in enumerate(captions)
        for b, block in enumerate(blocks)
        if distance(caption, block) <= MAX_CAPTION_DISTANCE
    )
    labels: dict[int, str] = {}
    paired: set[int] = set()
    for bad, _, c, b in pairs:
        if c in paired or b in labels:
            continue
        if bad:
            return [], numbers
        labels[b] = f"Table {captions[c][1]}"
        paired.add(c)
    if len(paired) < len(captions):
        # A caption with no table next to it: the table is likely an image.
        return [], numbers
    tables = [
        (labels[b], [CELL_SEPARATOR.join(_cells(line)) for line in block.lines if _cells(line)])
        for b, block in enumerate(blocks)
        if b in labels
    ]
    return tables, numbers


def _label(numbers: Sequence[str]) -> str:
    return "; ".join(f"Table {n}" for n in dict.fromkeys(numbers))


def add_layout_tables(paper: ParsedPaper, pages: Sequence[str] | None) -> ParsedPaper:
    """`paper` with the tables read from the layout text `pages`; with no layout text, an
    unread table per page where the parser found a table caption."""
    tables: list[Table] = []
    sentences: list[Sentence] = []
    if pages is not None:
        for number, page in enumerate(pages, start=1):
            found = _page_tables(page.splitlines())
            if found is None:
                continue
            read, captions = found
            if not read:
                tables.append(Table(id=f"layout-p{number}", label=_label(captions), page=number))
                continue
            for k, (label, rows) in enumerate(read):
                ids = []
                for index, row in enumerate(rows):
                    sentence_id = f"lt{number}-{k}-{index}"
                    ids.append(sentence_id)
                    sentences.append(Sentence(id=sentence_id, kind=SentenceKind.TABLE, text=row))
                tables.append(
                    Table(id=f"layout-p{number}-{k}", label=label, row_ids=tuple(ids), page=number)
                )
    else:
        captions_by_page: dict[int, list[str]] = {}
        for sentence in paper.sentences:
            match = _CAPTION.match(sentence.text)
            if sentence.kind is SentenceKind.CAPTION and match and sentence.pages:
                captions_by_page.setdefault(sentence.pages[0], []).append(
                    match.group(1) or match.group(2)
                )
        for number, captions in sorted(captions_by_page.items()):
            tables.append(Table(id=f"layout-p{number}", label=_label(captions), page=number))
    if not tables:
        return paper
    data = paper.model_dump()
    data["sentences"] = [*data["sentences"], *(s.model_dump() for s in sentences)]
    data["tables"] = [*data["tables"], *(t.model_dump() for t in tables)]
    return ParsedPaper.model_validate(data)
