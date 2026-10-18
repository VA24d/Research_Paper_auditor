"""The region of each captioned table on a PDF page, and that region as an image.

A whole page rendered at 150 dpi is a poor input for reading a table: the table may take a
fifth of the page, and small digits, signs and subscripts blur. This module finds each
table's bounding box with pdfplumber and renders only that box, at a high resolution.

For each page:

- the words are grouped into text lines, and a two-column page's gutter is found from
  where prose lines leave a blank band in the middle of the page;
- a caption is a line (or a column's part of a line) that starts with "Table N" or
  "Tab. N" followed by a colon, a full stop or a bar, or an IEEE style "TABLE IV" standing
  alone; its region runs on over the closely spaced lines below it;
- the caption's scope is its column, the full text width when the caption crosses the
  gutter, and half of that when another caption sits beside it (tables side by side);
- **rules**: the horizontal rules nearest the caption in its scope (booktabs top, middle
  and bottom rules, `\\hline`, or the edges of a frame) give the table's vertical extent
  and its width. The rules must have the width of the first one, and no prose or other
  caption may lie between them;
- **text**, when there are not two such rules: the lines next to the caption are taken
  while they look like table rows (two or more numbers, or three or more cells split by
  wide gaps) or are short, and the walk stops at a line of prose, another caption, or a
  wide vertical gap. A table whose rows run past the caption's column is taken at the
  full text width;
- **image**, when neither works: a raster image next to the caption;
- a region holding a plotted curve, or covered by a raster image with no table rows
  printed over it, is a figure and is not taken;
- both sides of the caption are tried. Rules beat text, then the side most of the
  paper's captions use wins, then the nearer table. Captions whose best option is surest
  choose first, and no two captions on a page take the same table.

The box includes the caption, is padded by a few points, and is clipped to the page.
Rotated tables (text set sideways) are not handled: their words are left out.
"""

import io
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pdfplumber
from pdfplumber.page import Page
from pdfplumber.pdf import PDF

# A box in PDF points from the page's top left corner: (x0, top, x1, bottom), as pdfplumber.
Bbox = tuple[float, float, float, float]
Method = Literal["rules", "text", "image"]
Side = Literal["above", "below"]  # where the caption is, relative to its table

# A word ends at a gap wider than this share of its font size. pdfplumber's default of 3 pt
# runs words together in PDFs that have no space characters.
WORD_GAP_RATIO = 0.15
# A gap between two words wider than this share of the line's height separates cells.
CELL_GAP_RATIO = 0.8
# How far, in line heights, a table may start from its caption, and the widest gap
# between two of its lines.
MAX_LEAD = 3.5
MAX_ROW_GAP = 1.8
PADDING = 4.0
MAX_CAPTION_LINES = 10

# "Table", "Table 3:", "Table3.", "Table.3", "TABLE IV", "Tab. 2", "Table S1:".
_CAPTION_WORD = re.compile(r"^(Table|TABLE|Tab\.)(\.)?(\d+|[IVX]+|[A-Z]\d+)?([:.|])?$")
_CAPTION_NUMBER = re.compile(r"^(\d+|[IVX]+|[A-Z]\d+)([:.|])?$")
_FLOAT_START = re.compile(r"^(Table|TABLE|Tab\.|Figure|FIGURE|Fig\.)\s*(\d+|[IVX]+|[A-Z]\d+)")
_MARKS = frozenset("\u2713\u2714\u2717\u2718\u00d7-\u2013\u2014\u2212")


@dataclass(frozen=True)
class TableBox:
    """Where a captioned table is: its page (from 1), label ("Table 3"), box in points,
    the method that found it, and whether the caption is above or below the table."""

    page: int
    label: str
    bbox: Bbox
    method: Method
    caption: Side


@dataclass(frozen=True)
class TableCrop(TableBox):
    """A `TableBox` with the box rendered as a PNG image, and the PDF's text inside it
    (`box_text`)."""

    png: bytes = field(repr=False, default=b"")
    text: str = field(repr=False, default="")


@dataclass(frozen=True)
class _Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass(frozen=True)
class _Line:
    """Words at one height, left to right: a whole printed line, or the part of one in a
    scope (a column)."""

    words: tuple[_Word, ...]

    @property
    def top(self) -> float:
        return min(w.top for w in self.words)

    @property
    def bottom(self) -> float:
        return max(w.bottom for w in self.words)

    @property
    def x0(self) -> float:
        return min(w.x0 for w in self.words)

    @property
    def x1(self) -> float:
        return max(w.x1 for w in self.words)

    @property
    def height(self) -> float:
        return statistics.median(w.height for w in self.words)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def view(self, x0: float, x1: float) -> "_Line | None":
        """The words whose centres lie between x0 and x1, or None."""
        words = tuple(w for w in self.words if x0 <= w.cx <= x1)
        return _Line(words) if words else None

    def segments(self) -> list[list[_Word]]:
        """Runs of words split at wide gaps: cells of a table row, or a run of prose."""
        runs: list[list[_Word]] = [[self.words[0]]]
        for previous, word in zip(self.words, self.words[1:], strict=False):
            if word.x0 - previous.x1 > CELL_GAP_RATIO * max(word.height, previous.height):
                runs.append([word])
            else:
                runs[-1].append(word)
        return runs


@dataclass(frozen=True)
class _Rule:
    x0: float
    x1: float
    y: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass(frozen=True)
class _Caption:
    label: str
    top: float
    bottom: float
    x0: float
    x1: float
    scope: tuple[float, float]
    height: float


@dataclass(frozen=True)
class _Candidate:
    """A table found on one side of a caption."""

    method: Method
    side: Side
    top: float
    bottom: float
    x0: float
    x1: float
    lead: float  # space between the caption and the table
    rows: int  # table-like lines inside


@dataclass
class _PageInfo:
    number: int
    bbox: Bbox
    lines: list[_Line]
    rules: list[_Rule]
    images: list[Bbox]
    text_x0: float
    text_x1: float
    gutter: float | None
    captions: list[_Caption] = field(default_factory=list)
    drawings: list[tuple[Bbox, int]] = field(default_factory=list)

    @property
    def columns(self) -> list[tuple[float, float]]:
        if self.gutter is None:
            return [(self.text_x0, self.text_x1)]
        return [(self.text_x0, self.gutter), (self.gutter, self.text_x1)]


def _words(page: Page) -> list[_Word]:
    found = page.extract_words(x_tolerance_ratio=WORD_GAP_RATIO, keep_blank_chars=False)
    return [
        _Word(str(w["text"]), float(w["x0"]), float(w["x1"]), float(w["top"]), float(w["bottom"]))
        for w in found
        if w.get("upright", True) and str(w["text"]).strip()
    ]


def _lines(words: Sequence[_Word]) -> list[_Line]:
    """Words grouped into printed lines, top to bottom: a word joins a line whose first
    word's centre is within a third of a line height of its own."""
    groups: list[list[_Word]] = []
    for word in sorted(words, key=lambda w: (w.top, w.x0)):
        centre = (word.top + word.bottom) / 2
        for group in reversed(groups[-4:]):
            first = group[0]
            if abs((first.top + first.bottom) / 2 - centre) <= 0.35 * max(
                first.height, word.height
            ):
                group.append(word)
                break
        else:
            groups.append([word])
    lines = sorted(
        (_Line(tuple(sorted(group, key=lambda w: w.x0))) for group in groups),
        key=lambda line: line.top,
    )
    # A line in a smaller font that sits mostly within the height of its neighbour holds
    # that line's subscripts or superscripts.
    merged: list[_Line] = []
    for line in lines:
        if merged:
            last = merged[-1]
            overlap = min(last.bottom, line.bottom) - max(last.top, line.top)
            small, big = sorted((line, last), key=lambda x: x.height)
            if small.height < 0.85 * big.height and overlap >= 0.5 * (small.bottom - small.top):
                merged[-1] = _Line(tuple(sorted(last.words + line.words, key=lambda w: w.x0)))
                continue
        merged.append(line)
    return merged


def _numeric(text: str) -> bool:
    """A number, perhaps with a unit or spread ("6.58M", "85.3±0.2"), or a mark used
    in place of one (a check mark, a dash)."""
    if text in _MARKS:
        return True
    return any(c.isdigit() for c in text) and sum(c.isalpha() for c in text) <= 3


def _table_like(line: _Line) -> bool:
    return sum(_numeric(w.text) for w in line.words) >= 2 or len(line.segments()) >= 3


def _prose(line: _Line, columns: Sequence[tuple[float, float]]) -> bool:
    """Running text: a run of six or more words, mostly not numbers, that fills most of
    one column of the page. A run that crosses the gutter is a table cell."""
    for run in line.segments():
        if len(run) < 6 or sum(_numeric(w.text) for w in run) > 0.4 * len(run):
            continue
        x0, x1 = run[0].x0, run[-1].x1
        for lo, hi in columns:
            if x0 >= lo - 2 and x1 <= hi + 2 and x1 - x0 >= 0.85 * (hi - lo):
                return True
    return False


def _float_caption(line: _Line) -> bool:
    return _FLOAT_START.match(line.text) is not None


def _gutter(lines: Sequence[_Line], x0: float, x1: float) -> float | None:
    """The x position of a two-column page's gutter: a point in the middle of the text
    area that few lines cross, with at least five lines of prose on each side."""
    width = x1 - x0
    if width <= 0 or len(lines) < 10:
        return None
    best: tuple[int, float] | None = None
    ties: list[float] = []
    for step in range(int(width * 0.3), int(width * 0.7) + 1):
        x = x0 + step
        crossing = 0
        for line in lines:
            words = line.words
            if any(w.x0 - 1 < x < w.x1 + 1 for w in words):
                crossing += 1
                continue
            left = [w for w in words if w.x1 <= x]
            right = [w for w in words if w.x0 >= x]
            if left and right and right[0].x0 - left[-1].x1 < 1.0 * line.height:
                crossing += 1
        if best is None or crossing < best[0]:
            best, ties = (crossing, x), [x]
        elif crossing == best[0] and ties and x - ties[-1] <= 1.5:
            ties.append(x)
    if best is None or best[0] > 0.2 * len(lines):
        return None
    gutter = (ties[0] + ties[-1]) / 2
    left_prose = right_prose = 0
    for line in lines:
        left_part = line.view(x0, gutter)
        right_part = line.view(gutter, x1)
        left_prose += left_part is not None and _prose(left_part, [(x0, gutter)])
        right_prose += right_part is not None and _prose(right_part, [(gutter, x1)])
    return gutter if left_prose >= 5 and right_prose >= 5 else None


def _rules(page: Page) -> list[_Rule]:
    """Horizontal rules: thin lines, thin rectangles and flat curves, merged where they
    meet at one height, and the top and bottom edges of rectangles at least 30 pt wide (a
    frame round a table). Edges of small rectangles, such as heatmap cells, are left out."""
    segments: list[_Rule] = []
    edges: list[_Rule] = []

    def add(into: list[_Rule], x0: Any, x1: Any, y: Any) -> None:
        a, b = sorted((float(x0), float(x1)))
        if b - a >= 10:
            into.append(_Rule(a, b, float(y)))

    for line in page.lines:
        top, bottom = float(line["top"]), float(line["bottom"])
        if abs(top - bottom) <= 1.5:
            add(segments, line["x0"], line["x1"], (top + bottom) / 2)
    for rect in page.rects:
        top, bottom = float(rect["top"]), float(rect["bottom"])
        if bottom - top <= 2.0:
            add(segments, rect["x0"], rect["x1"], (top + bottom) / 2)
        elif float(rect["x1"]) - float(rect["x0"]) >= 30:
            add(edges, rect["x0"], rect["x1"], top)
            add(edges, rect["x0"], rect["x1"], bottom)
    for curve in page.curves:
        points = curve.get("pts") or []
        if len(points) >= 2:
            ys = [float(p[1]) for p in points]
            xs = [float(p[0]) for p in points]
            if max(ys) - min(ys) <= 1.5:
                add(segments, min(xs), max(xs), float(curve["top"]))
    merged: list[_Rule] = []
    for rule in sorted(segments, key=lambda r: (r.y, r.x0)):
        if merged:
            last = merged[-1]
            if abs(rule.y - last.y) <= 1.0 and rule.x0 <= last.x1 + 2:
                merged[-1] = _Rule(last.x0, max(last.x1, rule.x1), last.y)
                continue
        merged.append(rule)
    return sorted(merged + edges, key=lambda r: (r.y, r.x0))


def _clear(info: _PageInfo, caption: _Caption, edge: float, lo: float, hi: float) -> bool:
    """Whether the space between the caption and a table edge at `edge` holds no prose
    and no other caption."""
    a, b = (caption.bottom, edge) if edge > caption.top else (edge, caption.top)
    for line in info.lines:
        middle = (line.top + line.bottom) / 2
        if a < middle < b and not caption.top <= middle <= caption.bottom:
            part = line.view(lo, hi)
            if part is not None and (_prose(part, info.columns) or _float_caption(part)):
                return False
    return True


def _drawings(page: Page) -> list[tuple[Bbox, int]]:
    """The boxes of curves at least 15 pt long that are not straight lines (plotted data,
    mostly), with their numbers of points."""
    found = []
    for curve in page.curves:
        x0, top, x1, bottom = (float(curve[k]) for k in ("x0", "top", "x1", "bottom"))
        # Small curves are drawn symbols (arrows, glyphs), not plots.
        if min(x1 - x0, bottom - top) > 1.5 and max(x1 - x0, bottom - top) >= 15:
            found.append(((x0, top, x1, bottom), len(curve.get("pts") or [])))
    return found


def _figure(info: _PageInfo, x0: float, top: float, x1: float, bottom: float) -> bool:
    """Whether a region is a figure: it holds a plotted curve or three smaller curves, or
    a raster image covers a third of it with no table rows printed over the image (some
    PDFs give images boxes far larger than what they show)."""
    curves = [
        points
        for (a, b, c, d), points in info.drawings
        if x0 <= (a + c) / 2 <= x1 and top <= (b + d) / 2 <= bottom
    ]
    if len(curves) >= 3 or any(points >= 10 for points in curves):
        return True
    area = max((x1 - x0) * (bottom - top), 1e-6)
    for a, b, c, d in info.images:
        left, right = max(x0, a), min(x1, c)
        upper, lower = max(top, b), min(bottom, d)
        if right <= left or lower <= upper or (right - left) * (lower - upper) <= area / 3:
            continue
        rows = sum(
            _table_like(part)
            for line in info.lines
            if upper <= (line.top + line.bottom) / 2 <= lower
            and (part := line.view(left, right)) is not None
        )
        if rows < 2:
            return True
    return False


def _caption_start(line: _Line, gutter: float | None = None) -> list[tuple[int, str]]:
    """(index of the first word, label) of each caption that starts in the line: at its
    start, after a wide gap, or after a gap over the `gutter` wider than a space."""
    found = []
    words = line.words
    for i, word in enumerate(words):
        match = _CAPTION_WORD.match(word.text)
        if match is None:
            continue
        if i > 0:
            gap = word.x0 - words[i - 1].x1
            across = gutter is not None and words[i - 1].x1 <= gutter <= word.x0
            if gap < 1.5 * word.height and not (across and gap > 0.6 * word.height):
                continue  # inside a sentence: "as Table 2: shows" is a reference
        number, mark = match.group(3), match.group(4) or match.group(2)
        after = i + 1
        if number is None:
            if after >= len(words):
                continue
            nxt = _CAPTION_NUMBER.match(words[after].text)
            if nxt is None or words[after].x0 - word.x1 > 1.5 * word.height:
                continue
            number, mark = nxt.group(1), nxt.group(2) or mark
            after += 1
        if mark is None:
            alone = after >= len(words) or words[after].x0 - words[after - 1].x1 > 1.5 * word.height
            punctuation = after < len(words) and words[after].text in {":", ".", "|"}
            if not (punctuation or (alone and (word.text == "TABLE" or number.isalpha()))):
                continue
        found.append((i, f"Table {number}"))
    return found


def _captions(info: _PageInfo) -> list[_Caption]:
    """The page's table captions. A caption's scope is cut in half where other text sits
    beside it on its line (a table set beside running text) or another caption sits next
    to it (tables side by side)."""
    raw: list[tuple[str, _Line, _Line]] = []  # label, printed line, the caption's words
    for line in info.lines:
        for i, label in _caption_start(line, info.gutter):
            run = [line.words[i]]
            for word in line.words[i + 1 :]:
                gap = word.x0 - run[-1].x1
                height = max(word.height, run[-1].height)
                # A caption ends at a wide gap, or at a gap over the gutter wider than a
                # space: the other column's text follows.
                across = info.gutter is not None and run[-1].x1 <= info.gutter <= word.x0
                if gap > 1.5 * height or (across and gap > 0.6 * height):
                    break
                run.append(word)
            raw.append((label, line, _Line(tuple(run))))
    captions = []
    for label, line, first in raw:
        scope = _scope(info, first)
        lo, hi = scope
        for word in line.words:
            if word in first.words or not lo <= word.cx <= hi:
                continue
            if word.x1 <= first.x0:
                lo = max(lo, (word.x1 + first.x0) / 2)
            elif word.x0 >= first.x1:
                hi = min(hi, (first.x1 + word.x0) / 2)
        for _, _, other in raw:
            if (
                other is not first
                and abs(other.top - first.top) <= 3 * first.height
                and _scope(info, other) == scope
            ):
                if other.x0 > first.x1:
                    hi = min(hi, (first.x1 + other.x0) / 2)
                elif other.x1 < first.x0:
                    lo = max(lo, (other.x1 + first.x0) / 2)
        captions.append(_caption_region(info, label, first, (lo, hi)))
    return captions


def _caption_region(
    info: _PageInfo, label: str, first: _Line, scope: tuple[float, float]
) -> _Caption:
    """The caption with the lines it runs on over: each follows at the caption's line
    spacing, with no rule in between, after a line that fills most of the scope, and is
    not a table row."""
    lo, hi = scope
    bottom, x0, x1 = first.bottom, first.x0, first.x1
    previous = first
    count = 0
    for line in info.lines:
        if line.top <= first.top + 0.5 * first.height:
            continue
        part = line.view(lo, hi)
        if part is None:
            continue
        if part.top < bottom - 0.3 * first.height:
            if part.height >= 0.85 * first.height:
                continue  # text beside the caption, such as a figure's caption
            # Sub- or superscripts of the line before.
            bottom, x0, x1 = max(bottom, part.bottom), min(x0, part.x0), max(x1, part.x1)
            continue
        if part.top - bottom > 0.6 * first.height or count >= MAX_CAPTION_LINES:
            break
        # A first line that holds only the label ("TABLE II") is followed by the title.
        wrapped = previous.x1 - previous.x0 >= 0.7 * (hi - lo) or (
            previous is first and len(first.words) <= 2
        )
        row = _table_like(part) and not _prose(part, info.columns)
        ruled = any(
            bottom - 1 <= r.y <= part.top + 1 and min(r.x1, hi) - max(r.x0, lo) > 0
            for r in info.rules
        )
        if not wrapped or row or ruled or _float_caption(part):
            break
        bottom, x0, x1 = part.bottom, min(x0, part.x0), max(x1, part.x1)
        previous = part
        count += 1
    return _Caption(label, first.top, bottom, x0, x1, scope, first.height)


def _scope(info: _PageInfo, first: _Line) -> tuple[float, float]:
    """The caption's column, or the whole text width when it crosses the gutter."""
    full = (info.text_x0, info.text_x1)
    g = info.gutter
    if g is None or (first.x0 < g - 2 and first.x1 > g + 2):
        return full
    return (info.text_x0, g) if first.x1 <= g + 2 else (g, info.text_x1)


def _by_rules(info: _PageInfo, caption: _Caption, side: Side) -> _Candidate | None:
    lo, hi = caption.scope
    span = hi - lo
    h = caption.height
    inside = [
        r
        for r in info.rules
        if r.width >= 0.2 * span and min(r.x1, hi) - max(r.x0, lo) >= 0.8 * r.width
    ]
    if side == "above":  # the caption is above: rules below it, top to bottom
        ordered = sorted(
            (r for r in inside if r.y > caption.bottom - 1), key=lambda r: (round(r.y), -r.width)
        )
    else:
        ordered = sorted(
            (r for r in inside if r.y < caption.top + 1), key=lambda r: (-round(r.y), -r.width)
        )

    def lead_of(rule: _Rule) -> float:
        return rule.y - caption.bottom if side == "above" else caption.top - rule.y

    if not ordered or lead_of(ordered[0]) > MAX_LEAD * h + 6:
        return None
    first = ordered[0]
    if not _clear(info, caption, first.y, first.x0, first.x1):
        return None
    full = [first]
    previous = first
    for rule in ordered[1:]:
        if rule.x0 < first.x0 - 3 or rule.x1 > first.x1 + 3:
            if abs(rule.y - previous.y) > 2:
                break
            continue
        a, b = sorted((previous.y, rule.y))
        between = [
            part
            for line in info.lines
            if a < (line.top + line.bottom) / 2 < b
            and (part := line.view(first.x0 - 3, first.x1 + 3)) is not None
        ]
        if any(_prose(p, info.columns) or _float_caption(p) for p in between):
            break
        previous = rule
        if abs(rule.x0 - first.x0) <= 3 and abs(rule.x1 - first.x1) <= 3:
            full.append(rule)
    if len(full) < 2:
        return None
    top, bottom = sorted((full[0].y, full[-1].y))
    rows = [
        part
        for line in info.lines
        if top < (line.top + line.bottom) / 2 < bottom
        and (part := line.view(first.x0 - 10, first.x1 + 10)) is not None
    ]
    if not rows or _figure(info, first.x0, top, first.x1, bottom):
        return None
    # Lines just past the outer rule, away from the caption, such as the group headers
    # above a table's top rule or a note below its bottom rule.
    far = _beyond(info, first.x0 - 10, first.x1 + 10, bottom if side == "above" else top, side)
    if far:
        top, bottom = min(top, *(p.top for p in far)), max(bottom, *(p.bottom for p in far))
        rows += far
    x0 = min([first.x0, *(p.x0 for p in rows)])
    x1 = max([first.x1, *(p.x1 for p in rows)])
    return _Candidate(
        "rules", side, top, bottom, x0, x1, lead_of(first), sum(map(_table_like, rows))
    )


def _beyond(info: _PageInfo, lo: float, hi: float, edge: float, side: Side) -> list[_Line]:
    """Up to two lines past `edge` (below it when the caption is above the table, else
    above it) at a row's spacing that are neither prose nor a caption."""
    parts = [p for line in info.lines if (p := line.view(lo, hi)) is not None]
    if side == "above":
        ordered = [p for p in parts if p.top >= edge - 1]
    else:
        ordered = [p for p in reversed(parts) if p.bottom <= edge + 1]
    taken: list[_Line] = []
    for part in ordered[:2]:
        gap = part.top - edge if side == "above" else edge - part.bottom
        if gap > 0.8 * part.height or _prose(part, info.columns) or _float_caption(part):
            break
        if part.x0 < lo or part.x1 > hi:
            break
        taken.append(part)
        edge = part.bottom if side == "above" else part.top
    return taken


def _walk(
    info: _PageInfo, caption: _Caption, side: Side, scope: tuple[float, float]
) -> list[_Line]:
    """The lines next to the caption, in scope, that make up a table, nearest first."""
    lo, hi = scope
    parts = [p for line in info.lines if (p := line.view(lo, hi)) is not None]
    if side == "above":
        ordered = [p for p in parts if p.top >= caption.bottom - 0.2 * caption.height]
        edge = caption.bottom
    else:
        ordered = [p for p in reversed(parts) if p.bottom <= caption.top + 0.2 * caption.height]
        edge = caption.top
    taken: list[_Line] = []
    gaps: list[float] = []
    for part in ordered:
        gap = part.top - edge if side == "above" else edge - part.bottom
        h = part.height
        if taken:
            limit = max(MAX_ROW_GAP * h, 3 * statistics.median(gaps) if gaps else 0.0)
        else:
            limit = MAX_LEAD * h
        if gap > limit:
            break
        if _float_caption(part) or _prose(part, info.columns):
            break
        if taken:
            gaps.append(max(gap, 0.0))
        taken.append(part)
        edge = part.bottom if side == "above" else part.top
    while taken and not _table_like(taken[-1]):
        taken.pop()
    return taken


def _by_text(info: _PageInfo, caption: _Caption, side: Side) -> _Candidate | None:
    scope = caption.scope
    taken = _walk(info, caption, side, scope)
    full = (info.text_x0, info.text_x1)
    if info.gutter is not None and scope != full and taken and _runs_past(info, taken, scope):
        scope = full
        taken = _walk(info, caption, side, scope)
    rows = sum(map(_table_like, taken))
    if rows < 2:
        return None
    lead = taken[0].top - caption.bottom if side == "above" else caption.top - taken[0].bottom
    top, bottom = min(p.top for p in taken), max(p.bottom for p in taken)
    x0, x1 = min(p.x0 for p in taken), max(p.x1 for p in taken)
    if _figure(info, x0, top, x1, bottom):
        return None
    return _Candidate("text", side, top, bottom, x0, x1, max(lead, 0.0), rows)


def _runs_past(info: _PageInfo, taken: Sequence[_Line], scope: tuple[float, float]) -> bool:
    """Whether the table goes on past the caption's column: most of its rows have a word
    across the gutter, or nearly all go on with table cells in the other column and that
    column has no caption of its own next to them."""
    rows = [p for p in taken if _table_like(p)]
    if not rows or info.gutter is None:
        return False
    lo, hi = scope
    gutter = info.gutter
    crossing = past = 0
    for part in rows:
        line = next(line for line in info.lines if part.words[0] in line.words)
        crossing += any(w.x0 < gutter < w.x1 for w in line.words)
        outside = tuple(w for w in line.words if not lo <= w.cx <= hi)
        if outside and not _prose(_Line(outside), info.columns):
            past += _table_like(_Line(outside)) or any(_numeric(w.text) for w in outside)
    if crossing >= 0.5 * len(rows):
        return True
    top, bottom = rows[0].top, rows[-1].bottom
    top, bottom = min(top, bottom), max(top, bottom)
    margin = 6 * rows[0].height
    other = (gutter, info.text_x1) if lo < gutter else (info.text_x0, gutter)
    for line in info.lines:
        if top - margin <= line.top <= bottom + margin:
            beside = line.view(*other)
            if beside is not None and _float_caption(beside):
                return False
    return past >= 0.7 * len(rows)


def _by_image(info: _PageInfo, caption: _Caption, side: Side) -> _Candidate | None:
    lo, hi = caption.scope
    h = caption.height
    best: _Candidate | None = None
    for x0, top, x1, bottom in info.images:
        if x1 - x0 < 20 or bottom - top < 10:
            continue
        if min(x1, hi) - max(x0, lo) < 0.5 * (x1 - x0):
            continue
        lead = top - caption.bottom if side == "above" else caption.top - bottom
        if not -h <= lead <= 2 * MAX_LEAD * h:
            continue
        if not _clear(info, caption, top if side == "above" else bottom, x0, x1):
            continue
        if best is None or lead < best.lead:
            best = _Candidate("image", side, top, bottom, x0, x1, max(lead, 0.0), 0)
    return best


def _candidate(info: _PageInfo, caption: _Caption, side: Side) -> _Candidate | None:
    return (
        _by_rules(info, caption, side)
        or _by_text(info, caption, side)
        or _by_image(info, caption, side)
    )


def _page_info(page: Page, number: int) -> _PageInfo | None:
    words = _words(page)
    if not words:
        return None
    lines = _lines(words)
    x0 = min(w.x0 for w in words)
    x1 = max(w.x1 for w in words)
    bbox = tuple(float(v) for v in page.bbox)
    images = [
        (float(i["x0"]), float(i["top"]), float(i["x1"]), float(i["bottom"])) for i in page.images
    ]
    info = _PageInfo(
        number=number,
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        lines=lines,
        rules=[],
        images=images,
        text_x0=x0,
        text_x1=x1,
        gutter=None,
    )
    if not any(_CAPTION_WORD.match(w.text) for w in words):
        return info
    info.gutter = _gutter(lines, x0, x1)
    if not any(_caption_start(line, info.gutter) for line in lines):
        return info
    info.rules = _rules(page)
    info.drawings = _drawings(page)
    info.captions = _captions(info)
    return info


_METHOD_RANK = {"rules": 2, "text": 1, "image": 0}


def _overlap(a: _Candidate, b: _Candidate) -> bool:
    """Whether two tables share more than half of the smaller one's area."""
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.bottom, b.bottom) - max(a.top, b.top)
    if width <= 0 or height <= 0:
        return False
    smaller = min((a.x1 - a.x0) * (a.bottom - a.top), (b.x1 - b.x0) * (b.bottom - b.top))
    return width * height > 0.5 * max(smaller, 1e-6)


def _resolve(
    pages: Sequence[_PageInfo],
) -> list[tuple[_PageInfo, _Caption, _Candidate]]:
    """The table chosen for each caption. A caption's options are ranked by method (rules,
    text, image), then by the side most unambiguous captions in the paper use, then by
    nearness. Captions whose best option is surer choose first (rules before text, the
    paper's usual side, fewer options), and a caption may not take a table another
    caption on the page has taken."""
    options: list[tuple[_PageInfo, _Caption, list[_Candidate]]] = []
    for info in pages:
        for caption in info.captions:
            found = [c for side in ("above", "below") if (c := _candidate(info, caption, side))]
            if found:
                options.append((info, caption, found))
    votes: Counter[str] = Counter()
    for _, _, found in options:
        best = max(_METHOD_RANK[c.method] for c in found)
        sides = {c.side for c in found if _METHOD_RANK[c.method] == best}
        if len(sides) == 1:
            votes[sides.pop()] += 1
    preferred: str | None = None
    if votes:
        (side, count), *rest = votes.most_common()
        if not rest or count > rest[0][1]:
            preferred = side
    for _, _, found in options:
        found.sort(key=lambda c: (-_METHOD_RANK[c.method], c.side != preferred, c.lead))
    order = sorted(
        range(len(options)),
        key=lambda k: (
            -_METHOD_RANK[options[k][2][0].method],
            options[k][2][0].side != preferred,
            len(options[k][2]),
            options[k][2][0].lead,
        ),
    )
    taken: dict[int, list[_Candidate]] = {}
    chosen: list[tuple[_PageInfo, _Caption, _Candidate]] = []
    for k in order:
        info, caption, found = options[k]
        page = taken.setdefault(info.number, [])
        for candidate in found:
            if not any(_overlap(candidate, other) for other in page):
                page.append(candidate)
                chosen.append((info, caption, candidate))
                break
    return chosen


def _box(info: _PageInfo, caption: _Caption, table: _Candidate) -> TableBox:
    px0, ptop, px1, pbottom = info.bbox
    bbox = (
        max(px0, min(table.x0, caption.x0) - PADDING),
        max(ptop, min(table.top, caption.top) - PADDING),
        min(px1, max(table.x1, caption.x1) + PADDING),
        min(pbottom, max(table.bottom, caption.bottom) + PADDING),
    )
    return TableBox(info.number, caption.label, bbox, table.method, table.side)


def _analyse(document: PDF) -> tuple[list[TableBox], list[tuple[int, str]]]:
    """The boxes found in the document, and (page, label) of every table caption."""
    pages = [
        info
        for number, page in enumerate(document.pages, start=1)
        if (info := _page_info(page, number)) is not None and info.captions
    ]
    boxes: dict[tuple[int, str], tuple[TableBox, int]] = {}
    for info, caption, table in _resolve(pages):
        key = (info.number, caption.label)
        quality = _METHOD_RANK[table.method] * 1000 + table.rows
        if key not in boxes or quality > boxes[key][1]:
            boxes[key] = (_box(info, caption, table), quality)
    found = sorted((box for box, _ in boxes.values()), key=lambda b: (b.page, b.bbox[1], b.bbox[0]))
    captions = list(
        dict.fromkeys((info.number, caption.label) for info in pages for caption in info.captions)
    )
    return found, captions


def find_tables(pdf: Path) -> tuple[list[TableBox], list[tuple[int, str]]]:
    """The box of every captioned table found in the PDF, and (page, label) of every table
    caption, found or not."""
    with pdfplumber.open(pdf) as document:
        return _analyse(document)


def crop_boxes(pdf: Path) -> list[TableBox]:
    """The box of every captioned table in the PDF, without rendering anything."""
    with pdfplumber.open(pdf) as document:
        return _analyse(document)[0]


def table_captions(pdf: Path) -> list[tuple[int, str]]:
    """(page, label) of every table caption in the PDF, whether or not its table was
    found."""
    with pdfplumber.open(pdf) as document:
        return _analyse(document)[1]


def box_text(page: Page, bbox: Bbox) -> str:
    """The PDF's text inside `bbox`: one printed line per line, with cells (runs of words
    split at wide gaps) joined by " | "."""
    x0, top, x1, bottom = bbox
    words = [
        w for w in _words(page) if x0 <= w.cx <= x1 and top <= (w.top + w.bottom) / 2 <= bottom
    ]
    return "\n".join(
        " | ".join(" ".join(w.text for w in run) for run in line.segments())
        for line in _lines(words)
    )


def table_crops(pdf: Path, dpi: int = 300) -> list[TableCrop]:
    """Every captioned table in the PDF, rendered on its own at `dpi`, with its text."""
    crops = []
    with pdfplumber.open(pdf) as document:
        for box in _analyse(document)[0]:
            page = document.pages[box.page - 1]
            image = page.crop(box.bbox).to_image(resolution=dpi)
            buffer = io.BytesIO()
            image.original.save(buffer, format="PNG")
            crops.append(
                TableCrop(
                    box.page,
                    box.label,
                    box.bbox,
                    box.method,
                    box.caption,
                    png=buffer.getvalue(),
                    text=box_text(page, box.bbox),
                )
            )
    return crops
