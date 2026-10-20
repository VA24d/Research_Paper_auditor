"""Table boxes on small PDFs written by hand: Courier text (6 pt per character at 10 pt, so
widths are exact), lines drawn as strokes."""

from pathlib import Path

from auditor.parser.table_crops import (
    TableBox,
    box_rows,
    crop_boxes,
    table_captions,
    table_crops,
)

PAGE_WIDTH, PAGE_HEIGHT = 612, 792
SIZE = 10
CHAR = 0.6 * SIZE  # Courier advance width


class Page:
    """Drawing operations in top-left coordinates, as pdfplumber reports them."""

    def __init__(self) -> None:
        self.ops: list[str] = []

    def text(self, x: float, top: float, text: str) -> None:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        baseline = PAGE_HEIGHT - top - 0.8 * SIZE
        self.ops.append(f"BT /F1 {SIZE} Tf {x:.2f} {baseline:.2f} Td ({escaped}) Tj ET")

    def rule(self, x0: float, x1: float, top: float) -> None:
        y = PAGE_HEIGHT - top
        self.ops.append(f"0.5 w {x0:.2f} {y:.2f} m {x1:.2f} {y:.2f} l S")

    def prose(self, x: float, top: float, lines: int, chars: int) -> float:
        """`lines` lines of running text `chars` characters wide; returns the next top."""
        words = [
            "the",
            "method",
            "reads",
            "every",
            "table",
            "in",
            "the",
            "paper",
            "and",
            "checks",
            "it",
            "again",
        ]
        for k in range(lines):
            line, i = "", k
            while len(line) < chars:
                line += words[i % len(words)] + " "
                i += 1
            self.text(x, top, line[:chars].rstrip())
            top += 12
        return top

    def row(self, x: float, top: float, cells: list[str], width: int = 12) -> None:
        self.text(x, top, "".join(cell.ljust(width) for cell in cells).rstrip())


def write_pdf(path: Path, pages: list[Page]) -> Path:
    """A PDF with one Courier font and the pages' drawing operations."""
    count = len(pages)
    kids = " ".join(f"{4 + 2 * k} 0 R" for k in range(count))
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    ]
    for k, page in enumerate(pages):
        stream = "\n".join(page.ops).encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Contents {5 + 2 * k} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n" + stream.decode("latin-1") + "\nendstream"
        )
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("latin-1")
    path.write_bytes(out)
    return path


def inside(box: TableBox, x0: float, top: float, x1: float, bottom: float) -> bool:
    """Whether the region lies within the box."""
    bx0, btop, bx1, bbottom = box.bbox
    return bx0 <= x0 and btop <= top and bx1 >= x1 and bbottom >= bottom


def ruled_table_page() -> Page:
    """One column: prose, a caption, a booktabs table below it, prose."""
    page = Page()
    top = page.prose(72, 72, 6, 78)
    page.text(150, top + 12, "Table 1: Accuracy of the three methods.")
    rules_top = top + 40
    page.rule(140, 470, rules_top)
    page.row(150, rules_top + 4, ["Method", "Dev", "Test", "Params"])
    page.rule(140, 470, rules_top + 18)
    for k, name in enumerate(["Base", "Large", "Ours"]):
        page.row(150, rules_top + 22 + 12 * k, [name, f"7{k}.1", f"6{k}.5", f"1{k}M"])
    page.rule(140, 470, rules_top + 60)
    page.prose(72, rules_top + 84, 6, 78)
    return page


def test_a_ruled_table_below_its_caption_is_cut_at_its_rules(tmp_path: Path) -> None:
    page = ruled_table_page()
    pdf = write_pdf(tmp_path / "ruled.pdf", [page])
    (box,) = crop_boxes(pdf)
    assert (box.page, box.label, box.method, box.caption) == (1, "Table 1", "rules", "above")
    top = 72 + 6 * 12
    # The caption and every rule are inside; the prose above and below is not.
    assert inside(box, 150, top + 12, 140 + 0.6 * 39 * SIZE, top + 40 + 60)
    assert box.bbox[1] > top - 2
    assert box.bbox[3] < top + 40 + 84
    assert box.bbox[0] >= 130
    assert box.bbox[2] <= 480


def test_an_unruled_table_above_its_caption_is_found_from_its_text(tmp_path: Path) -> None:
    page = Page()
    top = page.prose(72, 72, 6, 78)
    first_row = top + 24
    page.row(120, first_row, ["Variant", "BLEU", "chrF", "Speed"], width=14)
    for k, name in enumerate(["full", "no-attn", "no-copy", "small"]):
        page.row(
            120, first_row + 14 + 12 * k, [name, f"2{k}.4", f"5{k}.0", f"{k + 1}.3x"], width=14
        )
    caption_top = first_row + 14 + 12 * 4 + 10
    page.text(150, caption_top, "Table 2: Ablation of the model parts.")
    page.prose(72, caption_top + 34, 6, 78)
    pdf = write_pdf(tmp_path / "unruled.pdf", [page])
    (box,) = crop_boxes(pdf)
    assert (box.label, box.method, box.caption) == ("Table 2", "text", "below")
    assert inside(box, 120, first_row, 120 + 0.6 * 14 * 3 * SIZE, caption_top + SIZE)
    assert box.bbox[1] > top - 2  # the prose above is left out
    assert box.bbox[3] < caption_top + 34


def two_column_page() -> Page:
    """Two columns of prose; a ruled table in the left one, with prose beside it."""
    page = Page()
    left, right, chars = 50, 320, 40  # columns 240 pt wide
    page.prose(right, 72, 40, chars)
    top = page.prose(left, 72, 12, chars)
    page.text(left + 12, top + 12, "Table 3: Left column results.")
    rules_top = top + 34
    page.rule(left + 6, left + 234, rules_top)
    page.row(left + 12, rules_top + 4, ["Model", "F1", "AUC"], width=10)
    page.rule(left + 6, left + 234, rules_top + 18)
    for k in range(4):
        page.row(left + 12, rules_top + 22 + 12 * k, [f"m{k}", f"0.{k}5", f"0.9{k}"], width=10)
    page.rule(left + 6, left + 234, rules_top + 72)
    page.prose(left, rules_top + 90, 14, chars)
    return page


def test_a_table_in_one_column_keeps_to_its_column(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "columns.pdf", [two_column_page()])
    (box,) = crop_boxes(pdf)
    assert (box.label, box.method, box.caption) == ("Table 3", "rules", "above")
    assert box.bbox[0] >= 40
    assert box.bbox[2] < 320  # the right column's prose is left out
    assert box.bbox[2] > 50 + 228


def test_side_by_side_tables_get_one_box_each(tmp_path: Path) -> None:
    page = Page()
    top = page.prose(72, 72, 6, 78)
    for x, label in ((72, "Table 4: Left."), (330, "Table 5: Right.")):
        page.text(x + 10, top + 12, label)
        page.rule(x, x + 200, top + 32)
        page.row(x + 6, top + 36, ["Run", "Acc"], width=10)
        page.rule(x, x + 200, top + 50)
        for k in range(3):
            page.row(x + 6, top + 54 + 12 * k, [f"r{k}", f"9{k}.0"], width=10)
        page.rule(x, x + 200, top + 92)
    page.prose(72, top + 110, 6, 78)
    pdf = write_pdf(tmp_path / "beside.pdf", [page])
    left, right = sorted(crop_boxes(pdf), key=lambda b: b.bbox[0])
    assert (left.label, right.label) == ("Table 4", "Table 5")
    assert left.method == right.method == "rules"
    assert left.bbox[2] < 330
    assert right.bbox[0] > 272


def test_a_reference_in_running_text_is_not_a_caption(tmp_path: Path) -> None:
    page = Page()
    page.text(72, 72, "As shown in Table 1: the method works on every benchmark here.")
    page.prose(72, 84, 4, 78)
    pdf = write_pdf(tmp_path / "reference.pdf", [page])
    assert crop_boxes(pdf) == []
    assert table_captions(pdf) == []


def test_crops_are_rendered_with_their_text(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "ruled.pdf", [Page(), ruled_table_page()])
    (crop,) = table_crops(pdf, dpi=144)
    assert crop.page == 2
    assert crop.png.startswith(b"\x89PNG")
    width = int.from_bytes(crop.png[16:20], "big")
    assert abs(width - 2 * (crop.bbox[2] - crop.bbox[0])) <= 2
    assert crop.text.splitlines()[0].startswith("Table 1: Accuracy")
    assert "Base | 70.1 | 60.5 | 10M" in crop.text
    assert table_captions(pdf) == [(2, "Table 1")]


def test_an_ieee_caption_with_its_title_on_the_next_line(tmp_path: Path) -> None:
    page = Page()
    top = page.prose(72, 72, 6, 78)
    page.text(280, top + 12, "TABLE II")
    page.text(170, top + 24, "Statistics of the corpora used for the tests")
    rules_top = top + 44
    page.rule(150, 460, rules_top)
    page.row(160, rules_top + 4, ["Corpus", "Speakers", "Hours"])
    page.rule(150, 460, rules_top + 18)
    for k in range(3):
        page.row(160, rules_top + 22 + 12 * k, [f"set{k}", f"{k + 4}0", f"1{k}.5"])
    page.rule(150, 460, rules_top + 60)
    page.prose(72, rules_top + 80, 6, 78)
    pdf = write_pdf(tmp_path / "ieee.pdf", [page])
    (box,) = crop_boxes(pdf)
    assert (box.label, box.method, box.caption) == ("Table II", "rules", "above")
    assert inside(box, 170, top + 12, 450, rules_top + 60)


def wrapped_table_page() -> Page:
    """A one-column page with a ruled table set into the running text on the right, as
    LaTeX's wraptable sets it: each line holds a column of prose on the left and, a little
    more than a space away, a caption line or a table row on the right."""
    page = Page()
    top = page.prose(72, 72, 6, 78)
    left = [
        "the method reads every table in every",
        "paper and checks it again and the",
        "method reads every table in the paper",
        "and checks it again and the method",
        "reads every table in the paper and it",
        "checks it again and the method reads",
        "every table in the paper and checks",
        "it again and the method reads every",
        "table in the paper and checks it and",
        "the method reads every table in it",
    ]
    right = [
        "Table 6: Human ratings of the rationales",
        "from three audio sources. Higher scores",
        "are better. Rubric: 1-3 poor; 4-6 fair;",
        "7-8 good; 9-10 precise and grounded.",
        "Means over items.",
    ]
    for k, text in enumerate(left):
        page.text(72, top + 12 * k, text)
    for k, text in enumerate(right):
        page.text(72 + 39 * CHAR, top + 12 * k, text)
    rules_top = top + 12 * len(right) + 4
    page.rule(320, 520, rules_top)
    page.row(330, rules_top + 4, ["Source", "Mean"], width=20)
    page.rule(320, 520, rules_top + 17)
    rows = [("GPT-4", "7.07"), ("Moshi", "6.85"), ("Simulation audio", "6.30")]
    for k, (name, mean) in enumerate(rows):
        page.row(330, rules_top + 20 + 11 * k, [name, mean], width=20)
    page.rule(320, 520, rules_top + 54)
    page.prose(72, top + 12 * len(left) + 12, 6, 78)
    return page


def test_a_table_set_beside_running_text_is_found(tmp_path: Path) -> None:
    # arXiv 2512.21706 Table 6: the caption follows the prose on its line after a gap of
    # one line height, which was taken for a reference inside a sentence.
    pdf = write_pdf(tmp_path / "wrapped.pdf", [wrapped_table_page()])
    (crop,) = table_crops(pdf, dpi=72)
    assert (crop.label, crop.method, crop.caption) == ("Table 6", "rules", "above")
    assert crop.bbox[0] > 72 + 37 * CHAR  # the prose on the left is left out
    assert box_rows(crop.text, crop.label, crop.caption) == [
        "Source | Mean",
        "GPT-4 | 7.07",
        "Moshi | 6.85",
        "Simulation audio | 6.30",
    ]


def test_a_wide_space_in_a_sentence_does_not_start_a_caption(tmp_path: Path) -> None:
    page = Page()
    top = page.prose(72, 72, 4, 78)
    # The gap before "Table" is wider than a space, but the lines below have none there.
    page.text(72, top, "the method reads every table in the")
    page.text(72 + 37 * CHAR, top, "Table 1: shows it again and the method")
    page.prose(72, top + 12, 4, 78)
    pdf = write_pdf(tmp_path / "gap.pdf", [page])
    assert table_captions(pdf) == []


def test_the_short_last_line_of_a_caption_is_not_a_row() -> None:
    above = [
        "Table 2: Scores of the three systems on the test set, with the",
        "mean over five runs and the spread over the five seeds.",
        "Means across items.",
        "System | Score",
        "Base | 70.1",
    ]
    assert box_rows("\n".join(above), "Table 2", "above") == ["System | Score", "Base | 70.1"]
    below = [
        "System | Score",
        "Base | 70.1",
        "Table 2: Scores of the three systems on the test set, with the",
        "mean over five runs of the same amount (15%) of tokens.",
        "over five seeds.",
    ]
    assert box_rows("\n".join(below), "Table 2", "below") == ["System | Score", "Base | 70.1"]
    # A line of justified caption text with a wide space after a full stop.
    wide = ["Table 4: Ratings of the sources.", "Means across all items. | Higher is better."]
    assert box_rows("\n".join([*wide, "Source | Mean"]), "Table 4", "above") == ["Source | Mean"]
    # Rows of words with one wide gap are rows, not caption lines.
    rows = ["Table 3: Layers.", "Input: 28 x 28 vectors | Input: 128 noise"]
    assert box_rows("\n".join(rows), "Table 3", "above") == [rows[1]]
