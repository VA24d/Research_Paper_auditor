"""Build mid-submission.pdf with ReportLab. This script holds the report's text.

Run: uv run --no-project --with reportlab python docs/build_mid_submission.py
"""

# ruff: noqa: E501

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).parent
FONTS = Path("/System/Library/Fonts/Supplemental")
NAVY = colors.HexColor("#1F3A5F")
BLUE = colors.HexColor("#2F6DB5")
TINT = colors.HexColor("#F1F5FA")
LINE = colors.HexColor("#CBD6E4")
GREY = "#5B6776"
GREEN = "#1B7F4B"
RED = "#B3261E"
AMBER = "#A86200"
WIDTH = 180 * mm

pdfmetrics.registerFont(TTFont("Arial", FONTS / "Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", FONTS / "Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", FONTS / "Arial Italic.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", FONTS / "Arial Bold Italic.ttf"))
pdfmetrics.registerFontFamily(
    "Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic"
)

BODY = ParagraphStyle(
    "body", fontName="Arial", fontSize=9, leading=12, textColor="#1B2430", spaceAfter=4
)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=10, bulletIndent=1, spaceAfter=2)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
CELL_BULLET = ParagraphStyle("cell_bullet", parent=CELL, leftIndent=7, bulletIndent=0, spaceAfter=1)
HEAD = ParagraphStyle("head", parent=CELL, fontName="Arial-Bold", textColor=colors.white)
SECTION = ParagraphStyle(
    "section",
    parent=BODY,
    fontName="Arial-Bold",
    fontSize=12,
    leading=15,
    textColor=NAVY,
    spaceBefore=8,
    spaceAfter=0,
)
SUB = ParagraphStyle(
    "sub", parent=BODY, fontName="Arial-Bold", fontSize=9.5, textColor=NAVY, spaceBefore=8
)
CAPTION = ParagraphStyle("caption", parent=CELL, alignment=TA_CENTER, textColor=GREY)
TITLE = ParagraphStyle(
    "title", parent=BODY, fontName="Arial-Bold", fontSize=17, leading=21, textColor=colors.white
)
SUBTITLE = ParagraphStyle(
    "subtitle", parent=BODY, fontSize=9.5, textColor=colors.HexColor("#D5E2F2"), spaceAfter=0
)


def good(text: str) -> str:
    return f'<font color="{GREEN}"><b>{text}</b></font>'


def bad(text: str) -> str:
    return f'<font color="{RED}"><b>{text}</b></font>'


def warn(text: str) -> str:
    return f'<font color="{AMBER}"><b>{text}</b></font>'


def section(text: str) -> Table:
    """A section heading with a coloured rule under it."""
    table = Table([[Paragraph(text, SECTION)]], colWidths=[WIDTH], spaceBefore=14)
    table.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 1.2, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    table.keepWithNext = True
    return table


def bullets(*items: str) -> list[Paragraph]:
    return [Paragraph(item, BULLET, bulletText="•") for item in items]


def cell_body(cell: str | list[str]) -> Paragraph | list[Paragraph]:
    """A table cell: plain text, or a list of lines shown as bullets."""
    if isinstance(cell, str):
        return Paragraph(cell, CELL)
    return [Paragraph(line, CELL_BULLET, bulletText="\u2022") for line in cell]


def grid(
    header: list[str], rows: list[list], widths: list[float], extra: list | None = None
) -> Table:
    """A table with a navy header row and tinted alternate rows; widths are in mm."""
    data = [[Paragraph(cell, HEAD) for cell in header]]
    data += [[cell_body(cell) for cell in row] for row in rows]
    table = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TINT]),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                *(extra or []),
            ]
        )
    )
    return table


def banner() -> Table:
    table = Table(
        [
            [Paragraph("Research Paper Auditor", TITLE)],
            [
                Paragraph(
                    "Mid-project submission | Team Hood Research Department | 30 October 2026",
                    SUBTITLE,
                )
            ],
        ],
        colWidths=[WIDTH],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (0, 0), 9),
                ("BOTTOMPADDING", (0, 0), (0, 0), 1),
                ("TOPPADDING", (0, 1), (0, 1), 1),
                ("BOTTOMPADDING", (0, 1), (0, 1), 9),
            ]
        )
    )
    return table


def team() -> Table:
    url = "https://github.com/itikelabhaskar/Research_Paper_auditor"
    link = f'<a href="{url}" color="#2F6DB5"><u>github.com/itikelabhaskar/Research_Paper_auditor</u></a>'
    left = [
        Paragraph("Team members", SUB),
        *bullets(
            "Vijay Aravynthan (2023111007)",
            "Anurag Peddi (2023101090)",
            "Itikela Bhaskar (2023111032)",
            "Sudheera Y S (2023111002)",
        ),
    ]
    right = [Paragraph("Code repository", SUB), *bullets(link)]
    table = Table([[left, right]], colWidths=[WIDTH * 0.42, WIDTH * 0.58])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), TINT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def share(found: int, total: int) -> str:
    return f"{round(100 * found / total)}% ({found} of {total})"


class Bar(Flowable):
    """A percentage bar with its label to the right."""

    def __init__(
        self,
        pct: float,
        text: str,
        colour: colors.Color,
        width: float = 34 * mm,
        bar: float = 20 * mm,
    ) -> None:
        super().__init__()
        self.pct, self.text, self.colour, self.width, self.bar = pct, text, colour, width, bar

    def wrap(self, *_: float) -> tuple[float, float]:
        return self.width, 10

    def draw(self) -> None:
        c = self.canv
        c.setFillColor(LINE)
        c.rect(0, 2, self.bar, 6, stroke=0, fill=1)
        c.setFillColor(self.colour)
        c.rect(0, 2, self.bar * self.pct / 100, 6, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#1B2430"))
        c.setFont("Arial-Bold", 8)
        c.drawString(self.bar + 4, 2.2, self.text)


class SystemDiagram(Flowable):
    """The system as a vector drawing: inputs, specialists, then the parts not built yet."""

    INK = colors.HexColor("#1B2430")
    PALE = colors.HexColor("#E3EDF9")
    INPUT = colors.HexColor("#FFF4D6")
    INPUT_LINE = colors.HexColor("#D9A93A")
    MISSING = colors.HexColor(RED)

    def __init__(self) -> None:
        super().__init__()
        self.width, self.height = WIDTH, 176
        self.hAlign = "CENTER"

    def wrap(self, *_: float) -> tuple[float, float]:
        return self.width, self.height

    def box(
        self, x: float, y: float, w: float, h: float, title: str, text: str, style: str = "step"
    ) -> None:
        c = self.canv
        fill, line, ink = {
            "step": (NAVY, NAVY, colors.white),
            "specialist": (self.PALE, BLUE, self.INK),
            "input": (self.INPUT, self.INPUT_LINE, self.INK),
            "output": (colors.white, NAVY, self.INK),
            "missing": (colors.white, self.MISSING, self.INK),
        }[style]
        c.saveState()
        c.setFillColor(fill)
        c.setStrokeColor(line)
        c.setLineWidth(0.9)
        if style == "missing":
            c.setDash(3, 2)
        c.roundRect(x, y, w, h, 3, stroke=1, fill=1)
        c.restoreState()
        lines = simpleSplit(text, "Arial", 6.8, w - 8)
        top = y + h / 2 + (9 + 8 * len(lines)) / 2 - 8
        c.setFillColor(ink)
        c.setFont("Arial-Bold", 8)
        if style == "missing":
            tag = "  not built yet"
            full = stringWidth(title, "Arial-Bold", 8) + stringWidth(tag, "Arial-Bold", 6.5)
            start = x + (w - full) / 2
            c.drawString(start, top, title)
            c.setFont("Arial-Bold", 6.5)
            c.setFillColor(self.MISSING)
            c.drawString(start + stringWidth(title, "Arial-Bold", 8), top, tag)
            c.setFillColor(ink)
        else:
            c.drawCentredString(x + w / 2, top, title)
        c.setFont("Arial", 6.8)
        for i, line_text in enumerate(lines, 1):
            c.drawCentredString(x + w / 2, top - 8 * i - 1, line_text)

    def arrow(self, x1: float, y1: float, x2: float, y2: float) -> None:
        c = self.canv
        c.setStrokeColor(NAVY)
        c.setFillColor(NAVY)
        c.setLineWidth(0.9)
        c.line(x1, y1, x2, y2)
        size = 3.2
        path = c.beginPath()
        if x1 == x2:
            path.moveTo(x2, y2)
            path.lineTo(x2 - size, y2 + size * 1.5)
            path.lineTo(x2 + size, y2 + size * 1.5)
        else:
            path.moveTo(x2, y2)
            path.lineTo(x2 - size * 1.5, y2 - size)
            path.lineTo(x2 - size * 1.5, y2 + size)
        path.close()
        c.drawPath(path, stroke=0, fill=1)

    def draw(self) -> None:
        c = self.canv
        # Top row: what goes in.
        self.box(0, 140, 74, 34, "Submission", "paper as PDF or LaTeX", "input")
        self.box(92, 140, 86, 34, "Parser", "reads text, tables and references")
        self.box(196, 140, 86, 34, "Planner", "rules and model passes pick items to check")
        self.box(302, 140, 92, 34, "Code repository", "official code, URL from the paper", "input")
        self.box(420, 140, 90, 34, "Venue rules", "page limit, template", "input")
        self.arrow(74, 157, 92, 157)
        self.arrow(178, 157, 196, 157)

        # Middle row: the specialists.
        c.setFillColor(TINT)
        c.setStrokeColor(LINE)
        c.roundRect(0, 62, WIDTH, 66, 4, stroke=1, fill=1)
        c.setFillColor(NAVY)
        c.setFont("Arial-Bold", 8)
        c.drawString(7, 117, "Specialists: each checks its own items")
        specialists = [
            ("Citations", "references exist; the cited source supports the claim"),
            ("Theory", "proofs, derivations, stated results"),
            ("Integrity, results", "text against tables; placeholder text"),
            ("Code", "settings, metrics and method against the code; defects"),
            ("Data", "splits, sizes, labels, leakage"),
            ("Template", "pages, fonts, margins, anonymity"),
        ]
        gap = 5
        w = (WIDTH - 12 - gap * 5) / 6
        lefts = [6 + i * (w + gap) for i in range(6)]
        for left, (title, text) in zip(lefts, specialists, strict=True):
            self.box(left, 66, w, 46, title, text, "specialist")
        self.arrow(239, 140, 239, 128)
        code_mid, data_mid = lefts[3] + w * 0.75, lefts[4] + w * 0.25
        self.arrow(code_mid, 140, code_mid, 112)
        self.arrow(data_mid, 140, data_mid, 112)
        self.arrow(465, 140, 465, 112)

        # Bottom row: what comes out.
        self.box(
            40,
            24,
            150,
            30,
            "Findings",
            "each with its quoted evidence, or marked unverifiable",
            "output",
        )
        self.box(
            212,
            24,
            150,
            30,
            "Verifier",
            "re-checks evidence, merges repeats, ranks by severity",
            "missing",
        )
        self.box(384, 24, 126, 30, "Report", "ranked findings with evidence", "missing")
        self.arrow(115, 62, 115, 54)
        self.arrow(190, 39, 212, 39)
        self.arrow(362, 39, 384, 39)

        # Services every part uses.
        c.setFillColor(colors.HexColor("#E8ECF1"))
        c.roundRect(0, 0, WIDTH, 15, 3, stroke=0, fill=1)
        c.setFillColor(self.INK)
        c.setFont("Arial-Bold", 6.8)
        c.drawString(6, 4.6, "Shared:")
        c.setFont("Arial", 6.8)
        c.drawString(
            34,
            4.6,
            "model calls with checked replies  |  spend limits  |  web requests with rate limits  |  pinned code snapshots  |  "
            "local embeddings  |  sandbox  |  run logs",
        )


def rated(pct: float, text: str) -> Bar:
    """A result bar coloured by how much of the known errors it found."""
    colour = GREEN if pct >= 80 else AMBER if pct >= 50 else RED
    return Bar(pct, text, colors.HexColor(colour))


def score_cards() -> Table:
    """Overall score of the three systems as large numbers."""
    number = ParagraphStyle(
        "number", parent=BODY, fontName="Arial-Bold", fontSize=18, leading=21, alignment=TA_CENTER
    )
    cards = [("Full system", "68%", GREEN), ("B0", "14%", GREY), ("B1", "0%", GREY)]
    row = [
        [
            Paragraph(f'<font color="{colour}">{value}</font>', number),
            Paragraph(f"{name}: overall score (macro-F1)", CAPTION),
        ]
        for name, value, colour in cards
    ]
    table = Table([row], colWidths=[WIDTH / 3] * 3)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), TINT),
                ("LINEAFTER", (0, 0), (1, 0), 3, colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def glossary() -> Table:
    items = bullets(
        "<b>Caught:</b> the share of known errors a check found. Counts sit beside each %, as a % from few errors can move a lot.",
        "<b>False alarm:</b> a correct item that a check wrongly flags. Clean controls are unchanged items used to measure this.",
        "<b>F1, macro-F1:</b> a 0% to 100% score combining errors caught and flags that are right; macro-F1 averages over error types.",
    )
    table = Table([[[Paragraph("How to read the numbers", SUB), *items]]], colWidths=[WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), TINT),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def card(question: str, measure: str, rows: list[tuple], takeaway: str) -> Table:
    """One ablation: the question, then each variant with its own bar, then what it means."""
    width = 88 * mm
    data: list[list] = [
        [Paragraph(question, HEAD), ""],
        [Paragraph(f'<font color="{GREY}"><i>{measure}</i></font>', CELL), ""],
    ]
    for label, pct, text, best in rows:
        colour = colors.HexColor(GREEN) if best else BLUE
        value = (
            Paragraph(f"<b>{text}</b>", CELL)
            if pct is None
            else Bar(pct, text, colour, width * 0.52, 18 * mm)
        )
        data.append([Paragraph(label, CELL), value])
    data.append([Paragraph(f"<b>{takeaway}</b>", CELL), ""])
    table = Table(data, colWidths=[width * 0.48, width * 0.52])
    table.setStyle(
        TableStyle(
            [
                ("SPAN", (0, 0), (1, 0)),
                ("SPAN", (0, 1), (1, 1)),
                ("SPAN", (0, -1), (1, -1)),
                ("BACKGROUND", (0, 0), (1, 0), BLUE),
                ("BACKGROUND", (0, -1), (1, -1), TINT),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def results() -> Table:
    """Every check in plain words, grouped by what it compares, with a bar for the share caught."""
    groups = [
        (
            "Citations (set A, 831 references; marker links on 10 papers)",
            [
                (
                    "Citation marker linked to a reference",
                    96,
                    "96%",
                    "765 of 797",
                    "hand check of 100: 99 linked, all correct",
                ),
                (
                    "Reference that does not exist or has wrong details",
                    86,
                    "86%",
                    "",
                    "of flags raised, 90% were real; 16% of valid references wrongly flagged",
                ),
            ],
        ),
        (
            "Paper against its code (set C)",
            [
                (
                    "Code uses a different training setting than the paper",
                    100,
                    "100%",
                    "60 of 60",
                    "0 of 72 clean controls flagged; planted only where our tool had read the value",
                ),
                ("Code computes a different metric than the paper", 67, "67%", "2 of 3", ""),
                (
                    "Code does a different method than the paper describes",
                    30,
                    "30%",
                    "17 of 57",
                    "Gemini 3.8 Flash, about $1",
                ),
            ],
        ),
        (
            "Errors inside the paper (set D, planted by us from templates)",
            [
                ("Leftover placeholder text", 100, "100%", "40 of 40", ""),
                ("Mean impossible for the sample size (GRIM test)", 100, "100%", "40 of 40", ""),
                ("Text reverses a comparison its numbers show", 94, "94%", "31 of 33", ""),
                ("Text states a wrong gain between two table rows", 55, "55%", "18 of 33", ""),
                (
                    "Text quotes a number that differs from its table",
                    20,
                    "20%",
                    "2 of 10",
                    "no false alarm on 254 real papers; its only hit there is a real error",
                ),
            ],
        ),
        (
            "Broken code (set D, planted) and template rules",
            [
                ("Code uses an undefined name", 100, "100%", "26 of 26", ""),
                ("Code has a syntax error", 100, "100%", "13 of 13", ""),
                (
                    "Paper breaks a template rule",
                    100,
                    "100%",
                    "15 of 15",
                    "15 hand-built test papers, not set D",
                ),
            ],
        ),
        (
            "Proofs and derivations (SPOT test half)",
            [
                (
                    "Error in a proof, counted by one LLM judge",
                    57,
                    "57%",
                    "8 of 14",
                    "the judges disagree on some matches",
                ),
                ("Error in a proof, counted by both LLM judges", 36, "36%", "5 of 14", ""),
            ],
        ),
    ]
    data = [[Paragraph(h, HEAD) for h in ("What we test", "Result", "Count", "Notes")]]
    style: list = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for title, rows in groups:
        style += [
            ("SPAN", (0, len(data)), (-1, len(data))),
            ("BACKGROUND", (0, len(data)), (-1, len(data)), TINT),
        ]
        data.append([Paragraph(f'<font color="#1F3A5F"><b>{title}</b></font>', CELL), "", "", ""])
        for what, pct, text, count, note in rows:
            data.append(
                [
                    Paragraph(what, CELL),
                    rated(pct, text),
                    Paragraph(count, CELL),
                    Paragraph(note, CELL),
                ]
            )
    table = Table(data, colWidths=[72 * mm, 32 * mm, 20 * mm, 56 * mm], repeatRows=1)
    table.setStyle(TableStyle(style))
    return table


def padded(content: list) -> Table:
    table = Table([[content]], colWidths=[88 * mm])
    table.setStyle(
        TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 0)])
    )
    return table


def card_grid(pairs: list[tuple[object, object]]) -> Table:
    table = Table([[a, "", b] for a, b in pairs], colWidths=[88 * mm, 4 * mm, 88 * mm])
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]
    for r in range(len(pairs)):
        style += [("BOX", (0, r), (0, r), 0.5, LINE), ("BOX", (2, r), (2, r), 0.5, LINE)]
        if r:
            style += [("LINEABOVE", (1, r), (1, r), 0, colors.white)]
    table.setStyle(TableStyle(style))
    return table


def timeline() -> Table:
    """The plan as a table with a bar for each row over the days from 31 October to 14 November."""
    days = ["31", *(str(d) for d in range(1, 15))]
    plan = [
        (
            "31 Oct to 2 Nov",
            "Label checks, freeze sets B and C; agree the verifier design",
            "all",
            0,
            2,
        ),
        (
            "31 Oct to 4 Nov",
            "First verifier and audit report; evaluation runner over all sets",
            "Parts 4 and 1",
            0,
            4,
        ),
        (
            "3 to 6 Nov",
            "RQ1 runs: all systems on sets B, C, D and clean originals",
            "Parts 1 to 3",
            3,
            6,
        ),
        (
            "5 to 8 Nov",
            "Ablations: no verifier, model-only, abstract-only, single agent",
            "Parts 2 to 4",
            5,
            8,
        ),
        (
            "7 to 10 Nov",
            "Failure analysis; check unlabelled flags; method and table fixes",
            "Parts 1 and 3",
            7,
            10,
        ),
        ("10 to 13 Nov", "Write-up, figures, demo on real papers", "all", 10, 13),
        ("14 Nov", "Final submission, tagged release of the repository", "all", 14, 14),
    ]
    small = ParagraphStyle("small", parent=HEAD, fontSize=5, leading=8, alignment=TA_CENTER)
    head = [Paragraph(t, HEAD) for t in ("Dates", "Work", "Owner")] + [
        Paragraph(d, small) for d in days
    ]
    data = [head] + [
        [Paragraph(d, CELL), Paragraph(w, CELL), Paragraph(o, CELL)] + [""] * len(days)
        for d, w, o, _, _ in plan
    ]
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (2, -1), 4),
        ("RIGHTPADDING", (0, 0), (2, -1), 4),
        ("LEFTPADDING", (3, 0), (-1, -1), 0),
        ("RIGHTPADDING", (3, 0), (-1, -1), 0),
    ]
    for r, (_, _, _, start, end) in enumerate(plan, 1):
        style.append(
            (
                "BACKGROUND",
                (3 + start, r),
                (3 + end, r),
                BLUE if end < 14 else colors.HexColor(GREEN),
            )
        )
    table = Table(data, colWidths=[23 * mm, 89 * mm, 20 * mm] + [3.2 * mm] * len(days))
    table.setStyle(TableStyle(style))
    return table


def story() -> list:
    return [
        banner(),
        team(),
        section("1. Overview and progress"),
        Spacer(1, 5),
        Paragraph(
            "We are building an auditor for research-paper submissions. It reads the paper (PDF or LaTeX) and its code, plans checkable items, sends each to a specialist and reports findings with quoted evidence. RQ1 asks whether this agent system detects more errors, with no more false alarms, than the same model run as one agent in one pass (B1). B0, the same model with no tools, is a second reference.",
            BODY,
        ),
        *bullets(
            f"{good('Running end to end on real papers:')} parsing, planning, orchestration, citation checker, code "
            "and data agent, template checker, theory checker, integrity checks.",
            f"{bad('Not built yet:')} verifier, report, evaluation runner (Part 4).",
            "A placeholder verifier passes every finding to human review, so we score all runs before the verifier.",
        ),
        Spacer(1, 3),
        grid(
            ["Part", "Owner", "Done", "Still to do"],
            [
                [
                    "<b>1.</b> Parser, planner, orchestration, baselines",
                    "Bhaskar",
                    [
                        "Reads papers from PDF and LaTeX",
                        "Planner that picks what to check",
                        "Runs the specialists with retries, resume and a budget per task (LangGraph)",
                        "B0 and B1 baselines, batch runner, Docker image",
                    ],
                    [
                        "Test the planner's optional model step with a real model",
                        "Analyse where the system fails and why",
                    ],
                ],
                [
                    "<b>2.</b> Citation checker",
                    "Anurag",
                    [
                        "Checks that references exist and their details are right (arXiv, OpenAlex, Crossref, Semantic Scholar)",
                        "Checks that a cited source supports the claim, using its full text",
                        "Set A scored, set B built",
                    ],
                    [
                        "Second person checks a sample of set B's labels",
                        "Run the full system on set B",
                        "Compare keyword search (BM25) with embedding search for finding source passages",
                    ],
                ],
                [
                    "<b>3.</b> Code, data, template, theory, integrity",
                    "Vijay",
                    [
                        "Compares a paper's settings, metrics and method with its code",
                        "Checks data analysis and finds code defects",
                        "Agent that tests claims against the repository (claim loop)",
                        "Template, proof, integrity and results checks",
                        "Sets C and D built, SPOT scored",
                    ],
                    [
                        "Second person checks a sample of set C's labels",
                        "Decide whether flags that match no label are real errors",
                        "Run the full system, B0 and B1 on all of sets C and D",
                    ],
                ],
                [
                    "<b>4.</b> Verifier, report, evaluation",
                    "Sudheera",
                    bad("not built yet"),
                    [
                        "Build the verifier, which re-checks each finding against its evidence",
                        "Build the audit report and the runner that scores all sets",
                        "Measure what the verifier adds by removing it",
                    ],
                ],
            ],
            [28, 17, 70, 65],
        ),
        Spacer(1, 12),
        SystemDiagram(),
        Paragraph(
            "System overview: the planner sends each item to a specialist, and every finding carries its evidence.",
            CAPTION,
        ),
        KeepTogether(
            [
                section("2. Datasets"),
                Spacer(1, 5),
                grid(
                    ["Set", "What it tests", "Size", "State"],
                    [
                        [
                            "<b>A:</b> HALLMARK (public benchmark)",
                            "Reference existence and metadata",
                            "831 entries (test-public split)",
                            good("external, scored"),
                        ],
                        [
                            "<b>B:</b> claim-source pairs",
                            "Whether a cited source supports the claim",
                            "60 arXiv papers, 158 gold labels (112 injected, 46 clean controls)",
                            warn("second review pending"),
                        ],
                        [
                            "<b>C:</b> paper-code pairs (from the public SciCoQA benchmark)",
                            "Settings, metrics and method against the official code",
                            "40 pairs, 80 paper versions, 235 gold labels (63 setting and metric errors, 72 clean-control labels, 100 method labels)",
                            warn("second review pending"),
                        ],
                        [
                            "<b>D:</b> planted integrity errors",
                            "Placeholder text, impossible means, prose claims a table contradicts, code defects",
                            "the 40 set C papers, 195 plants",
                            good("built"),
                        ],
                        [
                            "<b>SPOT</b> (public benchmark)",
                            "Errors in proofs and derivations",
                            "26 papers, 29 annotated errors, dev and test halves of 13 papers",
                            good("external, LLM judge"),
                        ],
                        [
                            "<b>Parser fixtures</b>",
                            "Citation linking, metadata",
                            "10 arXiv papers, 100 citation markers",
                            good("hand-checked"),
                        ],
                        [
                            "<b>Real cases</b>",
                            "Errors authors later corrected",
                            "33 arXiv withdrawals and revisions, 30 ICLR review cases",
                            warn("automatic labels"),
                        ],
                    ],
                    [42, 46, 66, 26],
                ),
                Spacer(1, 4),
                *bullets(
                    "We built sets B and C from real papers with planted errors and clean controls, and checked each label automatically.",
                    "A pair is a paper and its official code. Each paper has two versions: the original and a copy with planted errors.",
                    "Sets B and C could be expanded; set B has few number-claim (4) and support-claim (3) labels.",
                ),
            ]
        ),
        KeepTogether(
            [
                section("3. Baseline experiments"),
                Spacer(1, 5),
                *bullets(
                    "<b>B0:</b> the same model in a single pass with no tools.",
                    "<b>B1:</b> the same model as one agent with every tool, in one pass, with no verifier.",
                ),
                Spacer(1, 2),
                glossary(),
                Spacer(1, 4),
                grid(
                    ["Run", "Test", "Errors caught", "Cost"],
                    [
                        [
                            "B0, Gemini 3.8 Flash, high thinking",
                            "SPOT, all 26 papers",
                            share(4, 29),
                            "$3.39 (26 papers)",
                        ],
                        [
                            "B0, same run, test half only",
                            "SPOT test half",
                            share(4, 14),
                            "",
                        ],
                        [
                            "<b>Full system</b> (theory checker), first judge",
                            "SPOT test half",
                            good(share(8, 14)),
                            "$1.68 (13 papers)",
                        ],
                        [
                            "B1, Gemini 2.5 Flash",
                            "Set B, 15 papers, old labels",
                            "macro-F1 2.6%; 0 of 14 clean controls flagged",
                            "",
                        ],
                    ],
                    [62, 38, 50, 30],
                ),
                Spacer(1, 4),
                *bullets(
                    "On the SPOT test half the full system catches 8 errors to B0's 4 by the first judge, at the same cost per paper (about $0.13). If a second judge must also agree, it keeps 5; B0 has no second grading."
                ),
            ]
        ),
        KeepTogether(
            [
                Paragraph("First RQ1 comparison, set C", SUB),
                Paragraph(
                    "We ran all three systems once on Gemini 3.8 Flash, on the first pair of the set and 4 pairs chosen "
                    "at random, with the citation checker off because this set does not test citations.",
                    BODY,
                ),
                score_cards(),
                Spacer(1, 4),
                grid(
                    ["Measure", "Full system", "B0", "B1"],
                    [
                        [
                            "Planted wrong settings caught",
                            good(share(5, 6)),
                            share(1, 6),
                            share(0, 6),
                        ],
                        [
                            "Method differences caught, by the scorer's count",
                            good(share(4, 14)),
                            share(0, 14),
                            share(0, 14),
                        ],
                        [
                            "Method differences caught, read by hand",
                            good(share(2, 14)),
                            share(0, 14),
                            share(0, 14),
                        ],
                        ["False alarms on clean controls", share(0, 6), share(0, 6), share(0, 6)],
                        ["Flags that match no known error, not yet checked", warn("23"), "0", "1"],
                        ["Model calls per paper version", "22", "1", "20"],
                        ["Cost per paper version", "$0.12", "$0.06", "$0.21"],
                    ],
                    [84, 32, 32, 32],
                ),
                Spacer(1, 4),
            ]
        ),
        KeepTogether(
            [
                Paragraph("Caveats on this comparison", SUB),
                *bullets(
                    "The full system leads, but one run on 5 pairs does not test the hypothesis.",
                    "We have not checked the 23 unmatched flags; each may be a real error we did not label or a false alarm.",
                    "The automatic scorer counts a method finding as correct if the paper has any method error, even a different one, so we also read each finding by hand.",
                    "B0 cannot read code, so it can catch a planted setting only from the paper's text.",
                    "B1 used all 20 of its model calls on every version, so we will test a larger limit before the final comparison.",
                    "A sixth pair failed on rate limits and was dropped for all systems. B0 also lost one clean version, so it covers 9 versions.",
                    "We reran the full system on two pairs after the local embedding model unloaded during the first run.",
                ),
            ]
        ),
        KeepTogether(
            [
                section("4. Results"),
                Spacer(1, 5),
                results(),
                Spacer(1, 2),
                Paragraph(
                    "Bar colour: green 80% or more, amber 50 to 79%, red below 50%.", CAPTION
                ),
                Spacer(1, 4),
                Paragraph("Errors our checks found on real papers", SUB),
                *bullets(
                    'arXiv 2312.11580 has a reversed comparison: "PlaNet-S (0.73) is higher than U-Net (0.78)".',
                    "arXiv 2512.21706 gives a mean score of 8.93 that its own Table 6 gives as 6.30.",
                    "arXiv 2006.06958 says it uses no data shuffling, while every training loader in its code shuffles.",
                ),
            ]
        ),
        section("5. Ablations: what changes when we swap one part"),
        Spacer(1, 2),
        Paragraph("A green bar marks the best variant in its card.", CAPTION),
        Spacer(1, 5),
        card_grid(
            [
                (
                    card(
                        "How should we read tables from a PDF?",
                        "Share of table numbers recovered, 12 papers",
                        [
                            ("GROBID parser", 41, "41%", False),
                            ("Text with page layout", 20, "20%", False),
                            ("Both combined", 51, "51%", False),
                            ("Text inside table regions", 98, "98%", True),
                            ("Table image read by a model", 96, "96%", True),
                        ],
                        "Table regions recover almost every number.",
                    ),
                    card(
                        "How should we find the code for a method sentence?",
                        "Share of 84 known differences where the changed line was shown",
                        [
                            ("Word overlap", 42, "42% (35)", False),
                            ("BM25 keyword search", 57, "57% (48)", False),
                            ("Embedding search (nomic)", 70, "70% (59)", True),
                            ("BM25 + embeddings", 68, "68% (57)", False),
                            ("Embeddings + reranker", 68, "68% (57)", False),
                        ],
                        "Embedding search alone did best, by 2 of 84.",
                    ),
                ),
                (
                    card(
                        "How should the model be shown the code?",
                        "Share of 28 known differences found, 10 pairs, earlier version",
                        [
                            ("Whole files", 7, "7% (2)", False),
                            ("One sentence with its code", 0, "0% (0)", False),
                            ("Retrieved chunks", 14, "14% (4)", True),
                            ("Agent with tools (claim loop)", 4, "4% (1)", False),
                        ],
                        "All are low; retrieved chunks found 2 more than whole files.",
                    ),
                    card(
                        "Did our citation fixes work?",
                        "Set A, before and after the author, year and venue fixes",
                        [
                            ("False alarms before", 31, "31%", False),
                            ("False alarms after", 16, "16%", True),
                            ("F1 before", 80, "80%", False),
                            ("F1 after", 88, "88%", True),
                        ],
                        "False alarms halved and F1 rose.",
                    ),
                ),
                (
                    card(
                        "Which model reads code best?",
                        "Share of 57 known differences found",
                        [
                            ("Gemini 3.8 Flash", 30, "30% (17)", False),
                            ("Gemini 3.1 Pro", 33, "33% (19)", True),
                            ("DeepSeek V4.1 Flash", 33, "33% (19)", True),
                            ("Kimi K2", 23, "23% (13)", False),
                            ("GLM-5", 19, "19% (11)", False),
                            ("Gemini 3.5 Flash-Lite", 12, "12% (7)", False),
                        ],
                        "The best two find only 2 more than 3.8 Flash.",
                    ),
                    card(
                        "How should the theory checker be organised?",
                        "Found by the first judge: first two rows of 29, the rest of 14 (test half)",
                        [
                            ("Cheap screen, then deep check", 7, "7% (2 of 29)", False),
                            ("One call", 14, "14% (4 of 29)", False),
                            ("Triage, then deep check", 57, "57% (8 of 14)", False),
                            ("Two triage calls pooled", 50, "50% (7 of 14)", False),
                            ("Two looks at each paper", 50, "50% (7 of 14)", False),
                        ],
                        "Triage beats a screen or one call. If both judges must agree, the last three are 36%, 36% and 43%.",
                    ),
                ),
                (
                    card(
                        "Does showing more code help?",
                        "Share of 57 known differences found",
                        [
                            ("60k characters", 30, "30% (17)", True),
                            ("120k characters", 28, "28% (16)", False),
                        ],
                        "No: doubling the code found one fewer.",
                    ),
                    card(
                        "How hard should the claim loop think?",
                        "Tasks that ran out of time and total cost, 3 papers",
                        [
                            ("Medium effort", None, "17 to 32 timeouts, $0.63", False),
                            ("Low effort, time limit per claim", None, "0 timeouts, $0.47", True),
                        ],
                        "Low effort plus a time limit: no timeouts, 25% less cost.",
                    ),
                ),
                (
                    card(
                        "Do citations need the full source text?",
                        "Set B, macro-F1",
                        [("Full text", 2.6, "2.6%", False), ("Abstract only", 2.6, "2.6%", False)],
                        "No difference, but B1 found almost nothing either way.",
                    ),
                    padded(
                        [
                            Paragraph("Not yet measured", SUB),
                            *bullets(
                                "Removing the verifier (needs Part 4)",
                                "Model-only against tool-using, on set B and the rest of set C",
                                "Whether the right passage is in the top 5 (7 records so far)",
                            ),
                        ]
                    ),
                ),
            ]
        ),
        KeepTogether(
            [
                section("6. Challenges and mitigation"),
                Spacer(1, 5),
                grid(
                    ["Challenge", "Effect", "Mitigation"],
                    [
                        [
                            "<b>Part 4 not built</b>",
                            "we cannot measure what the verifier adds; every finding goes to human review",
                            "we agree the verifier design by 2 November; its first version re-checks each finding's quote against the source",
                        ],
                        [
                            "<b>RQ1 on 5 of 40 set C pairs; B1 at its call limit</b>",
                            "we cannot yet say whether the full system beats B1 by our 10-point target",
                            "test a larger B1 limit, then run all systems on sets B, C and D in early November (B0 once, the other systems three times each)",
                        ],
                        [
                            "<b>Cost of those runs</b>",
                            "about $83 of shared credits for all of set C at those repeats",
                            "spend ledger and dollar cap per run; run B1 and the full system once first, repeat only where close",
                        ],
                        [
                            "<b>Label checks pending</b>",
                            "we cannot freeze sets B and C",
                            "we check 20% of each other's labels by 2 November",
                        ],
                        [
                            "<b>Models overlook code they are shown</b>",
                            "in an earlier run, the model saw the changed line but reported nothing for 46% (26 of 57)",
                            "show each sentence only its best-matching code (12 of 57 so far); repeat looks",
                        ],
                        [
                            "<b>Judges disagree</b>",
                            "on SPOT, 57% against 36% of errors matched",
                            "report both counts; add a third judge for the disagreements",
                        ],
                        [
                            "<b>Unlabelled flags</b>",
                            "152 settings-check flags on set C have no label",
                            "check a sample by hand; some are already real mismatches",
                        ],
                        [
                            "<b>Slow services</b>",
                            "lookups slow runs; rate limits failed papers",
                            "caches; time limits; fewer papers at once; rerun failures",
                        ],
                    ],
                    [36, 62, 82],
                ),
            ]
        ),
        KeepTogether(
            [section("7. Timeline to the final submission (14 November)"), Spacer(1, 5), timeline()]
        ),
    ]


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Arial", 7.5)
    canvas.setFillColor(colors.HexColor(GREY))
    canvas.drawString(15 * mm, 9 * mm, "Research Paper Auditor: mid-project submission")
    canvas.drawRightString(195 * mm, 9 * mm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(HERE / "mid-submission.pdf"),
    pagesize=A4,
    leftMargin=15 * mm,
    rightMargin=15 * mm,
    topMargin=12 * mm,
    bottomMargin=11 * mm,
    title="Research Paper Auditor: mid-project submission",
    author="Team Hood Research Department",
)
doc.build(story(), onFirstPage=footer, onLaterPages=footer)
