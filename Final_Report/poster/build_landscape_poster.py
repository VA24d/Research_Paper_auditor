"""Builds the A1 landscape poster on the IIIT Hyderabad template.

    uv run --with python-pptx --with matplotlib --with "qrcode[pil]" \
        python Final_Report/poster/build_landscape_poster.py

Starts from the untouched IIIT template (`template_A1_landscape_IIITH.pptx`), keeps its banner,
logo, footer rule, fonts and panel colours, and writes `poster_A1_landscape.pptx`
next to this file. Charts are drawn with matplotlib into `assets/landscape/`.
Every shape stays editable in PowerPoint.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qrcode
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Mm, Pt

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template_A1_landscape_IIITH.pptx"
OUT = HERE / "poster_A1_landscape.pptx"
ASSETS = HERE / "assets" / "landscape"
SHOTS = HERE.parent / "app_screenshots"

CODE_URL = "https://github.com/VA24d/Research_Paper_auditor"
SITE_URL = "https://va24d.github.io/Research_Paper_auditor/"
DEMO_URL = "https://researchpaperauditor.streamlit.app"
EMAIL = "vijay.s@research.iiit.ac.in"

# Template colours.
PURPLE = "47247B"
PANEL_A = "E1EDEB"  # grey-green
PANEL_B = "EFEAE4"  # beige
PANEL_C = "F2F2F2"  # light grey
INK = "1F1F1F"
MUTED = "5A5A5A"
# Chart colours: ours in the template purple, baselines in warm colours.
C_FULL = "#47247B"
C_NOVER = "#A992CF"
C_B0 = "#E08A2E"
C_B1 = "#B5443A"
GREEN, GREEN_BG = "2E7D4F", "DDEFE3"
YELLOW, YELLOW_BG = "9A6B00", "FFF1C2"
GREY, GREY_BG = "6B6B6B", "E4E4E4"
RED = "C0392B"
HILITE = "FFE066"
FONT = "Arial"
MONO = "Courier New"


def rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


# --- pptx helpers ----------------------------------------------------------------------


def box(slide, x, y, w, h, fill=None, line=None, line_w=1.0, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Mm(x), Mm(y), Mm(w), Mm(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(line_w)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def text(
    slide,
    x,
    y,
    w,
    h,
    paragraphs,
    size=24,
    color=INK,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    font=FONT,
    spacing=1.1,
    after=6,
    bold=False,
):
    """`paragraphs`: a list; each item is a string or a list of (text, style) runs, where
    style is a dict with any of bold, italic, color, size, font."""
    tb = slide.shapes.add_textbox(Mm(x), Mm(y), Mm(w), Mm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Mm(0))
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(after)
        runs = [(para, {})] if isinstance(para, str) else para
        for chunk, style in runs:
            r = p.add_run()
            r.text = chunk
            f = r.font
            f.name = style.get("font", font)
            f.size = Pt(style.get("size", size))
            f.bold = style.get("bold", bold)
            f.italic = style.get("italic", False)
            f.color.rgb = rgb(style.get("color", color))
    return tb


def heading(slide, x, y, w, number, title):
    text(slide, x, y, w, 16, [f"{number}  {title.upper()}"], size=32, color=PURPLE, bold=True)


def arrow(slide, x1, y1, x2, y2, color="7A7A7A", width=2.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Mm(x1), Mm(y1), Mm(x2), Mm(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(
        str(path), Mm(x), Mm(y), Mm(w) if w else None, Mm(h) if h else None
    )


# --- charts ----------------------------------------------------------------------------

MM = 1 / 25.4


def chart_style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#888888")
    ax.tick_params(colors="#333333", length=0, pad=6)


def grouped_f1(path):
    plt.rcParams.update({"font.family": ["Arial", "Noto Sans Kannada"], "font.size": 22})
    fig, ax = plt.subplots(figsize=(220 * MM, 112 * MM), dpi=300)
    groups = ["Method", "Metric", "Setting", "Macro-F1"]
    series = [
        ("Full system", C_FULL, [0.34, 0.80, 0.77, 0.636]),
        ("No verifier", C_NOVER, [0.34, 0.80, 0.86, 0.667]),
        ("B0 (LLM)", C_B0, [0, 0, 0.14, 0.048]),
        ("B1 (Agent)", C_B1, [0, 0, 0.14, 0.046]),
    ]
    width = 0.2
    for i, (name, color, values) in enumerate(series):
        xs = [g + (i - 1.5) * width for g in range(len(groups))]
        ax.bar(xs, values, width * 0.92, color=color, label=name)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("F1")
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=1)
    ax.set_axisbelow(True)
    chart_style(ax)
    ax.text(3 - 1.5 * width, 0.636 + 0.03, "0.636", ha="center", fontsize=20,
            weight="bold", color=C_FULL)
    fig.legend(ncol=4, loc="upper center", frameon=False, handlelength=1.0,
               columnspacing=1.4, fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, transparent=True)
    plt.close(fig)


def cost_scatter(path):
    plt.rcParams.update({"font.family": ["Arial", "Noto Sans Kannada"], "font.size": 22})
    fig, ax = plt.subplots(figsize=(128 * MM, 112 * MM), dpi=300)
    for cost, f1, color in ((7.4, 0.636, C_FULL), (4.4, 0.048, C_B0), (50.9, 0.046, C_B1)):
        ax.scatter([cost], [f1], s=420, color=color, zorder=3)
    ax.annotate("Ours  ₹7.4", (7.4, 0.636), xytext=(12, 0), textcoords="offset points",
                fontsize=21, weight="bold", color=C_FULL, va="center")
    ax.annotate("B0  ₹4.4", (4.4, 0.048), xytext=(12, 24), textcoords="offset points",
                fontsize=19, color=C_B0, ha="left")
    ax.annotate("B1  ₹51", (50.9, 0.046), xytext=(0, 22), textcoords="offset points",
                fontsize=19, color=C_B1, ha="center")
    ax.set_xscale("log")
    ax.set_xlim(2.5, 100)
    ax.set_xticks([5, 10, 20, 50], ["₹5", "₹10", "₹20", "₹50"])
    ax.minorticks_off()
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Cost per paper (log scale)")
    ax.set_ylabel("Macro-F1")
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=1)
    ax.set_axisbelow(True)
    chart_style(ax)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def each_check(path):
    plt.rcParams.update({"font.family": ["Arial", "Noto Sans Kannada"], "font.size": 21})
    fig, ax = plt.subplots(figsize=(172 * MM, 122 * MM), dpi=300)
    rows = [
        ("Number vs. table", 20),
        ("Wrong gain", 55),
        ("Proof step (SPOT)", 57),
        ("Fake reference (F1)", 88),
        ("Reversed comparison", 94),
        ("Impossible mean", 100),
        ("Placeholder text", 100),
        ("Setting vs. code", 100),
    ]
    values = [r[1] for r in rows]
    colors = [C_FULL if v >= 85 else C_NOVER for v in values]
    ax.barh(range(len(rows)), values, 0.68, color=colors)
    for i, v in enumerate(values):
        ax.text(v + 1.5, i, f"{v}", va="center", fontsize=20, weight="bold", color="#333333")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Errors found (%)")
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=1)
    ax.set_axisbelow(True)
    chart_style(ax)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def qr(path, url):
    qrcode.make(url, box_size=20, border=1).save(path)


# --- poster parts ----------------------------------------------------------------------


def stat(slide, x, y, w, h, number, label, note):
    box(slide, x, y, w, h, fill="FFFFFF", line="D6CFE3", line_w=1.5,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(slide, x, y + 3, w, 26, [number], size=60, color=PURPLE, align=PP_ALIGN.CENTER,
         bold=True)
    text(slide, x + 2, y + 28, w - 4, 10, [label], size=22, align=PP_ALIGN.CENTER, bold=True)
    text(slide, x + 2, y + 38, w - 4, 8, [note], size=18, color=MUTED, align=PP_ALIGN.CENTER)


def hook(slide, x, y):
    """The shuffle mismatch: the paper's sentence next to the line of code."""
    box(slide, x + 1.5, y + 1.5, 96, 64, fill="D9D9D9")
    box(slide, x, y, 96, 64, fill="FFFFFF", line="BBBBBB", line_w=1)
    text(slide, x + 5, y + 4, 88, 6, ["arXiv 2006.06958 · Stable SGD"], size=14, color=MUTED,
         font="Georgia")
    text(slide, x + 5, y + 11, 88, 8, ["B.2  Continual learning setup"], size=16,
         font="Georgia", bold=True)
    for i, frac in enumerate((0.95, 0.78)):
        box(slide, x + 5, y + 22 + i * 4.5, 86 * frac, 2.2, fill="E6E6E6")
    box(slide, x + 3, y + 31, 90, 23, fill=HILITE)
    text(slide, x + 5, y + 33, 86, 19,
         [[("…we have not employed any data augmentation technique or ", {"italic": True}),
           ("shuffling", {"italic": True, "bold": True}), (".", {"italic": True})]],
         size=17, font="Georgia", spacing=1.0)
    box(slide, x + 5, y + 57.5, 76, 2.2, fill="E6E6E6")
    text(slide, x, y + 67, 96, 8, ["What the paper says"], size=20, align=PP_ALIGN.CENTER,
         bold=True)
    badge = box(slide, x + 100, y + 25, 15, 15, fill=RED, shape=MSO_SHAPE.OVAL)
    tf = badge.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "≠"
    r.font.size, r.font.bold, r.font.name = Pt(34), True, FONT
    r.font.color.rgb = rgb("FFFFFF")
    ex = x + 119
    box(slide, ex, y, 98, 64, fill="F6F8FA", line="C9CED6", line_w=1)
    box(slide, ex, y, 98, 9, fill="EAEEF2")
    for i, c in enumerate(("E5534B", "D4A72C", "57AB5A")):
        box(slide, ex + 4 + i * 5, y + 3, 3, 3, fill=c, shape=MSO_SHAPE.OVAL)
    text(slide, ex + 22, y + 2, 74, 6, ["stable_sgd/data_utils.py"], size=14, color=MUTED,
         font=MONO)
    box(slide, ex, y + 20.5, 98, 21, fill="FFE2E0")
    lines = [
        ("26", " mnist_train = MNIST(…)", None),
        ("27", " train_loader = DataLoader(", None),
        ("  ", "     mnist_train, …,", None),
        ("  ", "     ", "shuffle=True)"),
        ("28", " test_loader = DataLoader(…)", None),
    ]
    paras = []
    for num, code, red in lines:
        runs = [(num, {"color": "8C959F"}), (code, {})]
        if red:
            runs.append((red, {"color": RED, "bold": True}))
        paras.append(runs)
    text(slide, ex + 3, y + 14, 94, 46, paras, size=14, font=MONO, spacing=1.0, after=3)
    text(slide, ex, y + 67, 98, 8, ["What the code does"], size=20, align=PP_ALIGN.CENTER,
         bold=True)


def flow(slide, x, y, w):
    """The system, top to bottom: inputs, parser, four checkers, verifier, outcomes."""
    border = "8E7AB5"

    def node(nx, ny, nw, nh, title, sub, fill="FFFFFF", line=border, title_color=INK):
        box(slide, nx, ny, nw, nh, fill=fill, line=line, line_w=1.8,
            shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        paras = [[(title, {"bold": True, "color": title_color})]]
        if sub:
            paras.append([(sub, {"size": 17, "color": MUTED})])
        text(slide, nx + 3, ny + 2, nw - 6, nh - 4, paras, size=21, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, after=2, spacing=1.0)

    half = (w - 8) / 2
    node(x, y, half, 20, "Paper", "PDF or LaTeX")
    node(x + half + 8, y, half, 20, "Code repository", "pinned commit, never run")
    arrow(slide, x + half / 2, y + 20, x + w / 2 - 20, y + 28)
    arrow(slide, x + half + 8 + half / 2, y + 20, x + w / 2 + 20, y + 28)
    node(x, y + 28, w, 20, "Parser and planner",
         "one task per reference, setting, claim…")
    arrow(slide, x + w / 2, y + 48, x + w / 2, y + 55)
    cont = box(slide, x - 2, y + 55, w + 4, 74, fill="F5F2FA", line="B9A9D6", line_w=1.5,
               shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    cont.line.dash_style = 4
    text(slide, x, y + 57, w, 7, ["Four checkers, run in parallel"], size=18, color=PURPLE,
         align=PP_ALIGN.CENTER)
    checkers = [
        ("Citations", "reference exists, supports the sentence"),
        ("Code and data", "settings, metrics, method vs. code"),
        ("Theory", "claims and proof steps"),
        ("Integrity", "placeholders, impossible means"),
    ]
    for i, (title, sub) in enumerate(checkers):
        node(x + (i % 2) * (half + 8), y + 65 + (i // 2) * 31, half, 27, title, sub)
    arrow(slide, x + w / 2, y + 129, x + w / 2, y + 136)
    node(x, y + 136, w, 22, "Verifier", "finds every quote again in its source",
         fill="EFE8F8", line=PURPLE)
    third = (w - 8) / 3
    outcomes = [
        ("Verified flag", "quote found", GREEN_BG, GREEN),
        ("Human review", "unsure, or not found", YELLOW_BG, YELLOW),
        ("Dropped", "quote not in source", GREY_BG, GREY),
    ]
    for i, (title, sub, fill, line) in enumerate(outcomes):
        ox = x + i * (third + 4)
        arrow(slide, x + w / 2, y + 158, ox + third / 2, y + 166)
        node(ox, y + 166, third, 24, title, sub, fill=fill, line=line, title_color=line)


def ablations(slide, x, y, w):
    rows = [
        ("Verifier", "flags left to check per run",
         [("with", 255, C_FULL), ("without", 391, "#B3B3B3")], 391,
         "35% fewer flags, for 3 points of F1"),
        ("Finding the code", "changed lines found, of 84",
         [("embeddings", 59, C_FULL), ("word overlap", 35, "#B3B3B3")], 84, None),
        ("Reading PDF tables", "numbers recovered (%)",
         [("table region", 98, C_FULL), ("GROBID", 41, "#B3B3B3")], 100, None),
        ("Model for code", "differences found, of 57",
         [("3.8 Flash", 17, C_FULL), ("3.1 Pro, 3.6× cost", 19, "#B3B3B3")], 57, None),
    ]
    label_w, bar_x0 = 52, x + 54
    bar_w = w - 54 - 20
    yy = y
    for title, sub, bars, scale, note in rows:
        text(slide, x, yy, w, 8, [[(title, {"bold": True}),
                                   (f"   {sub}", {"size": 17, "color": MUTED})]], size=21)
        yy += 9.5
        for label, value, color in bars:
            text(slide, x, yy, label_w, 8, [label], size=18, align=PP_ALIGN.RIGHT,
                 anchor=MSO_ANCHOR.MIDDLE)
            length = bar_w * value / scale
            box(slide, bar_x0, yy + 0.8, length, 6.4, fill=color.lstrip("#"))
            text(slide, bar_x0 + length + 2, yy, 18, 8, [str(value)], size=18,
                 anchor=MSO_ANCHOR.MIDDLE, bold=True)
            yy += 8.5
        if note:
            text(slide, x, yy + 0.5, w, 7, [note], size=18, color=PURPLE, bold=True)
            yy += 8.5
        yy += 13
    return yy


# --- poster ----------------------------------------------------------------------------


def bullets(slide, x, y, w, h, items, size=20, after=5):
    text(slide, x, y, w, h, [[("•  ", {"color": PURPLE, "bold": True}), (b, {})]
                             for b in items], size=size, after=after)


def build():
    ASSETS.mkdir(parents=True, exist_ok=True)
    grouped_f1(ASSETS / "f1.png")
    cost_scatter(ASSETS / "cost.png")
    each_check(ASSETS / "each_check.png")
    qr(ASSETS / "qr_site.png", SITE_URL)

    prs = Presentation(str(TEMPLATE))
    slide = prs.slides[0]
    # Keep only the IIIT logo picture; the banner and footer rule live in the layout.
    for shape in list(slide.shapes):
        if shape.shape_type != 13:
            shape._element.getparent().remove(shape._element)

    # header: title on the left, the team beside it, QR code and IIIT logo on the right
    text(slide, 12, 8, 380, 32, ["Research Paper Auditor"], size=80, bold=True)
    text(slide, 12, 46, 384, 14, ["Does a paper match the papers it cites and its own code?"],
         size=30, color=PURPLE)
    picture(slide, HERE.parent / "logo.png", 412, 9, h=52)
    text(slide, 466, 10, 225, 8, ["Team"], size=18, color=MUTED)
    text(slide, 466, 18, 225, 18, ["Hood Research Department"], size=36, color=PURPLE,
         bold=True)
    text(slide, 466, 38, 225, 22,
         ["CL3.410 Language Models and Agents", "Monsoon 2026 · IIIT Hyderabad"], size=17,
         color=INK, after=0, spacing=1.0)

    # left column
    box(slide, 11, 80, 237, 182, fill=PANEL_A)
    heading(slide, 19, 86, 224, "1", "The problem")
    text(slide, 19, 104, 222, 78, [
        [("Reviewers rarely open the papers a submission cites or the code it releases. ",
          {}), ("ICLR 2027 received 62,000+ abstracts.", {"bold": True})],
        [("Our auditor checks a paper against both, and ", {}),
         ("reports a problem only when it can quote both sides.", {"bold": True})],
        [("A real case it found ", {}),
         ("(Mirzadeh et al., NeurIPS 2020)", {"color": MUTED}), (":", {})],
    ], size=24, after=10)
    hook(slide, 18, 182)

    box(slide, 11, 268, 237, 292, fill=PANEL_A)
    heading(slide, 19, 274, 224, "2", "Our system")
    text(slide, 19, 292, 222, 22, [
        [("New: ", {"bold": True, "color": PURPLE}),
         ("one run checks references, claims, code and the paper's own numbers. "
          "Earlier tools check one of these.", {})],
    ], size=22)
    flow(slide, 21, 318, 217)
    box(slide, 19, 515, 221, 39, fill="FFFFFF", line="C9C2D6", line_w=1,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(slide, 23, 518, 213, 34, [
        [("Baselines, same model: ", {"bold": True, "color": PURPLE})],
        [("B0 (LLM)", {"bold": True}), (" reads the whole paper once, no tools.", {})],
        [("B1 (Agent)", {"bold": True}),
         (" one agent with the same tools, no verifier.", {})],
    ], size=19, after=2, spacing=1.0)

    # middle column: results
    box(slide, 254, 80, 382, 214, fill=PANEL_B)
    heading(slide, 263, 86, 365, "3", "Results on 80 paper versions")
    tiles = [
        ("0.636", "Macro-F1", "baselines 0.05"),
        ("0", "false alarms", "on 72 clean items"),
        ("100%", "flags with a quote", "baselines 0%"),
        ("₹7.4", "per paper", "B1 (Agent) ₹51"),
    ]
    tw = (365 - 3 * 6) / 4
    for i, (num, label, note) in enumerate(tiles):
        stat(slide, 263 + i * (tw + 6), 103, tw, 48, num, label, note)
    picture(slide, ASSETS / "f1.png", 258, 154, w=220)
    picture(slide, ASSETS / "cost.png", 482, 154, w=128)
    text(slide, 263, 268, 365, 24, [
        [("Target: +10 points of Macro-F1 over one LLM pass. We got ", {}),
         ("+59.", {"bold": True, "color": PURPLE}),
         ("  Neither baseline finds a method or metric error.", {})],
    ], size=22)

    # middle: what we tested on
    box(slide, 254, 300, 382, 58, fill=PANEL_C)
    heading(slide, 263, 305, 365, "4", "Datasets")
    data = [
        ("40", "papers + code", "each also with errors added"),
        ("163", "known errors", "and 72 clean items"),
        ("831", "references", "HALLMARK"),
        ("13", "papers with proofs", "SPOT"),
    ]
    dw = 365 / 4
    for i, (num, label, note) in enumerate(data):
        dx = 263 + i * dw
        text(slide, dx, 321, 30, 30, [num], size=44, color=PURPLE, bold=True,
             anchor=MSO_ANCHOR.MIDDLE)
        text(slide, dx + 31 + (6 if len(num) == 3 else 0), 323, dw - 33, 26,
             [[(label, {"bold": True})], [(note, {"size": 16, "color": MUTED})]], size=19,
             after=1, spacing=1.0, anchor=MSO_ANCHOR.MIDDLE)

    # middle bottom: ablations and checkers
    box(slide, 254, 364, 188, 196, fill=PANEL_C)
    heading(slide, 262, 370, 175, "5", "Ablations")
    ablations(slide, 262, 388, 175)
    text(slide, 262, 548, 175, 8,
         [[("Purple", {"bold": True, "color": PURPLE}), (": the variant we kept.", {})]],
         size=17, color=MUTED)

    box(slide, 447, 364, 189, 196, fill=PANEL_C)
    heading(slide, 455, 370, 176, "6", "Checker results")
    picture(slide, ASSETS / "each_check.png", 452, 386, w=178)
    text(slide, 455, 512, 175, 44, [
        [("Rule checks catch ", {}), ("every", {"bold": True}),
         (" planted placeholder and impossible mean.", {})],
        [("Mostly planted errors; real papers may score lower.",
          {"color": MUTED, "size": 17})],
    ], size=20, after=3)

    # right column
    box(slide, 642, 80, 188, 189, fill=PANEL_A)
    heading(slide, 650, 86, 176, "7", "The web app")
    picture(slide, SHOTS / "demo_counts.png", 650, 103, w=172)
    picture(slide, SHOTS / "demo_flag_shuffle.png", 650, 124, w=172)
    text(slide, 650, 247, 172, 18, [
        [("The same case in the app. ", {"bold": True}),
         ("Demo: saved reports, or audit with your own Gemini key or local model.", {})],
    ], size=17)

    box(slide, 642, 275, 188, 285, fill=PANEL_A)
    text(slide, 650, 281, 176, 12, ["CONCLUSION"], size=28, color=PURPLE, bold=True)
    bullets(slide, 650, 296, 174, 56, size=22, items=[
        "Tools plus verified quotes beat one pass over the whole paper: 0.636 against "
        "under 0.05.",
        "No clean item flagged; every flag carries a quote.",
    ])
    text(slide, 650, 350, 176, 12, ["LIMITATIONS AND NEXT STEPS"], size=28, color=PURPLE,
         bold=True)
    bullets(slide, 650, 365, 174, 80, size=22, items=[
        "Most labels are planted; 16 of 163 are real.",
        "One model family (Gemini). Method recall is low (F1 0.34).",
        "Next: a second model family, and checking unlabelled flags by hand.",
    ])
    text(slide, 650, 432, 176, 12, ["REFERENCES"], size=28, color=PURPLE, bold=True)
    refs = [
        "HALLMARK: fabricated and corrupted references.",
        "SciCoQA: paper–code discrepancies.",
        "SPOT: errors in scientific papers.",
        "Brown & Heathers, the GRIM test, 2017.",
        "GROBID: PDF to structured text.",
        "Mirzadeh et al., training regimes in continual learning, NeurIPS 2020.",
    ]
    text(slide, 650, 447, 176, 110, [f"[{i + 1}] {r}" for i, r in enumerate(refs)], size=21,
         after=9, color="333333")

    # footer: the template's author names and email fields
    text(slide, 19, 566, 560, 12, [[("Authors: ", {"bold": True}),
                                    ("Vijay Aravynthan · Itikela Bhaskar · Anurag Peddi · "
                                     "Sudheera Y S", {})]], size=26)
    text(slide, 580, 566, 248, 12, [[("Email: ", {"bold": True}), (EMAIL, {})]], size=26,
         align=PP_ALIGN.RIGHT)

    # header: one QR code, to the website (it links the code, demo, report and poster)
    box(slide, 697, 7, 52, 52, fill="FFFFFF")
    picture(slide, ASSETS / "qr_site.png", 699, 9, w=48)
    text(slide, 683, 60, 80, 7, ["Website, code and demo"], size=15, color=INK,
         align=PP_ALIGN.CENTER)

    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
