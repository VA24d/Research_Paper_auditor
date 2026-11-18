#!/usr/bin/env python3
"""Build the official template-matched, publication-grade A1 Landscape Research Poster.
Dimensions: 841mm x 594mm (Standard ISO A1 Landscape).
Faithfully reproduces the official IIIT-H template aesthetics:
- Authentic lavender header graphic with built-in official R&D Showcase 2026 & IIIT Hyderabad branding
- Bold ALL-CAPS section headers in #47247B (Official Template Deep Purple)
- Exact template card palette:
  * Col 1: #E1EDEB (Soft Sage / Mint)
  * Col 2 Top: #EFEAE4 (Warm Linen / Beige)
  * Col 2 Bottom: #F2F2F2 (Crisp Light Grey)
  * Col 3: #E1EDEB (Soft Sage / Mint)
- PROMINENT BIG TEAM NAME:
  * Top of Col 3: Dedicated box with #7030A0 outline
  * "HOOD RESEARCH DEPARTMENT" in 34pt Bold #47247B!
  * "SERC & CVIT LABS" in 18pt Bold #7030A0!
  * "Software Engineering Research Center • CVIT, IIIT Hyderabad" in 9.5pt!
- Flawless "fit" with seamless background integration (no white box clashes) and comfortable padding.
"""

import os
from PIL import Image
import pptx
from pptx.util import Pt, Mm
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement

# Palette matching the official IIIT-H template
INK = RGBColor(20, 33, 61)           # #14213D (Deep Ink for body)
PURPLE_TITLE = RGBColor(71, 36, 123) # #47247B (Template Deep Purple for headings)
PURPLE_ACCENT = RGBColor(112, 48, 160) # #7030A0 (Template Vibrant Purple line/borders)
GOOD = RGBColor(4, 120, 87)          # #047857 (Verified / 0 False Alarms)
AMBER = RGBColor(180, 83, 9)         # #B45309 (Needs Review)
BAD = RGBColor(192, 57, 43)          # #C0392B (Contradiction / Error)
GREY = RGBColor(100, 116, 139)       # #64748B (Subtext)
WHITE = RGBColor(255, 255, 255)

# Template card background fills
FILL_MINT = RGBColor(225, 237, 235)  # #E1EDEB (Col 1 & Col 3)
FILL_BEIGE = RGBColor(239, 234, 228) # #EFEAE4 (Col 2 Top)
FILL_GREY = RGBColor(242, 242, 242)  # #F2F2F2 (Col 2 Bottom)
BORDER_SUBTLE = RGBColor(203, 213, 225) # #CBD5E1

def clean_paragraph_bullets(p):
    """Strip all bullet and hanging indent XML properties from a paragraph."""
    pPr = p._p.get_or_add_pPr()
    pPr.attrib['indent'] = '0'
    pPr.attrib['marL'] = '0'
    for child in list(pPr):
        if child.tag.endswith(('buChar', 'buAutoNum', 'buFont', 'buBlip', 'buClr', 'buSzPct', 'buSzPts')):
            pPr.remove(child)
    buNone = OxmlElement('a:buNone')
    pPr.append(buNone)

def add_template_card(slide, left, top, width, height, fill_color, border_color=None, border_width=Pt(1.0)):
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left), int(top), int(width), int(height))
    card.fill.solid()
    card.fill.fore_color.rgb = fill_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = border_width
    else:
        card.line.fill.background()
    return card

def add_card_header(slide, left, top, width, title_str):
    tb = slide.shapes.add_textbox(int(left + Mm(7)), int(top + Mm(5)), int(width - Mm(14)), int(Mm(12)))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = title_str
    r.font.name = "Arial"
    r.font.size = Pt(17)
    r.font.bold = True
    r.font.color.rgb = PURPLE_TITLE

def add_image_fit(slide, img_path, left, top, max_width, max_height):
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found")
        return None
    with Image.open(img_path) as im:
        img_w, img_h = im.size
    aspect = img_w / img_h
    box_aspect = max_width / max_height
    if aspect > box_aspect:
        w = max_width
        h = max_width / aspect
        x = left
        y = top + (max_height - h) / 2
    else:
        h = max_height
        w = max_height * aspect
        y = top
        x = left + (max_width - w) / 2
    return slide.shapes.add_picture(img_path, int(x), int(y), int(w), int(h))

def add_stat_card(slide, left, top, width, height, num_str, label_str, note_str="", num_color=PURPLE_TITLE):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(left), int(top), int(width), int(height))
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = BORDER_SUBTLE
    box.line.width = Pt(1.0)

    tb = slide.shapes.add_textbox(int(left), int(top), int(width), int(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    p1 = tf.paragraphs[0]
    clean_paragraph_bullets(p1)
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = num_str
    r1.font.name = "Arial"
    r1.font.size = Pt(20)
    r1.font.bold = True
    r1.font.color.rgb = num_color

    p2 = tf.add_paragraph()
    clean_paragraph_bullets(p2)
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = label_str
    r2.font.name = "Arial"
    r2.font.size = Pt(9.5)
    r2.font.bold = True
    r2.font.color.rgb = INK

    if note_str:
        p3 = tf.add_paragraph()
        clean_paragraph_bullets(p3)
        p3.alignment = PP_ALIGN.CENTER
        r3 = p3.add_run()
        r3.text = note_str
        r3.font.name = "Arial"
        r3.font.size = Pt(8.0)
        r3.font.color.rgb = GREY

def main():
    prs = pptx.Presentation()
    prs.slide_width = Mm(841)
    prs.slide_height = Mm(594)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # ================================================================
    # 1. HEADER BANNER (Top: 0 to 68mm)
    # ================================================================
    # Master header graphic with IIIT-H branding (R&D Showcase removed)
    banner_path = "Final_Report/poster/assets/pure_header_banner.png"
    if not os.path.exists(banner_path):
        banner_path = "Final_Report/poster/assets/template_header_banner.png"
    slide.shapes.add_picture(banner_path, int(Mm(0)), int(Mm(0)), int(Mm(841)), int(Mm(68)))

    # Mascot on left
    mascot_path = "Final_Report/Hooded Blue Scholar Mascot.png"
    if os.path.exists(mascot_path):
        slide.shapes.add_picture(mascot_path, int(Mm(12)), int(Mm(8)), int(Mm(52)), int(Mm(52)))

    # Title, Subtitle, and Authors (now with ample breathing room with R&D Showcase removed)
    tb_title = slide.shapes.add_textbox(int(Mm(68)), int(Mm(7)), int(Mm(550)), int(Mm(56)))
    tf_t = tb_title.text_frame
    tf_t.word_wrap = True

    p_t = tf_t.paragraphs[0]
    clean_paragraph_bullets(p_t)
    r_t = p_t.add_run()
    r_t.text = "Research Paper Auditor: Evidence-Gated Verification"
    r_t.font.name = "Arial"
    r_t.font.size = Pt(30)
    r_t.font.bold = True
    r_t.font.color.rgb = INK

    p_sub = tf_t.add_paragraph()
    clean_paragraph_bullets(p_sub)
    r_sub = p_sub.add_run()
    r_sub.text = "Does the paper match its citations and its code? Pre-review factual auditing for AI conferences"
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(15.0)
    r_sub.font.bold = True
    r_sub.font.color.rgb = PURPLE_TITLE

    p_aut = tf_t.add_paragraph()
    clean_paragraph_bullets(p_aut)
    r_aut = p_aut.add_run()
    r_aut.text = "Vijay Aravynthan   •   Itikela Bhaskar   •   Anurag Peddi   •   Sudheera Y S"
    r_aut.font.name = "Arial"
    r_aut.font.size = Pt(13.0)
    r_aut.font.color.rgb = RGBColor(45, 55, 72)

    # Thin purple divider line under header
    sep = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(Mm(12)), int(Mm(67.5)), int(Mm(817)), int(Mm(1.5)))
    sep.fill.solid()
    sep.fill.fore_color.rgb = PURPLE_ACCENT
    sep.line.fill.background()

    # ================================================================
    # 2. COLUMN GEOMETRY
    # Col 1: 250mm (Mint #E1EDEB)
    # Col 2: 316mm (Beige #EFEAE4 Top, Grey #F2F2F2 Bottom)
    # Col 3: 235mm (Team Badge Box + Mint #E1EDEB)
    # ================================================================
    col1_x = Mm(12)
    col1_w = Mm(250)

    col2_x = Mm(270)
    col2_w = Mm(316)

    col3_x = Mm(594)
    col3_w = Mm(235)

    row1_y = Mm(74)
    row1_h = Mm(238)

    row2_y = Mm(318)
    row2_h = Mm(242)

    # ----------------------------------------------------------------
    # COLUMN 1: PROBLEM STATEMENT & PIPELINE ARCHITECTURE (Mint #E1EDEB)
    # ----------------------------------------------------------------
    # Card 1: Problem Statement
    add_template_card(slide, col1_x, row1_y, col1_w, row1_h, FILL_MINT)
    add_card_header(slide, col1_x, row1_y, col1_w, "1. PAPER SAYS ONE THING. CODE DOES ANOTHER.")

    tb_c1 = slide.shapes.add_textbox(int(col1_x + Mm(6)), int(Mm(89)), int(col1_w - Mm(12)), int(Mm(52)))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True

    p = tf_c1.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• The Reviewer Bottleneck: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Top AI conferences (ICLR, NeurIPS) receive 62,000+ submissions annually. Reviewers have only 15–20 minutes and cannot clone code repos or trace cited sources. Unaudited discrepancies slip into publication.\n\n"
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = INK

    p = tf_c1.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Accepted Paper Contradiction (Mirzadeh et al., NeurIPS 2020): "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "In an accepted continual learning paper, our auditor caught an unlabelled contradiction: the paper explicitly claims no shuffling, but released code sets "
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = INK
    r3 = p.add_run()
    r3.text = "shuffle=True"
    r3.font.bold = True
    r3.font.size = Pt(10.5)
    r3.font.color.rgb = BAD
    r4 = p.add_run()
    r4.text = ". Both lines were quoted verbatim without human prompt engineering:"
    r4.font.size = Pt(10.5)
    r4.font.color.rgb = INK

    add_image_fit(slide, "Final_Report/poster/assets/paper_vs_code_template.png", col1_x + Mm(6), Mm(145), col1_w - Mm(12), Mm(88))

    tb_c1_sub = slide.shapes.add_textbox(int(col1_x + Mm(6)), int(Mm(238)), int(col1_w - Mm(12)), int(Mm(68)))
    tf_c1s = tb_c1_sub.text_frame
    tf_c1s.word_wrap = True
    p = tf_c1s.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Automated Audit in 42 Seconds: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = PURPLE_ACCENT
    r2 = p.add_run()
    r2.text = "Neither LLMs nor reviewers routinely cross-check hyperparameter configs across paper text and code ASTs. Our system automates cross-modal verification in 42s.\n\n"
    r2.font.size = Pt(10.2)
    r2.font.color.rgb = INK

    p = tf_c1s.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Directional Finding (arXiv:2312.11580): "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = 'Text asserts "PlaNet-S (0.73) is higher than U-Net (0.78)" — detected automatically by table-consistency validation.'
    r2.font.size = Pt(10.2)
    r2.font.color.rgb = INK

    # Card 2: Modular Pipeline Architecture
    add_template_card(slide, col1_x, row2_y, col1_w, row2_h, FILL_MINT)
    add_card_header(slide, col1_x, row2_y, col1_w, "2. EVIDENCE-GATED PIPELINE ARCHITECTURE")

    tb_pdesc = slide.shapes.add_textbox(int(col1_x + Mm(6)), int(Mm(332)), int(col1_w - Mm(12)), int(Mm(22)))
    tf_pd = tb_pdesc.text_frame
    tf_pd.word_wrap = True
    p = tf_pd.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "Core Verification Invariant: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = PURPLE_TITLE
    r2 = p.add_run()
    r2.text = "A discrepancy is flagged ONLY when backed by exact dual verbatim quotes (paper sentence + source/code line). Unverified claims are strictly dropped."
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = INK

    add_image_fit(slide, "Final_Report/poster/assets/system_pipeline_template.png", col1_x + Mm(6), Mm(354), col1_w - Mm(12), Mm(118))

    tb_pfoot = slide.shapes.add_textbox(int(col1_x + Mm(6)), int(Mm(476)), int(col1_w - Mm(12)), int(Mm(80)))
    tf_pf = tb_pfoot.text_frame
    tf_pf.word_wrap = True
    p = tf_pf.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "4-Stage Modular Verification Process:\n"
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = PURPLE_TITLE

    stages = [
        ("1. Multi-Modal Ingestion: ", "GROBID parses paper text, references & formulas; Vision OCR extracts 98% of PDF tables; Python AST scans repository configs."),
        ("2. 4 Parallel Specialists: ", "Parallel execution across Citations (831 refs via Crossref/arXiv), Code (60 settings, 3 metrics), Theory (13 SPOT papers), and Integrity (GRIM test)."),
        ("3. Verification Gate: ", "Enforces verbatim substring quote matching. Suppresses 35% of noisy candidate flags, guaranteeing 0 false alarms on clean controls."),
        ("4. Actionable Audit Report: ", "Structured 3-state output (Verified Flag, Needs Human Review, Dropped) delivered to reviewers in 42s.")
    ]
    for st_title, st_desc in stages:
        p_st = tf_pf.add_paragraph()
        clean_paragraph_bullets(p_st)
        r1 = p_st.add_run()
        r1.text = "• " + st_title
        r1.font.bold = True
        r1.font.size = Pt(10.2)
        r1.font.color.rgb = PURPLE_ACCENT
        r2 = p_st.add_run()
        r2.text = st_desc
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = INK

    # ----------------------------------------------------------------
    # COLUMN 2: BENCHMARK CORPUS & EXPERIMENTAL RESULTS (Beige & Grey)
    # ----------------------------------------------------------------
    # Card 3: Quantitative Results vs Baselines (Top Card: Warm Beige #EFEAE4)
    add_template_card(slide, col2_x, row1_y, col2_w, row1_h, FILL_BEIGE)
    add_card_header(slide, col2_x, row1_y, col2_w, "3. RESULTS: 13× GAIN OVER BASELINES (0.636 F1)")

    # 4 Stat Badges
    bw = Mm(71)
    bh = Mm(33)
    add_stat_card(slide, col2_x + Mm(6), Mm(88), bw, bh, "0.636", "Macro-F1 Score", "Baselines: 0.05 (+59 pts)", PURPLE_TITLE)
    add_stat_card(slide, col2_x + Mm(82), Mm(88), bw, bh, "0", "False Alarms", "On 72 clean controls (100% P)", GOOD)
    add_stat_card(slide, col2_x + Mm(158), Mm(88), bw, bh, "100%", "Dual Quote Grounding", "Exact text + code quotes", PURPLE_ACCENT)
    add_stat_card(slide, col2_x + Mm(234), Mm(88), bw, bh, "₹7.4", "Cost / Paper Audit", "ReAct: ₹50.9 (7× cheaper)", INK)

    # Comparative Results Table
    tb_r_top = Mm(126)
    table_r_shape = slide.shapes.add_table(5, 7, int(col2_x + Mm(6)), int(tb_r_top), int(col2_w - Mm(12)), int(Mm(58)))
    table_r = table_r_shape.table
    table_r.columns[0].width = int(Mm(70))
    table_r.columns[1].width = int(Mm(39))
    table_r.columns[2].width = int(Mm(39))
    table_r.columns[3].width = int(Mm(39))
    table_r.columns[4].width = int(Mm(41))
    table_r.columns[5].width = int(Mm(37))
    table_r.columns[6].width = int(Mm(39))

    r_headers = ["System / Model", "Method F1", "Metric F1", "Setting F1", "Macro-F1", "Prec.", "Cost/Paper"]
    for c_idx, h_text in enumerate(r_headers):
        cell = table_r.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PURPLE_TITLE
        p = cell.text_frame.paragraphs[0]
        clean_paragraph_bullets(p)
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h_text
        r.font.name = "Arial"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = WHITE

    r_data = [
        ("Full System (Ours)", "0.34", "0.80", "0.77", "0.636", "1.000", "₹7.4 ($0.079)"),
        ("No Verifier (Ablation)", "0.34", "0.80", "0.86", "0.667", "0.912", "₹7.2 ($0.077)"),
        ("B0: Zero-Shot LLM", "0.00", "0.00", "0.14", "0.048", "0.038", "₹4.4 ($0.047)"),
        ("B1: ReAct Agent", "0.00", "0.00", "0.14", "0.046", "0.033", "₹50.9 ($0.542)")
    ]
    for r_idx, row_data in enumerate(r_data):
        for c_idx, val in enumerate(row_data):
            cell = table_r.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 0 else RGBColor(248, 246, 242)
            p = cell.text_frame.paragraphs[0]
            clean_paragraph_bullets(p)
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            r.font.name = "Arial"
            r.font.size = Pt(9.0)
            if r_idx == 0:
                r.font.bold = True
                r.font.color.rgb = PURPLE_ACCENT if c_idx == 4 else INK
            else:
                r.font.color.rgb = INK

    add_image_fit(slide, "Final_Report/poster/assets/chart_f1_template.png", col2_x + Mm(6), Mm(186), col2_w - Mm(12), Mm(76))

    tb_res_takeaway = slide.shapes.add_textbox(int(col2_x + Mm(6)), int(Mm(265)), int(col2_w - Mm(12)), int(Mm(42)))
    tf_rt = tb_res_takeaway.text_frame
    tf_rt.word_wrap = True
    p = tf_rt.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "Key Finding: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = PURPLE_ACCENT
    r2 = p.add_run()
    r2.text = "Both baselines fail completely on method and metric errors (0.00 F1). Our specialist routing achieves "
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = INK
    r3 = p.add_run()
    r3.text = "0.34 Method F1 and 0.80 Metric F1 at ₹7.4 per audit — 7× cheaper than ReAct agents."
    r3.font.bold = True
    r3.font.size = Pt(10.5)
    r3.font.color.rgb = PURPLE_TITLE

    # Card 4: Benchmark Corpus & Triage (Bottom Card: Crisp Grey #F2F2F2)
    add_template_card(slide, col2_x, row2_y, col2_w, row2_h, FILL_GREY)
    add_card_header(slide, col2_x, row2_y, col2_w, "4. BENCHMARK CORPUS & EVIDENCE TRIAGE")

    # 4 Stat Badges
    add_stat_card(slide, col2_x + Mm(6), Mm(332), bw, bh, "40", "Papers + Code", "Accepted papers + GitHub", PURPLE_TITLE)
    add_stat_card(slide, col2_x + Mm(82), Mm(332), bw, bh, "163", "Planted Errors", "Across 80 code versions", BAD)
    add_stat_card(slide, col2_x + Mm(158), Mm(332), bw, bh, "831", "Citations Checked", "HALLMARK benchmark", PURPLE_ACCENT)
    add_stat_card(slide, col2_x + Mm(234), Mm(332), bw, bh, "13", "Proof Papers (SPOT)", "Mathematical derivations", GOOD)

    # Benchmark Data Table (5 Rows)
    table_shape = slide.shapes.add_table(6, 5, int(col2_x + Mm(6)), int(Mm(370)), int(col2_w - Mm(12)), int(Mm(66)))
    table = table_shape.table
    table.columns[0].width = int(Mm(64))
    table.columns[1].width = int(Mm(53))
    table.columns[2].width = int(Mm(43))
    table.columns[3].width = int(Mm(70))
    table.columns[4].width = int(Mm(74))

    table_headers = ["Corpus / Set", "Domain / Source", "Evaluation Size", "Discrepancy Types", "Verification Standard"]
    for c_idx, h_text in enumerate(table_headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PURPLE_TITLE
        p = cell.text_frame.paragraphs[0]
        clean_paragraph_bullets(p)
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h_text
        r.font.name = "Arial"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = WHITE

    table_data = [
        ("Set A: Real Papers", "NeurIPS / ICLR", "40 papers + repos", "Unlabelled real mismatches", "Code repo inspection"),
        ("Set B: Clean Controls", "Peer-Reviewed", "72 papers / repos", "Zero injected errors", "100% Precision (0 False Alarms)"),
        ("Set C: Planted Corpus", "Method/Metric/Set", "80 code versions", "163 known perturbations", "Dual Verbatim Quotes Required"),
        ("Set D: HALLMARK", "Citation Verification", "831 cited papers", "Fake papers, wrong venue/yr", "Crossref & arXiv API lookups"),
        ("Set E: SPOT Corpus", "Mathematical Logic", "13 papers / proofs", "42 formal proof steps", "Step-by-step symbolic derivation")
    ]
    for r_idx, row_data in enumerate(table_data):
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 0 else RGBColor(246, 246, 246)
            p = cell.text_frame.paragraphs[0]
            clean_paragraph_bullets(p)
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            r.font.name = "Arial"
            r.font.size = Pt(8.8)
            if c_idx == 4 and "100%" in val:
                r.font.bold = True
                r.font.color.rgb = GOOD
            else:
                r.font.color.rgb = INK

    # 3-Box Triage Container (Eliminates dead space)
    t_box_y = Mm(442)
    t_box_h = Mm(52)
    card_w = Mm(98)
    
    # 1. Verified Card (Green)
    box_v = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(col2_x + Mm(6)), int(t_box_y), int(card_w), int(t_box_h))
    box_v.fill.solid()
    box_v.fill.fore_color.rgb = RGBColor(236, 253, 245)
    box_v.line.color.rgb = RGBColor(16, 185, 129)
    box_v.line.width = Pt(1.2)
    tf_v = box_v.text_frame
    tf_v.word_wrap = True
    p = tf_v.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "180 Verified Flags (37%)\n"
    r.font.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = GOOD
    p2 = tf_v.add_paragraph()
    clean_paragraph_bullets(p2)
    r2 = p2.add_run()
    r2.text = "Supported by exact dual verbatim quotes in both paper text and code AST. Immediately actionable for reviewers without manual search."
    r2.font.size = Pt(8.8)
    r2.font.color.rgb = INK

    # 2. Needs Review Card (Amber)
    box_r = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(col2_x + Mm(109)), int(t_box_y), int(card_w), int(t_box_h))
    box_r.fill.solid()
    box_r.fill.fore_color.rgb = RGBColor(255, 251, 235)
    box_r.line.color.rgb = RGBColor(245, 158, 11)
    box_r.line.width = Pt(1.2)
    tf_r = box_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "23 Needs Review (5%)\n"
    r.font.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = AMBER
    p2 = tf_r.add_paragraph()
    clean_paragraph_bullets(p2)
    r2 = p2.add_run()
    r2.text = "Borderline or unresolvable items (paywalled papers, uncertain references, private repos). Flagged for human judgment, never silently dropped."
    r2.font.size = Pt(8.8)
    r2.font.color.rgb = INK

    # 3. Dropped Card (Slate)
    box_d = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(col2_x + Mm(212)), int(t_box_y), int(card_w), int(t_box_h))
    box_d.fill.solid()
    box_d.fill.fore_color.rgb = RGBColor(241, 245, 249)
    box_d.line.color.rgb = RGBColor(148, 163, 184)
    box_d.line.width = Pt(1.2)
    tf_d = box_d.text_frame
    tf_d.word_wrap = True
    p = tf_d.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "286 Dropped (58%)\n"
    r.font.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(71, 85, 105)
    p2 = tf_d.add_paragraph()
    clean_paragraph_bullets(p2)
    r2 = p2.add_run()
    r2.text = "Suppressed hallucinations where model claims failed exact quote substring verification in source text or repository."
    r2.font.size = Pt(8.8)
    r2.font.color.rgb = INK

    # Bottom Invariant Callout in Card 4
    tb_bfoot = slide.shapes.add_textbox(int(col2_x + Mm(6)), int(Mm(498)), int(col2_w - Mm(12)), int(Mm(58)))
    tf_bf = tb_bfoot.text_frame
    tf_bf.word_wrap = True
    
    p = tf_bf.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Zero False Alarm Invariant: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = GOOD
    r2 = p.add_run()
    r2.text = "Across all 72 clean control papers, the evidence gate achieved "
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = INK
    r3 = p.add_run()
    r3.text = "100% precision (0 false alarms)"
    r3.font.bold = True
    r3.font.size = Pt(10.5)
    r3.font.color.rgb = GOOD
    r4 = p.add_run()
    r4.text = ". Clean publications are never penalized with spurious flags.\n\n"
    r4.font.size = Pt(10.5)
    r4.font.color.rgb = INK

    p = tf_bf.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Verified Real Discoveries: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = PURPLE_ACCENT
    r2 = p.add_run()
    r2.text = "Successfully surfaced unlabelled contradictions in accepted NeurIPS 2020 paper (training with shuffle=True despite explicit no-augmentation claim) and arXiv:2312.11580 (table directional error)."
    r2.font.size = Pt(10.2)
    r2.font.color.rgb = INK

    # ----------------------------------------------------------------
    # COLUMN 3: PROMINENT TEAM NAME, ABLATIONS & REFERENCES
    # ----------------------------------------------------------------
    # 1. TOP TEAM & LAB BADGE BOX (Matching Shape 99 in template!)
    # Left: 594mm, Top: 74mm, Width: 235mm, Height: 56mm
    box_team = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(col3_x), int(row1_y), int(col3_w), int(Mm(56)))
    box_team.fill.solid()
    box_team.fill.fore_color.rgb = WHITE
    box_team.line.color.rgb = PURPLE_ACCENT
    box_team.line.width = Pt(2.25)

    tf_team = box_team.text_frame
    tf_team.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    p_t1 = tf_team.paragraphs[0]
    clean_paragraph_bullets(p_t1)
    p_t1.alignment = PP_ALIGN.CENTER
    r_t1 = p_t1.add_run()
    r_t1.text = "HOOD RESEARCH DEPARTMENT"
    r_t1.font.name = "Arial"
    r_t1.font.size = Pt(24)
    r_t1.font.bold = True
    r_t1.font.color.rgb = PURPLE_TITLE

    p_t2 = tf_team.add_paragraph()
    clean_paragraph_bullets(p_t2)
    p_t2.alignment = PP_ALIGN.CENTER
    r_t2 = p_t2.add_run()
    r_t2.text = "SERC & CVIT LABS"
    r_t2.font.name = "Arial"
    r_t2.font.size = Pt(18)
    r_t2.font.bold = True
    r_t2.font.color.rgb = PURPLE_ACCENT

    p_t3 = tf_team.add_paragraph()
    clean_paragraph_bullets(p_t3)
    p_t3.alignment = PP_ALIGN.CENTER
    r_t3 = p_t3.add_run()
    r_t3.text = "Software Engineering Research Center  •  CVIT\nInternational Institute of Information Technology, Hyderabad"
    r_t3.font.name = "Arial"
    r_t3.font.size = Pt(9.5)
    r_t3.font.color.rgb = RGBColor(74, 85, 104)

    # 2. Card 5: Ablations & Specialist Checks (Mint #E1EDEB)
    card5_y = Mm(136)
    card5_h = Mm(198)
    add_template_card(slide, col3_x, card5_y, col3_w, card5_h, FILL_MINT)
    add_card_header(slide, col3_x, card5_y, col3_w, "5. ABLATIONS & SPECIALIST CHECKS")

    tb_abl = slide.shapes.add_textbox(int(col3_x + Mm(6)), int(card5_y + Mm(14)), int(col3_w - Mm(12)), int(Mm(48)))
    tf_ab = tb_abl.text_frame
    tf_ab.word_wrap = True
    p = tf_ab.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Method Code Retrieval: "
    r.font.bold = True
    r.font.size = Pt(10.8)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Dense embeddings achieve 70.2% (59/84) vs 41.7% (35/84) for BM25 lexical search (+28.5 pts).\n"
    r2.font.size = Pt(9.8)
    r2.font.color.rgb = INK

    p = tf_ab.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Table Parsing: "
    r.font.bold = True
    r.font.size = Pt(10.8)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Vision image cropping extracts 98% of tables vs 41% for GROBID structured parser (+57 pts).\n"
    r2.font.size = Pt(9.8)
    r2.font.color.rgb = INK

    p = tf_ab.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Model Pareto Frontier: "
    r.font.bold = True
    r.font.size = Pt(10.8)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Flash achieves 0.636 F1 at ₹7.4; Pro achieves 0.650 F1 at ₹26.6 (3.6× cost). Flash provides 7× lower cost."
    r2.font.size = Pt(9.8)
    r2.font.color.rgb = INK

    add_image_fit(slide, "Final_Report/poster/assets/chart_each_check_template.png", col3_x + Mm(6), card5_y + Mm(64), col3_w - Mm(12), Mm(82))

    tb_vimpact = slide.shapes.add_textbox(int(col3_x + Mm(6)), int(card5_y + Mm(148)), int(col3_w - Mm(12)), int(Mm(48)))
    tf_vi = tb_vimpact.text_frame
    tf_vi.word_wrap = True
    p = tf_vi.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Verifier Workload Impact: "
    r.font.bold = True
    r.font.size = Pt(10.8)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Evidence gate drops 35% of spurious flags (391 → 255 flags to review) with negligible recall penalty. Reviewers inspect only verified, actionable evidence.\n\n"
    r2.font.size = Pt(9.8)
    r2.font.color.rgb = INK

    p = tf_vi.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Theory & Proofs (SPOT): "
    r.font.bold = True
    r.font.size = Pt(10.8)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Two-stage triage + deep check catches 57% (8/14) of mathematical proof errors in accepted papers."
    r2.font.size = Pt(9.8)
    r2.font.color.rgb = INK

    # 3. Card 6: Web App, Conclusion & References (Mint #E1EDEB)
    card6_y = Mm(340)
    card6_h = Mm(220)
    add_template_card(slide, col3_x, card6_y, col3_w, card6_h, FILL_MINT)
    add_card_header(slide, col3_x, card6_y, col3_w, "6. STREAMLIT TRIAGE & REFERENCES")

    # Live Dual-Quote Box
    box_ui = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(col3_x + Mm(6)), int(card6_y + Mm(15)), int(col3_w - Mm(12)), int(Mm(48)))
    box_ui.fill.solid()
    box_ui.fill.fore_color.rgb = WHITE
    box_ui.line.color.rgb = PURPLE_ACCENT
    box_ui.line.width = Pt(1.0)

    tb_ui = slide.shapes.add_textbox(int(col3_x + Mm(8)), int(card6_y + Mm(16)), int(col3_w - Mm(16)), int(Mm(46)))
    tf_ui = tb_ui.text_frame
    tf_ui.word_wrap = True
    p = tf_ui.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "STREAMLIT AUDITOR: DUAL VERBATIM QUOTE TRIAGE\n"
    r.font.bold = True
    r.font.size = Pt(10.0)
    r.font.color.rgb = PURPLE_TITLE

    p = tf_ui.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = '• Paper Quote: "...we have not employed any data augmentation or shuffling."\n'
    r.font.size = Pt(8.8)
    r.font.color.rgb = INK
    r = p.add_run()
    r.text = '• Code Quote:  "train_loader = DataLoader(mnist_train, ..., shuffle=True)"\n'
    r.font.size = Pt(8.8)
    r.font.color.rgb = INK
    r = p.add_run()
    r.text = "• Outcome: [VERIFIED CONTRADICTION] • Flag confirmed with 0 false alarms."
    r.font.bold = True
    r.font.size = Pt(9.0)
    r.font.color.rgb = GOOD

    tb_app = slide.shapes.add_textbox(int(col3_x + Mm(6)), int(card6_y + Mm(66)), int(col3_w - Mm(12)), int(Mm(42)))
    tf_app = tb_app.text_frame
    tf_app.word_wrap = True

    p = tf_app.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Conclusion: "
    r.font.bold = True
    r.font.size = Pt(11.2)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Verbatim quote gating eliminates hallucination risk, providing trustworthy pre-review auditing for conference tracks at ₹7.4 per paper.\n"
    r2.font.size = Pt(10.0)
    r2.font.color.rgb = INK

    p = tf_app.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Mentorship & Lab: "
    r.font.bold = True
    r.font.size = Pt(11.2)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Software Engineering Research Center (SERC) & CVIT, IIIT Hyderabad. Course: CL3.410 LMA."
    r2.font.size = Pt(10.0)
    r2.font.color.rgb = INK

    tb_ref = slide.shapes.add_textbox(int(col3_x + Mm(6)), int(card6_y + Mm(110)), int(col3_w - Mm(12)), int(Mm(68)))
    tf_ref = tb_ref.text_frame
    tf_ref.word_wrap = True

    p = tf_ref.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "CORE REFERENCES (5 CRUCIAL)\n"
    r.font.bold = True
    r.font.size = Pt(10.8)
    r.font.color.rgb = PURPLE_TITLE

    refs = [
        ("[1] HALLMARK: ", "K. Kim et al., Diagnosing LLM citation verifiers, arXiv:2607.18360 (2026)."),
        ("[2] SciCoQA: ", "T. Baumgärtner & I. Gurevych, Scientific paper–code alignment, arXiv:2601.12910 (2026)."),
        ("[3] SPOT: ", "G. Son et al., Automated verification of scientific research, arXiv:2505.11855 (2025)."),
        ("[4] ReAct: ", "S. Yao et al., Synergizing reasoning and acting in language models, ICLR (2023)."),
        ("[5] GROBID: ", "P. Lopez, Automatic bibliographic data recognition, ECDL (2009).")
    ]
    for lbl, cit in refs:
        p_r = tf_ref.add_paragraph()
        clean_paragraph_bullets(p_r)
        r1 = p_r.add_run()
        r1.text = lbl
        r1.font.bold = True
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = PURPLE_ACCENT
        r2 = p_r.add_run()
        r2.text = cit
        r2.font.size = Pt(8.8)
        r2.font.color.rgb = INK

    # QR Codes
    add_image_fit(slide, "Final_Report/poster/assets/qr_code.png", col3_x + Mm(22), card6_y + Mm(178), Mm(23), Mm(23))
    add_image_fit(slide, "Final_Report/poster/assets/qr_website.png", col3_x + Mm(138), card6_y + Mm(178), Mm(23), Mm(23))

    tb_q1 = slide.shapes.add_textbox(int(col3_x + Mm(10)), int(card6_y + Mm(201)), int(Mm(48)), int(Mm(12)))
    tf_q1 = tb_q1.text_frame
    p = tf_q1.paragraphs[0]
    clean_paragraph_bullets(p)
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "GitHub Code"
    r.font.bold = True
    r.font.size = Pt(10.0)
    r.font.color.rgb = INK

    tb_q2 = slide.shapes.add_textbox(int(col3_x + Mm(126)), int(card6_y + Mm(201)), int(Mm(48)), int(Mm(12)))
    tf_q2 = tb_q2.text_frame
    p = tf_q2.paragraphs[0]
    clean_paragraph_bullets(p)
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "Project Website"
    r.font.bold = True
    r.font.size = Pt(10.0)
    r.font.color.rgb = INK

    # ================================================================
    # 3. FOOTER
    # ================================================================
    sep_f = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(Mm(12)), int(Mm(566)), int(Mm(817)), int(Mm(1.5)))
    sep_f.fill.solid()
    sep_f.fill.fore_color.rgb = PURPLE_ACCENT
    sep_f.line.fill.background()

    tb_foot = slide.shapes.add_textbox(int(Mm(14)), int(Mm(569)), int(Mm(813)), int(Mm(14)))
    tf_ft = tb_foot.text_frame
    p_ft = tf_ft.paragraphs[0]
    clean_paragraph_bullets(p_ft)
    
    r_f1 = p_ft.add_run()
    r_f1.text = "Authors: "
    r_f1.font.bold = True
    r_f1.font.size = Pt(11.5)
    r_f1.font.color.rgb = PURPLE_TITLE
    r_f2 = p_ft.add_run()
    r_f2.text = "Vijay Aravynthan, Itikela Bhaskar, Anurag Peddi, Sudheera Y S   •   Team Hood Research Department   •   SERC & CVIT, IIIT Hyderabad               "
    r_f2.font.size = Pt(11.5)
    r_f2.font.color.rgb = INK

    r_f3 = p_ft.add_run()
    r_f3.text = "Contact: "
    r_f3.font.bold = True
    r_f3.font.size = Pt(11.5)
    r_f3.font.color.rgb = PURPLE_TITLE
    r_f4 = p_ft.add_run()
    r_f4.text = "{vijay.aravynthan, itikela.bhaskar, anurag.peddi, sudheera.ys}@research.iiit.ac.in"
    r_f4.font.size = Pt(11.5)
    r_f4.font.color.rgb = GREY

    out_path = "Final_Report/poster/A1 size Landscape Research poster - new.pptx"
    prs.save(out_path)
    print(f"Successfully generated template-matched poster: {out_path}")

if __name__ == "__main__":
    main()
