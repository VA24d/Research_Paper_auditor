#!/usr/bin/env python3
"""Build a world-class, high-density 3-column A1 Landscape Research Poster.
Dimensions: 841mm x 594mm (Standard ISO A1 Landscape).
Balances column widths to exactly 261mm each.
Eliminates all empty diagram space, uses bold 14pt-22pt poster typography,
and fills every card with rigorous empirical results.
"""

import os
from PIL import Image
import pptx
from pptx.util import Pt, Mm
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement

# Palette
INK = RGBColor(20, 33, 61)         # #14213D (Deep Ink / Body)
BLUE = RGBColor(11, 92, 173)       # #0B5CAD (Report Primary Blue)
NAVY = RGBColor(15, 44, 89)        # #0F2C59 (Deep Navy for Headings)
CFULL = RGBColor(31, 111, 178)     # #1F6FB2 (Full System)
GOOD = RGBColor(46, 125, 79)       # #2E7D4F (Success / 0 False Alarms)
BAD = RGBColor(192, 57, 43)        # #C0392B (Error / Baseline)
GREY = RGBColor(100, 116, 139)     # #64748B (Slate subtext)
WHITE = RGBColor(255, 255, 255)
BORDER_CARD = RGBColor(11, 92, 173) # Primary blue card border
BORDER_SUBTLE = RGBColor(203, 213, 225)
BG_BADGE = RGBColor(241, 245, 249)

def clean_paragraph_bullets(p):
    pPr = p._p.get_or_add_pPr()
    pPr.attrib['indent'] = '0'
    pPr.attrib['marL'] = '0'
    for child in list(pPr):
        if child.tag.endswith(('buChar', 'buAutoNum', 'buFont', 'buBlip', 'buClr', 'buSzPct', 'buSzPts')):
            pPr.remove(child)
    buNone = OxmlElement('a:buNone')
    pPr.append(buNone)

def add_card(slide, left, top, width, height, border_color=BORDER_CARD, border_width=Pt(1.2)):
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left), int(top), int(width), int(height))
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = border_color
    card.line.width = border_width
    return card

def add_section_header(slide, left, top, width, num_str, title_str):
    badge_size = Mm(11)
    badge = slide.shapes.add_shape(MSO_SHAPE.OVAL, int(left + Mm(5)), int(top + Mm(3.5)), int(badge_size), int(badge_size))
    badge.fill.solid()
    badge.fill.fore_color.rgb = BLUE
    badge.line.fill.background()
    tf_b = badge.text_frame
    tf_b.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_b = tf_b.paragraphs[0]
    clean_paragraph_bullets(p_b)
    p_b.alignment = PP_ALIGN.CENTER
    r_b = p_b.add_run()
    r_b.text = str(num_str)
    r_b.font.name = "Arial"
    r_b.font.size = Pt(14)
    r_b.font.bold = True
    r_b.font.color.rgb = WHITE

    tb = slide.shapes.add_textbox(int(left + Mm(18)), int(top + Mm(2)), int(width - Mm(22)), int(Mm(13)))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = title_str
    r.font.name = "Arial"
    r.font.size = Pt(16.5)
    r.font.bold = True
    r.font.color.rgb = INK

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left + Mm(5)), int(top + Mm(16)), int(width - Mm(10)), int(Mm(1.0)))
    line.fill.solid()
    line.fill.fore_color.rgb = BLUE
    line.line.fill.background()

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

def add_mini_stat(slide, left, top, width, height, num_str, label_str, note_str="", num_color=BLUE):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(left), int(top), int(width), int(height))
    box.fill.solid()
    box.fill.fore_color.rgb = BG_BADGE
    box.line.color.rgb = BORDER_SUBTLE
    box.line.width = Pt(1.2)

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
    r1.font.size = Pt(24)
    r1.font.bold = True
    r1.font.color.rgb = num_color

    p2 = tf.add_paragraph()
    clean_paragraph_bullets(p2)
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = label_str
    r2.font.name = "Arial"
    r2.font.size = Pt(11)
    r2.font.bold = True
    r2.font.color.rgb = INK

    if note_str:
        p3 = tf.add_paragraph()
        clean_paragraph_bullets(p3)
        p3.alignment = PP_ALIGN.CENTER
        r3 = p3.add_run()
        r3.text = note_str
        r3.font.name = "Arial"
        r3.font.size = Pt(9.2)
        r3.font.color.rgb = GREY

def main():
    prs = pptx.Presentation()
    prs.slide_width = Mm(841)
    prs.slide_height = Mm(594)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Outer border (Crisp Blue)
    border = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(Mm(6)), int(Mm(6)), int(Mm(829)), int(Mm(582)))
    border.fill.background()
    border.line.color.rgb = BLUE
    border.line.width = Pt(3.5)

    # ================================================================
    # HEADER BANNER
    # ================================================================
    if os.path.exists("Final_Report/Hooded Blue Scholar Mascot.png"):
        slide.shapes.add_picture("Final_Report/Hooded Blue Scholar Mascot.png", int(Mm(14)), int(Mm(10)), int(Mm(50)), int(Mm(50)))
    elif os.path.exists("Final_Report/logo.png"):
        slide.shapes.add_picture("Final_Report/logo.png", int(Mm(14)), int(Mm(10)), int(Mm(50)), int(Mm(50)))

    tb_title = slide.shapes.add_textbox(int(Mm(68)), int(Mm(9)), int(Mm(565)), int(Mm(54)))
    tf_t = tb_title.text_frame
    tf_t.word_wrap = True

    p_t = tf_t.paragraphs[0]
    clean_paragraph_bullets(p_t)
    r_t = p_t.add_run()
    r_t.text = "Research Paper Auditor: Evidence-Gated Verification"
    r_t.font.name = "Arial"
    r_t.font.size = Pt(34)
    r_t.font.bold = True
    r_t.font.color.rgb = INK

    p_sub = tf_t.add_paragraph()
    clean_paragraph_bullets(p_sub)
    r_sub = p_sub.add_run()
    r_sub.text = "Does the paper match its citations and its code? Pre-review factual auditing for AI conferences"
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(16.5)
    r_sub.font.bold = True
    r_sub.font.color.rgb = BLUE

    p_aut = tf_t.add_paragraph()
    clean_paragraph_bullets(p_aut)
    r_aut = p_aut.add_run()
    r_aut.text = "Vijay Aravynthan   •   Itikela Bhaskar   •   Anurag Peddi   •   Sudheera Y S"
    r_aut.font.name = "Arial"
    r_aut.font.size = Pt(14)
    r_aut.font.color.rgb = INK

    # Lab Box (SERC & CVIT, IIIT Hyderabad)
    box_lab = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(Mm(642)), int(Mm(10)), int(Mm(184)), int(Mm(52)))
    box_lab.fill.solid()
    box_lab.fill.fore_color.rgb = BLUE
    box_lab.line.fill.background()

    tf_lab = box_lab.text_frame
    tf_lab.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_l1 = tf_lab.paragraphs[0]
    clean_paragraph_bullets(p_l1)
    p_l1.alignment = PP_ALIGN.CENTER
    r_l1 = p_l1.add_run()
    r_l1.text = "SERC & CVIT"
    r_l1.font.name = "Arial"
    r_l1.font.size = Pt(22)
    r_l1.font.bold = True
    r_l1.font.color.rgb = WHITE

    p_l2 = tf_lab.add_paragraph()
    clean_paragraph_bullets(p_l2)
    p_l2.alignment = PP_ALIGN.CENTER
    r_l2 = p_l2.add_run()
    r_l2.text = "Software Engineering Research Center  •  CVIT\nInternational Institute of Information Technology, Hyderabad\nTeam: Hood Research Department  •  CL3.410 LMA"
    r_l2.font.name = "Arial"
    r_l2.font.size = Pt(9.0)
    r_l2.font.color.rgb = WHITE

    sep = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(Mm(14)), int(Mm(65)), int(Mm(812)), int(Mm(1.8)))
    sep.fill.solid()
    sep.fill.fore_color.rgb = INK
    sep.line.fill.background()

    # ================================================================
    # 3 EQUAL BALANCED COLUMNS (Width: 261mm each)
    # ================================================================
    c_w = Mm(261)
    col1_x = Mm(15)
    col2_x = Mm(290)
    col3_x = Mm(565)

    # ----------------------------------------------------------------
    # COLUMN 1: THE PROBLEM & BENCHMARK TESTBED
    # ----------------------------------------------------------------
    # Card 1: Motivation (Height: 232mm)
    add_card(slide, col1_x, Mm(70), c_w, Mm(232))
    add_section_header(slide, col1_x, Mm(70), c_w, "1", "The Paper Says One Thing. Code Does Another.")

    tb_c1 = slide.shapes.add_textbox(int(col1_x + Mm(6)), int(Mm(88)), int(c_w - Mm(12)), int(Mm(52)))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• The Reviewer Bottleneck: "
    r.font.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Top AI venues (ICLR, NeurIPS) receive 62,000+ submissions annually. Reviewers have only 15–20 minutes per paper and cannot inspect external codebases or trace cited literature. Discrepancies and ungrounded claims slip into peer-reviewed publication.\n\n"
    r2.font.size = Pt(12)
    r2.font.color.rgb = INK

    p = tf_c1.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Real Discovery (Mirzadeh et al., NeurIPS 2020): "
    r.font.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "In an accepted continual learning paper, our auditor autonomously caught an unlabelled contradiction: the paper explicitly claims no shuffling, but the code sets "
    r2.font.size = Pt(12)
    r2.font.color.rgb = INK
    r3 = p.add_run()
    r3.text = "shuffle=True"
    r3.font.bold = True
    r3.font.size = Pt(12)
    r3.font.color.rgb = BAD
    r4 = p.add_run()
    r4.text = ". Both lines quoted verbatim without human guidance:"
    r4.font.size = Pt(12)
    r4.font.color.rgb = INK

    add_image_fit(slide, "Final_Report/poster/assets/paper_vs_code_tight.png", col1_x + Mm(6), Mm(168), c_w - Mm(12), Mm(82))

    tb_note = slide.shapes.add_textbox(int(col1_x + Mm(6)), int(Mm(254)), int(c_w - Mm(12)), int(Mm(44)))
    tf_n = tb_note.text_frame
    tf_n.word_wrap = True
    p = tf_n.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "Audit Significance: "
    r.font.bold = True
    r.font.size = Pt(12.5)
    r.font.color.rgb = BLUE
    r2 = p.add_run()
    r2.text = "Neither LLMs nor humans routinely cross-check hyperparameter configs across paper text and code repos. Our system provides automated verification in seconds."
    r2.font.size = Pt(12)
    r2.font.color.rgb = INK

    # Card 2: Benchmark Testbed (Height: 258mm)
    add_card(slide, col1_x, Mm(310), c_w, Mm(258))
    add_section_header(slide, col1_x, Mm(310), c_w, "2", "Benchmark Testbed & Multi-Source Corpus")

    bw = Mm(122)
    bh = Mm(38)
    add_mini_stat(slide, col1_x + Mm(6), Mm(328), bw, bh, "40", "Papers + Codebases", "Real accepted papers + GitHub", CFULL)
    add_mini_stat(slide, col1_x + Mm(133), Mm(328), bw, bh, "163", "Planted Errors", "Across 80 code versions", BAD)
    add_mini_stat(slide, col1_x + Mm(6), Mm(370), bw, bh, "831", "Citations Checked", "HALLMARK benchmark corpus", BLUE)
    add_mini_stat(slide, col1_x + Mm(133), Mm(370), bw, bh, "13", "Proof Papers (SPOT)", "Mathematical derivation corpus", GOOD)

    tb_bench = slide.shapes.add_textbox(int(col1_x + Mm(6)), int(Mm(414)), int(c_w - Mm(12)), int(Mm(148)))
    tf_b = tb_bench.text_frame
    tf_b.word_wrap = True

    p = tf_b.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Multi-Domain Benchmark Corpus:\n"
    r.font.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "  - Set A: Real accepted conference papers (NeurIPS, ICLR) with code.\n" \
              "  - Set B: 72 clean control papers to strictly verify zero false alarms.\n" \
              "  - Set C: 163 known errors across Method, Metric, and Setting.\n" \
              "  - Set D: 831 citations cross-checked against Crossref and arXiv.\n" \
              "  - Set E: 13 papers evaluated on multi-step mathematical logic.\n\n"
    r2.font.size = Pt(11.5)
    r2.font.color.rgb = INK

    p = tf_b.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Precision-First Standard: "
    r.font.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Every planted error must be confirmed by dual quotes. On the 72 clean controls, the verifier achieved "
    r2.font.size = Pt(11.5)
    r2.font.color.rgb = INK
    r3 = p.add_run()
    r3.text = "100% precision (0 false alarms)."
    r3.font.bold = True
    r3.font.size = Pt(11.5)
    r3.font.color.rgb = GOOD

    # ----------------------------------------------------------------
    # COLUMN 2: SYSTEM ARCHITECTURE & RESULTS
    # ----------------------------------------------------------------
    # Card 3: Pipeline (Height: 232mm)
    add_card(slide, col2_x, Mm(70), c_w, Mm(232))
    add_section_header(slide, col2_x, Mm(70), c_w, "3", "Evidence-Gated Pipeline Architecture")

    tb_pdesc = slide.shapes.add_textbox(int(col2_x + Mm(6)), int(Mm(88)), int(c_w - Mm(12)), int(Mm(28)))
    tf_pd = tb_pdesc.text_frame
    tf_pd.word_wrap = True
    p = tf_pd.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "Core Invariant: "
    r.font.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = BLUE
    r2 = p.add_run()
    r2.text = "A discrepancy is flagged ONLY when backed by exact dual verbatim quotes (paper text + code/source). Unverified claims are strictly dropped."
    r2.font.size = Pt(12)
    r2.font.color.rgb = INK

    add_image_fit(slide, "Final_Report/poster/assets/system_pipeline_tight.png", col2_x + Mm(6), Mm(118), c_w - Mm(12), Mm(138))

    tb_pfoot = slide.shapes.add_textbox(int(col2_x + Mm(6)), int(Mm(260)), int(c_w - Mm(12)), int(Mm(38)))
    tf_pf = tb_pfoot.text_frame
    tf_pf.word_wrap = True
    p = tf_pf.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• 4 Modular Stages: "
    r.font.bold = True
    r.font.size = Pt(12.5)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "(1) Ingestion  →  (2) 4 Parallel Specialists  →  (3) Quote Verification Gate  →  (4) Actionable Audit Report."
    r2.font.size = Pt(11.5)
    r2.font.color.rgb = INK

    # Card 4: Quantitative Results vs Baselines (Height: 258mm)
    add_card(slide, col2_x, Mm(310), c_w, Mm(258))
    add_section_header(slide, col2_x, Mm(310), c_w, "4", "Results: 13× Gain Over Baselines (0.636 F1)")

    add_mini_stat(slide, col2_x + Mm(6), Mm(328), bw, bh, "0.636", "Macro-F1 Score", "Baselines: 0.05 (+59 pts)", CFULL)
    add_mini_stat(slide, col2_x + Mm(133), Mm(328), bw, bh, "0", "False Alarms", "On 72 clean controls (100% P)", GOOD)
    add_mini_stat(slide, col2_x + Mm(6), Mm(370), bw, bh, "100%", "Dual Quote Grounding", "Exact text + code quotes", BLUE)
    add_mini_stat(slide, col2_x + Mm(133), Mm(370), bw, bh, "₹7.4", "Cost / Paper Audit", "ReAct: ₹50.9 (7× cheaper)", INK)

    add_image_fit(slide, "Final_Report/poster/assets/chart_f1_tight.png", col2_x + Mm(6), Mm(414), c_w - Mm(12), Mm(98))

    tb_res = slide.shapes.add_textbox(int(col2_x + Mm(6)), int(Mm(514)), int(c_w - Mm(12)), int(Mm(50)))
    tf_r = tb_res.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Massive Performance Leap: "
    r.font.bold = True
    r.font.size = Pt(12.5)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Zero-shot LLM (0.048 F1) and ReAct Agent (0.046 F1) fail completely on Method and Metric discrepancies (0.00 F1). Our specialist routing achieves "
    r2.font.size = Pt(11.5)
    r2.font.color.rgb = INK
    r3 = p.add_run()
    r3.text = "0.34 Method F1 and 0.80 Metric F1."
    r3.font.bold = True
    r3.font.size = Pt(11.5)
    r3.font.color.rgb = CFULL

    # ----------------------------------------------------------------
    # COLUMN 3: ABLATIONS, WEB APP & REFERENCES
    # ----------------------------------------------------------------
    # Card 5: Ablations & Specialist Accuracy (Height: 232mm)
    add_card(slide, col3_x, Mm(70), c_w, Mm(232))
    add_section_header(slide, col3_x, Mm(70), c_w, "5", "Ablation Studies & Specialist Checks")

    tb_abl = slide.shapes.add_textbox(int(col3_x + Mm(6)), int(Mm(88)), int(c_w - Mm(12)), int(Mm(52)))
    tf_ab = tb_abl.text_frame
    tf_ab.word_wrap = True
    p = tf_ab.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Method Code Retrieval: "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Dense embeddings achieve 70.2% (59/84) vs 41.7% (35/84) for BM25 lexical search (+28.5 pts).\n"
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    p = tf_ab.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Table Parsing: "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Vision image cropping extracts 98% of tables vs 41% for GROBID structured parser.\n"
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    p = tf_ab.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Model Pareto: "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Flash achieves 0.636 F1 at ₹7.4; Pro achieves 0.650 F1 at ₹26.6 (3.6× cost). Flash provides 7× lower cost."
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    add_image_fit(slide, "Final_Report/poster/assets/chart_each_check_tight.png", col3_x + Mm(6), Mm(144), c_w - Mm(12), Mm(94))

    tb_vimpact = slide.shapes.add_textbox(int(col3_x + Mm(6)), int(Mm(242)), int(c_w - Mm(12)), int(Mm(54)))
    tf_vi = tb_vimpact.text_frame
    tf_vi.word_wrap = True
    p = tf_vi.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Verifier Workload Impact: "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Evidence gate drops 35% of spurious flags (391 → 255 flags to review) with negligible recall penalty. Reviewers inspect only verified, actionable evidence."
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    # Card 6: Web App, Conclusion & References (Height: 258mm)
    add_card(slide, col3_x, Mm(310), c_w, Mm(258))
    add_section_header(slide, col3_x, Mm(310), c_w, "6", "Web App, Conclusion & References")

    # Clean UI Inspection Box
    box_ui = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(col3_x + Mm(6)), int(Mm(328)), int(c_w - Mm(12)), int(Mm(42)))
    box_ui.fill.solid()
    box_ui.fill.fore_color.rgb = BG_BADGE
    box_ui.line.color.rgb = BLUE
    box_ui.line.width = Pt(1.0)

    tb_ui = slide.shapes.add_textbox(int(col3_x + Mm(8)), int(Mm(329)), int(c_w - Mm(16)), int(Mm(40)))
    tf_ui = tb_ui.text_frame
    tf_ui.word_wrap = True
    p = tf_ui.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "STREAMLIT AUDITOR: DUAL VERBATIM QUOTE TRIAGE\n"
    r.font.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = BLUE

    p = tf_ui.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = '• Paper Quote: "...we have not employed any data augmentation or shuffling."\n'
    r.font.size = Pt(9.5)
    r.font.color.rgb = INK
    r = p.add_run()
    r.text = '• Code Quote:  "train_loader = DataLoader(mnist_train, ..., shuffle=True)"\n'
    r.font.size = Pt(9.5)
    r.font.color.rgb = INK
    r = p.add_run()
    r.text = "• Outcome: [VERIFIED CONTRADICTION] • Flag confirmed with 0 false alarms."
    r.font.bold = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = GOOD

    tb_app = slide.shapes.add_textbox(int(col3_x + Mm(6)), int(Mm(372)), int(c_w - Mm(12)), int(Mm(50)))
    tf_app = tb_app.text_frame
    tf_app.word_wrap = True

    p = tf_app.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Conclusion: "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Verbatim quote gating eliminates hallucination risk, providing trustworthy pre-review auditing at ₹7.4 per paper.\n"
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    p = tf_app.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Acknowledgment: "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "We thank the faculty and mentors at SERC & CVIT, IIIT Hyderabad, and CL3.410 LMA course staff."
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    tb_ref = slide.shapes.add_textbox(int(col3_x + Mm(6)), int(Mm(424)), int(c_w - Mm(12)), int(Mm(96)))
    tf_ref = tb_ref.text_frame
    tf_ref.word_wrap = True

    p = tf_ref.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "CORE REFERENCES (5 CRUCIAL)\n"
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = NAVY

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
        r1.font.size = Pt(10)
        r1.font.color.rgb = BLUE
        r2 = p_r.add_run()
        r2.text = cit
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = INK

    # QR Codes
    add_image_fit(slide, "Final_Report/poster/assets/qr_code.png", col3_x + Mm(26), Mm(522), Mm(26), Mm(26))
    add_image_fit(slide, "Final_Report/poster/assets/qr_website.png", col3_x + Mm(148), Mm(522), Mm(26), Mm(26))

    tb_q1 = slide.shapes.add_textbox(int(col3_x + Mm(14)), int(Mm(549)), int(Mm(50)), int(Mm(12)))
    tf_q1 = tb_q1.text_frame
    p = tf_q1.paragraphs[0]
    clean_paragraph_bullets(p)
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "GitHub Code"
    r.font.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = INK

    tb_q2 = slide.shapes.add_textbox(int(col3_x + Mm(136)), int(Mm(549)), int(Mm(50)), int(Mm(12)))
    tf_q2 = tb_q2.text_frame
    p = tf_q2.paragraphs[0]
    clean_paragraph_bullets(p)
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "Project Website"
    r.font.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = INK

    # ================================================================
    # FOOTER
    # ================================================================
    sep_f = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(Mm(14)), int(Mm(571)), int(Mm(812)), int(Mm(1.0)))
    sep_f.fill.solid()
    sep_f.fill.fore_color.rgb = BLUE
    sep_f.line.fill.background()

    tb_foot = slide.shapes.add_textbox(int(Mm(15)), int(Mm(573)), int(Mm(811)), int(Mm(12)))
    tf_ft = tb_foot.text_frame
    p_ft = tf_ft.paragraphs[0]
    clean_paragraph_bullets(p_ft)
    
    r_f1 = p_ft.add_run()
    r_f1.text = "Authors: "
    r_f1.font.bold = True
    r_f1.font.size = Pt(11.5)
    r_f1.font.color.rgb = INK
    r_f2 = p_ft.add_run()
    r_f2.text = "Vijay Aravynthan, Itikela Bhaskar, Anurag Peddi, Sudheera Y S   •   SERC & CVIT, IIIT Hyderabad               "
    r_f2.font.size = Pt(11.5)
    r_f2.font.color.rgb = INK

    r_f3 = p_ft.add_run()
    r_f3.text = "Contact: "
    r_f3.font.bold = True
    r_f3.font.size = Pt(11.5)
    r_f3.font.color.rgb = INK
    r_f4 = p_ft.add_run()
    r_f4.text = "{vijay.aravynthan, itikela.bhaskar, anurag.peddi, sudheera.ys}@research.iiit.ac.in"
    r_f4.font.size = Pt(11.5)
    r_f4.font.color.rgb = GREY

    out_path = "Final_Report/poster/A1 size Landscape Research poster - new.pptx"
    prs.save(out_path)
    print(f"Successfully generated bold balanced poster: {out_path}")

if __name__ == "__main__":
    main()
