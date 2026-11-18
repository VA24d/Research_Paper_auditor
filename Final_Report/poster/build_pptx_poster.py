#!/usr/bin/env python3
"""Build A1 Landscape PowerPoint Research Poster for Research Paper Auditor.
Dimensions: 841 mm x 594 mm (Standard A1 Landscape).
Colors and typography match the project's PDF report, SERC/CVIT lab affiliations,
and reduced dead space.
"""

import os
import sys
from PIL import Image
from pptx import Presentation
from pptx.util import Mm, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

# Colors
INK = RGBColor(20, 33, 61)       # #14213D
BLUE = RGBColor(11, 92, 173)     # #0B5CAD
CFULL = RGBColor(31, 111, 178)   # #1F6FB2
CNOVER = RGBColor(141, 187, 227) # #8DBBE3
CBZERO = RGBColor(224, 138, 46)  # #E08A2E
CBONE = RGBColor(181, 68, 58)    # #B5443A
GOOD = RGBColor(46, 125, 79)     # #2E7D4F
BAD = RGBColor(192, 57, 43)      # #C0392B
HILITE = RGBColor(255, 224, 102) # #FFE066
RULEGREY = RGBColor(201, 206, 214) # #C9CED6
WHITE = RGBColor(255, 255, 255)
BG_LIGHT = RGBColor(248, 250, 252)
SLATE_TEXT = RGBColor(71, 85, 105)
SLATE_LIGHT = RGBColor(241, 245, 249)
BOX_BG = RGBColor(239, 246, 255)

def add_image_fit(slide, img_path, left, top, max_width, max_height):
    """Add image centered and scaled to fit within bounding box."""
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_path} not found.")
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

def add_card(slide, left, top, width, height, border_color=RULEGREY, border_width=Pt(1.2), bg_color=WHITE):
    """Add rectangular card container."""
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left), int(top), int(width), int(height))
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = border_width
    return card

def add_card_header(slide, left, top, width, step_num, title_text):
    """Add step number circle, title text, and horizontal dividing rule."""
    circle_size = Mm(9.5)
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, int(left + Mm(3.5)), int(top + Mm(3)), int(circle_size), int(circle_size))
    circle.fill.solid()
    circle.fill.fore_color.rgb = BLUE
    circle.line.fill.background()
    tf_c = circle.text_frame
    tf_c.word_wrap = False
    tf_c.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_c = tf_c.paragraphs[0]
    p_c.alignment = PP_ALIGN.CENTER
    r_c = p_c.add_run()
    r_c.text = str(step_num)
    r_c.font.size = Pt(13)
    r_c.font.bold = True
    r_c.font.color.rgb = WHITE
    r_c.font.name = "Helvetica"

    tb = slide.shapes.add_textbox(int(left + Mm(14.5)), int(top + Mm(2)), int(width - Mm(18)), int(Mm(11)))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title_text
    r.font.size = Pt(13.5)
    r.font.bold = True
    r.font.color.rgb = INK
    r.font.name = "Helvetica"

    # Blue divider rule
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left + Mm(3.5)), int(top + Mm(14)), int(width - Mm(7)), int(Mm(0.9)))
    rule.fill.solid()
    rule.fill.fore_color.rgb = BLUE
    rule.line.fill.background()

def add_stat_box(slide, left, top, width, height, number_str, label_str, note_str, num_color=BLUE, bg_color=BOX_BG):
    """Add stat box with number, label, and note."""
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(left), int(top), int(width), int(height))
    box.fill.solid()
    box.fill.fore_color.rgb = bg_color
    box.line.color.rgb = RULEGREY
    box.line.width = Pt(0.8)

    tb = slide.shapes.add_textbox(int(left), int(top + Mm(1)), int(width), int(height - Mm(2)))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # Number
    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = number_str
    r1.font.size = Pt(21)
    r1.font.bold = True
    r1.font.color.rgb = num_color
    r1.font.name = "Helvetica"
    
    # Label
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = label_str
    r2.font.size = Pt(10)
    r2.font.bold = True
    r2.font.color.rgb = INK
    r2.font.name = "Helvetica"
    
    # Note
    if note_str:
        p3 = tf.add_paragraph()
        p3.alignment = PP_ALIGN.CENTER
        r3 = p3.add_run()
        r3.text = note_str
        r3.font.size = Pt(8.5)
        r3.font.color.rgb = SLATE_TEXT
        r3.font.name = "Helvetica"

def main():
    prs = Presentation()
    prs.slide_width = Mm(841)
    prs.slide_height = Mm(594)

    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Outer background & blue boundary frame
    outer_frame = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(Mm(5)), int(Mm(5)), int(Mm(831)), int(Mm(584)))
    outer_frame.fill.solid()
    outer_frame.fill.fore_color.rgb = WHITE
    outer_frame.line.color.rgb = BLUE
    outer_frame.line.width = Pt(3.5)

    # ================================================================
    # HEADER (y: 8 mm to 78 mm, h: 70 mm)
    # ================================================================
    mascot_path = "Final_Report/Hooded Blue Scholar Mascot.png"
    if not os.path.exists(mascot_path):
        mascot_path = "Final_Report/logo.png"
    add_image_fit(slide, mascot_path, Mm(14), Mm(10), Mm(58), Mm(62))

    # Header Title Text Block
    tb_title = slide.shapes.add_textbox(int(Mm(76)), int(Mm(9)), int(Mm(580)), int(Mm(66)))
    tf_t = tb_title.text_frame
    tf_t.word_wrap = True

    p_main = tf_t.paragraphs[0]
    r_main = p_main.add_run()
    r_main.text = "Research Paper Auditor"
    r_main.font.size = Pt(36)
    r_main.font.bold = True
    r_main.font.color.rgb = INK
    r_main.font.name = "Helvetica"

    p_sub = tf_t.add_paragraph()
    r_sub = p_sub.add_run()
    r_sub.text = "Does the paper match its citations and its code?  Evidence-Gated Verification"
    r_sub.font.size = Pt(17.5)
    r_sub.font.bold = True
    r_sub.font.color.rgb = BLUE
    r_sub.font.name = "Helvetica"

    p_auth = tf_t.add_paragraph()
    r_auth = p_auth.add_run()
    r_auth.text = "Vijay Aravynthan        Itikela Bhaskar        Anurag Peddi        Sudheera Y S"
    r_auth.font.size = Pt(14)
    r_auth.font.bold = True
    r_auth.font.color.rgb = INK
    r_auth.font.name = "Helvetica"

    p_inst = tf_t.add_paragraph()
    r_inst = p_inst.add_run()
    r_inst.text = "SERC (Software Engineering Research Center)  &  CVIT (Centre for Visual Information Technology)\nInternational Institute of Information Technology, Hyderabad (IIIT-H)   •   CL3.410 LMA Final Project"
    r_inst.font.size = Pt(11)
    r_inst.font.color.rgb = SLATE_TEXT
    r_inst.font.name = "Helvetica"

    # Team & Lab Badge Card on Right
    team_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(Mm(662)), int(Mm(10)), int(Mm(165)), int(Mm(62)))
    team_box.fill.solid()
    team_box.fill.fore_color.rgb = BLUE
    team_box.line.fill.background()
    tf_team = team_box.text_frame
    tf_team.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_t1 = tf_team.paragraphs[0]
    p_t1.alignment = PP_ALIGN.CENTER
    r_t1 = p_t1.add_run()
    r_t1.text = "Labs: SERC & CVIT\n"
    r_t1.font.size = Pt(13)
    r_t1.font.color.rgb = RGBColor(191, 219, 254)
    r_t1.font.name = "Helvetica"

    r_t2 = p_t1.add_run()
    r_t2.text = "Hood Research\nDepartment\n"
    r_t2.font.size = Pt(21)
    r_t2.font.bold = True
    r_t2.font.color.rgb = WHITE
    r_t2.font.name = "Helvetica"

    r_t3 = p_t1.add_run()
    r_t3.text = "IIIT Hyderabad"
    r_t3.font.size = Pt(12)
    r_t3.font.color.rgb = RGBColor(191, 219, 254)
    r_t3.font.name = "Helvetica"

    # Header Divider Rule
    h_rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(Mm(12)), int(Mm(76.5)), int(Mm(817)), int(Mm(1.5)))
    h_rule.fill.solid()
    h_rule.fill.fore_color.rgb = INK
    h_rule.line.fill.background()

    # ================================================================
    # ROW 1 (y: 81 mm to 230 mm, h: 149 mm)
    # Card 1 (Problem): Left (w: 308 mm)
    # Card 2 (System Pipeline): Right (w: 501 mm)
    # ================================================================
    # --- CARD 1: The Problem ---
    add_card(slide, Mm(12), Mm(81), Mm(308), Mm(149), border_color=BLUE)
    add_card_header(slide, Mm(12), Mm(81), Mm(308), 1, "The paper says one thing. Its code does another.")
    
    # Graphic: Paper vs Code
    add_image_fit(slide, "Final_Report/poster/assets/paper_vs_code.png", Mm(16), Mm(97), Mm(300), Mm(66))

    # Text explanation
    tb_c1 = slide.shapes.add_textbox(int(Mm(16)), int(Mm(165)), int(Mm(300)), int(Mm(60)))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    p1 = tf_c1.paragraphs[0]
    r1 = p1.add_run()
    r1.text = "Real Discovery: "
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = INK
    r2 = p1.add_run()
    r2.text = "Mirzadeh et al., NeurIPS 2020. "
    r2.font.bold = True
    r2.font.size = Pt(11)
    r2.font.color.rgb = BLUE
    r3 = p1.add_run()
    r3.text = "An unlabelled mismatch discovered automatically in an accepted paper. Paper claimed no shuffling; repository sets "
    r3.font.size = Pt(10)
    r3.font.color.rgb = INK
    r4 = p1.add_run()
    r4.text = "shuffle=True"
    r4.font.bold = True
    r4.font.color.rgb = BAD
    r5 = p1.add_run()
    r5.text = ". Both lines were quoted verbatim without human guidance.\n"
    r5.font.size = Pt(10)
    
    p2 = tf_c1.add_paragraph()
    r6 = p2.add_run()
    r6.text = "The Reviewer Bottleneck: "
    r6.font.bold = True
    r6.font.size = Pt(11)
    r6.font.color.rgb = INK
    r7 = p2.add_run()
    r7.text = "ICLR received "
    r7.font.size = Pt(10)
    r8 = p2.add_run()
    r8.text = "62,000+ submissions"
    r8.font.bold = True
    r8.font.color.rgb = BAD
    r9 = p2.add_run()
    r9.text = ". Reviewers lack time to inspect repos or verify cited text. Unaudited claims slip through peer review at scale."
    r9.font.size = Pt(10)

    # --- CARD 2: Modular System Architecture ---
    add_card(slide, Mm(328), Mm(81), Mm(501), Mm(149), border_color=BLUE)
    add_card_header(slide, Mm(328), Mm(81), Mm(501), 2, "Our system: Modular Evidence-Gated Architecture")
    
    # Graphic: System Pipeline
    add_image_fit(slide, "Final_Report/poster/assets/system_pipeline.png", Mm(332), Mm(97), Mm(493), Mm(96))

    tb_c2 = slide.shapes.add_textbox(int(Mm(332)), int(Mm(195)), int(Mm(493)), int(Mm(30)))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True
    p_c2 = tf_c2.paragraphs[0]
    r_c2_a = p_c2.add_run()
    r_c2_a.text = "Evidence-Gated Principle: "
    r_c2_a.font.bold = True
    r_c2_a.font.size = Pt(11)
    r_c2_a.font.color.rgb = BLUE
    r_c2_b = p_c2.add_run()
    r_c2_b.text = "The auditor reports a problem "
    r_c2_b.font.size = Pt(10)
    r_c2_c = p_c2.add_run()
    r_c2_c.text = "only when it can quote both sides verbatim"
    r_c2_c.font.bold = True
    r_c2_c.font.color.rgb = INK
    r_c2_d = p_c2.add_run()
    r_c2_d.text = " (paper quote + cited paper/code). All unverified flags are sent to Human Review or Dropped, eliminating false alarms."
    r_c2_d.font.size = Pt(10)

    # ================================================================
    # ROW 2: THREE COLUMNS (y: 236 mm to 556 mm, h: 320 mm)
    # Col 1: x = 12 mm, w = 263 mm
    # Col 2: x = 289 mm, w = 263 mm
    # Col 3: x = 566 mm, w = 263 mm
    # ================================================================

    # ----------------------------------------------------------------
    # COLUMN 1: BENCHMARK & ENGINEERING CHOICES
    # ----------------------------------------------------------------
    # Card 3: What We Tested On (h: 148 mm)
    add_card(slide, Mm(12), Mm(236), Mm(263), Mm(148), border_color=BLUE)
    add_card_header(slide, Mm(12), Mm(236), Mm(263), 3, "What we tested on: Diverse Benchmark")
    
    # 4 Stat boxes (2x2)
    bw, bh = Mm(123), Mm(44)
    add_stat_box(slide, Mm(17), Mm(254), bw, bh, "40", "papers + code", "realistic planted errors & clean", num_color=CFULL)
    add_stat_box(slide, Mm(147), Mm(254), bw, bh, "163", "known errors", "and 72 clean controls", num_color=BAD)
    add_stat_box(slide, Mm(17), Mm(303), bw, bh, "831", "references", "HALLMARK benchmark", num_color=BLUE)
    add_stat_box(slide, Mm(147), Mm(303), bw, bh, "13", "papers with proofs", "SPOT math benchmark", num_color=GOOD)

    tb_c3 = slide.shapes.add_textbox(int(Mm(16)), int(Mm(351)), int(Mm(255)), int(Mm(28)))
    tf_c3 = tb_c3.text_frame
    tf_c3.word_wrap = True
    p_c3 = tf_c3.paragraphs[0]
    r_c3 = p_c3.add_run()
    r_c3.text = "Testbed spans ML, NLP, vision, and theory papers with paired GitHub repositories and gold error annotations."
    r_c3.font.size = Pt(9.5)
    r_c3.font.color.rgb = SLATE_TEXT

    # Card 4: Our Choices & Ablations (h: 166 mm)
    add_card(slide, Mm(12), Mm(390), Mm(263), Mm(166), border_color=BLUE)
    add_card_header(slide, Mm(12), Mm(390), Mm(263), 4, "Ablation insights: What works & why")

    tb_c4 = slide.shapes.add_textbox(int(Mm(16)), int(Mm(406)), int(Mm(255)), int(Mm(146)))
    tf_c4 = tb_c4.text_frame
    tf_c4.word_wrap = True

    items = [
        ("Finding Code for Methods", "Dense embeddings locate 59 of 84 method changes (70.2% recall) vs only 35 (41.7%) for lexical word overlap.", CFULL),
        ("PDF Table Extraction", "Table region image extraction succeeds on 98% of tables, whereas GROBID structured parser extracts only 41%.", BLUE),
        ("LLM Model Efficiency", "Gemini 2.5/3.8 Flash catches 17 of 57 code changes at 1x cost; Gemini Pro catches 19 at 3.6x cost. Flash retained for 7x lower cost.", GOOD),
        ("Deterministic Rules", "Rule-based checks catch 100% of formatting, placeholder text, and impossible means at $0 model cost.", INK)
    ]
    for idx, (head, desc, col) in enumerate(items):
        p_h = tf_c4.paragraphs[0] if idx == 0 else tf_c4.add_paragraph()
        r_h = p_h.add_run()
        r_h.text = f"• {head}: "
        r_h.font.bold = True
        r_h.font.size = Pt(10)
        r_h.font.color.rgb = col
        r_d = p_h.add_run()
        r_d.text = desc + "\n"
        r_d.font.size = Pt(9)
        r_d.font.color.rgb = INK

    # ----------------------------------------------------------------
    # COLUMN 2: CORE EMPIRICAL RESULTS & BASELINES
    # ----------------------------------------------------------------
    # Card 5: Headline Results (h: 148 mm)
    add_card(slide, Mm(289), Mm(236), Mm(263), Mm(148), border_color=BLUE)
    add_card_header(slide, Mm(289), Mm(236), Mm(263), 5, "Headline results: 13x baseline leap")

    add_stat_box(slide, Mm(294), Mm(254), bw, bh, "0.636", "Macro-F1", "baselines score 0.05", num_color=CFULL)
    add_stat_box(slide, Mm(424), Mm(254), bw, bh, "0", "false alarms", "on 72 clean control items", num_color=GOOD)
    add_stat_box(slide, Mm(294), Mm(303), bw, bh, "100%", "flags with quote", "baselines provide 0%", num_color=BLUE)
    add_stat_box(slide, Mm(424), Mm(303), bw, bh, "Rs 7.4", "per paper audit", "B1 Agent costs Rs 51", num_color=INK)

    tb_c5 = slide.shapes.add_textbox(int(Mm(293)), int(Mm(351)), int(Mm(255)), int(Mm(28)))
    tf_c5 = tb_c5.text_frame
    tf_c5.word_wrap = True
    p_c5 = tf_c5.paragraphs[0]
    r_c5_a = p_c5.add_run()
    r_c5_a.text = "Goal exceeded: "
    r_c5_a.font.bold = True
    r_c5_a.font.size = Pt(10)
    r_c5_a.font.color.rgb = BLUE
    r_c5_b = p_c5.add_run()
    r_c5_b.text = "Target was +10 F1 points over baselines. We achieved "
    r_c5_b.font.size = Pt(9.5)
    r_c5_c = p_c5.add_run()
    r_c5_c.text = "+59 F1 points "
    r_c5_c.font.bold = True
    r_c5_c.font.color.rgb = BAD
    r_c5_d = p_c5.add_run()
    r_c5_d.text = "with 0 false alarms on clean controls."
    r_c5_d.font.size = Pt(9.5)

    # Card 6: Task Breakdown (h: 166 mm)
    add_card(slide, Mm(289), Mm(390), Mm(263), Mm(166), border_color=BLUE)
    add_card_header(slide, Mm(289), Mm(390), Mm(263), 6, "Finds what an LLM misses")

    # Graphic: Chart F1
    add_image_fit(slide, "Final_Report/poster/assets/chart_f1.png", Mm(293), Mm(405), Mm(255), Mm(94))

    tb_c6 = slide.shapes.add_textbox(int(Mm(293)), int(Mm(502)), int(Mm(255)), int(Mm(50)))
    tf_c6 = tb_c6.text_frame
    tf_c6.word_wrap = True
    p_c6 = tf_c6.paragraphs[0]
    r_c6_1 = p_c6.add_run()
    r_c6_1.text = "Zero baseline recall on code & math: "
    r_c6_1.font.bold = True
    r_c6_1.font.size = Pt(10)
    r_c6_1.font.color.rgb = INK
    r_c6_2 = p_c6.add_run()
    r_c6_2.text = "Neither zero-shot LLM (B0) nor ReAct agent (B1) found a single Method or Metric error (0.00 F1). Our modular pipeline scores "
    r_c6_2.font.size = Pt(9)
    r_c6_3 = p_c6.add_run()
    r_c6_3.text = "0.80 on Metric "
    r_c6_3.font.bold = True
    r_c6_3.font.color.rgb = BLUE
    r_c6_4 = p_c6.add_run()
    r_c6_4.text = "and "
    r_c6_4.font.size = Pt(9)
    r_c6_5 = p_c6.add_run()
    r_c6_5.text = "0.77 on Setting"
    r_c6_5.font.bold = True
    r_c6_5.font.color.rgb = BLUE
    r_c6_6 = p_c6.add_run()
    r_c6_6.text = "."
    r_c6_6.font.size = Pt(9)

    # ----------------------------------------------------------------
    # COLUMN 3: VERIFIER IMPACT, PARETO & WEB UI
    # ----------------------------------------------------------------
    # Card 7: Specialist Breakdown & Fewer Flags (h: 154 mm)
    add_card(slide, Mm(566), Mm(236), Mm(263), Mm(154), border_color=BLUE)
    add_card_header(slide, Mm(566), Mm(236), Mm(263), 7, "Specialist accuracy & 35% fewer flags")

    # Graphic: Chart each check
    add_image_fit(slide, "Final_Report/poster/assets/chart_each_check.png", Mm(570), Mm(251), Mm(255), Mm(86))

    tb_c7 = slide.shapes.add_textbox(int(Mm(570)), int(Mm(338)), int(Mm(255)), int(Mm(48)))
    tf_c7 = tb_c7.text_frame
    tf_c7.word_wrap = True
    p_c7 = tf_c7.paragraphs[0]
    r_c7_a = p_c7.add_run()
    r_c7_a.text = "Perfect recovery: "
    r_c7_a.font.bold = True
    r_c7_a.font.size = Pt(10)
    r_c7_a.font.color.rgb = GOOD
    r_c7_b = p_c7.add_run()
    r_c7_b.text = "Rule checks achieve 100% detection on setting discrepancies, impossible means, and placeholder text.\n"
    r_c7_b.font.size = Pt(9.5)
    r_c7_c = p_c7.add_run()
    r_c7_c.text = "Verifier Impact: "
    r_c7_c.font.bold = True
    r_c7_c.font.size = Pt(10)
    r_c7_c.font.color.rgb = BLUE
    r_c7_d = p_c7.add_run()
    r_c7_d.text = "Evidence gating reduces human review burden by "
    r_c7_d.font.size = Pt(9.5)
    r_c7_e = p_c7.add_run()
    r_c7_e.text = "35% fewer flags "
    r_c7_e.font.bold = True
    r_c7_e.font.color.rgb = INK
    r_c7_f = p_c7.add_run()
    r_c7_f.text = "(391 -> 255) while preserving high recall."
    r_c7_f.font.size = Pt(9.5)

    # Card 8: Pareto Dominance & Web App (h: 160 mm)
    add_card(slide, Mm(566), Mm(396), Mm(263), Mm(160), border_color=BLUE)
    add_card_header(slide, Mm(566), Mm(396), Mm(263), 8, "Better, cheaper & interactive web app")

    # Graphic: Cost vs F1 on Left, App screenshot on Right
    add_image_fit(slide, "Final_Report/poster/assets/chart_cost.png", Mm(570), Mm(412), Mm(122), Mm(76))
    add_image_fit(slide, "Final_Report/app_screenshots/light_claim.png", Mm(696), Mm(412), Mm(129), Mm(76))

    tb_c8 = slide.shapes.add_textbox(int(Mm(570)), int(Mm(492)), int(Mm(255)), int(Mm(60)))
    tf_c8 = tb_c8.text_frame
    tf_c8.word_wrap = True
    p_c8 = tf_c8.paragraphs[0]
    r_c8_1 = p_c8.add_run()
    r_c8_1.text = "Pareto dominant: "
    r_c8_1.font.bold = True
    r_c8_1.font.size = Pt(10)
    r_c8_1.font.color.rgb = BLUE
    r_c8_2 = p_c8.add_run()
    r_c8_2.text = "At Rs 7.4 ($0.079) per paper, our system is "
    r_c8_2.font.size = Pt(9.5)
    r_c8_3 = p_c8.add_run()
    r_c8_3.text = "13x more accurate and 7x cheaper "
    r_c8_3.font.bold = True
    r_c8_3.font.color.rgb = GOOD
    r_c8_4 = p_c8.add_run()
    r_c8_4.text = "than agentic search (Rs 50.9).\n"
    r_c8_4.font.size = Pt(9.5)
    r_c8_5 = p_c8.add_run()
    r_c8_5.text = "Working Web UI: "
    r_c8_5.font.bold = True
    r_c8_5.font.size = Pt(10)
    r_c8_5.font.color.rgb = INK
    r_c8_6 = p_c8.add_run()
    r_c8_6.text = "Streamlit dashboard displays verified flags with dual verbatim quotes, triage outcomes, and token spend tracking."
    r_c8_6.font.size = Pt(9.5)

    # ================================================================
    # FOOTER (y: 560 mm to 585 mm, h: 25 mm)
    # ================================================================
    f_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(Mm(12)), int(Mm(560)), int(Mm(817)), int(Mm(24)))
    f_bar.fill.solid()
    f_bar.fill.fore_color.rgb = BG_LIGHT
    f_bar.line.color.rgb = RULEGREY
    f_bar.line.width = Pt(1.0)

    # Footer Left Text
    tb_foot_l = slide.shapes.add_textbox(int(Mm(16)), int(Mm(560)), int(Mm(525)), int(Mm(23)))
    tf_foot_l = tb_foot_l.text_frame
    tf_foot_l.word_wrap = True
    p_fl1 = tf_foot_l.paragraphs[0]
    r_fl1 = p_fl1.add_run()
    r_fl1.text = "SERC & CVIT, International Institute of Information Technology, Hyderabad (IIIT-H)   •   CL3.410 LMA Project"
    r_fl1.font.bold = True
    r_fl1.font.size = Pt(10.5)
    r_fl1.font.color.rgb = INK

    p_fl2 = tf_foot_l.add_paragraph()
    r_ref_title = p_fl2.add_run()
    r_ref_title.text = "Key References: "
    r_ref_title.font.bold = True
    r_ref_title.font.size = Pt(8.5)
    r_ref_title.font.color.rgb = BLUE

    r_refs = p_fl2.add_run()
    r_refs.text = "[1] Kim et al. (HALLMARK), arXiv 2026   [2] Baumgärtner & Gurevych (SciCoQA), arXiv 2026   [3] Son et al. (SPOT), arXiv 2025   [4] Yao et al. (ReAct), ICLR 2023   [5] Lopez (GROBID), ECDL 2009"
    r_refs.font.size = Pt(8.5)
    r_refs.font.color.rgb = SLATE_TEXT

    # Footer QR Codes on Right
    add_image_fit(slide, "Final_Report/poster/assets/qr_code.png", Mm(565), Mm(561), Mm(21), Mm(21))
    tb_qr1 = slide.shapes.add_textbox(int(Mm(588)), int(Mm(561)), int(Mm(96)), int(Mm(21)))
    tf_qr1 = tb_qr1.text_frame
    p_q1 = tf_qr1.paragraphs[0]
    r_q1_a = p_q1.add_run()
    r_q1_a.text = "Code Repository\n"
    r_q1_a.font.bold = True
    r_q1_a.font.size = Pt(10)
    r_q1_a.font.color.rgb = INK
    r_q1_b = p_q1.add_run()
    r_q1_b.text = "github.com/itikelabhaskar/\nResearch_Paper_auditor"
    r_q1_b.font.size = Pt(8)
    r_q1_b.font.color.rgb = SLATE_TEXT

    add_image_fit(slide, "Final_Report/poster/assets/qr_website.png", Mm(692), Mm(561), Mm(21), Mm(21))
    tb_qr2 = slide.shapes.add_textbox(int(Mm(715)), int(Mm(561)), int(Mm(108)), int(Mm(21)))
    tf_qr2 = tb_qr2.text_frame
    p_q2 = tf_qr2.paragraphs[0]
    r_q2_a = p_q2.add_run()
    r_q2_a.text = "Project Website\n"
    r_q2_a.font.bold = True
    r_q2_a.font.size = Pt(10)
    r_q2_a.font.color.rgb = INK
    r_q2_b = p_q2.add_run()
    r_q2_b.text = "va24d.github.io/\nResearch_Paper_auditor"
    r_q2_b.font.size = Pt(8)
    r_q2_b.font.color.rgb = SLATE_TEXT

    out_path = "Final_Report/poster/Research_Paper_Auditor_Poster_A1_Landscape.pptx"
    prs.save(out_path)
    print(f"Saved landscape poster to: {out_path}")

    new_template_path = "Final_Report/poster/A1 size Landscape Research poster - new.pptx"
    prs.save(new_template_path)
    print(f"Also updated: {new_template_path}")

if __name__ == "__main__":
    main()
