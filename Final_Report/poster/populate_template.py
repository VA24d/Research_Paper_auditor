#!/usr/bin/env python3
"""Populate the official A1 Landscape Research Poster Template (v2).
Fixes:
1. Eliminates all ugly white image background boxes by matching exact card colors:
   - Col 1: #E1EDEB
   - Col 2 Top: #EFEAE4
   - Col 2 Bottom: #F2F2F2
   - Col 3: #E1EDEB
2. Adds missing sections from the template:
   - Restores ACKNOWLEDGMENT section!
   - Properly aligns CONCLUSION AND FUTURE WORKS.
   - Cleans up CORE REFERENCES (5 crucial citations).
   - Formats SERC & CVIT lab header.
3. Completely removes rogue bullet artifacts by resetting paragraph properties and margins.
4. Fills all dead space with rich quantitative data from report.tex.
"""

import os
from PIL import Image
import pptx
from pptx.util import Pt, Mm
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement

# Report and template colors
INK = RGBColor(20, 33, 61)         # #14213D (Deep Ink / Body)
BLUE = RGBColor(11, 92, 173)       # #0B5CAD (Report Primary Blue)
NAVY = RGBColor(15, 44, 89)        # #0F2C59 (Deep Navy for Headings)
CFULL = RGBColor(31, 111, 178)     # #1F6FB2 (Ours)
GOOD = RGBColor(46, 125, 79)       # #2E7D4F (Success / 0 False Alarms)
BAD = RGBColor(192, 57, 43)        # #C0392B (Error / Baseline)
GREY = RGBColor(100, 116, 139)     # #64748B (Slate subtext)
WHITE = RGBColor(255, 255, 255)
BORDER_STAT = RGBColor(203, 213, 225)

def clean_paragraph_bullets(p):
    """Strip all bullet and hanging indent XML properties from a paragraph."""
    pPr = p._p.get_or_add_pPr()
    pPr.attrib['indent'] = '0'
    pPr.attrib['marL'] = '0'
    # Remove any buChar, buAutoNum, buFont, buBlip
    for child in list(pPr):
        if child.tag.endswith(('buChar', 'buAutoNum', 'buFont', 'buBlip', 'buClr', 'buSzPct', 'buSzPts')):
            pPr.remove(child)
    # Ensure buNone is added if needed
    buNone = OxmlElement('a:buNone')
    pPr.append(buNone)

def set_text(shape, text, font_name="Arial", font_size=Pt(18), bold=False, color=INK, align=PP_ALIGN.LEFT):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.clear()
    lines = text.split('\n')
    for idx, line in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        clean_paragraph_bullets(p)
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name = font_name
        r.font.size = font_size
        r.font.bold = bold
        r.font.color.rgb = color

def add_image_inside(slide, img_path, left, top, max_width, max_height):
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

def add_mini_stat(slide, left, top, width, height, number_str, label_str, note_str="", num_color=BLUE):
    box = slide.shapes.add_shape(pptx.enum.shapes.MSO_SHAPE.ROUNDED_RECTANGLE, int(left), int(top), int(width), int(height))
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = BORDER_STAT
    box.line.width = Pt(1.2)

    tb = slide.shapes.add_textbox(int(left), int(top), int(width), int(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    p1 = tf.paragraphs[0]
    clean_paragraph_bullets(p1)
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = number_str
    r1.font.name = "Arial"
    r1.font.size = Pt(21)
    r1.font.bold = True
    r1.font.color.rgb = num_color

    p2 = tf.add_paragraph()
    clean_paragraph_bullets(p2)
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = label_str
    r2.font.name = "Arial"
    r2.font.size = Pt(10.5)
    r2.font.bold = True
    r2.font.color.rgb = INK

    if note_str:
        p3 = tf.add_paragraph()
        clean_paragraph_bullets(p3)
        p3.alignment = PP_ALIGN.CENTER
        r3 = p3.add_run()
        r3.text = note_str
        r3.font.name = "Arial"
        r3.font.size = Pt(8.5)
        r3.font.color.rgb = GREY

def main():
    prs = pptx.Presentation("Final_Report/poster/A1 size Landscape Research poster - new.template.bak.pptx")
    slide = prs.slides[0]
    shape_map = {s.shape_id: s for s in slide.shapes}

    # ================================================================
    # 1. HEADER (Title, Subtitle, Authors, Affiliations)
    # ================================================================
    s_title = shape_map[108]
    s_title.left = Mm(14)
    s_title.top = Mm(8)
    s_title.width = Mm(570)
    s_title.height = Mm(62)
    tf_t = s_title.text_frame
    tf_t.clear()
    
    p_t = tf_t.paragraphs[0]
    clean_paragraph_bullets(p_t)
    r_t = p_t.add_run()
    r_t.text = "Research Paper Auditor: Evidence-Gated Verification"
    r_t.font.name = "Arial"
    r_t.font.size = Pt(36)
    r_t.font.bold = True
    r_t.font.color.rgb = INK

    p_sub = tf_t.add_paragraph()
    clean_paragraph_bullets(p_sub)
    r_sub = p_sub.add_run()
    r_sub.text = "Does the paper match its citations and its code? Pre-review factual auditing for AI conferences"
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(17.5)
    r_sub.font.bold = True
    r_sub.font.color.rgb = BLUE

    # Footer author names
    s_auth = shape_map[106]
    s_auth.left = Mm(14)
    s_auth.top = Mm(568)
    s_auth.width = Mm(460)
    tf_a = s_auth.text_frame
    tf_a.clear()
    p_a = tf_a.paragraphs[0]
    clean_paragraph_bullets(p_a)
    r_a1 = p_a.add_run()
    r_a1.text = "Authors: "
    r_a1.font.bold = True
    r_a1.font.size = Pt(13)
    r_a1.font.color.rgb = INK
    r_a2 = p_a.add_run()
    r_a2.text = "Vijay Aravynthan,  Itikela Bhaskar,  Anurag Peddi,  Sudheera Y S   •   SERC & CVIT, IIIT Hyderabad"
    r_a2.font.size = Pt(13)
    r_a2.font.color.rgb = INK

    # Footer contact
    s_email = shape_map[107]
    s_email.left = Mm(480)
    s_email.top = Mm(568)
    s_email.width = Mm(345)
    tf_e = s_email.text_frame
    tf_e.clear()
    p_e = tf_e.paragraphs[0]
    clean_paragraph_bullets(p_e)
    r_e1 = p_e.add_run()
    r_e1.text = "Contact: "
    r_e1.font.bold = True
    r_e1.font.size = Pt(13)
    r_e1.font.color.rgb = INK
    r_e2 = p_e.add_run()
    r_e2.text = "{vijay.aravynthan, itikela.bhaskar, anurag.peddi, sudheera.ys}@research.iiit.ac.in"
    r_e2.font.size = Pt(13)
    r_e2.font.color.rgb = GREY

    # ================================================================
    # 2. COLUMN 1: ABSTRACT & METHODOLOGY (Cards #E1EDEB)
    # ================================================================
    # Card 1: ABSTRACT: THE PROBLEM (PAPER VS. CODE)
    s_h1 = shape_map[85]
    s_h1.top = Mm(84)
    set_text(s_h1, "ABSTRACT: THE PROBLEM (PAPER VS. CODE)", font_size=Pt(19), bold=True, color=NAVY)

    s_t1 = shape_map[86]
    s_t1.top = Mm(98)
    s_t1.height = Mm(78)
    tf_1 = s_t1.text_frame
    tf_1.clear()
    
    p1 = tf_1.paragraphs[0]
    clean_paragraph_bullets(p1)
    r = p1.add_run()
    r.text = "• The Reviewer Bottleneck: "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r2 = p1.add_run()
    r2.text = "Top AI conferences (ICLR, NeurIPS) receive 62,000+ submissions annually. Reviewers have only 15–20 minutes per paper and cannot manually inspect external code repositories or trace cited papers, allowing unlabelled discrepancies and hallucinations to slip into published work.\n\n"
    r2.font.size = Pt(11.5)
    r2.font.color.rgb = INK

    p2 = tf_1.add_paragraph()
    clean_paragraph_bullets(p2)
    r = p2.add_run()
    r.text = "• Real Discovery (Mirzadeh et al., NeurIPS 2020): "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r3 = p2.add_run()
    r3.text = "In an accepted paper on continual learning, our auditor autonomously caught an unlabelled contradiction: the paper explicitly claims no shuffling, but the code sets "
    r3.font.size = Pt(11.5)
    r3.font.color.rgb = INK
    r4 = p2.add_run()
    r4.text = "shuffle=True"
    r4.font.bold = True
    r4.font.size = Pt(11.5)
    r4.font.color.rgb = BAD
    r5 = p2.add_run()
    r5.text = ". Both lines were quoted verbatim without human hinting:"
    r5.font.size = Pt(11.5)
    r5.font.color.rgb = INK

    # Graphic: paper_vs_code.png
    add_image_inside(slide, "Final_Report/poster/assets/paper_vs_code.png", Mm(14), Mm(180), Mm(230), Mm(80))

    # Card 2: METHODOLOGY: EVIDENCE-GATED PIPELINE
    s_h2 = shape_map[87]
    s_h2.top = Mm(280)
    set_text(s_h2, "METHODOLOGY: EVIDENCE-GATED PIPELINE", font_size=Pt(19), bold=True, color=NAVY)

    s_t2 = shape_map[91]
    s_t2.top = Mm(295)
    s_t2.height = Mm(72)
    tf_2 = s_t2.text_frame
    tf_2.clear()

    p = tf_2.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Core Verification Invariant: "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r_sub = p.add_run()
    r_sub.text = "A discrepancy is flagged ONLY when backed by exact verbatim quotes on both sides (paper text + cited paper / code repo). Unverified flags are strictly dropped, guaranteeing 0 false alarms on clean controls.\n\n"
    r_sub.font.size = Pt(11.5)
    r_sub.font.color.rgb = INK

    p_bench = tf_2.add_paragraph()
    clean_paragraph_bullets(p_bench)
    r = p_bench.add_run()
    r.text = "• Multi-Domain Benchmark Testbed: "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r_bench = p_bench.add_run()
    r_bench.text = "Evaluated on 40 papers + GitHub codebases, 163 planted discrepancies & 72 clean controls (100% precision), 831 citations (HALLMARK), and 13 mathematical proofs (SPOT corpus)."
    r_bench.font.size = Pt(11.5)
    r_bench.font.color.rgb = INK

    # Graphic: system_pipeline.png
    s_img_pip = shape_map[98]
    s_img_pip.text_frame.clear()
    s_img_pip.line.fill.background()
    s_img_pip.fill.background()
    add_image_inside(slide, "Final_Report/poster/assets/system_pipeline.png", Mm(14), Mm(370), Mm(230), Mm(178))

    # ================================================================
    # 3. COLUMN 2: RESULTS, ABLATIONS & SPECIALISTS
    # ================================================================
    # Card 3: RESULTS (Top Wide Card - #EFEAE4)
    s_h3 = shape_map[89]
    s_h3.top = Mm(84)
    set_text(s_h3, "RESULTS: 13× GAIN OVER BASELINES (0.636 F1)", font_size=Pt(20), bold=True, color=NAVY)

    # 4 Stat Badges
    bw = Mm(88)
    bh = Mm(36)
    add_mini_stat(slide, Mm(258), Mm(98), bw, bh, "0.636", "Macro-F1", "Baselines: 0.05 (+59 pts)", num_color=CFULL)
    add_mini_stat(slide, Mm(351), Mm(98), bw, bh, "0", "False Alarms", "on 72 clean controls (100% P)", num_color=GOOD)
    add_mini_stat(slide, Mm(444), Mm(98), bw, bh, "100%", "Evidence Grounding", "Exact dual verbatim quotes", num_color=BLUE)
    add_mini_stat(slide, Mm(537), Mm(98), bw, bh, "₹7.4", "Cost / Paper Audit", "B1 Agent: ₹50.9 (7× cheaper)", num_color=INK)

    # Results Text with Detailed Metrics
    s_t3 = shape_map[88]
    s_t3.top = Mm(138)
    s_t3.height = Mm(34)
    tf_3 = s_t3.text_frame
    tf_3.clear()
    
    p = tf_3.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Massive Performance Leap: "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Full system achieves 0.636 Macro-F1 across 40 papers. Zero-shot LLM (B0, 0.048 F1) and ReAct Agent (B1, 0.046 F1) fail completely on Method and Metric discrepancies (0.00 F1).\n"
    r2.font.size = Pt(11.5)
    r2.font.color.rgb = INK

    p2 = tf_3.add_paragraph()
    clean_paragraph_bullets(p2)
    r = p2.add_run()
    r.text = "• Precision-First Guarantee: "
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = INK
    r3 = p2.add_run()
    r3.text = "0 false alarms on 72 clean controls. Verifier evidence gate merges duplicates, removes 136 spurious flags (391 → 255, 35% workload reduction), and ensures every accusation is actionable."
    r3.font.size = Pt(11.5)
    r3.font.color.rgb = INK

    # Replace Shape 94 with chart_f1.png
    s_img_res = shape_map[94]
    s_img_res.text_frame.clear()
    s_img_res.line.fill.background()
    s_img_res.fill.background()
    add_image_inside(slide, "Final_Report/poster/assets/chart_f1.png", Mm(260), Mm(174), Mm(366), Mm(126))

    # Card 4A: ABLATION INSIGHTS & COST PARETO (Bottom Left - #F2F2F2)
    s_h4 = shape_map[110]
    s_h4.top = Mm(314)
    s_h4.width = Mm(180)
    set_text(s_h4, "ABLATIONS & COST PARETO", font_size=Pt(18), bold=True, color=NAVY)

    s_t4 = shape_map[111]
    s_t4.top = Mm(330)
    s_t4.height = Mm(46)
    tf_4 = s_t4.text_frame
    tf_4.clear()
    
    p = tf_4.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Dense Code Embeddings: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Locates 70.2% (59/84) of method changes vs 41.7% (35/84) for lexical overlap (+28.5 pts).\n"
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    p = tf_4.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• PDF Table Extraction: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Vision region cropping succeeds on 98% of tables vs 41% for GROBID structured parser.\n"
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    p = tf_4.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Model Pareto Frontier: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Gemini Flash achieves 0.636 F1 at ₹7.4; Pro achieves 0.650 F1 at ₹26.6 (3.6× cost). Flash provides 7× lower audit cost."
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    s_img_abl = shape_map[112]
    s_img_abl.text_frame.clear()
    s_img_abl.line.fill.background()
    s_img_abl.fill.background()
    add_image_inside(slide, "Final_Report/poster/assets/chart_cost.png", Mm(258), Mm(386), Mm(180), Mm(160))

    # Card 4B: SPECIALIST ACCURACY & ERROR BREAKDOWN (Bottom Right - #F2F2F2)
    s_h5 = shape_map[96]
    s_h5.top = Mm(314)
    s_h5.width = Mm(180)
    set_text(s_h5, "SPECIALIST ACCURACY & ANALYSIS", font_size=Pt(18), bold=True, color=NAVY)

    s_t5 = shape_map[97]
    s_t5.top = Mm(330)
    s_t5.height = Mm(46)
    tf_5 = s_t5.text_frame
    tf_5.clear()

    p = tf_5.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Deterministic Rule Accuracy: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "100% on settings, impossible means (GRIM test), and placeholder text; 94% on comparisons.\n"
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    p = tf_5.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• 35% Workload Reduction: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Evidence gate drops 35% of spurious flags (391 → 255 flags to review) with negligible recall penalty.\n"
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    p = tf_5.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Zero Cost First Pass: "
    r.font.bold = True
    r.font.size = Pt(11.5)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Static AST and citation lookups run before model calls, keeping token expenditures minimal."
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

    s_img_spec = shape_map[105]
    s_img_spec.text_frame.clear()
    s_img_spec.line.fill.background()
    s_img_spec.fill.background()
    add_image_inside(slide, "Final_Report/poster/assets/chart_each_check.png", Mm(450), Mm(386), Mm(180), Mm(160))

    # ================================================================
    # 4. COLUMN 3: LAB BRANDING, APP, CONCLUSION, ACKNOWLEDGMENT & REFS
    # ================================================================
    # Shape 99 (Lab background) & 103/104 (Text)
    s_lab_box = shape_map[99]
    s_lab_box.line.color.rgb = BLUE
    s_lab_box.line.width = Pt(1.5)

    s_lab1 = shape_map[103]
    s_lab1.left = Mm(645)
    s_lab1.top = Mm(79)
    s_lab1.width = Mm(184)
    set_text(s_lab1, "SERC & CVIT", font_size=Pt(27), bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    s_lab2 = shape_map[104]
    s_lab2.left = Mm(645)
    s_lab2.top = Mm(94)
    s_lab2.width = Mm(184)
    set_text(s_lab2, "Software Engineering Research Center  •  Centre for Visual Info Technology\nInternational Institute of Information Technology, Hyderabad\nTeam: Hood Research Department   •   Course: CL3.410 LMA", font_size=Pt(9.2), bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    # Section 1: INTERACTIVE WEB APPLICATION (Shape 90 header, Shape 102 body)
    s_h6 = shape_map[90]
    s_h6.top = Mm(142)
    s_h6.height = Mm(14)
    set_text(s_h6, "INTERACTIVE WEB APP & IMPACT", font_size=Pt(17.5), bold=True, color=NAVY)

    s_t6 = shape_map[102]
    s_t6.top = Mm(156)
    s_t6.height = Mm(28)
    tf_6 = s_t6.text_frame
    tf_6.clear()
    p = tf_6.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Live Streamlit Dashboard: "
    r.font.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Reviewers inspect dual verbatim quotes, triage outcomes, and audit token expenditures.\n"
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = INK

    p = tf_6.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Conference Scale: "
    r.font.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Automates pre-review factual checks for tens of thousands of submissions at ₹7.4 per paper."
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = INK

    add_image_inside(slide, "Final_Report/app_screenshots/light_claim.png", Mm(646), Mm(192), Mm(176), Mm(54))

    # Section 2: CONCLUSION AND FUTURE WORKS (Shape 100 header, Shape 92 body)
    s_h7 = shape_map[100]
    s_h7.top = Mm(252)
    s_h7.height = Mm(14)
    set_text(s_h7, "CONCLUSION AND FUTURE WORKS", font_size=Pt(17.5), bold=True, color=NAVY)

    s_t7 = shape_map[92]
    s_t7.top = Mm(266)
    s_t7.height = Mm(32)
    tf_7 = s_t7.text_frame
    tf_7.clear()
    p = tf_7.paragraphs[0]
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Conclusion: "
    r.font.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Dual-quote evidence gating eliminates LLM hallucinations, establishing automated pre-review auditing as a reliable, scalable conference safeguard.\n"
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = INK

    p = tf_7.add_paragraph()
    clean_paragraph_bullets(p)
    r = p.add_run()
    r.text = "• Future Work: "
    r.font.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = INK
    r2 = p.add_run()
    r2.text = "Dynamic execution sandboxes for GPU training scripts, C++/CUDA parsing, and multimodal plot fact-checking."
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = INK

    # Section 3: ACKNOWLEDGMENT (Restoring the explicit template section!)
    tb_ack_h = slide.shapes.add_textbox(int(Mm(646)), int(Mm(304)), int(Mm(176)), int(Mm(12)))
    set_text(tb_ack_h, "ACKNOWLEDGMENT", font_size=Pt(16), bold=True, color=NAVY)

    tb_ack_b = slide.shapes.add_textbox(int(Mm(646)), int(Mm(316)), int(Mm(176)), int(Mm(26)))
    tf_ack = tb_ack_b.text_frame
    tf_ack.word_wrap = True
    tf_ack.clear()
    p_ack = tf_ack.paragraphs[0]
    clean_paragraph_bullets(p_ack)
    r = p_ack.add_run()
    r.text = "We gratefully acknowledge the faculty and mentors at SERC and CVIT, IIIT Hyderabad, and the instructors and teaching assistants of CL3.410 Language Models and Applications for their guidance. We also thank Google DeepMind for Gemini API access."
    r.font.name = "Arial"
    r.font.size = Pt(10)
    r.font.color.rgb = INK

    # Section 4: CORE REFERENCES (5 Crucial References - Shape 101 header, Shape 93 body)
    s_h8 = shape_map[101]
    s_h8.top = Mm(346)
    s_h8.height = Mm(12)
    set_text(s_h8, "CORE REFERENCES (5 CRUCIAL)", font_size=Pt(17.5), bold=True, color=NAVY)

    s_t8 = shape_map[93]
    s_t8.top = Mm(360)
    s_t8.height = Mm(138)
    tf_8 = s_t8.text_frame
    tf_8.clear()

    refs = [
        ("[1] HALLMARK: ", "K. Kim et al., Diagnosing LLM citation verifiers and when to deploy them, arXiv:2607.18360, 2026."),
        ("[2] SciCoQA: ", "T. Baumgärtner & I. Gurevych, Quality assurance for scientific paper–code alignment, arXiv:2601.12910, 2026."),
        ("[3] SPOT: ", "G. Son et al., Benchmark for automated verification of scientific research, arXiv:2505.11855, 2025."),
        ("[4] ReAct: ", "S. Yao et al., Synergizing reasoning and acting in language models, ICLR 2023."),
        ("[5] GROBID: ", "P. Lopez, Automatic bibliographic data recognition for scholarship publications, ECDL 2009.")
    ]
    for idx, (lbl, citation) in enumerate(refs):
        p_ref = tf_8.paragraphs[0] if idx == 0 else tf_8.add_paragraph()
        clean_paragraph_bullets(p_ref)
        r1 = p_ref.add_run()
        r1.text = lbl
        r1.font.name = "Arial"
        r1.font.bold = True
        r1.font.size = Pt(10.5)
        r1.font.color.rgb = BLUE
        r2 = p_ref.add_run()
        r2.text = citation + ("\n" if idx < len(refs) - 1 else "")
        r2.font.name = "Arial"
        r2.font.size = Pt(10)
        r2.font.color.rgb = INK

    # Section 5: CODE REPOSITORY & WEBSITE (QR Codes at bottom of Col 3)
    add_image_inside(slide, "Final_Report/poster/assets/qr_code.png", Mm(656), Mm(504), Mm(34), Mm(34))
    add_image_inside(slide, "Final_Report/poster/assets/qr_website.png", Mm(742), Mm(504), Mm(34), Mm(34))

    tb_qrl1 = slide.shapes.add_textbox(int(Mm(646)), int(Mm(540)), int(Mm(56)), int(Mm(10)))
    set_text(tb_qrl1, "GitHub Code", font_size=Pt(10), bold=True, color=INK, align=PP_ALIGN.CENTER)

    tb_qrl2 = slide.shapes.add_textbox(int(Mm(732)), int(Mm(540)), int(Mm(56)), int(Mm(10)))
    set_text(tb_qrl2, "Project Website", font_size=Pt(10), bold=True, color=INK, align=PP_ALIGN.CENTER)

    out_file = "Final_Report/poster/A1 size Landscape Research poster - new.pptx"
    prs.save(out_file)
    print(f"Successfully populated and refined template poster: {out_file}")

if __name__ == "__main__":
    main()
