#!/usr/bin/env python3
"""Generate high-resolution, perfectly color-matched visual assets for the poster.
Matches exact card background colors:
- Col 1 Cards: #E1EDEB
- Col 2 Top Card: #EFEAE4
- Col 2 Bottom Cards: #F2F2F2
- Col 3 Card: #E1EDEB
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# ================================================================
# 1. CHART F1 (Top Center Card - #EFEAE4)
# ================================================================
def make_chart_f1():
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=300)
    bg = '#EFEAE4'
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    categories = ['Method Verification', 'Metric Verification', 'Setting & Hyperparams', 'Overall Macro-F1']
    x = np.arange(len(categories))
    width = 0.20

    full = [0.34, 0.80, 0.77, 0.636]
    nover = [0.34, 0.80, 0.86, 0.667]
    b0 = [0.00, 0.00, 0.14, 0.048]
    b1 = [0.00, 0.00, 0.14, 0.046]

    c_full = '#1F6FB2'
    c_nover = '#8DBBE3'
    c_b0 = '#E08A2E'
    c_b1 = '#B5443A'

    r1 = ax.bar(x - 1.5*width, full, width, label='Full System (Ours)', color=c_full, edgecolor='none', zorder=3)
    r2 = ax.bar(x - 0.5*width, nover, width, label='No Verifier (Ablation)', color=c_nover, edgecolor='none', zorder=3)
    r3 = ax.bar(x + 0.5*width, b0, width, label='B0: Zero-Shot LLM', color=c_b0, edgecolor='none', zorder=3)
    r4 = ax.bar(x + 1.5*width, b1, width, label='B1: ReAct Agent', color=c_b1, edgecolor='none', zorder=3)

    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            h = rect.get_height()
            if h > 0.02:
                lbl = f'{h:.2f}' if h != 0.636 else '0.636'
                ax.annotate(lbl,
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 4), textcoords='offset points',
                            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#14213D')
            elif h == 0.00:
                ax.annotate('0.00',
                            xy=(rect.get_x() + rect.get_width() / 2, 0),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, color='#64748B')

    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold', color='#14213D')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold', color='#14213D')
    ax.set_ylim(0, 1.08)
    ax.grid(axis='y', linestyle='--', alpha=0.55, color='#CBD5E1', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#94A3B8')
    ax.spines['bottom'].set_color('#94A3B8')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig('Final_Report/poster/assets/chart_f1.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Saved chart_f1.png")

# ================================================================
# 2. CHART COST (Bottom-Left Card - #F2F2F2)
# ================================================================
def make_chart_cost():
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=300)
    bg = '#F2F2F2'
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # Data
    # Full System: ₹7.4, 0.636 F1
    # B0 Zero-Shot: ₹4.4, 0.048 F1
    # B1 ReAct Agent: ₹50.9, 0.046 F1
    # Gemini Pro: ₹26.6, 0.650 F1
    ax.set_xscale('log')
    ax.set_xlim(2.8, 85)
    ax.set_ylim(-0.04, 0.82)

    ax.scatter([7.4], [0.636], s=260, color='#1F6FB2', edgecolors='#0B5CAD', linewidth=2.5, zorder=5)
    ax.scatter([4.4], [0.048], s=160, color='#E08A2E', edgecolors='#C26B1E', linewidth=1.5, zorder=5)
    ax.scatter([50.9], [0.046], s=160, color='#B5443A', edgecolors='#8F2A21', linewidth=1.5, zorder=5)
    ax.scatter([26.6], [0.650], s=160, color='#64748B', edgecolors='#475569', linewidth=1.5, zorder=5)

    # Annotations
    ax.annotate("Ours: Full System\n₹7.4 (0.636 F1)", xy=(7.4, 0.636), xytext=(15, 15),
                textcoords='offset points', fontsize=11, fontweight='bold', color='#1F6FB2',
                bbox=dict(boxstyle="round,pad=0.3", fc="#E1EDEB", ec="#1F6FB2", lw=1.2))

    ax.annotate("B0: Zero-Shot\n₹4.4 (0.048 F1)", xy=(4.4, 0.048), xytext=(-10, 18),
                textcoords='offset points', fontsize=9.5, fontweight='bold', color='#E08A2E')

    ax.annotate("B1: ReAct Agent\n₹50.9 (0.046 F1, 7× Cost)", xy=(50.9, 0.046), xytext=(-95, 22),
                textcoords='offset points', fontsize=9.5, fontweight='bold', color='#B5443A')

    ax.annotate("Gemini Pro\n₹26.6 (0.650 F1)", xy=(26.6, 0.650), xytext=(-75, -28),
                textcoords='offset points', fontsize=9.5, color='#475569')

    # Connecting curve / Pareto line
    ax.plot([4.4, 7.4, 26.6], [0.048, 0.636, 0.650], linestyle='--', color='#94A3B8', linewidth=1.5, zorder=2)

    import matplotlib.ticker as ticker
    ax.set_xlabel('Audit Cost per Paper (INR, log scale)', fontsize=12, fontweight='bold', color='#14213D')
    ax.set_ylabel('Macro-F1 Score', fontsize=12, fontweight='bold', color='#14213D')
    ax.xaxis.set_major_locator(ticker.FixedLocator([4, 7.4, 15, 30, 60]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(['₹4', '₹7.4', '₹15', '₹30', '₹60']))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, which="major", ls="--", color='#CBD5E1', alpha=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#94A3B8')
    ax.spines['bottom'].set_color('#94A3B8')

    plt.tight_layout()
    plt.savefig('Final_Report/poster/assets/chart_cost.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Saved chart_cost.png")

# ================================================================
# 3. CHART EACH CHECK (Bottom-Right Card - #F2F2F2)
# ================================================================
def make_chart_each_check():
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=300)
    bg = '#F2F2F2'
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    checks = [
        'Number vs Table',
        'Wrong Claim Gain',
        'SPOT Proof Step',
        'HALLMARK Fake Ref',
        'Reversed Comparison',
        'Impossible Mean (GRIM)',
        'Placeholder Text',
        'Setting vs Code'
    ]
    scores = [20, 55, 57, 88, 94, 100, 100, 100]
    colors = [
        '#64748B', '#64748B', '#3B82F6', '#3B82F6',
        '#1F6FB2', '#2E7D4F', '#2E7D4F', '#2E7D4F'
    ]

    y = np.arange(len(checks))
    bars = ax.barh(y, scores, height=0.65, color=colors, edgecolor='none', zorder=3)

    for bar, val in zip(bars, scores):
        ax.annotate(f'{val}%',
                    xy=(val, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=10.5, fontweight='bold',
                    color='#14213D')

    ax.set_yticks(y)
    ax.set_yticklabels(checks, fontsize=11, fontweight='bold', color='#14213D')
    ax.set_xlabel('Planted Errors Detected (%)', fontsize=12, fontweight='bold', color='#14213D')
    ax.set_xlim(0, 115)
    ax.grid(axis='x', linestyle='--', alpha=0.55, color='#CBD5E1', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#94A3B8')
    ax.spines['bottom'].set_color('#94A3B8')

    plt.tight_layout()
    plt.savefig('Final_Report/poster/assets/chart_each_check.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Saved chart_each_check.png")

# ================================================================
# 4. PAPER VS CODE (Column 1 Top Card - #E1EDEB)
# ================================================================
def make_paper_vs_code():
    fig, ax = plt.subplots(figsize=(10.0, 3.8), dpi=300)
    bg = '#E1EDEB'
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Card 1: Paper snippet (Left)
    paper_box = patches.FancyBboxPatch((0.2, 0.4), 4.2, 3.2, boxstyle="round,pad=0.15,rounding_size=0.12",
                                      facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.5, zorder=1)
    ax.add_patch(paper_box)

    # Paper Header
    ax.text(0.4, 3.3, "NeurIPS 2020 • Mirzadeh et al.", fontsize=11, fontweight='bold', color='#64748B')
    ax.text(0.4, 3.0, "§B.2 Continual Learning Setup (p. 14)", fontsize=10.5, color='#94A3B8')
    
    # Paper Mock Lines
    ax.add_patch(patches.Rectangle((0.4, 2.7), 3.8, 0.08, facecolor='#E2E8F0', edgecolor='none'))
    ax.add_patch(patches.Rectangle((0.4, 2.5), 3.4, 0.08, facecolor='#E2E8F0', edgecolor='none'))

    # Highlighted quote
    quote_box = patches.FancyBboxPatch((0.35, 1.3), 3.9, 1.0, boxstyle="round,pad=0.1",
                                       facecolor='#FEF3C7', edgecolor='#F59E0B', linewidth=1.2)
    ax.add_patch(quote_box)
    ax.text(0.45, 1.85, '"...we have not employed any', fontsize=11, style='italic', color='#1E293B')
    ax.text(0.45, 1.50, 'data augmentation technique or ', fontsize=11, style='italic', color='#1E293B')
    ax.text(3.15, 1.50, 'shuffling."', fontsize=12, style='italic', fontweight='bold', color='#B45309')

    ax.text(0.4, 0.75, "Explicit claim in published text", fontsize=10.5, fontweight='bold', color='#0B5CAD')

    # Center != Badge
    badge = patches.Circle((5.0, 2.0), 0.55, facecolor='#EF4444', edgecolor='#B91C1C', linewidth=2.0, zorder=3)
    ax.add_patch(badge)
    ax.text(5.0, 1.95, "≠", fontsize=28, fontweight='bold', color='#FFFFFF', ha='center', va='center', zorder=4)
    ax.text(5.0, 1.15, "CONTRADICTION", fontsize=9.5, fontweight='bold', color='#DC2626', ha='center', va='center')

    # Card 2: Code Editor snippet (Right)
    code_box = patches.FancyBboxPatch((5.6, 0.4), 4.2, 3.2, boxstyle="round,pad=0.15,rounding_size=0.12",
                                     facecolor='#1E2329', edgecolor='#334155', linewidth=1.5, zorder=1)
    ax.add_patch(code_box)

    # Window dots
    ax.add_patch(patches.Circle((5.85, 3.3), 0.06, facecolor='#EF4444'))
    ax.add_patch(patches.Circle((6.05, 3.3), 0.06, facecolor='#F59E0B'))
    ax.add_patch(patches.Circle((6.25, 3.3), 0.06, facecolor='#10B981'))
    ax.text(6.45, 3.25, "stable_sgd/data_utils.py", fontsize=10.5, family='monospace', color='#94A3B8')

    # Code text
    ax.text(5.8, 2.65, "25  mnist_train = MNIST(...)", fontsize=10, family='monospace', color='#94A3B8')
    ax.text(5.8, 2.30, "26  train_loader = DataLoader(", fontsize=10, family='monospace', color='#E2E8F0')
    ax.text(5.8, 1.95, "27      mnist_train, batch_size=64,", fontsize=10, family='monospace', color='#E2E8F0')

    # Highlighted line
    code_hl = patches.Rectangle((5.75, 1.48), 3.8, 0.35, facecolor='#7F1D1D', edgecolor='#EF4444', linewidth=1)
    ax.add_patch(code_hl)
    ax.text(5.8, 1.58, "28      shuffle=True)", fontsize=11, family='monospace', fontweight='bold', color='#FCA5A5')

    ax.text(5.8, 0.75, "Actual execution code repository", fontsize=10.5, fontweight='bold', color='#38BDF8')

    plt.tight_layout()
    plt.savefig('Final_Report/poster/assets/paper_vs_code.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Saved paper_vs_code.png")

# ================================================================
# 5. SYSTEM PIPELINE (Column 1 Bottom Card - #E1EDEB)
# ================================================================
def make_system_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 7.8), dpi=300)
    bg = '#E1EDEB'
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 7.8)
    ax.axis('off')

    # 4 Columns for the 4 Stages
    # Stage 1: Ingestion
    ax.text(1.2, 7.3, "1. INGESTION", fontsize=12.5, fontweight='bold', color='#0F2C59', ha='center')
    b1_paper = patches.FancyBboxPatch((0.2, 5.0), 2.0, 2.0, boxstyle="round,pad=0.1", fc='#FFFFFF', ec='#0B5CAD', lw=1.5)
    ax.add_patch(b1_paper)
    ax.text(1.2, 6.3, "Research Paper", fontsize=11, fontweight='bold', color='#14213D', ha='center')
    ax.text(1.2, 5.8, "PDF / LaTeX Source", fontsize=9.5, color='#64748B', ha='center')
    ax.text(1.2, 5.3, "Section & Math Extraction", fontsize=8.5, color='#0B5CAD', ha='center')

    b1_code = patches.FancyBboxPatch((0.2, 2.5), 2.0, 2.0, boxstyle="round,pad=0.1", fc='#FFFFFF', ec='#0B5CAD', lw=1.5)
    ax.add_patch(b1_code)
    ax.text(1.2, 3.8, "Code Repository", fontsize=11, fontweight='bold', color='#14213D', ha='center')
    ax.text(1.2, 3.3, "GitHub Repository", fontsize=9.5, color='#64748B', ha='center')
    ax.text(1.2, 2.8, "AST & Call-Graph Parser", fontsize=8.5, color='#0B5CAD', ha='center')

    # Stage 2: Specialist Checkers
    ax.text(4.2, 7.3, "2. PARALLEL SPECIALISTS", fontsize=12.5, fontweight='bold', color='#0F2C59', ha='center')
    
    spec_box = patches.FancyBboxPatch((2.7, 0.5), 3.0, 6.5, boxstyle="round,pad=0.1", fc='#F8FAFC', ec='#94A3B8', lw=1.2, ls='--')
    ax.add_patch(spec_box)

    specs = [
        ("Citation Specialist", "HALLMARK: 831 refs\nFake papers, venue, year", "#1E40AF"),
        ("Code & Setting Checker", "Static AST + regex\nlr, batch, optimizer, seed", "#047857"),
        ("Theory & Proof Checker", "SPOT: 13 papers\nStep-by-step math audit", "#6D28D9"),
        ("Integrity & Tables", "GRIM test, table numbers\nContradiction detection", "#B45309")
    ]
    for i, (title, desc, tc) in enumerate(specs):
        y_pos = 5.4 - i * 1.55
        sb = patches.FancyBboxPatch((2.9, y_pos), 2.6, 1.3, boxstyle="round,pad=0.08", fc='#FFFFFF', ec=tc, lw=1.3)
        ax.add_patch(sb)
        ax.text(4.2, y_pos + 0.9, title, fontsize=10.5, fontweight='bold', color=tc, ha='center')
        ax.text(4.2, y_pos + 0.4, desc, fontsize=8.8, color='#475569', ha='center')

    # Stage 3: Verifier Gate
    ax.text(7.1, 7.3, "3. DUAL-QUOTE GATE", fontsize=12.5, fontweight='bold', color='#0F2C59', ha='center')
    b3_ver = patches.FancyBboxPatch((6.0, 1.8), 2.2, 5.2, boxstyle="round,pad=0.12", fc='#EFF6FF', ec='#1D4ED8', lw=2.0)
    ax.add_patch(b3_ver)
    ax.text(7.1, 6.3, "Evidence Verifier", fontsize=12, fontweight='bold', color='#1D4ED8', ha='center')
    ax.text(7.1, 5.8, "Quote Substring Search", fontsize=10, fontweight='bold', color='#1E293B', ha='center')
    ax.text(7.1, 4.8, "• Must quote paper text\n• Must quote source\n• Drops spurious flags\n• 35% reduction (391 to 255)\n• 0 False Alarms!",
            fontsize=9.2, color='#334155', ha='center')
    ax.text(7.1, 2.4, "Precision Guarantee:\n100% on Controls", fontsize=10, fontweight='bold', color='#047857', ha='center')

    # Stage 4: Report Outcomes
    ax.text(9.5, 7.3, "4. AUDIT REPORT", fontsize=12.5, fontweight='bold', color='#0F2C59', ha='center')
    
    # 3 outcome badges
    b4_flag = patches.FancyBboxPatch((8.6, 5.1), 1.7, 1.8, boxstyle="round,pad=0.08", fc='#ECFDF5', ec='#10B981', lw=1.5)
    ax.add_patch(b4_flag)
    ax.text(9.45, 6.2, "Verified Flag", fontsize=11, fontweight='bold', color='#047857', ha='center')
    ax.text(9.45, 5.5, "Dual quotes verified\nHuman actionable", fontsize=8.8, color='#065F46', ha='center')

    b4_rev = patches.FancyBboxPatch((8.6, 2.9), 1.7, 1.8, boxstyle="round,pad=0.08", fc='#FFFBEB', ec='#F59E0B', lw=1.5)
    ax.add_patch(b4_rev)
    ax.text(9.45, 4.0, "Needs Review", fontsize=11, fontweight='bold', color='#B45309', ha='center')
    ax.text(9.45, 3.3, "Source paywall or\nmissing file", fontsize=8.8, color='#92400E', ha='center')

    b4_drop = patches.FancyBboxPatch((8.6, 0.7), 1.7, 1.8, boxstyle="round,pad=0.08", fc='#F1F5F9', ec='#94A3B8', lw=1.5)
    ax.add_patch(b4_drop)
    ax.text(9.45, 1.8, "Dropped Flag", fontsize=11, fontweight='bold', color='#475569', ha='center')
    ax.text(9.45, 1.1, "Unverifiable claim\n(Zero false alarms)", fontsize=8.8, color='#64748B', ha='center')

    # Connecting Arrows
    arrow_kw = dict(arrowstyle="-|>", color='#64748B', lw=1.8, mutation_scale=15)
    ax.annotate("", xy=(2.7, 5.0), xytext=(2.2, 5.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(2.7, 3.5), xytext=(2.2, 3.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.0, 4.4), xytext=(5.7, 4.4), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.6, 6.0), xytext=(8.2, 5.0), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.6, 3.8), xytext=(8.2, 4.4), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.6, 1.6), xytext=(8.2, 3.8), arrowprops=arrow_kw)

    plt.tight_layout()
    plt.savefig('Final_Report/poster/assets/system_pipeline.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Saved system_pipeline.png")

if __name__ == "__main__":
    make_chart_f1()
    make_chart_cost()
    make_chart_each_check()
    make_paper_vs_code()
    make_system_pipeline()
    print("All poster assets successfully generated.")
