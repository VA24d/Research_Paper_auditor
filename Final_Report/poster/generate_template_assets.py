#!/usr/bin/env python3
"""Generate bold, high-density, perfectly matched visual assets for the template-styled poster.
Eliminates all white-box artifacts by matching the exact template card backgrounds:
- Col 1 (Mint): #E1EDEB
- Col 2 Top (Beige): #EFEAE4
- Col 2 Bottom (Crisp Grey): #F2F2F2
- Col 3 (Mint): #E1EDEB
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

COLOR_MINT = '#E1EDEB'
COLOR_BEIGE = '#EFEAE4'
COLOR_GREY = '#F2F2F2'
PURPLE = '#47247B'
PURPLE_LINE = '#7030A0'

# ================================================================
# 1. PAPER VS CODE (Matching #E1EDEB background)
# ================================================================
def make_paper_vs_code():
    fig, ax = plt.subplots(figsize=(8.8, 2.85), dpi=300)
    fig.patch.set_facecolor(COLOR_MINT)
    ax.set_facecolor(COLOR_MINT)
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 2.85)
    ax.axis('off')

    # Card 1: Paper snippet (Left)
    paper_box = patches.FancyBboxPatch((0.05, 0.1), 3.8, 2.65, boxstyle="round,pad=0.06,rounding_size=0.1",
                                      facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.5, zorder=1)
    ax.add_patch(paper_box)

    ax.text(0.2, 2.45, "Accepted Paper: NeurIPS 2020", fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(0.2, 2.18, "Mirzadeh et al. • §B.2 Continual Learning", fontsize=9.5, color='#64748B')
    
    # Text lines
    ax.add_patch(patches.Rectangle((0.2, 1.95), 3.4, 0.06, facecolor='#E2E8F0', edgecolor='none'))
    ax.add_patch(patches.Rectangle((0.2, 1.80), 3.1, 0.06, facecolor='#E2E8F0', edgecolor='none'))

    # Quote highlight box
    qbox = patches.FancyBboxPatch((0.15, 0.75), 3.55, 0.90, boxstyle="round,pad=0.06",
                                  facecolor='#FEF3C7', edgecolor='#F59E0B', linewidth=1.2)
    ax.add_patch(qbox)
    ax.text(0.25, 1.28, '"...we have not employed any', fontsize=10.5, style='italic', color='#1E293B')
    ax.text(0.25, 0.95, 'data augmentation or ', fontsize=10.5, style='italic', color='#1E293B')
    ax.text(2.25, 0.95, 'shuffling."', fontsize=11.5, style='italic', fontweight='bold', color='#B45309')

    ax.text(0.2, 0.35, "Paper Text: Explicit claim in Section B.2", fontsize=9.5, fontweight='bold', color='#47247B')

    # Center != Badge
    badge = patches.Circle((4.4, 1.42), 0.40, facecolor='#DC2626', edgecolor='#991B1B', linewidth=2.0, zorder=3)
    ax.add_patch(badge)
    ax.text(4.4, 1.38, "≠", fontsize=24, fontweight='bold', color='#FFFFFF', ha='center', va='center', zorder=4)
    ax.text(4.4, 0.80, "CONTRADICTION", fontsize=8.0, fontweight='bold', color='#DC2626', ha='center', va='center')

    # Card 2: Code Editor snippet (Right)
    code_box = patches.FancyBboxPatch((4.95, 0.1), 3.8, 2.65, boxstyle="round,pad=0.06,rounding_size=0.1",
                                     facecolor='#1E2329', edgecolor='#334155', linewidth=1.5, zorder=1)
    ax.add_patch(code_box)

    # Window dots
    ax.add_patch(patches.Circle((5.15, 2.48), 0.06, facecolor='#EF4444'))
    ax.add_patch(patches.Circle((5.35, 2.48), 0.06, facecolor='#F59E0B'))
    ax.add_patch(patches.Circle((5.55, 2.48), 0.06, facecolor='#10B981'))
    ax.text(5.75, 2.43, "stable_sgd/data_utils.py", fontsize=9.5, family='monospace', color='#94A3B8')

    # Code lines
    ax.text(5.1, 1.95, "25  mnist_train = MNIST(...)", fontsize=9.5, family='monospace', color='#94A3B8')
    ax.text(5.1, 1.65, "26  train_loader = DataLoader(", fontsize=9.5, family='monospace', color='#E2E8F0')
    ax.text(5.1, 1.35, "27      mnist_train, batch_size=64,", fontsize=9.5, family='monospace', color='#E2E8F0')

    # Highlighted line
    hl = patches.Rectangle((5.05, 0.95), 3.55, 0.32, facecolor='#7F1D1D', edgecolor='#EF4444', linewidth=1.0)
    ax.add_patch(hl)
    ax.text(5.1, 1.03, "28      shuffle=True)", fontsize=10.5, family='monospace', fontweight='bold', color='#FCA5A5')

    ax.text(5.1, 0.35, "Code Repo: Configured with shuffle=True", fontsize=9.5, fontweight='bold', color='#38BDF8')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('Final_Report/poster/assets/paper_vs_code_template.png', facecolor=COLOR_MINT, edgecolor='none', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print("Saved paper_vs_code_template.png")

# ================================================================
# 2. SYSTEM PIPELINE (Matching #E1EDEB background)
# ================================================================
def make_system_pipeline():
    fig, ax = plt.subplots(figsize=(8.8, 4.3), dpi=300)
    fig.patch.set_facecolor(COLOR_MINT)
    ax.set_facecolor(COLOR_MINT)
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 4.3)
    ax.axis('off')

    # Stage 1: Ingestion
    ax.text(0.95, 4.08, "1. INGESTION", fontsize=10.5, fontweight='bold', color=PURPLE, ha='center')
    b_paper = patches.FancyBboxPatch((0.05, 2.60), 1.75, 1.30, boxstyle="round,pad=0.05", fc='#FFFFFF', ec='#47247B', lw=1.3)
    ax.add_patch(b_paper)
    ax.text(0.92, 3.52, "Paper PDF / LaTeX", fontsize=9.5, fontweight='bold', color='#14213D', ha='center')
    ax.text(0.92, 3.12, "Section & Table Extractor", fontsize=8.5, color='#475569', ha='center')
    ax.text(0.92, 2.78, "Extracts Claims & Settings", fontsize=8.0, fontweight='bold', color='#7030A0', ha='center')

    b_code = patches.FancyBboxPatch((0.05, 1.10), 1.75, 1.30, boxstyle="round,pad=0.05", fc='#FFFFFF', ec='#47247B', lw=1.3)
    ax.add_patch(b_code)
    ax.text(0.92, 2.02, "GitHub Repo", fontsize=9.5, fontweight='bold', color='#14213D', ha='center')
    ax.text(0.92, 1.62, "AST & Call Graph", fontsize=8.5, color='#475569', ha='center')
    ax.text(0.92, 1.28, "Static regex & configs", fontsize=8.0, fontweight='bold', color='#7030A0', ha='center')

    # Stage 2: Parallel Specialists
    ax.text(3.45, 4.08, "2. 4 SPECIALISTS", fontsize=10.5, fontweight='bold', color=PURPLE, ha='center')
    spec_container = patches.FancyBboxPatch((2.05, 0.08), 2.8, 3.82, boxstyle="round,pad=0.06", fc='#FFFFFF', ec='#94A3B8', lw=1.1, ls='--')
    ax.add_patch(spec_container)

    specs = [
        ("Citations (HALLMARK)", "831 refs • Fake papers, year, venue", "#1E40AF"),
        ("Code & Setting Checker", "Static AST • lr, batch, optimizer, seed", "#047857"),
        ("Theory & Proofs (SPOT)", "13 papers • Step-by-step logic check", "#6D28D9"),
        ("Data & Integrity", "GRIM test, table numbers, placeholders", "#B45309")
    ]
    for i, (title, desc, tc) in enumerate(specs):
        y_pos = 2.96 - i * 0.91
        sb = patches.FancyBboxPatch((2.15, y_pos), 2.6, 0.80, boxstyle="round,pad=0.05", fc='#F8FAFC', ec=tc, lw=1.2)
        ax.add_patch(sb)
        ax.text(3.45, y_pos + 0.50, title, fontsize=9.0, fontweight='bold', color=tc, ha='center')
        ax.text(3.45, y_pos + 0.20, desc, fontsize=7.8, color='#334155', ha='center')

    # Stage 3: Verifier Gate
    ax.text(5.95, 4.08, "3. VERIFIER GATE", fontsize=10.5, fontweight='bold', color=PURPLE, ha='center')
    b_ver = patches.FancyBboxPatch((5.10, 0.75), 1.7, 3.15, boxstyle="round,pad=0.08", fc='#F5F3FF', ec=PURPLE_LINE, lw=1.8)
    ax.add_patch(b_ver)
    ax.text(5.95, 3.50, "Evidence Verifier", fontsize=10.5, fontweight='bold', color=PURPLE, ha='center')
    ax.text(5.95, 3.15, "Quote Substring Match", fontsize=9.0, fontweight='bold', color='#1E293B', ha='center')
    ax.text(5.95, 2.30, "• Dual verbatim quotes\n• Drops spurious flags\n• 35% workload cut\n• 0 False Alarms!",
            fontsize=8.2, color='#334155', ha='center')
    ax.text(5.95, 1.10, "100% Precision\non Clean Controls", fontsize=8.8, fontweight='bold', color='#047857', ha='center')

    # Stage 4: Outcomes
    ax.text(7.90, 4.08, "4. AUDIT REPORT", fontsize=10.5, fontweight='bold', color=PURPLE, ha='center')
    
    b_flag = patches.FancyBboxPatch((7.05, 2.70), 1.65, 1.20, boxstyle="round,pad=0.05", fc='#ECFDF5', ec='#10B981', lw=1.4)
    ax.add_patch(b_flag)
    ax.text(7.87, 3.50, "Verified Flag", fontsize=9.5, fontweight='bold', color='#047857', ha='center')
    ax.text(7.87, 3.05, "Dual quotes verified\nHuman actionable", fontsize=8.0, color='#065F46', ha='center')

    b_rev = patches.FancyBboxPatch((7.05, 1.40), 1.65, 1.20, boxstyle="round,pad=0.05", fc='#FFFBEB', ec='#F59E0B', lw=1.4)
    ax.add_patch(b_rev)
    ax.text(7.87, 2.20, "Needs Review", fontsize=9.5, fontweight='bold', color='#B45309', ha='center')
    ax.text(7.87, 1.75, "Paywall / missing code\nFlagged for author", fontsize=8.0, color='#92400E', ha='center')

    b_drop = patches.FancyBboxPatch((7.05, 0.10), 1.65, 1.20, boxstyle="round,pad=0.05", fc='#F1F5F9', ec='#94A3B8', lw=1.4)
    ax.add_patch(b_drop)
    ax.text(7.87, 0.90, "Dropped Flag", fontsize=9.5, fontweight='bold', color='#475569', ha='center')
    ax.text(7.87, 0.45, "Unverifiable claim\nZero false alarms", fontsize=8.0, color='#64748B', ha='center')

    # Arrows
    akw = dict(arrowstyle="-|>", color='#64748B', lw=1.5, mutation_scale=13)
    ax.annotate("", xy=(2.05, 3.05), xytext=(1.8, 3.25), arrowprops=akw)
    ax.annotate("", xy=(2.05, 1.75), xytext=(1.8, 1.75), arrowprops=akw)
    ax.annotate("", xy=(5.10, 2.35), xytext=(4.85, 2.35), arrowprops=akw)
    ax.annotate("", xy=(7.05, 3.30), xytext=(6.80, 2.75), arrowprops=akw)
    ax.annotate("", xy=(7.05, 2.00), xytext=(6.80, 2.35), arrowprops=akw)
    ax.annotate("", xy=(7.05, 0.70), xytext=(6.80, 1.95), arrowprops=akw)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('Final_Report/poster/assets/system_pipeline_template.png', facecolor=COLOR_MINT, edgecolor='none', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print("Saved system_pipeline_template.png")

# ================================================================
# 3. CHART F1 (Matching #EFEAE4 background)
# ================================================================
def make_chart_f1():
    fig, ax = plt.subplots(figsize=(8.8, 3.1), dpi=300)
    fig.patch.set_facecolor(COLOR_BEIGE)
    ax.set_facecolor(COLOR_BEIGE)

    categories = ['Method', 'Metric', 'Setting', 'Macro-F1']
    x = np.arange(len(categories))
    width = 0.20

    full = [0.34, 0.80, 0.77, 0.636]
    nover = [0.34, 0.80, 0.86, 0.667]
    b0 = [0.00, 0.00, 0.14, 0.048]
    b1 = [0.00, 0.00, 0.14, 0.046]

    r1 = ax.bar(x - 1.5*width, full, width, label='Full System (Ours)', color='#47247B', edgecolor='none', zorder=3)
    r2 = ax.bar(x - 0.5*width, nover, width, label='No Verifier', color='#8A65B8', edgecolor='none', zorder=3)
    r3 = ax.bar(x + 0.5*width, b0, width, label='B0: Zero-Shot', color='#D97706', edgecolor='none', zorder=3)
    r4 = ax.bar(x + 1.5*width, b1, width, label='B1: ReAct Agent', color='#DC2626', edgecolor='none', zorder=3)

    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            h = rect.get_height()
            if h > 0.02:
                lbl = f'{h:.2f}' if h != 0.636 else '0.636'
                ax.annotate(lbl,
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#14213D')
            elif h == 0.00:
                ax.annotate('0.00',
                            xy=(rect.get_x() + rect.get_width() / 2, 0),
                            xytext=(0, 2), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8.5, color='#94A3B8')

    ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold', color='#14213D')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold', color='#14213D')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.55, color='#CBD5E1', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#94A3B8')
    ax.spines['bottom'].set_color('#94A3B8')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False, fontsize=9.5)

    plt.tight_layout()
    plt.savefig('Final_Report/poster/assets/chart_f1_template.png', facecolor=COLOR_BEIGE, edgecolor='none', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print("Saved chart_f1_template.png")

# ================================================================
# 4. CHART EACH CHECK (Matching #E1EDEB background)
# ================================================================
def make_chart_each_check():
    fig, ax = plt.subplots(figsize=(8.8, 3.15), dpi=300)
    fig.patch.set_facecolor(COLOR_MINT)
    ax.set_facecolor(COLOR_MINT)

    checks = [
        'Number vs Table',
        'Wrong Gain Claim',
        'SPOT Proof Step',
        'HALLMARK Fake Ref',
        'Reversed Comparison',
        'Impossible Mean (GRIM)',
        'Placeholder Text',
        'Setting vs Code'
    ]
    scores = [20, 55, 57, 88, 94, 100, 100, 100]
    colors = ['#64748B', '#64748B', '#7030A0', '#7030A0', '#47247B', '#047857', '#047857', '#047857']

    y = np.arange(len(checks))
    bars = ax.barh(y, scores, height=0.70, color=colors, edgecolor='none', zorder=3)

    for bar, val in zip(bars, scores):
        ax.annotate(f'{val}%',
                    xy=(val, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=9.5, fontweight='bold', color='#14213D')

    ax.set_yticks(y)
    ax.set_yticklabels(checks, fontsize=10, fontweight='bold', color='#14213D')
    ax.set_xlabel('Planted Errors Detected (%)', fontsize=10.5, fontweight='bold', color='#14213D')
    ax.set_xlim(0, 115)
    ax.grid(axis='x', linestyle='--', alpha=0.55, color='#CBD5E1', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#94A3B8')
    ax.spines['bottom'].set_color('#94A3B8')

    plt.tight_layout()
    plt.savefig('Final_Report/poster/assets/chart_each_check_template.png', facecolor=COLOR_MINT, edgecolor='none', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print("Saved chart_each_check_template.png")

if __name__ == "__main__":
    make_paper_vs_code()
    make_system_pipeline()
    make_chart_f1()
    make_chart_each_check()
    print("All template-matched assets successfully generated.")
