#!/usr/bin/env python3
"""Generate bold, high-density, zero-padding visual assets for the balanced 3-column poster.
Optimized for readability from 3-5 feet away on an A1 poster.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# ================================================================
# 1. PAPER VS CODE (Bold, High Contrast, Tight Framing)
# ================================================================
def make_paper_vs_code():
    fig, ax = plt.subplots(figsize=(8.8, 2.9), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 2.9)
    ax.axis('off')

    # Card 1: Paper snippet (Left)
    paper_box = patches.FancyBboxPatch((0.05, 0.1), 3.8, 2.65, boxstyle="round,pad=0.06,rounding_size=0.1",
                                      facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.5, zorder=1)
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

    ax.text(0.2, 0.35, "Paper Text: Explicit claim in Section B.2", fontsize=9.5, fontweight='bold', color='#0B5CAD')

    # Center != Badge
    badge = patches.Circle((4.4, 1.45), 0.42, facecolor='#EF4444', edgecolor='#B91C1C', linewidth=2.0, zorder=3)
    ax.add_patch(badge)
    ax.text(4.4, 1.41, "≠", fontsize=24, fontweight='bold', color='#FFFFFF', ha='center', va='center', zorder=4)
    ax.text(4.4, 0.82, "CONTRADICTION", fontsize=8.0, fontweight='bold', color='#DC2626', ha='center', va='center')

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

    ax.text(5.1, 0.35, "Code Repo: Training configured with shuffle=True", fontsize=9.5, fontweight='bold', color='#38BDF8')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('Final_Report/poster/assets/paper_vs_code_tight.png', facecolor='#FFFFFF', edgecolor='none', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print("Saved paper_vs_code_tight.png")

# ================================================================
# 2. SYSTEM PIPELINE (Bold, Clear, Zero-Air Architecture)
# ================================================================
def make_system_pipeline():
    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 4.4)
    ax.axis('off')

    # Stage 1: Ingestion
    ax.text(0.95, 4.15, "1. INGEST", fontsize=11, fontweight='bold', color='#0F2C59', ha='center')
    b_paper = patches.FancyBboxPatch((0.05, 2.65), 1.75, 1.30, boxstyle="round,pad=0.05", fc='#F8FAFC', ec='#0B5CAD', lw=1.3)
    ax.add_patch(b_paper)
    ax.text(0.92, 3.55, "Paper PDF / LaTeX", fontsize=9.5, fontweight='bold', color='#14213D', ha='center')
    ax.text(0.92, 3.15, "Section & Table Extractor", fontsize=8.5, color='#475569', ha='center')
    ax.text(0.92, 2.80, "Extracts Claims & Settings", fontsize=8.0, fontweight='bold', color='#0B5CAD', ha='center')

    b_code = patches.FancyBboxPatch((0.05, 1.15), 1.75, 1.30, boxstyle="round,pad=0.05", fc='#F8FAFC', ec='#0B5CAD', lw=1.3)
    ax.add_patch(b_code)
    ax.text(0.92, 2.05, "GitHub Repo", fontsize=9.5, fontweight='bold', color='#14213D', ha='center')
    ax.text(0.92, 1.65, "AST & Call Graph", fontsize=8.5, color='#475569', ha='center')
    ax.text(0.92, 1.30, "Static regex & configs", fontsize=8.0, fontweight='bold', color='#0B5CAD', ha='center')

    # Stage 2: Parallel Specialists
    ax.text(3.45, 4.15, "2. PARALLEL SPECIALISTS", fontsize=11, fontweight='bold', color='#0F2C59', ha='center')
    spec_container = patches.FancyBboxPatch((2.05, 0.1), 2.8, 3.85, boxstyle="round,pad=0.06", fc='#F1F5F9', ec='#94A3B8', lw=1.1, ls='--')
    ax.add_patch(spec_container)

    specs = [
        ("Citations (HALLMARK)", "831 refs • Fake papers, year, venue", "#1E40AF"),
        ("Code & Setting Checker", "Static AST • lr, batch, optimizer, seed", "#047857"),
        ("Theory & Proofs (SPOT)", "13 papers • Step-by-step logic check", "#6D28D9"),
        ("Data & Integrity", "GRIM test, table numbers, placeholders", "#B45309")
    ]
    for i, (title, desc, tc) in enumerate(specs):
        y_pos = 3.0 - i * 0.92
        sb = patches.FancyBboxPatch((2.15, y_pos), 2.6, 0.80, boxstyle="round,pad=0.05", fc='#FFFFFF', ec=tc, lw=1.2)
        ax.add_patch(sb)
        ax.text(3.45, y_pos + 0.50, title, fontsize=9.0, fontweight='bold', color=tc, ha='center')
        ax.text(3.45, y_pos + 0.20, desc, fontsize=7.8, color='#334155', ha='center')

    # Stage 3: Verifier Gate
    ax.text(5.95, 4.15, "3. VERIFIER GATE", fontsize=11, fontweight='bold', color='#0F2C59', ha='center')
    b_ver = patches.FancyBboxPatch((5.10, 0.8), 1.7, 3.15, boxstyle="round,pad=0.08", fc='#EFF6FF', ec='#1D4ED8', lw=1.8)
    ax.add_patch(b_ver)
    ax.text(5.95, 3.55, "Evidence Verifier", fontsize=10.5, fontweight='bold', color='#1D4ED8', ha='center')
    ax.text(5.95, 3.20, "Quote Substring Match", fontsize=9.0, fontweight='bold', color='#1E293B', ha='center')
    ax.text(5.95, 2.35, "• Dual verbatim quotes\n• Drops spurious flags\n• 35% workload cut\n• 0 False Alarms!",
            fontsize=8.2, color='#334155', ha='center')
    ax.text(5.95, 1.15, "100% Precision\non Clean Controls", fontsize=8.8, fontweight='bold', color='#047857', ha='center')

    # Stage 4: Outcomes
    ax.text(7.90, 4.15, "4. REPORT", fontsize=11, fontweight='bold', color='#0F2C59', ha='center')
    
    b_flag = patches.FancyBboxPatch((7.05, 2.75), 1.65, 1.20, boxstyle="round,pad=0.05", fc='#ECFDF5', ec='#10B981', lw=1.4)
    ax.add_patch(b_flag)
    ax.text(7.87, 3.55, "Verified Flag", fontsize=9.5, fontweight='bold', color='#047857', ha='center')
    ax.text(7.87, 3.10, "Dual quotes verified\nHuman actionable", fontsize=8.0, color='#065F46', ha='center')

    b_rev = patches.FancyBboxPatch((7.05, 1.45), 1.65, 1.20, boxstyle="round,pad=0.05", fc='#FFFBEB', ec='#F59E0B', lw=1.4)
    ax.add_patch(b_rev)
    ax.text(7.87, 2.25, "Needs Review", fontsize=9.5, fontweight='bold', color='#B45309', ha='center')
    ax.text(7.87, 1.80, "Paywall / missing code\nFlagged for author", fontsize=8.0, color='#92400E', ha='center')

    b_drop = patches.FancyBboxPatch((7.05, 0.15), 1.65, 1.20, boxstyle="round,pad=0.05", fc='#F1F5F9', ec='#94A3B8', lw=1.4)
    ax.add_patch(b_drop)
    ax.text(7.87, 0.95, "Dropped Flag", fontsize=9.5, fontweight='bold', color='#475569', ha='center')
    ax.text(7.87, 0.50, "Unverifiable claim\nZero false alarms", fontsize=8.0, color='#64748B', ha='center')

    # Arrows
    akw = dict(arrowstyle="-|>", color='#64748B', lw=1.5, mutation_scale=13)
    ax.annotate("", xy=(2.05, 3.1), xytext=(1.8, 3.3), arrowprops=akw)
    ax.annotate("", xy=(2.05, 1.8), xytext=(1.8, 1.8), arrowprops=akw)
    ax.annotate("", xy=(5.10, 2.4), xytext=(4.85, 2.4), arrowprops=akw)
    ax.annotate("", xy=(7.05, 3.35), xytext=(6.80, 2.8), arrowprops=akw)
    ax.annotate("", xy=(7.05, 2.05), xytext=(6.80, 2.4), arrowprops=akw)
    ax.annotate("", xy=(7.05, 0.75), xytext=(6.80, 2.0), arrowprops=akw)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('Final_Report/poster/assets/system_pipeline_tight.png', facecolor='#FFFFFF', edgecolor='none', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print("Saved system_pipeline_tight.png")

# ================================================================
# 3. CHART F1 (Bold Direct Values, Clear Bars)
# ================================================================
def make_chart_f1():
    fig, ax = plt.subplots(figsize=(8.8, 3.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    categories = ['Method', 'Metric', 'Setting', 'Macro-F1']
    x = np.arange(len(categories))
    width = 0.20

    full = [0.34, 0.80, 0.77, 0.636]
    nover = [0.34, 0.80, 0.86, 0.667]
    b0 = [0.00, 0.00, 0.14, 0.048]
    b1 = [0.00, 0.00, 0.14, 0.046]

    r1 = ax.bar(x - 1.5*width, full, width, label='Full System (Ours)', color='#1F6FB2', edgecolor='none', zorder=3)
    r2 = ax.bar(x - 0.5*width, nover, width, label='No Verifier', color='#8DBBE3', edgecolor='none', zorder=3)
    r3 = ax.bar(x + 0.5*width, b0, width, label='B0: Zero-Shot', color='#E08A2E', edgecolor='none', zorder=3)
    r4 = ax.bar(x + 1.5*width, b1, width, label='B1: ReAct Agent', color='#B5443A', edgecolor='none', zorder=3)

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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=4, frameon=False, fontsize=9.5)

    plt.tight_layout()
    plt.savefig('Final_Report/poster/assets/chart_f1_tight.png', facecolor='#FFFFFF', edgecolor='none', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print("Saved chart_f1_tight.png")

# ================================================================
# 4. CHART EACH CHECK (Horizontal Bars with Direct Numbers)
# ================================================================
def make_chart_each_check():
    fig, ax = plt.subplots(figsize=(8.8, 3.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

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
    colors = ['#64748B', '#64748B', '#3B82F6', '#3B82F6', '#1F6FB2', '#2E7D4F', '#2E7D4F', '#2E7D4F']

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
    plt.savefig('Final_Report/poster/assets/chart_each_check_tight.png', facecolor='#FFFFFF', edgecolor='none', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print("Saved chart_each_check_tight.png")

if __name__ == "__main__":
    make_paper_vs_code()
    make_system_pipeline()
    make_chart_f1()
    make_chart_each_check()
    print("All bold tight assets successfully generated.")
