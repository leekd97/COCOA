#!/usr/bin/env python3
"""
Logprob Gap Visualization

Creates two figures from logprob gap analysis results:
  Figure A: Context-Awareness bar chart (all cultures)
  Figure B: Gap distribution violin plot (selected cultures)

Usage:
    python analysis/plot_logprob_gap.py \
        --input analysis/logprob_gap \
        --output analysis/figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"


def load_results(input_dir):
    """Load all per-culture gap results."""
    input_dir = Path(input_dir)
    results = {}

    for f in sorted(input_dir.glob("gap_*.json")):
        with open(f) as fh:
            d = json.load(fh)
        key = (d["culture"], d["model"])
        results[key] = d

    return results


# ============================================================================
# Figure A: Context-Awareness Bar Chart
# ============================================================================

def plot_context_awareness(results, output_path):
    """
    Grouped bar chart: Before vs After Context-Awareness per culture.
    CA = mean grounded gap - mean neutral gap
    """
    # Sort by culture order
    order = ["ko", "ja", "zh", "hi", "mr", "ml", "gu", "vi", "ur", "ar"]
    items = sorted(results.items(), key=lambda x: order.index(x[0][0]) if x[0][0] in order else 99)

    cultures = []
    before_ca = []
    after_ca = []

    for (culture, model), r in items:
        ca = r["context_awareness"]
        label = f"{culture.upper()}"
        cultures.append(label)
        before_ca.append(ca["before"])
        after_ca.append(ca["after"])

    x = np.arange(len(cultures))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(cultures) * 1.2), 5))

    bars1 = ax.bar(x - width/2, before_ca, width, label="Before COCOA",
                   color="#94a3b8", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, after_ca, width, label="After COCOA",
                   color="#3b82f6", edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#64748b")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#1e40af")

    ax.set_ylabel("Context-Awareness\n(Grounded Gap − Neutral Gap)", fontsize=11)
    ax.set_title("Context-Awareness Before vs After COCOA", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cultures, fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Figure B: Violin Plot (Gap Distribution)
# ============================================================================

def plot_gap_distribution(results, output_path, cultures_to_plot=None):
    """
    Violin plot showing gap distributions for selected cultures.
    2×2 grid: Before/After × Grounded/Neutral
    """
    if cultures_to_plot is None:
        # Pick top 2 by score
        by_score = sorted(results.items(), key=lambda x: x[1]["score"])
        cultures_to_plot = [k[0] for k, _ in by_score[:2]]

    n_cultures = len(cultures_to_plot)
    fig, axes = plt.subplots(1, n_cultures, figsize=(6 * n_cultures, 5), squeeze=False)

    for idx, culture in enumerate(cultures_to_plot):
        # Find matching result
        match = None
        for (c, m), r in results.items():
            if c == culture:
                match = r
                break
        if match is None:
            continue

        ax = axes[0, idx]

        # Extract gap values
        data_groups = []
        labels = []
        colors = []

        for phase, phase_label in [("before", "Before"), ("after", "After")]:
            for ctx_type in ["grounded", "neutral"]:
                gaps = [g["gap"] for g in match[phase]["gaps"] if g["context_type"] == ctx_type]
                if gaps:
                    data_groups.append(gaps)
                    labels.append(f"{phase_label}\n{ctx_type.capitalize()}")
                    if phase == "before":
                        colors.append("#94a3b8" if ctx_type == "grounded" else "#cbd5e1")
                    else:
                        colors.append("#3b82f6" if ctx_type == "grounded" else "#93c5fd")

        if not data_groups:
            continue

        positions = list(range(len(data_groups)))
        parts = ax.violinplot(data_groups, positions=positions, showmeans=True, showmedians=True)

        # Color violins
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Logprob Gap\nlog P(Asian) − log P(Western)", fontsize=10)
        ax.set_title(f"{culture.upper()} ({match['model']})", fontsize=12, fontweight="bold")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="No preference")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add mean annotations
        for i, grp in enumerate(data_groups):
            mean_val = np.mean(grp)
            ax.text(i, ax.get_ylim()[1] * 0.95, f"μ={mean_val:+.1f}",
                    ha="center", fontsize=8, color="black")

    plt.suptitle("Logprob Gap Distribution: Before vs After COCOA", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Figure C: Grounded vs Neutral Scatter (Before/After)
# ============================================================================

def plot_scatter(results, output_path, culture="ko"):
    """
    Scatter plot: x=grounded gap, y=neutral gap per example pair.
    Before (gray) vs After (blue).
    Ideal after: x >> 0, y ≈ 0 (bottom-right quadrant).
    """
    match = None
    for (c, m), r in results.items():
        if c == culture:
            match = r
            break
    if match is None:
        print(f"No data for {culture}")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # Aggregate by entity_type: mean gap per context_type
    for phase, color, marker, label in [
        ("before", "#94a3b8", "o", "Before COCOA"),
        ("after", "#3b82f6", "^", "After COCOA"),
    ]:
        grounded_gaps = [g["gap"] for g in match[phase]["gaps"] if g["context_type"] == "grounded"]
        neutral_gaps = [g["gap"] for g in match[phase]["gaps"] if g["context_type"] == "neutral"]

        # Trim to same length for pairing
        n = min(len(grounded_gaps), len(neutral_gaps))
        if n == 0:
            continue

        ax.scatter(grounded_gaps[:n], neutral_gaps[:n],
                   c=color, marker=marker, alpha=0.4, s=30, label=label, edgecolors="none")

        # Add mean point
        gm = np.mean(grounded_gaps[:n])
        nm = np.mean(neutral_gaps[:n])
        ax.scatter([gm], [nm], c=color, marker=marker, s=200,
                   edgecolors="black", linewidths=1.5, zorder=10)
        ax.annotate(f"μ=({gm:.1f}, {nm:.1f})", (gm, nm),
                    textcoords="offset points", xytext=(10, 10), fontsize=9)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Grounded Context Gap\nlog P(Asian) − log P(Western)", fontsize=11)
    ax.set_ylabel("Neutral Context Gap\nlog P(Asian) − log P(Western)", fontsize=11)
    ax.set_title(f"{culture.upper()} — Context-Conditional Behavior Shift", fontsize=13, fontweight="bold")

    # Annotate quadrants
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1]*0.7, ylim[0]*0.8, "✓ Ideal\n(Asian in grounded,\nneutral in neutral)",
            fontsize=8, color="green", ha="center", alpha=0.7)

    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="analysis/logprob_gap",
                   help="Directory with gap_*.json files")
    p.add_argument("--output", default="analysis/figures",
                   help="Output directory for figures")
    p.add_argument("--violin_cultures", nargs="*", default=None,
                   help="Cultures for violin plot (default: top 2)")
    p.add_argument("--scatter_culture", default="ko",
                   help="Culture for scatter plot")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(args.input)
    print(f"  Found {len(results)} culture×model results")

    if not results:
        print("No results found!")
        return

    # Figure A: Context-Awareness bar chart
    print("\nFigure A: Context-Awareness bar chart...")
    plot_context_awareness(results, output_dir / "context_awareness_bar.png")

    # Figure B: Violin plot
    print("\nFigure B: Gap distribution violin plot...")
    plot_gap_distribution(results, output_dir / "gap_distribution_violin.png",
                          cultures_to_plot=args.violin_cultures)

    # Figure C: Scatter plot
    print("\nFigure C: Scatter plot...")
    plot_scatter(results, output_dir / "gap_scatter.png",
                 culture=args.scatter_culture)

    print(f"\nAll figures saved in: {output_dir}/")


if __name__ == "__main__":
    main()