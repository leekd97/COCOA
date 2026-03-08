#!/usr/bin/env python3
"""
Paired Logprob Gap — Figure Generation (v2)

Creates publication-quality figures from paired analysis results.
Generates per-model figures with model name in filenames.

Usage:
    # Llama only (main paper)
    python analysis/plot_paired_gap.py --input analysis/logprob_gap_paired --model llama

    # Qwen only (appendix)
    python analysis/plot_paired_gap.py --input analysis/logprob_gap_paired --model qwen

    # Both (generates separate files for each)
    python analysis/plot_paired_gap.py --input analysis/logprob_gap_paired
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False


CULTURE_ORDER = ["ko", "ja", "zh", "ar", "hi", "mr", "ml", "gu", "ur", "vi"]
CULTURE_LABELS = {
    "ko": "KO", "ja": "JA", "zh": "ZH", "ar": "AR", "hi": "HI",
    "mr": "MR", "ml": "ML", "gu": "GU", "ur": "UR", "vi": "VI",
}

MODEL_DISPLAY = {"llama": "Llama-3.1-8B", "qwen": "Qwen3-8B"}


def load_paired_results(input_dir, model_filter=None):
    """Load paired_*.json results, optionally filtering by model."""
    input_dir = Path(input_dir)
    results = {}

    for f in sorted(input_dir.glob("paired_*.json")):
        if "summary" in f.name:
            continue
        with open(f) as fh:
            d = json.load(fh)
        
        file_model = d.get("model", "")
        if model_filter and file_model != model_filter:
            continue

        key = d["culture"]
        results[key] = d

    return results


def plot_context_effect(results, output_path, model_name=""):
    items = []
    for culture in CULTURE_ORDER:
        if culture not in results:
            continue
        a = results[culture]["analysis"]
        items.append({
            "culture": culture,
            "before_ce": a["before"]["mean_context_effect"],
            "after_ce": a["after"]["mean_context_effect"],
            "delta": a["delta_context_effect"],
        })

    items.sort(key=lambda x: -x["delta"])

    cultures = [CULTURE_LABELS[it["culture"]] for it in items]
    before = [it["before_ce"] for it in items]
    after = [it["after_ce"] for it in items]
    deltas = [it["delta"] for it in items]

    x = np.arange(len(cultures))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width/2, before, width,
           label="Before COCOA", color="#cbd5e1", edgecolor="#94a3b8", linewidth=0.8)
    bars2 = ax.bar(x + width/2, after, width,
                   label="After COCOA", color="#3b82f6", edgecolor="#1e40af", linewidth=0.8)

    for i, (bar, delta) in enumerate(zip(bars2, deltas)):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"+{delta:.1f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#1e40af")

    display = MODEL_DISPLAY.get(model_name, model_name)
    ax.set_ylabel("Context Effect (CE)\nGrounded Gap − Neutral Gap", fontsize=12)
    ax.set_xlabel("Culture", fontsize=12)
    ax.set_title(f"Context-Awareness Before vs After COCOA ({display})", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cultures, fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_gap_breakdown(results, output_path, model_name=""):
    items = []
    for culture in CULTURE_ORDER:
        if culture not in results:
            continue
        a = results[culture]["analysis"]
        items.append({
            "culture": culture,
            "before_g": a["before"]["mean_grounded_gap"],
            "after_g": a["after"]["mean_grounded_gap"],
            "before_n": a["before"]["mean_neutral_gap"],
            "after_n": a["after"]["mean_neutral_gap"],
            "delta_ce": a["delta_context_effect"],
        })

    items.sort(key=lambda x: -x["delta_ce"])

    cultures = [CULTURE_LABELS[it["culture"]] for it in items]
    x = np.arange(len(cultures))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    display = MODEL_DISPLAY.get(model_name, model_name)

    # Left: Grounded Gap
    before_g = [it["before_g"] for it in items]
    after_g = [it["after_g"] for it in items]

    ax1.bar(x - width/2, before_g, width,
            label="Before", color="#cbd5e1", edgecolor="#94a3b8", linewidth=0.8)
    ax1.bar(x + width/2, after_g, width,
            label="After", color="#22c55e", edgecolor="#15803d", linewidth=0.8)

    for i, (bg, ag) in enumerate(zip(before_g, after_g)):
        delta = ag - bg
        y_pos = max(bg, ag) + 0.2
        color = "#15803d" if delta >= 0 else "#dc2626"
        ax1.text(i, y_pos, f"{delta:+.1f}", ha="center", fontsize=7, color=color, fontweight="bold")

    ax1.set_ylabel("log P(Asian) − log P(Western)", fontsize=11)
    ax1.set_title("Grounded Context\n(should increase ↑)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cultures, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Right: Neutral Gap
    before_n = [it["before_n"] for it in items]
    after_n = [it["after_n"] for it in items]

    ax2.bar(x - width/2, before_n, width,
            label="Before", color="#cbd5e1", edgecolor="#94a3b8", linewidth=0.8)
    ax2.bar(x + width/2, after_n, width,
            label="After", color="#f97316", edgecolor="#c2410c", linewidth=0.8)

    for i, (bn, an) in enumerate(zip(before_n, after_n)):
        delta = an - bn
        y_pos = max(bn, an) + 0.2
        ax2.text(i, y_pos, f"{delta:+.1f}", ha="center", fontsize=7, color="#c2410c", fontweight="bold")

    ax2.set_title("Neutral Context\n(should approach 0 →)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cultures, fontsize=10)
    ax2.legend(fontsize=9)
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.suptitle(f"Logprob Gap Breakdown ({display})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_effect_size(results, output_path, model_name=""):
    items = []
    for culture in CULTURE_ORDER:
        if culture not in results:
            continue
        a = results[culture]["analysis"]
        items.append({"culture": culture, "d": a["cohens_d"]})

    items.sort(key=lambda x: x["d"])

    cultures = [CULTURE_LABELS[it["culture"]] for it in items]
    d_vals = [it["d"] for it in items]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#3b82f6" if d >= 0.8 else "#93c5fd" if d >= 0.5 else "#cbd5e1" for d in d_vals]
    bars = ax.barh(cultures, d_vals, color=colors, edgecolor="white", linewidth=0.5)

    ax.axvline(x=0.5, color="orange", linestyle="--", linewidth=0.8, alpha=0.7, label="Medium (0.5)")
    ax.axvline(x=0.8, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label="Large (0.8)")

    for bar, d in zip(bars, d_vals):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                f"{d:.2f}", ha="left", va="center", fontsize=9, fontweight="bold")

    display = MODEL_DISPLAY.get(model_name, model_name)
    ax.set_xlabel("Cohen's d (effect size)", fontsize=11)
    ax.set_title(f"Effect Size of COCOA on Context-Awareness ({display})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_figures(input_dir, output_dir, model_name):
    print(f"\n{'='*50}")
    print(f"Generating figures for: {MODEL_DISPLAY.get(model_name, model_name)}")
    print(f"{'='*50}")

    results = load_paired_results(input_dir, model_filter=model_name)
    print(f"  Found {len(results)} cultures")

    if not results:
        print("  No results, skipping.")
        return

    tag = f"_{model_name}"

    print("\n  Figure 1: Context Effect...")
    plot_context_effect(results, output_dir / f"fig_context_effect{tag}.png", model_name)
    plot_context_effect(results, output_dir / f"fig_context_effect{tag}.pdf", model_name)

    print("  Figure 2: Gap breakdown...")
    plot_gap_breakdown(results, output_dir / f"fig_gap_breakdown{tag}.png", model_name)
    plot_gap_breakdown(results, output_dir / f"fig_gap_breakdown{tag}.pdf", model_name)

    print("  Figure 3: Effect size...")
    plot_effect_size(results, output_dir / f"fig_effect_size{tag}.png", model_name)
    plot_effect_size(results, output_dir / f"fig_effect_size{tag}.pdf", model_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="analysis/logprob_gap_paired")
    p.add_argument("--output", default="analysis/figures")
    p.add_argument("--model", default=None,
                   help="Filter by model: llama, qwen. Default: both.")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        generate_figures(args.input, output_dir, args.model)
    else:
        for m in ["llama", "qwen"]:
            generate_figures(args.input, output_dir, m)

    print(f"\nAll figures saved in: {output_dir}/")


if __name__ == "__main__":
    main()