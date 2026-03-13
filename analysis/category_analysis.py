#!/usr/bin/env python3
"""
Category-Level Analysis

Analyzes COCOA performance broken down by entity category
(authors, beverage, food, locations, names-female, names-male, sports).

Two analyses:
  A) Per-culture: which categories improve most/least
  B) Cross-culture: which categories are consistently easy/hard

Usage:
    python analysis/category_analysis.py \
        --summary experiments/summary.json \
        --exp_dir experiments \
        --output analysis/categories
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

CULTURES_ORDER = ["ko", "ja", "zh", "ar", "hi", "mr", "ml", "gu", "ur", "vi"]
CATEGORIES = ["authors", "beverage", "food", "locations", "names-female", "names-male", "sports"]
CAT_SHORT = {
    "authors": "Auth", "beverage": "Bev", "food": "Food",
    "locations": "Loc", "names-female": "Name-F", "names-male": "Name-M", "sports": "Sport",
}

MODEL_NORMALIZE = {
    "llama3_8b": "llama", "llama3-8b": "llama",
    "meta-llama/llama-3.1-8b": "llama",
    "qwen3_8b": "qwen", "qwen3-8b": "qwen",
    "qwen/qwen3-8b": "qwen",
}
MODEL_FOLDER = {"llama": "llama3-8b", "qwen": "qwen3-8b"}

BEST_SEEDS = {
    ("ar", "llama"): 45, ("ar", "qwen"): 45,
    ("gu", "llama"): 42, ("gu", "qwen"): 42,
    ("hi", "llama"): 45, ("hi", "qwen"): 45,
    ("ja", "llama"): 45, ("ja", "qwen"): 45,
    ("ko", "llama"): 45, ("ko", "qwen"): 45,
    ("ml", "llama"): 42, ("ml", "qwen"): 42,
    ("mr", "llama"): 42, ("mr", "qwen"): 42,
    ("ur", "llama"): 42, ("ur", "qwen"): 42,
    ("vi", "llama"): 48, ("vi", "qwen"): 48,
    ("zh", "llama"): 42, ("zh", "qwen"): 42,
}


def find_best_exp_name(summary_path, culture, model):
    """Find best experiment name from summary.json."""
    with open(summary_path) as f:
        data = json.load(f)

    best = None
    for e in data["experiments"]:
        if e.get("lang") == "en":
            continue
        if "_weighted_" in e.get("name", "") or "_pcgrad_" in e.get("name", ""):
            continue
        if e.get("culture", "") in ("indian", "indian_combined"):
            continue
        c = e["culture"]
        m_raw = e.get("model", "")
        m = MODEL_NORMALIZE.get(m_raw.lower(), m_raw)

        if c == culture and m == model:
            score = e["final"]["score"]
            if best is None or score < best["final"]["score"]:
                best = e

    return best["name"] if best else None


def load_category_data(exp_dir, exp_name):
    """Load baseline and final by_category from results.json."""
    rfile = exp_dir / exp_name / "results.json"
    if not rfile.exists():
        return None

    with open(rfile) as f:
        d = json.load(f)

    baseline = d.get("baseline", {}).get("by_category", {})
    final = d.get("final", {}).get("by_category", {})

    result = {}
    for cat in CATEGORIES:
        bl_g = baseline.get("grounded", {}).get("by_category", {}).get(cat)
        bl_n = baseline.get("neutral", {}).get("by_category", {}).get(cat)
        fn_g = final.get("grounded", {}).get("by_category", {}).get(cat)
        fn_n = final.get("neutral", {}).get("by_category", {}).get(cat)

        if all(v is not None for v in [bl_g, bl_n, fn_g, fn_n]):
            result[cat] = {
                "baseline_g": bl_g,
                "baseline_n": bl_n,
                "final_g": fn_g,
                "final_n": fn_n,
                "baseline_score": bl_g + abs(bl_n - 50),
                "final_score": fn_g + abs(fn_n - 50),
                "delta_g": fn_g - bl_g,
                "delta_n": fn_n - bl_n,
                "delta_score": (fn_g + abs(fn_n - 50)) - (bl_g + abs(bl_n - 50)),
            }

    return result


def print_culture_table(all_data, model):
    """Print per-culture category breakdown."""
    print(f"\n{'='*120}")
    print(f"  Category Analysis — {model.upper()} (Score = CBS_g + |CBS_n - 50|, lower is better)")
    print(f"{'='*120}")

    for culture in CULTURES_ORDER:
        cats = all_data.get((culture, model))
        if not cats:
            continue

        print(f"\n  --- {culture.upper()} ---")
        print(f"  {'Category':>12} │ {'BL_g':>6} {'BL_n':>6} {'BL_Sc':>7} │ {'Fn_g':>6} {'Fn_n':>6} {'Fn_Sc':>7} │ {'Δ_g':>7} {'Δ_n':>7} {'Δ_Sc':>7}")
        print(f"  {'─'*95}")

        for cat in CATEGORIES:
            c = cats.get(cat)
            if not c:
                continue
            print(
                f"  {CAT_SHORT.get(cat, cat):>12} │"
                f" {c['baseline_g']:6.1f} {c['baseline_n']:6.1f} {c['baseline_score']:7.1f} │"
                f" {c['final_g']:6.1f} {c['final_n']:6.1f} {c['final_score']:7.1f} │"
                f" {c['delta_g']:+7.1f} {c['delta_n']:+7.1f} {c['delta_score']:+7.1f}"
            )


def print_cross_culture_table(all_data, model):
    """Print cross-culture summary: average delta per category."""
    print(f"\n{'='*90}")
    print(f"  Cross-Culture Category Summary — {model.upper()}")
    print(f"  (Average Δ Score across cultures, negative = improvement)")
    print(f"{'='*90}")
    print(f"  {'Category':>12} │ {'Avg Δ_g':>8} {'Avg Δ_n':>8} {'Avg Δ_Sc':>9} │ {'Improved':>8} {'Total':>6} │ {'Verdict':>8}")
    print(f"  {'─'*75}")

    cat_stats = {}

    for cat in CATEGORIES:
        deltas_g = []
        deltas_n = []
        deltas_score = []
        improved = 0
        total = 0

        for culture in CULTURES_ORDER:
            cats = all_data.get((culture, model))
            if not cats or cat not in cats:
                continue
            c = cats[cat]
            deltas_g.append(c["delta_g"])
            deltas_n.append(c["delta_n"])
            deltas_score.append(c["delta_score"])
            if c["delta_score"] < 0:
                improved += 1
            total += 1

        if deltas_score:
            avg_dg = np.mean(deltas_g)
            avg_dn = np.mean(deltas_n)
            avg_ds = np.mean(deltas_score)
            verdict = "✓" if avg_ds < -5 else "△" if avg_ds < 0 else "✗"

            print(
                f"  {CAT_SHORT.get(cat, cat):>12} │"
                f" {avg_dg:>+8.1f} {avg_dn:>+8.1f} {avg_ds:>+9.1f} │"
                f" {improved:>8}/{total:<5} │ {verdict:>8}"
            )

            cat_stats[cat] = {
                "avg_delta_g": avg_dg,
                "avg_delta_n": avg_dn,
                "avg_delta_score": avg_ds,
                "improved_ratio": improved / total if total > 0 else 0,
            }

    return cat_stats


def plot_heatmap(all_data, output_path, model, metric="delta_score"):
    """Heatmap: cultures × categories, colored by improvement."""
    cultures = [c for c in CULTURES_ORDER if (c, model) in all_data]
    cats = CATEGORIES

    matrix = np.full((len(cultures), len(cats)), np.nan)

    for i, culture in enumerate(cultures):
        cat_data = all_data.get((culture, model), {})
        for j, cat in enumerate(cats):
            if cat in cat_data:
                matrix[i, j] = cat_data[cat][metric]

    fig, ax = plt.subplots(figsize=(10, 6))

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([CAT_SHORT.get(c, c) for c in cats], fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(cultures)))
    ax.set_yticklabels([c.upper() for c in cultures], fontsize=11)

    # Add value annotations
    for i in range(len(cultures)):
        for j in range(len(cats)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.0f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Δ Score (negative = improvement)", shrink=0.8)

    model_disp = "Llama-3.1-8B" if model == "llama" else "Qwen3-8B"
    ax.set_title(f"Category-Level Improvement ({model_disp})\nΔ Score = Final − Baseline (green = better)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_category_bars(cat_stats, output_path, model):
    """Bar chart: average delta score per category across cultures."""
    cats = sorted(cat_stats.keys(), key=lambda c: cat_stats[c]["avg_delta_score"])

    labels = [CAT_SHORT.get(c, c) for c in cats]
    scores = [cat_stats[c]["avg_delta_score"] for c in cats]
    ratios = [cat_stats[c]["improved_ratio"] for c in cats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: avg delta score
    colors = ["#22c55e" if s < 0 else "#ef4444" for s in scores]
    bars = ax1.barh(labels, scores, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, scores):
        x = bar.get_width()
        ax1.text(x + (0.5 if x >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                 f"{val:+.1f}", ha="left" if x >= 0 else "right", va="center", fontsize=9, fontweight="bold")

    ax1.axvline(x=0, color="gray", linewidth=0.5)
    ax1.set_xlabel("Average Δ Score (negative = improvement)", fontsize=11)
    ax1.set_title("Avg Improvement by Category", fontsize=12, fontweight="bold")

    # Right: improvement ratio
    colors2 = ["#3b82f6" for _ in ratios]
    bars2 = ax2.barh(labels, [r * 100 for r in ratios], color=colors2, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, ratios):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{val*100:.0f}%", ha="left", va="center", fontsize=9, fontweight="bold")

    ax2.set_xlabel("% Cultures Improved", fontsize=11)
    ax2.set_title("Improvement Rate by Category", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 110)

    model_disp = "Llama-3.1-8B" if model == "llama" else "Qwen3-8B"
    plt.suptitle(f"Category Analysis ({model_disp})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="experiments/summary.json")
    p.add_argument("--exp_dir", default="experiments")
    p.add_argument("--output", default="analysis/categories")
    p.add_argument("--model", default=None, help="llama, qwen, or both (default)")
    args = p.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else ["llama", "qwen"]

    for model in models:
        print(f"\n{'='*60}")
        print(f"Loading category data for {model}...")
        print(f"{'='*60}")

        all_data = {}

        for culture in CULTURES_ORDER:
            exp_name = find_best_exp_name(args.summary, culture, model)
            if not exp_name:
                continue

            cat_data = load_category_data(exp_dir, exp_name)
            if cat_data and len(cat_data) == len(CATEGORIES):
                all_data[(culture, model)] = cat_data
                print(f"  {culture}: {len(cat_data)} categories loaded")

        if not all_data:
            print(f"  No data for {model}")
            continue

        # Tables
        print_culture_table(all_data, model)
        cat_stats = print_cross_culture_table(all_data, model)

        # Figures
        tag = f"_{model}"

        print(f"\nGenerating figures...")
        plot_heatmap(all_data, output_dir / f"fig_category_heatmap{tag}.png", model)
        plot_heatmap(all_data, output_dir / f"fig_category_heatmap{tag}.pdf", model)

        if cat_stats:
            plot_category_bars(cat_stats, output_dir / f"fig_category_bars{tag}.png", model)
            plot_category_bars(cat_stats, output_dir / f"fig_category_bars{tag}.pdf", model)

        # Save JSON
        save_data = {}
        for (c, m), cats in all_data.items():
            if m == model:
                save_data[c] = cats
        with open(output_dir / f"categories_{model}.json", "w") as f:
            json.dump({"data": save_data, "cross_culture": cat_stats}, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_dir}/categories_{model}.json")

    print("\nDone!")


if __name__ == "__main__":
    main()