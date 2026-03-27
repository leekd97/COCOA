"""
CoCoA: Entity Prior Variance Visualization

Generates:
  1. Strip plot: individual entity priors per culture (Asian vs Western)
  2. Box plot: distribution summary per culture
  3. Paired extreme examples: bar chart of most extreme entity pairs

Usage:
    python analysis/visualize_prior_variance.py --model llama3_8b
    python analysis/visualize_prior_variance.py --model qwen3_8b
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'NanumGothic', 'Noto Sans CJK KR', 'sans-serif']

CULTURES = ["ko", "ja", "zh", "hi", "mr", "gu", "ml", "vi", "ur", "ar"]
CULTURE_LABELS = {
    "ko": "KO", "ja": "JA", "zh": "ZH", "hi": "HI", "mr": "MR",
    "gu": "GU", "ml": "ML", "vi": "VI", "ur": "UR", "ar": "AR",
}
CATEGORIES = ["authors", "beverage", "food", "locations", "names-female", "names-male", "sports"]


def load_raw_priors(priors_root, model, culture, lang="cu"):
    """Load raw entity priors from JSON."""
    path = Path(priors_root) / model / f"{culture}_{lang}" / "entity_priors.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_all_priors(priors_root, model, lang="cu"):
    """Collect all entity priors into flat structure."""
    all_data = {}
    for culture in CULTURES:
        data = load_raw_priors(priors_root, model, culture, lang)
        if data is None:
            continue
        asian_vals, western_vals = [], []
        asian_by_cat, western_by_cat = defaultdict(list), defaultdict(list)

        for cat, sides in data["priors"].items():
            for ent, val in sides.get("asian", {}).items():
                asian_vals.append(val)
                asian_by_cat[cat].append((ent, val))
            for ent, val in sides.get("western", {}).items():
                western_vals.append(val)
                western_by_cat[cat].append((ent, val))

        all_data[culture] = {
            "asian": asian_vals,
            "western": western_vals,
            "asian_by_cat": dict(asian_by_cat),
            "western_by_cat": dict(western_by_cat),
        }
    return all_data


# =========================================================================
# Figure 1: Strip Plot — all entities per culture
# =========================================================================
def plot_strip(all_data, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(14, 6))

    positions = []
    pos = 0
    for i, culture in enumerate(CULTURES):
        if culture not in all_data:
            continue

        asian = all_data[culture]["asian"]
        western = all_data[culture]["western"]

        # Asian dots (left of center)
        x_a = np.random.normal(pos - 0.15, 0.06, len(asian))
        ax.scatter(x_a, asian, c="#2196F3", alpha=0.3, s=8, edgecolors="none", zorder=3)

        # Western dots (right of center)
        x_w = np.random.normal(pos + 0.15, 0.06, len(western))
        ax.scatter(x_w, western, c="#F44336", alpha=0.3, s=8, edgecolors="none", zorder=3)

        # Mean markers
        ax.scatter(pos - 0.15, np.mean(asian), c="#0D47A1", s=60, marker="_", linewidths=2, zorder=5)
        ax.scatter(pos + 0.15, np.mean(western), c="#B71C1C", s=60, marker="_", linewidths=2, zorder=5)

        positions.append(pos)
        pos += 1

    ax.set_xticks(positions)
    ax.set_xticklabels([CULTURE_LABELS[c] for c in CULTURES if c in all_data], fontsize=12)
    ax.set_ylabel("log P(entity | BOS)", fontsize=13)
    ax.set_title(f"Entity Prior Distribution by Culture — {model_name}", fontsize=14)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=8, label='Asian entities'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=8, label='Western entities'),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=11)

    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-0.5, len(positions) - 0.5)
    plt.tight_layout()

    out_path = Path(output_dir) / f"fig_prior_strip_{model_name.lower().replace('-','_')}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =========================================================================
# Figure 2: Box Plot — distribution per culture/side
# =========================================================================
def plot_box(all_data, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(14, 6))

    positions = []
    pos = 0
    bp_data_a, bp_data_w = [], []
    bp_pos_a, bp_pos_w = [], []

    for culture in CULTURES:
        if culture not in all_data:
            continue
        bp_data_a.append(all_data[culture]["asian"])
        bp_data_w.append(all_data[culture]["western"])
        bp_pos_a.append(pos - 0.18)
        bp_pos_w.append(pos + 0.18)
        positions.append(pos)
        pos += 1

    bp_a = ax.boxplot(bp_data_a, positions=bp_pos_a, widths=0.3,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor="#BBDEFB", edgecolor="#1565C0"),
                       medianprops=dict(color="#0D47A1", linewidth=2),
                       whiskerprops=dict(color="#1565C0"),
                       capprops=dict(color="#1565C0"))

    bp_w = ax.boxplot(bp_data_w, positions=bp_pos_w, widths=0.3,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor="#FFCDD2", edgecolor="#C62828"),
                       medianprops=dict(color="#B71C1C", linewidth=2),
                       whiskerprops=dict(color="#C62828"),
                       capprops=dict(color="#C62828"))

    ax.set_xticks(positions)
    ax.set_xticklabels([CULTURE_LABELS[c] for c in CULTURES if c in all_data], fontsize=12)
    ax.set_ylabel("log P(entity | BOS)", fontsize=13)
    ax.set_title(f"Entity Prior Distribution — {model_name}", fontsize=14)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#BBDEFB", edgecolor="#1565C0", label="Asian entities"),
        Patch(facecolor="#FFCDD2", edgecolor="#C62828", label="Western entities"),
    ], loc="lower left", fontsize=11)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = Path(output_dir) / f"fig_prior_box_{model_name.lower().replace('-','_')}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =========================================================================
# Figure 3: Category-level strip plot (single culture, e.g., ko)
# =========================================================================
def plot_category_strip(all_data, culture, model_name, output_dir):
    if culture not in all_data:
        print(f"  {culture} not found, skipping category strip")
        return

    cat_data = all_data[culture]
    cats_present = [c for c in CATEGORIES if c in cat_data["asian_by_cat"]]

    fig, ax = plt.subplots(figsize=(12, 6))

    CAT_SHORT = {
        "authors": "Auth", "beverage": "Bev", "food": "Food",
        "locations": "Loc", "names-female": "Name-F",
        "names-male": "Name-M", "sports": "Sport",
    }

    positions = []
    pos = 0
    for cat in cats_present:
        asian_ents = cat_data["asian_by_cat"].get(cat, [])
        western_ents = cat_data["western_by_cat"].get(cat, [])

        a_vals = [v for _, v in asian_ents]
        w_vals = [v for _, v in western_ents]

        x_a = np.random.normal(pos - 0.15, 0.06, len(a_vals))
        ax.scatter(x_a, a_vals, c="#2196F3", alpha=0.4, s=15, edgecolors="none", zorder=3)

        x_w = np.random.normal(pos + 0.15, 0.06, len(w_vals))
        ax.scatter(x_w, w_vals, c="#F44336", alpha=0.4, s=15, edgecolors="none", zorder=3)

        if a_vals:
            ax.scatter(pos - 0.15, np.mean(a_vals), c="#0D47A1", s=80, marker="_", linewidths=2.5, zorder=5)
        if w_vals:
            ax.scatter(pos + 0.15, np.mean(w_vals), c="#B71C1C", s=80, marker="_", linewidths=2.5, zorder=5)

        # Range annotation
        if a_vals:
            rng = max(a_vals) - min(a_vals)
            ax.annotate(f"Δ={rng:.0f}", xy=(pos - 0.15, max(a_vals) + 2),
                       ha="center", fontsize=8, color="#0D47A1")

        positions.append(pos)
        pos += 1

    ax.set_xticks(positions)
    ax.set_xticklabels([CAT_SHORT.get(c, c) for c in cats_present], fontsize=12)
    ax.set_ylabel("log P(entity | BOS)", fontsize=13)
    ax.set_title(f"Entity Prior by Category — {CULTURE_LABELS[culture]}/{model_name}", fontsize=14)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=8, label='Asian'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=8, label='Western'),
    ], loc="lower left", fontsize=11)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = Path(output_dir) / f"fig_prior_cat_{culture}_{model_name.lower().replace('-','_')}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =========================================================================
# Figure 4: Extreme pair examples (horizontal bar)
# =========================================================================
def plot_extreme_pairs(priors_root, model, model_name, output_dir, top_n=15):
    """Show most extreme Asian-Western prior gaps."""
    pairs = []
    for culture in CULTURES:
        data = load_raw_priors(priors_root, model, culture)
        if data is None:
            continue
        for cat, sides in data["priors"].items():
            a_ents = sides.get("asian", {})
            w_ents = sides.get("western", {})
            # Pick min Asian and max Western per category (most extreme pair)
            if a_ents and w_ents:
                a_min_name = min(a_ents, key=a_ents.get)
                a_min_val = a_ents[a_min_name]
                w_max_name = max(w_ents, key=w_ents.get)
                w_max_val = w_ents[w_max_name]
                pairs.append({
                    "culture": culture, "category": cat,
                    "asian": a_min_name, "a_val": a_min_val,
                    "western": w_max_name, "w_val": w_max_val,
                    "gap": abs(a_min_val - w_max_val),
                })

    pairs.sort(key=lambda x: x["gap"], reverse=True)
    pairs = pairs[:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(pairs))
    labels = []

    for i, p in enumerate(pairs):
        # Asian bar (extending left from 0)
        ax.barh(i, p["a_val"], height=0.35, color="#2196F3", alpha=0.8, align="center")
        # Western bar
        ax.barh(i, p["w_val"], height=0.35, color="#F44336", alpha=0.8, align="center",
                left=0)

        # Markers
        ax.plot(p["a_val"], i, "s", color="#0D47A1", markersize=6, zorder=5)
        ax.plot(p["w_val"], i, "s", color="#B71C1C", markersize=6, zorder=5)

        # Gap annotation
        mid = (p["a_val"] + p["w_val"]) / 2
        ax.annotate(f"gap={p['gap']:.0f}", xy=(mid, i + 0.2),
                   ha="center", fontsize=8, color="#333")

        cat_short = p["category"][:4].title()
        a_short = p["asian"][:12]
        w_short = p["western"][:12]
        labels.append(f"{p['culture'].upper()}/{cat_short}: {a_short} vs {w_short}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("log P(entity | BOS)", fontsize=12)
    ax.set_title(f"Most Extreme Prior Gaps (Asian min vs Western max) — {model_name}", fontsize=13)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2196F3", label="Asian entity (worst prior)"),
        Patch(facecolor="#F44336", label="Western entity (best prior)"),
    ], loc="lower right", fontsize=10)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out_path = Path(output_dir) / f"fig_prior_extremes_{model_name.lower().replace('-','_')}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--priors_root", default="./dataset/priors")
    parser.add_argument("--output_dir", default="./analysis/prior_variance")
    parser.add_argument("--lang", default="cu")
    args = parser.parse_args()

    MODEL_DISPLAY = {
        "llama3_8b": "Llama-3.1-8B",
        "qwen3_8b": "Qwen-3-8B",
    }
    model_name = MODEL_DISPLAY.get(args.model, args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading priors for {model_name}...")
    all_data = collect_all_priors(args.priors_root, args.model, args.lang)
    print(f"  Loaded {len(all_data)} cultures")

    print("\n1. Strip plot (all cultures)...")
    plot_strip(all_data, model_name, output_dir)

    print("2. Box plot (all cultures)...")
    plot_box(all_data, model_name, output_dir)

    print("3. Category strip (ko)...")
    plot_category_strip(all_data, "ko", model_name, output_dir)

    print("4. Extreme pairs...")
    plot_extreme_pairs(args.priors_root, args.model, model_name, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()