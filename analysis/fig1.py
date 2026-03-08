#!/usr/bin/env python3
"""
fig1.py — Baseline CBS dot plot for CoCoA Figure 1 (bottom panel).
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CULTURES = ["ko", "ja", "zh", "hi", "mr", "vi", "gu", "ml", "ur", "ar"]
CULTURE_LABELS = {
    "ko": "Korean (ko)",    "ja": "Japanese (ja)",  "zh": "Chinese (zh)",
    "hi": "Hindi (hi)",     "mr": "Marathi (mr)",   "gu": "Gujarati (gu)",
    "ml": "Malayalam (ml)", "vi": "Vietnamese (vi)", "ur": "Urdu (ur)",
    "ar": "Arabic (ar)",
}
MODEL_DISPLAY = {"llama": "Llama-3.1-8B", "qwen": "Qwen-3-8B"}

BLUE   = "#2171B5"
ORANGE = "#E6550D"
GRAY   = "#AAAAAA"


def load_baseline(json_path: str, model: str) -> dict:
    with open(json_path) as f:
        rows = json.load(f)["rows"]
    result = {}
    for row in rows:
        c = row["culture"]
        if c not in CULTURES or row["model"] != model:
            continue
        base = row.get("baseline")
        if base is None:
            continue
        result[c] = (base["cbs_g"], base["cbs_n"])
    for c in CULTURES:
        if c not in result:
            result[c] = None
    return result


def make_plot(data: dict, model: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 5.5))

    y_pos = list(range(len(CULTURES) - 1, -1, -1))

    for i, c in enumerate(CULTURES):
        y = y_pos[i]
        val = data[c]

        if val is None:
            ax.text(-1, y, CULTURE_LABELS[c], ha="right", va="center",
                    fontsize=10, color="#999999", fontstyle="italic")
            continue

        g, n = val
        delta = abs(n - g)

        ax.text(-1, y, CULTURE_LABELS[c], ha="right", va="center",
                fontsize=10, color="#222222")

        ax.plot([g, n], [y, y], color=GRAY, linewidth=1.8,
                solid_capstyle="round", zorder=3)

        ax.scatter(g, y, color=BLUE,   s=50, zorder=4)
        ax.scatter(n, y, color=ORANGE, s=50, zorder=4)

        hi = max(g, n)
        ax.text(hi + 2.5, y, f"\u0394={delta:.1f}", ha="left", va="center",
                fontsize=8, color="#666666")

    ax.axvline(x=0,  color="#cccccc", ls="--", lw=0.7, zorder=0)
    ax.axvline(x=50, color="#cccccc", ls="--", lw=0.7, zorder=0)
    for x in range(10, 50, 10):
        ax.axvline(x=x, color="#eeeeee", lw=0.5, zorder=0)

    ax.set_xlim(-1, 53)
    ax.set_ylim(-0.8, len(CULTURES) - 0.2)
    ax.set_xticks(range(0, 55, 10))
    ax.set_xticklabels([f"{x}" for x in range(0, 55, 10)], fontsize=9)
    ax.set_xlabel("CBS", fontsize=10)
    ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.tick_params(axis="x", colors="#666666")

    display = MODEL_DISPLAY.get(model, model)
    ax.set_title(
        f"Baseline CBS across Asian languages\n"
        f"(Ideal \u0394 \u2248 50)",
        fontsize=11, fontweight="bold", pad=22,
    )

    # legend — horizontal, tucked between title and data
    h1 = ax.scatter([], [], color=BLUE,   s=30)
    h2 = ax.scatter([], [], color=ORANGE, s=30)
    ax.legend([h1, h2],
              [r"CBS$_g$ (Grounded)", r"CBS$_n$ (Neutral)"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              fontsize=7.5, frameon=False,
              handletextpad=0.2, columnspacing=1.0, borderpad=0)

    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved -> {out_path}")


def main():
    here = Path(__file__).resolve().parent
    default_json = here.parent / "main_table.json"
    default_out  = here / "figures" / "fig1.png"

    parser = argparse.ArgumentParser(description="Generate Fig 1 dot plot")
    parser.add_argument("--json",  type=str, default=str(default_json))
    parser.add_argument("--out",   type=str, default=str(default_out))
    parser.add_argument("--model", type=str, default="llama",
                        choices=["llama", "qwen"])
    args = parser.parse_args()

    data = load_baseline(args.json, args.model)
    print(f"Model: {args.model}")
    for c in CULTURES:
        v = data[c]
        if v:
            print(f"  {c}: CBS_g={v[0]:.1f}  CBS_n={v[1]:.1f}  delta={abs(v[1]-v[0]):.1f}")
        else:
            print(f"  {c}: (no data)")

    make_plot(data, args.model, args.out)


if __name__ == "__main__":
    main()