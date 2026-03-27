"""
CoCoA: Entity Prior Variance Analysis

Shows that even within the same culture/category, entity priors vary wildly.
This motivates per-entity prior normalization (not just culture-level).

Usage:
    python analysis/analyze_entity_prior_variance.py \
        --model llama3_8b --output analysis/prior_variance/

Reads from: dataset/priors/{model}/{culture}_cu/entity_priors.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

CULTURES = ["ko", "ja", "zh", "hi", "mr", "gu", "ml", "vi", "ur", "ar"]
CATEGORIES = ["authors", "beverage", "food", "locations", "names-female", "names-male", "sports"]
CAT_SHORT = {
    "authors": "Auth", "beverage": "Bev", "food": "Food",
    "locations": "Loc", "names-female": "Name-F",
    "names-male": "Name-M", "sports": "Sport",
}


def load_priors(priors_root, model, culture, lang="cu"):
    path = Path(priors_root) / model / f"{culture}_{lang}" / "entity_priors.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--priors_root", default="./dataset/priors")
    parser.add_argument("--lang", default="cu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    all_results = {}

    # ================================================================
    # Table 1: Per-culture, per-category, per-side stats
    # ================================================================
    print(f"\n{'='*120}")
    print(f"  Entity Prior Variance — {args.model}")
    print(f"  (log P(entity|BOS), within same culture/category/side)")
    print(f"{'='*120}")
    print(f"  {'Cult':<4} {'Cat':<7} {'Side':<7} │ {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8} │ {'N':>4} │ Extreme entities")
    print(f"  {'─'*110}")

    for culture in CULTURES:
        data = load_priors(args.priors_root, args.model, culture, args.lang)
        if data is None:
            print(f"  {culture.upper()}: priors not found, skipping")
            continue

        culture_results = {}
        for cat, sides in data["priors"].items():
            for side, entities in sides.items():
                vals = list(entities.values())
                names = list(entities.keys())
                if not vals:
                    continue

                arr = np.array(vals)
                mean, std = np.mean(arr), np.std(arr)
                mn, mx = np.min(arr), np.max(arr)
                rng = mx - mn

                # Find extreme entities
                min_idx = np.argmin(arr)
                max_idx = np.argmax(arr)
                min_ent = names[min_idx]
                max_ent = names[max_idx]

                cat_short = CAT_SHORT.get(cat, cat[:6])
                print(f"  {culture.upper():<4} {cat_short:<7} {side:<7} │ "
                      f"{mean:>8.1f} {std:>8.1f} {mn:>8.1f} {mx:>8.1f} {rng:>8.1f} │ "
                      f"{len(vals):>4} │ "
                      f"worst={min_ent[:15]}({mn:.0f}) best={max_ent[:15]}({mx:.0f})")

                culture_results[f"{cat}_{side}"] = {
                    "mean": round(mean, 2), "std": round(std, 2),
                    "min": round(mn, 2), "max": round(mx, 2), "range": round(rng, 2),
                    "n": len(vals),
                    "min_entity": min_ent, "max_entity": max_ent,
                }

        all_results[culture] = culture_results

    # ================================================================
    # Table 2: Summary — avg range per culture (Asian vs Western)
    # ================================================================
    print(f"\n{'='*90}")
    print(f"  Summary: Average Prior Range within Category")
    print(f"  (how much entity priors vary WITHIN the same culture/category)")
    print(f"{'='*90}")
    print(f"  {'Culture':<8} │ {'Asian Range':>12} {'Asian Std':>10} │ {'Western Range':>14} {'Western Std':>12} │ {'Asian-Western Gap':>18}")
    print(f"  {'─'*80}")

    for culture in CULTURES:
        if culture not in all_results:
            continue
        r = all_results[culture]

        asian_ranges, western_ranges = [], []
        asian_stds, western_stds = [], []
        asian_means, western_means = [], []

        for key, v in r.items():
            if key.endswith("_asian"):
                asian_ranges.append(v["range"])
                asian_stds.append(v["std"])
                asian_means.append(v["mean"])
            elif key.endswith("_western"):
                western_ranges.append(v["range"])
                western_stds.append(v["std"])
                western_means.append(v["mean"])

        a_rng = np.mean(asian_ranges) if asian_ranges else 0
        a_std = np.mean(asian_stds) if asian_stds else 0
        w_rng = np.mean(western_ranges) if western_ranges else 0
        w_std = np.mean(western_stds) if western_stds else 0
        gap = np.mean(asian_means) - np.mean(western_means) if asian_means and western_means else 0

        print(f"  {culture.upper():<8} │ {a_rng:>12.1f} {a_std:>10.1f} │ {w_rng:>14.1f} {w_std:>12.1f} │ {gap:>+18.1f}")

    # ================================================================
    # Table 3: Top-10 most extreme prior gaps (same category, A vs W)
    # ================================================================
    print(f"\n{'='*90}")
    print(f"  Top Extreme Entity Pairs (largest |prior_asian - prior_western| in same category)")
    print(f"{'='*90}")

    pair_gaps = []
    for culture in CULTURES:
        data = load_priors(args.priors_root, args.model, culture, args.lang)
        if data is None:
            continue
        for cat, sides in data["priors"].items():
            asian_ents = sides.get("asian", {})
            western_ents = sides.get("western", {})
            for a_name, a_val in asian_ents.items():
                for w_name, w_val in western_ents.items():
                    pair_gaps.append({
                        "culture": culture, "category": cat,
                        "asian": a_name, "asian_prior": a_val,
                        "western": w_name, "western_prior": w_val,
                        "gap": a_val - w_val,
                    })

    pair_gaps.sort(key=lambda x: abs(x["gap"]), reverse=True)
    print(f"  {'Culture':<4} {'Cat':<7} {'Asian Entity':<20} {'A Prior':>8} {'Western Entity':<20} {'W Prior':>8} {'Gap':>8}")
    print(f"  {'─'*82}")
    for p in pair_gaps[:20]:
        print(f"  {p['culture'].upper():<4} {CAT_SHORT.get(p['category'],p['category'][:6]):<7} "
              f"{p['asian'][:19]:<20} {p['asian_prior']:>8.1f} "
              f"{p['western'][:19]:<20} {p['western_prior']:>8.1f} "
              f"{p['gap']:>+8.1f}")

    # Save
    if args.output:
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / f"prior_variance_{args.model}.json", "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved: {out_path / f'prior_variance_{args.model}.json'}")


if __name__ == "__main__":
    main()