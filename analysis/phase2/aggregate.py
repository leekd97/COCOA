"""
Phase 2 Result Aggregator

Usage:
    python aggregate_phase2.py --input_dir ./analysis/phase2
    python aggregate_phase2.py --input_dir ./analysis/phase2 --output ./analysis/phase2/phase2_summary.json
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./analysis/phase2")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("phase2_*.json"))

    if not files:
        print(f"No phase2_*.json files found in {input_dir}")
        return

    print(f"Found {len(files)} files\n")

    all_results = {}
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        key = f"{data['culture']}_{data['lang']}_{data['model_key']}"
        all_results[key] = data

    # =====================================================================
    # Table 1: Prior Gap Summary
    # =====================================================================
    print("=" * 85)
    print("  Entity Prior Gap: mean log P(entity) — Western minus Asian")
    print("  Positive = Western entities have higher unconditional probability")
    print("=" * 85)
    print(f"{'Key':<30} {'Overall':>10} {'authors':>10} {'food':>10} {'beverage':>10} {'sports':>10} {'names-m':>10}")
    print("-" * 92)

    for key, r in sorted(all_results.items()):
        gap = r["prior_gap_analysis"]
        overall = gap.get("_overall", {}).get("prior_gap_w_minus_a", "N/A")
        cats = {}
        for cat in ["authors", "food", "beverage", "sports", "names-male"]:
            if cat in gap:
                cats[cat] = gap[cat]["prior_gap_w_minus_a"]
            else:
                cats[cat] = None

        def fmt(v):
            return f"{v:>+10.3f}" if isinstance(v, (int, float)) else f"{'N/A':>10}"

        print(f"{key:<30} {fmt(overall)} {fmt(cats.get('authors'))} "
              f"{fmt(cats.get('food'))} {fmt(cats.get('beverage'))} "
              f"{fmt(cats.get('sports'))} {fmt(cats.get('names-male'))}")

    # =====================================================================
    # Table 2: CBS Comparison (Standard vs Normalized)
    # =====================================================================
    print()
    print("=" * 100)
    print("  CBS Comparison: Standard vs Prior-Normalized")
    print("=" * 100)
    print(f"{'Key':<30} {'Std_g':>8} {'Norm_g':>8} {'Δg':>7} {'Flip_g%':>8} "
          f"{'Std_n':>8} {'Norm_n':>8} {'Δn':>7} {'Flip_n%':>8}")
    print("-" * 95)

    for key, r in sorted(all_results.items()):
        cbs = r["cbs_comparison"]
        g = cbs["grounded"]
        n = cbs["neutral"]
        print(f"{key:<30} "
              f"{g['standard_cbs']:>8.1f} {g['normalized_cbs']:>8.1f} {g['delta']:>+7.1f} {g['flip_rate_pct']:>8.1f} "
              f"{n['standard_cbs']:>8.1f} {n['normalized_cbs']:>8.1f} {n['delta']:>+7.1f} {n['flip_rate_pct']:>8.1f}")

    # =====================================================================
    # Table 3: Per-Category CBS Detail (for each culture×model)
    # =====================================================================
    print()
    print("=" * 100)
    print("  Per-Category Detail")
    print("=" * 100)

    for key, r in sorted(all_results.items()):
        print(f"\n--- {key} ---")
        for ctx_type in ["grounded", "neutral"]:
            cats = r["cbs_comparison"][ctx_type].get("by_category", {})
            if not cats:
                continue
            print(f"  [{ctx_type}]")
            print(f"    {'Category':<18} {'Std':>8} {'Norm':>8} {'Flip%':>8} {'N_comp':>10}")
            for cat, v in sorted(cats.items()):
                print(f"    {cat:<18} {v['std_cbs']:>8.1f} {v['norm_cbs']:>8.1f} "
                      f"{v['prior_flip_rate']:>8.1f} {v['n_comparisons']:>10}")

    # =====================================================================
    # Save combined
    # =====================================================================
    output_path = args.output or str(input_dir / "phase2_summary.json")
    
    # Build compact summary for JSON
    summary = {}
    for key, r in all_results.items():
        gap = r["prior_gap_analysis"]
        cbs = r["cbs_comparison"]
        summary[key] = {
            "culture": r["culture"],
            "lang": r["lang"],
            "model": r["model_key"],
            "prior_gap_overall": gap.get("_overall", {}).get("prior_gap_w_minus_a"),
            "prior_gap_by_cat": {
                cat: v["prior_gap_w_minus_a"]
                for cat, v in gap.items() if not cat.startswith("_")
            },
            "grounded": {
                "std_cbs": cbs["grounded"]["standard_cbs"],
                "norm_cbs": cbs["grounded"]["normalized_cbs"],
                "delta": cbs["grounded"]["delta"],
                "flip_pct": cbs["grounded"]["flip_rate_pct"],
            },
            "neutral": {
                "std_cbs": cbs["neutral"]["standard_cbs"],
                "norm_cbs": cbs["neutral"]["normalized_cbs"],
                "delta": cbs["neutral"]["delta"],
                "flip_pct": cbs["neutral"]["flip_rate_pct"],
            },
        }

    with open(output_path, "w") as f:
        json.dump({"results": summary, "raw": all_results}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()