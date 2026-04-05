#!/usr/bin/env python3
"""
CoCoA: Cross-Experiment Best Finder

Scans ALL experiment subdirectories and finds the best config
per culture × model, with CBS_n ≈ 50 as hard constraint.

Usage:
    python analysis/find_best.py experiments/
    python analysis/find_best.py experiments/ --n_threshold 15
    python analysis/find_best.py experiments/ --model llama
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


CULTURES_ORDER = ["ko", "ja", "zh", "ar", "hi", "mr", "gu", "ml", "vi", "ur"]
MODEL_NORM = {"llama3-8b": "llama", "qwen3-8b": "qwen"}


def load_all_results(exp_root):
    """Scan all subdirectories for results.json files."""
    results = []
    for rfile in sorted(Path(exp_root).rglob("results.json")):
        # Get experiment subdirectory name (e.g., sweep_drift_1.0)
        # rfile: experiments/sweep_drift_1.0/ko_cu_llama3-8b_.../results.json
        parts = rfile.relative_to(exp_root).parts
        if len(parts) < 2:
            continue
        exp_subdir = parts[0]  # e.g., "sweep_drift_1.0"
        folder = parts[1] if len(parts) >= 3 else parts[0]

        try:
            with open(rfile) as f:
                data = json.load(f)
        except Exception:
            continue

        # Parse folder name for culture/model
        folder_name = rfile.parent.name
        folder_parts = folder_name.split("_")
        culture = folder_parts[0] if folder_parts else "?"

        model = "?"
        for m_key, m_val in MODEL_NORM.items():
            if m_key in folder_name:
                model = m_val
                break

        # Extract baseline and final
        baseline = data.get("baseline") or data.get("baseline_test")
        final = data.get("final")
        if not baseline or not final:
            continue

        def get_vals(d):
            g = d.get("cbs_grounded", d.get("cbs_g"))
            n = d.get("cbs_neutral", d.get("cbs_n"))
            if g is None or n is None:
                return None
            return {"g": g, "n": n}

        bl = get_vals(baseline)
        fn = get_vals(final)
        if not bl or not fn:
            continue

        results.append({
            "exp": exp_subdir,
            "folder": folder_name,
            "culture": culture,
            "model": model,
            "bl_g": bl["g"], "bl_n": bl["n"],
            "fn_g": fn["g"], "fn_n": fn["n"],
            "fn_n50": abs(fn["n"] - 50),
            "score": fn["g"] + abs(fn["n"] - 50),
            "delta_g": fn["g"] - bl["g"],
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_root", help="Root experiments directory")
    parser.add_argument("--n_threshold", type=float, default=15.0,
                        help="|CBS_n - 50| threshold for acceptable neutral (default 15)")
    parser.add_argument("--model", default=None, choices=["llama", "qwen"])
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = load_all_results(args.exp_root)
    if not results:
        print("No results found.")
        sys.exit(1)

    print(f"Loaded {len(results)} results from {args.exp_root}")

    # Group by (culture, model)
    groups = defaultdict(list)
    for r in results:
        if args.model and r["model"] != args.model:
            continue
        groups[(r["culture"], r["model"])].append(r)

    models = sorted(set(k[1] for k in groups))

    W = 130
    output_lines = []
    def out(line=""):
        print(line)
        output_lines.append(line)

    for model in models:
        out(f"\n{'='*W}")
        out(f"  {model.upper()} — Best Config per Culture")
        out(f"  (Constraint: |CBS_n - 50| < {args.n_threshold})")
        out(f"{'='*W}")

        # ── Best by Score (g + |n-50|), with n constraint ──
        out(f"\n  ★ Best by SCORE (CBS_g + |CBS_n - 50|, lower = better):")
        out(f"  {'Cult':<5} │ {'BL_g':>6} {'BL_n':>6} │ {'CBS_g':>6} {'CBS_n':>6} {'|n50|':>6} {'Score':>6} │ {'Δg':>7} │ Experiment")
        out(f"  {'─'*115}")

        best_configs = {}
        for culture in CULTURES_ORDER:
            key = (culture, model)
            if key not in groups:
                continue

            # Filter: |n-50| < threshold
            valid = [r for r in groups[key] if r["fn_n50"] < args.n_threshold]

            if not valid:
                # No valid result — show best anyway with warning
                best = min(groups[key], key=lambda x: x["score"])
                out(f"  {culture.upper():<5} │ {best['bl_g']:>6.1f} {best['bl_n']:>6.1f} │ "
                    f"{best['fn_g']:>6.1f} {best['fn_n']:>6.1f} {best['fn_n50']:>6.1f} {best['score']:>6.1f} │ "
                    f"{best['delta_g']:>+7.1f} │ {best['exp']}  ⚠️ NO VALID (|n50|={best['fn_n50']:.1f})")
                best_configs[culture] = best
            else:
                best = min(valid, key=lambda x: x["score"])
                marker = "✓" if best["fn_n50"] < 5 else "△" if best["fn_n50"] < 10 else ""
                out(f"  {culture.upper():<5} │ {best['bl_g']:>6.1f} {best['bl_n']:>6.1f} │ "
                    f"{best['fn_g']:>6.1f} {best['fn_n']:>6.1f} {best['fn_n50']:>6.1f} {best['score']:>6.1f} │ "
                    f"{best['delta_g']:>+7.1f} │ {best['exp']}  {marker}")
                best_configs[culture] = best

        # Summary
        valid_bests = [v for v in best_configs.values() if v["fn_n50"] < args.n_threshold]
        if valid_bests:
            avg_g = np.mean([r["fn_g"] for r in valid_bests])
            avg_n50 = np.mean([r["fn_n50"] for r in valid_bests])
            avg_sc = np.mean([r["score"] for r in valid_bests])
            out(f"  {'─'*115}")
            out(f"  AVG ({len(valid_bests)} valid) │ {'':>13} │ {avg_g:>6.1f} {'':>6} {avg_n50:>6.1f} {avg_sc:>6.1f} │")

        # ── Best by CBS_g only, with n constraint ──
        out(f"\n  🎯 Best by CBS_g (grounded only, with |n-50| < {args.n_threshold}):")
        out(f"  {'Cult':<5} │ {'CBS_g':>6} {'CBS_n':>6} {'|n50|':>6} │ Experiment")
        out(f"  {'─'*80}")

        for culture in CULTURES_ORDER:
            key = (culture, model)
            if key not in groups:
                continue
            valid = [r for r in groups[key] if r["fn_n50"] < args.n_threshold]
            if not valid:
                out(f"  {culture.upper():<5} │ {'—':>6} {'—':>6} {'—':>6} │ NO VALID")
                continue
            best = min(valid, key=lambda x: x["fn_g"])
            out(f"  {culture.upper():<5} │ {best['fn_g']:>6.1f} {best['fn_n']:>6.1f} {best['fn_n50']:>6.1f} │ {best['exp']}")

        # ── Best by CBS_n (closest to 50) ──
        out(f"\n  ⚖️  Best by CBS_n (closest to 50):")
        out(f"  {'Cult':<5} │ {'CBS_g':>6} {'CBS_n':>6} {'|n50|':>6} │ Experiment")
        out(f"  {'─'*80}")

        for culture in CULTURES_ORDER:
            key = (culture, model)
            if key not in groups:
                continue
            best = min(groups[key], key=lambda x: x["fn_n50"])
            out(f"  {culture.upper():<5} │ {best['fn_g']:>6.1f} {best['fn_n']:>6.1f} {best['fn_n50']:>6.1f} │ {best['exp']}")

        # ── Top 3 per culture ──
        out(f"\n  📊 Top 3 per Culture (by Score, |n-50| < {args.n_threshold}):")
        for culture in CULTURES_ORDER:
            key = (culture, model)
            if key not in groups:
                continue
            valid = [r for r in groups[key] if r["fn_n50"] < args.n_threshold]
            if not valid:
                out(f"  [{culture.upper()}] — no valid results")
                continue

            top3 = sorted(valid, key=lambda x: x["score"])[:3]
            out(f"  [{culture.upper()}]")
            for i, r in enumerate(top3):
                out(f"    {i+1}. g={r['fn_g']:.1f} n={r['fn_n']:.1f} |n50|={r['fn_n50']:.1f} score={r['score']:.1f} ← {r['exp']}")

        # ── Experiment frequency (which config appears most) ──
        out(f"\n  📈 Most frequent best experiment:")
        exp_counts = defaultdict(int)
        for v in best_configs.values():
            exp_counts[v["exp"]] += 1
        for exp, cnt in sorted(exp_counts.items(), key=lambda x: -x[1]):
            out(f"    {exp}: {cnt} cultures")

    out(f"\n{'='*W}")

    if args.save:
        out_path = Path(args.exp_root) / "best_configs.txt"
        out_path.write_text("\n".join(output_lines), encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()