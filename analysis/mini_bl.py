#!/usr/bin/env python3
"""
CoCoA Baseline Result Table

Shows Base Model vs BiasEdit/BiasUnlearn, grouped by culture × model.
Computes mean±SE across folds.

Usage:
    python analysis/baseline_result_table.py baselines/results
    python analysis/baseline_result_table.py baselines/results --method biasedit
    python analysis/baseline_result_table.py baselines/results --model llama
    python analysis/baseline_result_table.py baselines/results --save
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


CULTURES_ORDER = ["ko", "ja", "zh", "ar", "hi", "mr", "gu", "ml", "vi", "ur"]
MODEL_NORM = {"llama3-8b": "llama", "qwen3-8b": "qwen"}


def load_results(results_root):
    """Load all results.json from baselines/results/{method}/*/results.json."""
    results = []
    root = Path(results_root)

    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir() or method_dir.name.startswith("_"):
            continue
        method = method_dir.name  # biasedit or biasunlearn

        for rfile in sorted(method_dir.rglob("results.json")):
            folder = rfile.parent.name
            try:
                with open(rfile) as f:
                    data = json.load(f)
            except Exception:
                continue

            # Parse folder name for culture/model
            parts = folder.split("_")
            culture = parts[0] if parts else "?"

            model = "?"
            for m_key, m_val in MODEL_NORM.items():
                if m_key in folder:
                    model = m_val
                    break

            # Baseline results use "baseline" and "trained" keys
            baseline = data.get("baseline")
            trained = data.get("trained")
            if not baseline or not trained:
                continue

            bl_g = baseline.get("cbs_g")
            bl_n = baseline.get("cbs_n")
            tr_g = trained.get("cbs_g")
            tr_n = trained.get("cbs_n")

            if any(v is None for v in [bl_g, bl_n, tr_g, tr_n]):
                continue

            results.append({
                "method": method,
                "culture": culture,
                "model": model,
                "folder": folder,
                "bl_g": bl_g, "bl_n": bl_n,
                "bl_score": bl_g + abs(bl_n - 50),
                "tr_g": tr_g, "tr_n": tr_n,
                "tr_score": tr_g + abs(tr_n - 50),
            })

    return results


def fmt(mean, se):
    if se > 0:
        return f"{mean:>5.1f}±{se:<4.1f}"
    else:
        return f"{mean:>5.1f}     "


def print_table(results, method_filter=None, model_filter=None):
    """Print result tables grouped by method × model."""
    methods = sorted(set(r["method"] for r in results))
    if method_filter:
        methods = [m for m in methods if m == method_filter]

    output_lines = []
    def out(line=""):
        print(line)
        output_lines.append(line)

    for method in methods:
        method_results = [r for r in results if r["method"] == method]

        # Group by (culture, model)
        groups = defaultdict(list)
        for r in method_results:
            if model_filter and r["model"] != model_filter:
                continue
            groups[(r["culture"], r["model"])].append(r)

        if not groups:
            continue

        models = sorted(set(k[1] for k in groups))
        method_label = {"biasedit": "BiasEdit", "biasunlearn": "BiasUnlearn"}.get(method, method)

        out(f"\n{'='*105}")
        out(f"  {method_label}")
        out(f"{'='*105}")

        for model in models:
            out(f"\n  {model.upper()}")
            out(f"  {'Cult':<5} {'K':>3} │ {'BL g':>10} {'BL n':>10} │ {method_label + ' g':>12} {method_label + ' n':>12} {'Score':>10} │ {'Δg':>7} {'Δn→50':>7} {'ΔSc':>7}")
            out(f"  {'─'*105}")

            all_tr_g, all_tr_n, all_tr_sc = [], [], []
            all_bl_g, all_bl_n, all_bl_sc = [], [], []

            for culture in CULTURES_ORDER:
                key = (culture, model)
                if key not in groups:
                    continue

                g = groups[key]
                k = len(g)

                bl_g = [r["bl_g"] for r in g]
                bl_n = [r["bl_n"] for r in g]
                tr_g = [r["tr_g"] for r in g]
                tr_n = [r["tr_n"] for r in g]
                tr_sc = [r["tr_score"] for r in g]

                se = lambda x: np.std(x)/np.sqrt(len(x)) if len(x) > 1 else 0

                bl_n50 = np.mean([abs(n-50) for n in bl_n])
                tr_n50 = np.mean([abs(n-50) for n in tr_n])
                delta_g = np.mean(tr_g) - np.mean(bl_g)
                delta_n50 = tr_n50 - bl_n50
                delta_sc = np.mean(tr_sc) - np.mean([r["bl_score"] for r in g])

                out(f"  {culture.upper():<5} {k:>3} │ "
                    f"{fmt(np.mean(bl_g), se(bl_g))} {fmt(np.mean(bl_n), se(bl_n))} │ "
                    f"{fmt(np.mean(tr_g), se(tr_g)):>12} {fmt(np.mean(tr_n), se(tr_n)):>12} {fmt(np.mean(tr_sc), se(tr_sc)):>10} │ "
                    f"{delta_g:>+7.1f} {delta_n50:>+7.1f} {delta_sc:>+7.1f}")

                all_tr_g.extend(tr_g)
                all_tr_n.extend(tr_n)
                all_tr_sc.extend(tr_sc)
                all_bl_g.extend(bl_g)
                all_bl_n.extend(bl_n)
                all_bl_sc.extend([r["bl_score"] for r in g])

            if all_tr_g:
                out(f"  {'─'*105}")
                avg_tr_n50 = np.mean([abs(n-50) for n in all_tr_n])
                avg_bl_n50 = np.mean([abs(n-50) for n in all_bl_n])
                out(f"  {'AVG':<5} {len(all_tr_g):>3} │ "
                    f"{np.mean(all_bl_g):>5.1f}      {np.mean(all_bl_n):>5.1f}      │ "
                    f"{np.mean(all_tr_g):>12.1f} {np.mean(all_tr_n):>12.1f} {np.mean(all_tr_sc):>10.1f} │ "
                    f"{np.mean(all_tr_g)-np.mean(all_bl_g):>+7.1f} {avg_tr_n50-avg_bl_n50:>+7.1f} {np.mean(all_tr_sc)-np.mean(all_bl_sc):>+7.1f}")
                out(f"  |n-50| avg: {avg_tr_n50:.1f}")

    out(f"\n{'='*105}")
    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(description="CoCoA Baseline Result Table")
    parser.add_argument("results_root", help="baselines/results directory")
    parser.add_argument("--method", default=None, choices=["biasedit", "biasunlearn"])
    parser.add_argument("--model", default=None, choices=["llama", "qwen"])
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        print(f"Not found: {root}")
        sys.exit(1)

    results = load_results(root)
    if not results:
        print(f"No results found in {root}")
        sys.exit(1)

    n_be = sum(1 for r in results if r["method"] == "biasedit")
    n_bu = sum(1 for r in results if r["method"] == "biasunlearn")
    print(f"Loaded {len(results)} results (BiasEdit: {n_be}, BiasUnlearn: {n_bu})")

    table_text = print_table(results, args.method, args.model)

    if args.save and table_text:
        out_path = root / "baseline_result_table.txt"
        out_path.write_text(table_text, encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()