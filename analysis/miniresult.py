#!/usr/bin/env python3
"""
CoCoA Quick Result Table

Shows Base Model vs CoCoA for a single experiment condition.
Groups by culture × model, computes mean±SE across folds.

Usage:
    python analysis/result_table.py experiments/kfold_nxn_scaled_0.3
    python analysis/result_table.py experiments/kfold_nxn_scaled_0.3 --model llama
    python analysis/result_table.py experiments/kfold_nxn_scaled_0.3 --save
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


CULTURES_ORDER = ["ko", "ja", "zh", "ar", "hi", "mr", "gu", "ml", "vi", "ur"]
MODEL_NORM = {"llama3-8b": "llama", "qwen3-8b": "qwen"}


def load_results(exp_dir):
    """Load all results.json from experiment directory."""
    results = []
    for rfile in sorted(Path(exp_dir).rglob("results.json")):
        folder = rfile.parent.name
        try:
            with open(rfile) as f:
                data = json.load(f)
        except Exception:
            continue

        # Parse folder name
        parts = folder.split("_")
        culture = parts[0] if parts else "?"
        
        # Find model
        model = "?"
        for m_key, m_val in MODEL_NORM.items():
            if m_key in folder:
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
            return {"g": g, "n": n, "score": g + abs(n - 50)}
        
        bl = get_vals(baseline)
        fn = get_vals(final)
        if not bl or not fn:
            continue
        
        results.append({
            "culture": culture, "model": model, "folder": folder,
            "bl_g": bl["g"], "bl_n": bl["n"], "bl_score": bl["score"],
            "fn_g": fn["g"], "fn_n": fn["n"], "fn_score": fn["score"],
        })
    
    return results


def fmt(mean, se):
    """Format as mean±se."""
    if se > 0:
        return f"{mean:>5.1f}±{se:<4.1f}"
    else:
        return f"{mean:>5.1f}     "


def print_table(exp_dir, results, model_filter=None):
    """Print compact result table."""
    # Group by (culture, model)
    groups = defaultdict(list)
    for r in results:
        if model_filter and r["model"] != model_filter:
            continue
        groups[(r["culture"], r["model"])].append(r)
    
    if not groups:
        print(f"  No results found{' for model=' + model_filter if model_filter else ''}.")
        return None
    
    models = sorted(set(k[1] for k in groups))
    
    output_lines = []
    def out(line=""):
        print(line)
        output_lines.append(line)
    
    exp_name = Path(exp_dir).name
    out(f"\n{'='*105}")
    out(f"  {exp_name}")
    out(f"{'='*105}")
    
    all_model_stats = {}
    
    for model in models:
        out(f"\n  {model.upper()}")
        out(f"  {'Cult':<5} {'K':>3} │ {'BL g':>10} {'BL n':>10} │ {'CoCoA g':>10} {'CoCoA n':>10} {'Score':>10} │ {'Δg':>7} {'Δn→50':>7} {'ΔSc':>7}")
        out(f"  {'─'*98}")
        
        model_fn_g, model_fn_n, model_fn_sc = [], [], []
        model_bl_g, model_bl_n = [], []
        
        for culture in CULTURES_ORDER:
            key = (culture, model)
            if key not in groups:
                continue
            
            g = groups[key]
            k = len(g)
            
            bl_g = [r["bl_g"] for r in g]
            bl_n = [r["bl_n"] for r in g]
            fn_g = [r["fn_g"] for r in g]
            fn_n = [r["fn_n"] for r in g]
            fn_sc = [r["fn_score"] for r in g]
            
            se = lambda x: np.std(x)/np.sqrt(len(x)) if len(x) > 1 else 0
            
            # Delta: improvement in |n-50|
            bl_n50 = np.mean([abs(n-50) for n in bl_n])
            fn_n50 = np.mean([abs(n-50) for n in fn_n])
            delta_n50 = fn_n50 - bl_n50  # negative = better
            delta_g = np.mean(fn_g) - np.mean(bl_g)
            delta_sc = np.mean(fn_sc) - np.mean([r["bl_score"] for r in g])
            
            out(f"  {culture.upper():<5} {k:>3} │ "
                f"{fmt(np.mean(bl_g), se(bl_g))} {fmt(np.mean(bl_n), se(bl_n))} │ "
                f"{fmt(np.mean(fn_g), se(fn_g))} {fmt(np.mean(fn_n), se(fn_n))} {fmt(np.mean(fn_sc), se(fn_sc))} │ "
                f"{delta_g:>+7.1f} {delta_n50:>+7.1f} {delta_sc:>+7.1f}")
            
            model_fn_g.extend(fn_g)
            model_fn_n.extend(fn_n)
            model_fn_sc.extend(fn_sc)
            model_bl_g.extend(bl_g)
            model_bl_n.extend(bl_n)
        
        # Model average
        if model_fn_g:
            out(f"  {'─'*98}")
            avg_fn_g = np.mean(model_fn_g)
            avg_fn_n = np.mean(model_fn_n)
            avg_fn_sc = np.mean(model_fn_sc)
            avg_n50 = np.mean([abs(n-50) for n in model_fn_n])
            avg_bl_n50 = np.mean([abs(n-50) for n in model_bl_n])
            
            out(f"  {'AVG':<5} {len(model_fn_g):>3} │ "
                f"{np.mean(model_bl_g):>5.1f}      {np.mean(model_bl_n):>5.1f}      │ "
                f"{avg_fn_g:>5.1f}      {avg_fn_n:>5.1f}      {avg_fn_sc:>5.1f}      │ "
                f"{avg_fn_g - np.mean(model_bl_g):>+7.1f} {avg_n50 - avg_bl_n50:>+7.1f} {avg_fn_sc - np.mean([r['bl_score'] for r in results if r['model']==model]):>+7.1f}")
            out(f"  |n-50| avg: {avg_n50:.1f}")
            
            all_model_stats[model] = {
                "avg_g": avg_fn_g, "avg_n": avg_fn_n, 
                "avg_score": avg_fn_sc, "avg_n50": avg_n50,
                "n_runs": len(model_fn_g),
            }
    
    out(f"\n{'='*105}")
    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(description="CoCoA Quick Result Table")
    parser.add_argument("exp_dir", help="Experiment subdirectory path")
    parser.add_argument("--model", default=None, choices=["llama", "qwen"],
                        help="Filter by model")
    parser.add_argument("--save", action="store_true",
                        help="Save table to exp_dir/result_table.txt")
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Not found: {exp_dir}")
        sys.exit(1)
    
    results = load_results(exp_dir)
    if not results:
        print(f"No results.json found in {exp_dir}")
        sys.exit(1)
    
    print(f"Loaded {len(results)} results from {exp_dir.name}")
    
    table_text = print_table(exp_dir, results, args.model)
    
    if args.save and table_text:
        out_path = exp_dir / "result_table.txt"
        out_path.write_text(table_text, encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()