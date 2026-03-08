#!/usr/bin/env python3
"""
PCGrad Ablation Comparison

Compares Weighted Sum vs PCGrad vs Goal-Aware PCGrad across all cultures.
Reads results from experiment folders.

Usage:
    python analysis/compare_pcgrad_ablation.py \
        --exp_dir experiments \
        --summary experiments/summary.json \
        --output analysis/ablation
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

CULTURES_ORDER = ["ko", "ja", "zh", "ar", "hi", "mr", "ml", "gu", "ur", "vi"]
METHODS_ORDER = ["weighted", "pcgrad", "goal_aware_pcgrad"]
METHOD_DISPLAY = {
    "weighted": "Weighted",
    "pcgrad": "PCGrad",
    "goal_aware_pcgrad": "GA-PCGrad",
}
MODEL_DISPLAY = {"llama": "Llama", "qwen": "Qwen"}

# Folder naming: goal_aware has no suffix, others have _weighted / _pcgrad
METHOD_SUFFIX = {
    "weighted": "_weighted",
    "pcgrad": "_pcgrad",
    "goal_aware_pcgrad": "",
}

MODEL_FOLDER_MAP = {
    "llama": "llama3-8b",
    "qwen": "qwen3-8b",
}


def find_experiment_folder(exp_dir, culture, model, seed, method):
    """Find the experiment folder for a given config."""
    model_folder = MODEL_FOLDER_MAP.get(model, model)
    suffix = METHOD_SUFFIX[method]

    # Pattern: {culture}_cu_{model}_mse_wg1.0_wn2.0_tau1.0_r16{suffix}_seed{seed}
    pattern = f"{culture}_cu_{model_folder}_mse_wg1.0_wn2.0_tau1.0_r16{suffix}_seed{seed}"
    folder = exp_dir / pattern

    if folder.exists():
        return folder
    return None


def load_result(folder):
    """Load results.json from an experiment folder."""
    rfile = folder / "results.json"
    if not rfile.exists():
        return None

    with open(rfile) as f:
        d = json.load(f)

    final = d.get("final", {})
    return {
        "cbs_g": final.get("cbs_grounded"),
        "cbs_n": final.get("cbs_neutral"),
        "score": final.get("combined_score"),
    }


def load_goal_aware_from_summary(summary_path):
    """Load goal_aware_pcgrad results from summary.json as fallback."""
    if not Path(summary_path).exists():
        return {}

    with open(summary_path) as f:
        data = json.load(f)

    results = {}
    for e in data["experiments"]:
        if e.get("lang") == "en":
            continue
        culture = e["culture"]
        model_raw = e.get("model", "")
        model = "llama" if "llama" in model_raw.lower() else "qwen" if "qwen" in model_raw.lower() else model_raw
        seed = e.get("seed")
        key = (culture, model)

        score = e["final"]["score"]
        if key not in results or score < results[key]["score"]:
            results[key] = {
                "cbs_g": e["final"]["cbs_g"],
                "cbs_n": e["final"]["cbs_n"],
                "score": score,
                "seed": seed,
            }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", default="experiments")
    p.add_argument("--summary", default="experiments/summary.json")
    p.add_argument("--output", default="analysis/ablation")
    args = p.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best seeds per culture×model (from main experiments)
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

    # Load goal_aware results from summary as fallback
    ga_summary = load_goal_aware_from_summary(args.summary)

    # Collect all results
    all_data = {}  # (culture, model, method) -> {cbs_g, cbs_n, score}

    for (culture, model), seed in BEST_SEEDS.items():
        for method in METHODS_ORDER:
            folder = find_experiment_folder(exp_dir, culture, model, seed, method)

            result = None
            if folder:
                result = load_result(folder)

            # Fallback for goal_aware from summary.json
            if result is None and method == "goal_aware_pcgrad":
                ga = ga_summary.get((culture, model))
                if ga:
                    result = {"cbs_g": ga["cbs_g"], "cbs_n": ga["cbs_n"], "score": ga["score"]}

            if result:
                all_data[(culture, model, method)] = result

    # =========================================================================
    # Print comparison table (per model)
    # =========================================================================
    json_rows = []

    for model in ["llama", "qwen"]:
        mdisp = MODEL_DISPLAY[model]
        print(f"\n{'='*95}")
        print(f"  PCGrad Ablation — {mdisp}")
        print(f"{'='*95}")
        print(f"  {'':>5} │ {'Weighted':^22} │ {'PCGrad':^22} │ {'GA-PCGrad (Ours)':^22}")
        print(f"  {'Cult':>5} │ {'CBS_g':>6} {'CBS_n':>6} {'Score':>7} │ {'CBS_g':>6} {'CBS_n':>6} {'Score':>7} │ {'CBS_g':>6} {'CBS_n':>6} {'Score':>7}")
        print(f"  {'─'*90}")

        method_scores = defaultdict(list)

        for culture in CULTURES_ORDER:
            parts = []
            row = {"culture": culture, "model": model}

            for method in METHODS_ORDER:
                r = all_data.get((culture, model, method))
                if r:
                    cg = r["cbs_g"]
                    cn = r["cbs_n"]
                    sc = r["score"]
                    parts.append(f" {cg:6.1f} {cn:6.1f} {sc:7.1f}")
                    method_scores[method].append(sc)
                    row[method] = r
                else:
                    parts.append(f" {'—':>6} {'—':>6} {'—':>7}")
                    row[method] = None

            # Mark best
            scores = {}
            for method in METHODS_ORDER:
                r = all_data.get((culture, model, method))
                if r:
                    scores[method] = r["score"]
            best = min(scores, key=scores.get) if scores else None
            row["best"] = best

            # Format with marker
            formatted = []
            for i, method in enumerate(METHODS_ORDER):
                s = parts[i]
                if method == best:
                    formatted.append(s + "*")
                else:
                    formatted.append(s + " ")

            line = f"  {culture:>5} │{'│'.join(formatted)}"
            print(line)

            json_rows.append(row)

        # Averages
        print(f"  {'─'*90}")
        avg_line_parts = []
        for method in METHODS_ORDER:
            scores = method_scores.get(method, [])
            if scores:
                avg_line_parts.append(f" {'':>6} {'':>6} {np.mean(scores):7.1f}")
            else:
                avg_line_parts.append(f" {'':>6} {'':>6} {'—':>7}")
        print(f"  {'Avg':>5} │{'│'.join(avg_line_parts)}")

        # Win count
        wins = defaultdict(int)
        for row in json_rows:
            if row["model"] == model and row.get("best"):
                wins[row["best"]] += 1
        total = sum(wins.values())
        win_str = "  ".join(f"{METHOD_DISPLAY[m]}={wins.get(m,0)}" for m in METHODS_ORDER)
        print(f"  * = best    Wins ({total}): {win_str}")

    # =========================================================================
    # Save JSON
    # =========================================================================
    out_json = output_dir / "pcgrad_ablation.json"
    with open(out_json, "w") as f:
        json.dump({"rows": json_rows}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_json}")

    # =========================================================================
    # LaTeX table
    # =========================================================================
    print("\n\n% === LaTeX Table ===")
    print(r"\begin{table*}[t]")
    print(r"\centering\small")
    print(r"\caption{Gradient method ablation. Score = $|\text{CBS}_g| + |\text{CBS}_n - 50|$ (lower is better). Best result per row in \textbf{bold}.}")
    print(r"\label{tab:pcgrad_ablation}")
    print(r"\begin{tabular}{ll rrr rrr rrr}")
    print(r"\toprule")
    print(r"& & \multicolumn{3}{c}{Weighted Sum} & \multicolumn{3}{c}{PCGrad} & \multicolumn{3}{c}{GA-PCGrad (Ours)} \\")
    print(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}")
    print(r"Culture & Model & CBS$_g$ & CBS$_n$ & Score & CBS$_g$ & CBS$_n$ & Score & CBS$_g$ & CBS$_n$ & Score \\")
    print(r"\midrule")

    for row in json_rows:
        culture = row["culture"].upper()
        model = MODEL_DISPLAY[row["model"]]
        best = row.get("best")

        cells = []
        for method in METHODS_ORDER:
            r = row.get(method)
            if r:
                sc_str = f"\\textbf{{{r['score']:.1f}}}" if method == best else f"{r['score']:.1f}"
                cells.append(f"{r['cbs_g']:.1f} & {r['cbs_n']:.1f} & {sc_str}")
            else:
                cells.append("— & — & —")

        print(f"{culture} & {model} & {' & '.join(cells)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


if __name__ == "__main__":
    main()