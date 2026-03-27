#!/usr/bin/env python3
"""
CoCoA Experiment Results Summary

Usage:
    # Specific subdirectory
    python summarize.py experiments/kfold_std
    python summarize.py experiments/kfold_nxm

    # All subdirectories at once
    python summarize.py experiments/ --all

    # List available subdirectories
    python summarize.py experiments/ --list

Outputs:
    {target_dir}/summary.txt   — human-readable summary
    {target_dir}/summary.json  — structured data for analysis
"""

import re
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics
from io import StringIO


def parse_train_log(log_path):
    """Parse a single train.log file."""
    text = log_path.read_text()
    
    result = {"path": str(log_path.parent.name)}
    
    m = re.search(r"Epochs: (\d+)", text)
    result["epochs"] = int(m.group(1)) if m else None
    
    m = re.search(r"Grounded loss: .+?\(w=([\d.]+)\)", text)
    result["w_g"] = float(m.group(1)) if m else None
    
    m = re.search(r"Neutral loss: .+?\(w=([\d.]+)\)", text)
    result["w_n"] = float(m.group(1)) if m else None
    
    baseline_section = re.split(r"\[Epoch|\bstep\b", text)[0]
    g_match = re.findall(r"CBS_g: ([\d.]+)%", baseline_section)
    n_match = re.findall(r"CBS_n: ([\d.]+)%", baseline_section)
    s_match = re.findall(r"Score: ([\d.]+)", baseline_section)
    if g_match and n_match and s_match:
        result["base_g"] = float(g_match[-1])
        result["base_n"] = float(n_match[-1])
        result["base_score"] = float(s_match[-1])
    
    best_scores = re.findall(r"New best! \(score=([\d.]+)\)", text)
    if best_scores:
        result["best_val_score"] = float(best_scores[-1])
    
    val_evals = re.findall(r"\[Val@(\d+)\] CBS_g=([\d.]+)%, CBS_n=([\d.]+)%, score=([\d.]+)", text)
    if val_evals:
        best_val = min(val_evals, key=lambda x: float(x[3]))
        result["best_val_step"] = int(best_val[0])
        result["best_val_g"] = float(best_val[1])
        result["best_val_n"] = float(best_val[2])
    
    final_block = re.search(
        r"\[Final Evaluation\].*?CBS_g: ([\d.]+)%.*?CBS_n: ([\d.]+)%.*?Score: ([\d.]+)",
        text, re.DOTALL
    )
    if final_block:
        result["test_g"] = float(final_block.group(1))
        result["test_n"] = float(final_block.group(2))
        result["test_score"] = float(final_block.group(3))
    
    complete = re.search(
        r"Training Complete!.*?CBS_g: [\d.]+% → ([\d.]+)%.*?CBS_n: [\d.]+% → ([\d.]+)%.*?Score: [\d.]+ → ([\d.]+)",
        text, re.DOTALL
    )
    if complete:
        result["final_g"] = float(complete.group(1))
        result["final_n"] = float(complete.group(2))
        result["final_score"] = float(complete.group(3))
    
    return result


def parse_exp_name(name):
    info = {}
    parts = name.split("_")
    
    if len(parts) >= 4:
        info["culture"] = parts[0]
        info["lang"] = parts[1]
        
        model_parts = []
        i = 2
        while i < len(parts) and not parts[i].startswith(("mse", "npo", "wg", "wn", "tau", "seed")):
            model_parts.append(parts[i])
            i += 1
        info["model"] = "_".join(model_parts)
        
        for p in parts:
            if p.startswith("seed"):
                info["seed"] = p.replace("seed", "")
        
        for p in parts:
            if p.startswith("wg"): info["wg"] = p.replace("wg", "")
            if p.startswith("wn"): info["wn"] = p.replace("wn", "")
            if p.startswith("tau"): info["tau"] = p.replace("tau", "")
            if p.startswith("r") and p[1:].isdigit(): info["r"] = p.replace("r", "")
    
    return info


def get_score(r):
    return r.get("final_score", r.get("test_score", 999))

def get_g(r):
    return r.get("final_g", r.get("test_g", 999))

def get_n(r):
    return r.get("final_n", r.get("test_n", -1))

def ind_g(v):
    return "✓" if v < 10 else "△" if v < 20 else "✗"

def ind_n(v):
    return "✓" if 45 <= v <= 55 else "△" if 40 <= v <= 60 else "✗"


class TeeWriter:
    def __init__(self):
        self.buffer = StringIO()
    
    def print(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.buffer)
    
    def get_text(self):
        return self.buffer.getvalue()


def build_json_output(results, groups, by_culture_model, sorted_keys):
    experiments = []
    for r in results:
        experiments.append({
            "name": r.get("path", "?"),
            "culture": r.get("culture", "?"),
            "lang": r.get("lang", "?"),
            "model": r.get("model", "?"),
            "seed": r.get("seed", "?"),
            "baseline": {
                "cbs_g": r.get("base_g"),
                "cbs_n": r.get("base_n"),
                "score": r.get("base_score"),
            },
            "final": {
                "cbs_g": get_g(r),
                "cbs_n": get_n(r),
                "score": get_score(r),
            },
            "delta": {
                "cbs_g": round(get_g(r) - r.get("base_g", 0), 2) if r.get("base_g") else None,
                "cbs_n": round(get_n(r) - r.get("base_n", 0), 2) if r.get("base_n") else None,
                "score": round(get_score(r) - r.get("base_score", 0), 2) if r.get("base_score") else None,
            },
            "best_val_step": r.get("best_val_step"),
            "config": {
                "epochs": r.get("epochs"),
                "w_g": r.get("wg", r.get("w_g")),
                "w_n": r.get("wn", r.get("w_n")),
            },
        })
    
    grouped = []
    for key, group in groups.items():
        culture, lang, model, wg, wn = key
        scores = [get_score(r) for r in group]
        gs = [get_g(r) for r in group]
        ns = [get_n(r) for r in group]
        n = len(scores)
        grouped.append({
            "setting": f"{culture}_{lang}_{model}_wg{wg}_wn{wn}",
            "culture": culture, "lang": lang, "model": model,
            "n_seeds": n,
            "score": {"mean": round(statistics.mean(scores), 2),
                      "std": round(statistics.stdev(scores), 2) if n > 1 else 0},
            "cbs_g": {"mean": round(statistics.mean(gs), 2),
                      "std": round(statistics.stdev(gs), 2) if n > 1 else 0},
            "cbs_n": {"mean": round(statistics.mean(ns), 2),
                      "std": round(statistics.stdev(ns), 2) if n > 1 else 0},
        })
    grouped.sort(key=lambda x: x["score"]["mean"])
    
    rankings = {}
    for (culture, model) in sorted_keys:
        group = by_culture_model[(culture, model)]
        key = f"{culture}_{model}"
        
        def make_entry(r):
            return {
                "name": r.get("path", "?"),
                "seed": r.get("seed", "?"),
                "lang": r.get("lang", "?"),
                "baseline": {"cbs_g": r.get("base_g"), "cbs_n": r.get("base_n"), "score": r.get("base_score")},
                "final": {"cbs_g": get_g(r), "cbs_n": get_n(r), "score": get_score(r)},
            }
        
        rankings[key] = {
            "n_runs": len(group),
            "top3_score": [make_entry(r) for r in sorted(group, key=lambda x: get_score(x))[:3]],
            "top3_cbs_g": [make_entry(r) for r in sorted(group, key=lambda x: get_g(x))[:3]],
            "top3_cbs_n": [make_entry(r) for r in sorted(group, key=lambda x: abs(get_n(x) - 50))[:3]],
        }
    
    return {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": len(results),
        "experiments": experiments,
        "grouped_by_setting": grouped,
        "rankings_by_culture_model": rankings,
    }


def summarize_dir(exp_dir):
    """Run summary on a single directory."""
    logs = sorted(exp_dir.rglob("train.log"))
    
    if not logs:
        print(f"  No train.log files in {exp_dir}")
        return None
    
    results = []
    for log_path in logs:
        try:
            r = parse_train_log(log_path)
            r.update(parse_exp_name(log_path.parent.name))
            results.append(r)
        except Exception as e:
            print(f"  Error parsing {log_path}: {e}")
    
    results.sort(key=lambda x: get_score(x))
    
    groups = defaultdict(list)
    for r in results:
        key = (r.get("culture", "?"), r.get("lang", "?"), r.get("model", "?"),
               r.get("wg", "?"), r.get("wn", "?"))
        groups[key].append(r)
    
    by_culture_model = defaultdict(list)
    for r in results:
        key = (r.get("culture", "?"), r.get("model", "?"))
        by_culture_model[key].append(r)
    
    culture_order = ["ko", "ja", "zh", "hi", "ml", "mr", "gu", "vi", "ur", "ar"]
    sorted_keys = sorted(
        by_culture_model.keys(),
        key=lambda k: (
            culture_order.index(k[0]) if k[0] in culture_order else 999, k[1]
        )
    )
    
    out = TeeWriter()
    W = 140
    
    out.print("=" * W)
    out.print(f"CoCoA Experiment Results Summary — {exp_dir.name}")
    out.print(f"Directory: {exp_dir}")
    out.print(f"Total experiments: {len(results)}")
    out.print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.print("=" * W)
    
    out.print(f"\n{'Experiment':<60} {'Base':>10} {'Final':>10} {'':>6} {'CBS_g':>12} {'CBS_n':>12} {'Step':>6}")
    out.print(f"{'':60} {'Score':>10} {'Score':>10} {'Δ':>6} {'(→0%)':>12} {'(→50%)':>12} {'best':>6}")
    out.print("-" * W)
    
    for r in results:
        name = r.get("path", "?")
        if len(name) > 58:
            name = name[:55] + "..."
        base_s = r.get("base_score", 0)
        final_s = get_score(r)
        delta = final_s - base_s
        final_g = get_g(r)
        final_n = get_n(r)
        best_step = r.get("best_val_step", -1)
        out.print(f"{name:<60} {base_s:>10.1f} {final_s:>10.1f} {delta:>+6.1f} "
                  f"{final_g:>7.1f}% {ind_g(final_g):>3} {final_n:>7.1f}% {ind_n(final_n):>3} {best_step:>6}")
    
    out.print(f"\n{'=' * W}")
    out.print("Grouped by Setting (averaged over seeds/folds)")
    out.print("=" * W)
    
    out.print(f"\n{'Setting':<50} {'N':>5} {'Score':>12} {'CBS_g':>12} {'CBS_n':>12}")
    out.print(f"{'':50} {'':>5} {'mean±std':>12} {'mean±std':>12} {'mean±std':>12}")
    out.print("-" * 100)
    
    group_avgs = []
    for key, group in groups.items():
        culture, lang, model, wg, wn = key
        scores = [get_score(r) for r in group]
        gs = [get_g(r) for r in group]
        ns = [get_n(r) for r in group]
        n = len(scores)
        s_mean, g_mean, n_mean = statistics.mean(scores), statistics.mean(gs), statistics.mean(ns)
        s_std = statistics.stdev(scores) if n > 1 else 0
        g_std = statistics.stdev(gs) if n > 1 else 0
        n_std = statistics.stdev(ns) if n > 1 else 0
        group_avgs.append((s_mean, culture, lang, model, wg, wn, n, s_mean, s_std, g_mean, g_std, n_mean, n_std))
    
    group_avgs.sort(key=lambda x: x[0])
    
    for _, culture, lang, model, wg, wn, n, s_mean, s_std, g_mean, g_std, n_mean, n_std in group_avgs:
        label = f"{culture}_{lang}_{model}_wg{wg}_wn{wn}"
        if len(label) > 48:
            label = label[:45] + "..."
        out.print(f"{label:<50} {n:>5} {s_mean:>6.1f}±{s_std:<4.1f} {g_mean:>6.1f}±{g_std:<4.1f} {n_mean:>6.1f}±{n_std:<4.1f}")
    
    out.print(f"\n{'=' * W}")
    out.print("Per Culture × Model: Top 3 for Score / CBS_g / CBS_n")
    out.print("=" * W)
    
    for (culture, model) in sorted_keys:
        group = by_culture_model[(culture, model)]
        langs = sorted(set(r.get("lang", "?") for r in group))
        
        out.print(f"\n{'━' * W}")
        out.print(f"  [{culture.upper()}] {model}  ({len(group)} runs, lang={','.join(langs)})")
        out.print(f"{'━' * W}")
        
        sorted_by_score = sorted(group, key=lambda x: get_score(x))
        out.print(f"\n    📊 Final Score Top 3 (→ 0)")
        out.print(f"    {'#':<4} {'Experiment':<72} {'Score (base→final)':>20} {'CBS_g (base→final)':>22} {'CBS_n (base→final)':>22}")
        out.print(f"    {'─' * (W - 8)}")
        for i, r in enumerate(sorted_by_score[:3]):
            name = r.get("path", "?")
            if len(name) > 70: name = name[:67] + "..."
            base_s = r.get("base_score", 0)
            base_g = r.get("base_g", -1)
            base_n = r.get("base_n", -1)
            out.print(f"    {i+1:<4} {name:<72} {base_s:>6.1f} → {get_score(r):<6.1f}"
                      f"     {base_g:>5.1f}% → {get_g(r):<5.1f}% {ind_g(get_g(r))}"
                      f"    {base_n:>5.1f}% → {get_n(r):<5.1f}% {ind_n(get_n(r))}")
        
        sorted_by_g = sorted(group, key=lambda x: get_g(x))
        out.print(f"\n    🎯 CBS_g Top 3 (→ 0%)")
        out.print(f"    {'#':<4} {'Experiment':<72} {'CBS_g (base→final)':>22} {'CBS_n (base→final)':>22} {'Score (base→final)':>20}")
        out.print(f"    {'─' * (W - 8)}")
        for i, r in enumerate(sorted_by_g[:3]):
            name = r.get("path", "?")
            if len(name) > 70: name = name[:67] + "..."
            base_s = r.get("base_score", 0)
            base_g = r.get("base_g", -1)
            base_n = r.get("base_n", -1)
            out.print(f"    {i+1:<4} {name:<72} {base_g:>5.1f}% → {get_g(r):<5.1f}% {ind_g(get_g(r))}"
                      f"    {base_n:>5.1f}% → {get_n(r):<5.1f}% {ind_n(get_n(r))}"
                      f"   {base_s:>6.1f} → {get_score(r):<6.1f}")
        
        sorted_by_n = sorted(group, key=lambda x: abs(get_n(x) - 50))
        out.print(f"\n    ⚖️  CBS_n Top 3 (→ 50%)")
        out.print(f"    {'#':<4} {'Experiment':<72} {'CBS_n (base→final)':>22} {'CBS_g (base→final)':>22} {'Score (base→final)':>20}")
        out.print(f"    {'─' * (W - 8)}")
        for i, r in enumerate(sorted_by_n[:3]):
            name = r.get("path", "?")
            if len(name) > 70: name = name[:67] + "..."
            base_s = r.get("base_score", 0)
            base_g = r.get("base_g", -1)
            base_n = r.get("base_n", -1)
            out.print(f"    {i+1:<4} {name:<72} {base_n:>5.1f}% → {get_n(r):<5.1f}% {ind_n(get_n(r))}"
                      f"    {base_g:>5.1f}% → {get_g(r):<5.1f}% {ind_g(get_g(r))}"
                      f"   {base_s:>6.1f} → {get_score(r):<6.1f}")
    
    # Save
    txt_path = exp_dir / "summary.txt"
    txt_path.write_text(out.get_text(), encoding="utf-8")
    
    json_data = build_json_output(results, groups, by_culture_model, sorted_keys)
    json_path = exp_dir / "summary.json"
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"\n  📁 Saved: {txt_path}")
    print(f"  📁 Saved: {json_path}")
    
    return json_data


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python summarize.py experiments/kfold_nxm       # specific subdir")
        print("  python summarize.py experiments/ --all           # all subdirs")
        print("  python summarize.py experiments/ --list          # list subdirs")
        sys.exit(1)
    
    exp_dir = Path(sys.argv[1])
    
    if not exp_dir.exists():
        print(f"Directory not found: {exp_dir}")
        sys.exit(1)
    
    # --list: show available subdirectories
    if "--list" in sys.argv:
        print(f"Subdirectories in {exp_dir}:")
        for d in sorted(exp_dir.iterdir()):
            if d.is_dir():
                n_logs = len(list(d.rglob("train.log")))
                if n_logs > 0:
                    print(f"  {d.name:<30} ({n_logs} experiments)")
        sys.exit(0)
    
    # --all: summarize each subdirectory separately
    if "--all" in sys.argv:
        subdirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
        for subdir in subdirs:
            n_logs = len(list(subdir.rglob("train.log")))
            if n_logs == 0:
                continue
            print(f"\n{'='*80}")
            print(f"  Summarizing: {subdir.name} ({n_logs} experiments)")
            print(f"{'='*80}")
            summarize_dir(subdir)
        print(f"\nDone! Summarized {len(subdirs)} subdirectories.")
    else:
        # Single directory
        summarize_dir(exp_dir)


if __name__ == "__main__":
    main()