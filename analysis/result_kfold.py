"""
CoCoA K-Fold Results Aggregation

Scans experiments/ for fold results, computes mean ± SE across folds.

Usage:
    python analysis/aggregate_kfold.py --exp_dir ./experiments
    python analysis/aggregate_kfold.py --exp_dir ./experiments --seed 42
    python analysis/aggregate_kfold.py --exp_dir ./experiments --model llama
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


CULTURES_ORDER = ["ko", "ja", "zh", "ar", "hi", "mr", "ml", "gu", "ur", "vi"]
MODEL_NORM = {"llama3-8b": "llama", "qwen3-8b": "qwen"}


def parse_fold_folder(name: str) -> dict:
    """Parse experiment folder name for fold runs."""
    if "fold" not in name:
        return None
    if name.endswith(".log") or "ablation" in name:
        return None

    fold_match = re.search(r"fold(\d+)", name)
    seed_match = re.search(r"seed(\d+)", name)
    if not fold_match or not seed_match:
        return None

    # Extract culture and lang
    if "_cu_" in name:
        lang = "cu"
    elif "_en_" in name:
        lang = "en"
    else:
        return None

    culture = name.split(f"_{lang}_")[0]

    # Extract model
    model = None
    for m in ["llama3-8b", "qwen3-8b"]:
        if m in name:
            model = MODEL_NORM[m]
            break
    if not model:
        return None

    return {
        "name": name,
        "culture": culture,
        "lang": lang,
        "model": model,
        "fold": int(fold_match.group(1)),
        "seed": seed_match.group(1),
    }


def load_results(exp_dir: Path, folder: str) -> dict:
    rfile = exp_dir / folder / "results.json"
    if not rfile.exists():
        return None
    with open(rfile) as f:
        return json.load(f)


def extract_scores(results: dict) -> dict:
    baseline = results.get("baseline") or results.get("baseline_test")
    final = results.get("final")
    if not baseline or not final:
        return None

    def get(d):
        g = d.get("cbs_grounded", d.get("cbs_g"))
        n = d.get("cbs_neutral", d.get("cbs_n"))
        if g is None or n is None:
            return None
        return {"cbs_g": g, "cbs_n": n, "score": g + abs(n - 50)}

    bl = get(baseline)
    fn = get(final)
    if not bl or not fn:
        return None

    return {
        "baseline": bl,
        "final": fn,
        "delta": {
            "cbs_g": fn["cbs_g"] - bl["cbs_g"],
            "cbs_n": fn["cbs_n"] - bl["cbs_n"],
            "score": fn["score"] - bl["score"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="./experiments")
    parser.add_argument("--seed", default=None, help="Filter by seed")
    parser.add_argument("--model", default=None, help="llama or qwen")
    parser.add_argument("--lang", default="cu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    # Scan
    all_data = defaultdict(list)  # (culture, lang, model, seed) -> [fold_results]
    fold_count = 0

    for folder in sorted(exp_dir.iterdir()):
        if not folder.is_dir():
            continue
        parsed = parse_fold_folder(folder.name)
        if not parsed:
            continue
        if args.seed and parsed["seed"] != args.seed:
            continue
        if args.model and parsed["model"] != args.model:
            continue
        if args.lang and parsed["lang"] != args.lang:
            continue

        results = load_results(exp_dir, folder.name)
        if not results:
            continue
        scores = extract_scores(results)
        if not scores:
            continue

        key = (parsed["culture"], parsed["lang"], parsed["model"], parsed["seed"])
        all_data[key].append({"fold": parsed["fold"], **scores})
        fold_count += 1

    print(f"Loaded {fold_count} fold results")

    # Group by (culture, lang, model, seed)
    models = sorted(set(k[2] for k in all_data.keys()))

    for model in models:
        print(f"\n{'='*110}")
        print(f"  K-Fold Results — {model.upper()} (mean ± SE across folds)")
        print(f"{'='*110}")

        # Detailed per-fold table
        print(f"\n  Per-Fold Detail:")
        print(f"  {'Cult':<6} {'Seed':<6} {'Fold':<6} │ {'BL_g':>7} {'BL_n':>7} │ {'Fn_g':>7} {'Fn_n':>7} {'Fn_Sc':>7} │ {'Δg':>7} {'Δn':>7} {'ΔSc':>7}")
        print(f"  {'─'*88}")

        for key in sorted(all_data.keys()):
            culture, lang, m, seed = key
            if m != model:
                continue
            folds = sorted(all_data[key], key=lambda x: x["fold"])
            for f in folds:
                print(f"  {culture.upper():<6} {seed:<6} {f['fold']:<6} │ "
                      f"{f['baseline']['cbs_g']:>7.1f} {f['baseline']['cbs_n']:>7.1f} │ "
                      f"{f['final']['cbs_g']:>7.1f} {f['final']['cbs_n']:>7.1f} {f['final']['score']:>7.1f} │ "
                      f"{f['delta']['cbs_g']:>+7.1f} {f['delta']['cbs_n']:>+7.1f} {f['delta']['score']:>+7.1f}")

        # Summary: mean ± SE per (culture, seed)
        print(f"\n{'─'*110}")
        print(f"  Summary (mean ± SE):")
        print(f"{'─'*110}")
        print(f"  {'Cult':<6} {'Seed':<6} {'K':>3} │ {'BL CBS_g':>12} {'BL CBS_n':>12} │ "
              f"{'Fn CBS_g':>12} {'Fn CBS_n':>12} {'Fn Score':>12} │ {'Δ Score':>12}")
        print(f"  {'─'*96}")

        summary_rows = []
        for key in sorted(all_data.keys()):
            culture, lang, m, seed = key
            if m != model:
                continue
            folds = all_data[key]
            K = len(folds)

            bl_g = [f["baseline"]["cbs_g"] for f in folds]
            bl_n = [f["baseline"]["cbs_n"] for f in folds]
            fn_g = [f["final"]["cbs_g"] for f in folds]
            fn_n = [f["final"]["cbs_n"] for f in folds]
            fn_sc = [f["final"]["score"] for f in folds]
            d_sc = [f["delta"]["score"] for f in folds]

            def fmt(vals):
                m = np.mean(vals)
                se = np.std(vals) / np.sqrt(len(vals))
                return f"{m:>5.1f}±{se:>4.1f}"

            print(f"  {culture.upper():<6} {seed:<6} {K:>3} │ "
                  f"{fmt(bl_g):>12} {fmt(bl_n):>12} │ "
                  f"{fmt(fn_g):>12} {fmt(fn_n):>12} {fmt(fn_sc):>12} │ "
                  f"{fmt(d_sc):>12}")

            summary_rows.append({
                "culture": culture, "lang": lang, "model": model, "seed": seed, "K": K,
                "baseline_cbs_g": f"{np.mean(bl_g):.1f}±{np.std(bl_g)/np.sqrt(K):.1f}",
                "baseline_cbs_n": f"{np.mean(bl_n):.1f}±{np.std(bl_n)/np.sqrt(K):.1f}",
                "final_cbs_g": f"{np.mean(fn_g):.1f}±{np.std(fn_g)/np.sqrt(K):.1f}",
                "final_cbs_n": f"{np.mean(fn_n):.1f}±{np.std(fn_n)/np.sqrt(K):.1f}",
                "final_score": f"{np.mean(fn_sc):.1f}±{np.std(fn_sc)/np.sqrt(K):.1f}",
                "delta_score": f"{np.mean(d_sc):.1f}±{np.std(d_sc)/np.sqrt(K):.1f}",
                "raw_final_g": [round(v, 2) for v in fn_g],
                "raw_final_n": [round(v, 2) for v in fn_n],
            })

        # Overall average across cultures
        print(f"  {'─'*96}")
        all_fn_g = [f["final"]["cbs_g"] for key in all_data for f in all_data[key] if key[2] == model]
        all_fn_n = [f["final"]["cbs_n"] for key in all_data for f in all_data[key] if key[2] == model]
        all_fn_sc = [f["final"]["score"] for key in all_data for f in all_data[key] if key[2] == model]
        all_d_sc = [f["delta"]["score"] for key in all_data for f in all_data[key] if key[2] == model]

        if all_fn_g:
            n = len(all_fn_g)
            print(f"  {'ALL':<6} {'—':<6} {n:>3} │ "
                  f"{'':>12} {'':>12} │ "
                  f"{np.mean(all_fn_g):>5.1f}±{np.std(all_fn_g)/np.sqrt(n):>4.1f}  "
                  f"{np.mean(all_fn_n):>5.1f}±{np.std(all_fn_n)/np.sqrt(n):>4.1f}  "
                  f"{np.mean(all_fn_sc):>5.1f}±{np.std(all_fn_sc)/np.sqrt(n):>4.1f}  │ "
                  f"{np.mean(all_d_sc):>5.1f}±{np.std(all_d_sc)/np.sqrt(n):>4.1f}")

        # Save
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"model": model, "summary": summary_rows}, f, indent=2)
            print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()