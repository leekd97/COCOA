#!/usr/bin/env python3
"""
Main Table Generator — COCOA + BiasEdit + BiasUnlearn

Usage:
    python make_main_table.py \
        --cocoa experiments/summary.json \
        --biasedit baselines/results/biasedit \
        --biasunlearn baselines/results/biasunlearn
"""

import argparse
import json
from pathlib import Path

CULTURES_ORDER = ["ko", "ja", "zh", "hi", "mr", "ml", "gu", "vi", "ur", "ar"]

CULTURE_NORMALIZE = {
    "korean": "ko", "japanese": "ja", "chinese": "zh",
    "hindi": "hi", "vietnamese": "vi", "urdu": "ur",
    "gujarati": "gu", "marathi": "mr", "malayalam": "ml",
    "arabic": "ar", "arab": "ar",
}

MODEL_NORMALIZE = {
    "llama3_8b": "llama", "llama3-8b": "llama",
    "meta-llama/llama-3.1-8b": "llama",
    "qwen3_8b": "qwen", "qwen3-8b": "qwen",
    "qwen/qwen3-8b": "qwen",
}

MODEL_DISPLAY = {"llama": "Llama", "qwen": "Qwen"}
MODELS_ORDER = ["llama", "qwen"]


def norm_culture(c):
    c = c.lower().strip()
    return CULTURE_NORMALIZE.get(c, c)


def norm_model(m):
    m = m.lower().strip()
    return MODEL_NORMALIZE.get(m, "llama" if "llama" in m else "qwen" if "qwen" in m else m)


def load_cocoa(path):
    with open(path) as f:
        data = json.load(f)

    best = {}
    for e in data["experiments"]:
        if e.get("lang") == "en":
            continue
        c = norm_culture(e["culture"])
        m = norm_model(e.get("model", ""))
        key = (c, m)
        score = e["final"]["score"]
        if key not in best or score < best[key]["final"]["score"]:
            best[key] = e

    results = {}
    for key, e in best.items():
        results[key] = {
            "baseline": e["baseline"],
            "result": {"cbs_g": e["final"]["cbs_g"], "cbs_n": e["final"]["cbs_n"], "score": e["final"]["score"]},
            "seed": e.get("seed"),
        }
    return results


def load_baseline_results(base_dir):
    """Load baseline results.json files. Structure: config.culture/model, baseline{}, trained{}."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return {}

    best = {}
    for rfile in base_dir.rglob("results.json"):
        try:
            with open(rfile) as f:
                d = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        cfg = d.get("config", {})
        c = norm_culture(cfg.get("culture", ""))
        m = norm_model(cfg.get("model", cfg.get("model_full", "")))
        if not c:
            continue

        trained = d.get("trained", {})
        cbs_g = trained.get("cbs_g")
        cbs_n = trained.get("cbs_n")
        if cbs_g is None or cbs_n is None:
            continue

        score = trained.get("score", abs(cbs_g) + abs(cbs_n - 50))
        key = (c, m)

        if key not in best or score < best[key]["result"]["score"]:
            best[key] = {
                "baseline": d.get("baseline", {}),
                "result": {"cbs_g": cbs_g, "cbs_n": cbs_n, "score": score},
                "seed": cfg.get("seed"),
            }
    return best


def render_table(cocoa, biasedit, biasunlearn, output_path=None):
    all_keys = set(cocoa) | set(biasedit) | set(biasunlearn)
    cultures_present = sorted(
        {c for c, m in all_keys},
        key=lambda c: CULTURES_ORDER.index(c) if c in CULTURES_ORDER else 99
    )

    sep = "─" * 108
    lines = []
    lines.append("=" * 108)
    lines.append("  Main Table: Cultural Bias Mitigation  (CBS_g → 0%,  CBS_n → 50%,  Score = |CBS_g| + |CBS_n - 50|)")
    lines.append("=" * 108)
    lines.append("")
    lines.append(f"  {'':>5} {'':>6} │ {'Base Model':^20} │ {'BiasEdit':^20} │ {'BiasUnlearn':^20} │ {'COCOA (Ours)':^20}")
    lines.append(f"  {'Cult':>5} {'Model':>6} │ {'CBS_g':>6} {'CBS_n':>6} {'Score':>6} │ {'CBS_g':>6} {'CBS_n':>6} {'Score':>6} │ {'CBS_g':>6} {'CBS_n':>6} {'Score':>6} │ {'CBS_g':>6} {'CBS_n':>6} {'Score':>6}")
    lines.append(f"  {sep}")

    json_rows = []

    for culture in cultures_present:
        for model in MODELS_ORDER:
            key = (culture, model)
            mdisp = MODEL_DISPLAY.get(model, model)

            bl = None
            for src in [cocoa, biasedit, biasunlearn]:
                if key in src and src[key].get("baseline"):
                    bl = src[key]["baseline"]
                    break

            be = biasedit.get(key, {}).get("result")
            bu = biasunlearn.get(key, {}).get("result")
            co = cocoa.get(key, {}).get("result")

            if bl is None and be is None and bu is None and co is None:
                continue

            def fmt(val):
                return f"{val:6.1f}" if val is not None else f"{'—':>6}"

            def get(d, k):
                return d.get(k) if d else None

            methods = {}
            if be and get(be, "score") is not None: methods["biasedit"] = be["score"]
            if bu and get(bu, "score") is not None: methods["biasunlearn"] = bu["score"]
            if co and get(co, "score") is not None: methods["cocoa"] = co["score"]
            best_m = min(methods, key=methods.get) if methods else None

            def sfmt(mname, d):
                s = fmt(get(d, "score"))
                return s + "*" if best_m == mname else s + " "

            line = (
                f"  {culture:>5} {mdisp:>6} │"
                f" {fmt(get(bl,'cbs_g'))} {fmt(get(bl,'cbs_n'))} {fmt(get(bl,'score'))}  │"
                f" {fmt(get(be,'cbs_g'))} {fmt(get(be,'cbs_n'))} {sfmt('biasedit',be)} │"
                f" {fmt(get(bu,'cbs_g'))} {fmt(get(bu,'cbs_n'))} {sfmt('biasunlearn',bu)} │"
                f" {fmt(get(co,'cbs_g'))} {fmt(get(co,'cbs_n'))} {sfmt('cocoa',co)}"
            )
            lines.append(line)

            json_rows.append({
                "culture": culture, "model": model,
                "baseline": bl, "biasedit": be, "biasunlearn": bu, "cocoa": co,
                "best_method": best_m,
                "seeds": {
                    "biasedit": biasedit.get(key, {}).get("seed"),
                    "biasunlearn": biasunlearn.get(key, {}).get("seed"),
                    "cocoa": cocoa.get(key, {}).get("seed"),
                },
            })
        lines.append("")

    cw = sum(1 for r in json_rows if r["best_method"] == "cocoa")
    bew = sum(1 for r in json_rows if r["best_method"] == "biasedit")
    buw = sum(1 for r in json_rows if r["best_method"] == "biasunlearn")
    tot = len([r for r in json_rows if r.get("best_method")])
    lines.append(f"  * = best    Wins: COCOA {cw}/{tot}  BiasEdit {bew}/{tot}  BiasUnlearn {buw}/{tot}")

    table_str = "\n".join(lines)
    print(table_str)

    if output_path:
        with open(f"{output_path}.txt", "w") as f:
            f.write(table_str)
        with open(f"{output_path}.json", "w") as f:
            json.dump({"rows": json_rows}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {output_path}.txt / {output_path}.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cocoa", default="experiments/summary.json")
    p.add_argument("--biasedit", default="baselines/results/biasedit")
    p.add_argument("--biasunlearn", default="baselines/results/biasunlearn")
    p.add_argument("--output", default="main_table")
    args = p.parse_args()

    print("Loading COCOA...")
    cocoa = load_cocoa(args.cocoa)
    print(f"  {len(cocoa)} culture*model entries")

    print("Loading BiasEdit...")
    biasedit = load_baseline_results(args.biasedit)
    print(f"  {len(biasedit)} culture*model entries")

    print("Loading BiasUnlearn...")
    biasunlearn = load_baseline_results(args.biasunlearn)
    print(f"  {len(biasunlearn)} culture*model entries")
    print()

    render_table(cocoa, biasedit, biasunlearn, args.output)


if __name__ == "__main__":
    main()