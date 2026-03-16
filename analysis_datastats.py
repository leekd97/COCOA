"""
COCOA Analysis Phase 1: Data Statistics (No GPU Required)

Analyzes:
  1. Per-culture × per-category: Asian/Western entity counts (raw + post-split)
  2. PairedDataset entity frequency distribution (who gets paired how often)
  3. min(N_asian, N_western) vs pairs_per_category cap analysis
  4. N×M evaluation imbalance

Usage:
    # All cultures, cu language
    python analysis_phase1_data_stats.py --lang cu

    # Specific cultures
    python analysis_phase1_data_stats.py --cultures ko zh ja ar --lang cu

    # Both languages
    python analysis_phase1_data_stats.py --lang cu --output_dir ./analysis/phase1
    python analysis_phase1_data_stats.py --lang en --output_dir ./analysis/phase1
"""

import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.data import (
    load_camellia_data, split_data, build_category_data,
    CamelliaData, CategoryData, ENTITY_TYPE_ALIASES,
)


# =========================================================================
# 1. Raw Entity Count Analysis
# =========================================================================

def analyze_raw_entities(data: CamelliaData, culture: str, lang: str) -> Dict:
    """Analyze raw entity counts before splitting."""
    results = {}
    for etype, ent_dict in data.entities.items():
        n_asian = len(ent_dict["asian"])
        n_western = len(ent_dict["western"])
        ratio = n_asian / n_western if n_western > 0 else float("inf")
        results[etype] = {
            "n_asian": n_asian,
            "n_western": n_western,
            "ratio_a_w": round(ratio, 2),
            "diff": n_asian - n_western,
        }
    
    total_a = sum(v["n_asian"] for v in results.values())
    total_w = sum(v["n_western"] for v in results.values())
    results["_total"] = {
        "n_asian": total_a,
        "n_western": total_w,
        "ratio_a_w": round(total_a / total_w, 2) if total_w > 0 else 0,
        "diff": total_a - total_w,
    }
    return results


# =========================================================================
# 2. Post-Split Entity Count Analysis
# =========================================================================

def analyze_split_entities(split_info: Dict) -> Dict:
    """Analyze entity counts after train/val/test split."""
    results = {}
    for split_name in ["train", "val", "test"]:
        entities = split_info[f"{split_name}_entities"]
        split_data = {}
        for etype, ent_dict in entities.items():
            n_a = len(ent_dict["asian"])
            n_w = len(ent_dict["western"])
            split_data[etype] = {
                "n_asian": n_a,
                "n_western": n_w,
                "min_nw": min(n_a, n_w),
                "ratio": round(n_a / n_w, 2) if n_w > 0 else float("inf"),
            }
        results[split_name] = split_data
    return results


# =========================================================================
# 3. Pairing Frequency Analysis (core of feedback #2)
# =========================================================================

def analyze_pairing_frequency(
    category_data: Dict[str, CategoryData],
    pairs_per_category: int = 200,
    seed: int = 42,
) -> Dict:
    """
    Simulate PairedDataset pair generation and measure entity frequency.
    Mirrors PairedDataset.__init__ exactly.
    """
    rng = random.Random(seed)
    
    all_results = {}
    
    for cat_key, cdata in category_data.items():
        n_asian = len(cdata.asian_entities)
        n_western = len(cdata.western_entities)
        n_pairs = min(n_asian, n_western, pairs_per_category)
        
        # Replicate exact pairing logic
        asian_indices = list(range(n_asian))
        western_indices = list(range(n_western))
        rng.shuffle(asian_indices)
        rng.shuffle(western_indices)
        
        asian_counter = Counter()
        western_counter = Counter()
        
        for i in range(n_pairs):
            a_idx = asian_indices[i % n_asian]
            w_idx = western_indices[i % n_western]
            asian_counter[cdata.asian_entities[a_idx]] += 1
            western_counter[cdata.western_entities[w_idx]] += 1
        
        # Stats
        a_counts = list(asian_counter.values())
        w_counts = list(western_counter.values())
        a_unused = n_asian - len(asian_counter)
        w_unused = n_western - len(western_counter)
        
        def count_stats(counts, total_entities):
            if not counts:
                return {"min": 0, "max": 0, "mean": 0, "std": 0,
                        "used": 0, "total": total_entities, "unused": total_entities}
            return {
                "min": min(counts),
                "max": max(counts),
                "mean": round(np.mean(counts), 2),
                "std": round(np.std(counts), 2),
                "used": len(counts),
                "total": total_entities,
                "unused": total_entities - len(counts),
            }
        
        capped = n_pairs < pairs_per_category
        cap_reason = None
        if capped:
            if n_asian <= n_western and n_asian <= pairs_per_category:
                cap_reason = f"asian_count({n_asian})"
            elif n_western < n_asian and n_western <= pairs_per_category:
                cap_reason = f"western_count({n_western})"
        
        all_results[cat_key] = {
            "n_asian": n_asian,
            "n_western": n_western,
            "n_pairs_generated": n_pairs,
            "pairs_per_category_limit": pairs_per_category,
            "capped_by_entity_count": capped,
            "cap_reason": cap_reason,
            "asian_freq": count_stats(a_counts, n_asian),
            "western_freq": count_stats(w_counts, n_western),
            "max_imbalance_ratio_asian": (
                round(max(a_counts) / min(a_counts), 2) if a_counts and min(a_counts) > 0 else None
            ),
            "max_imbalance_ratio_western": (
                round(max(w_counts) / min(w_counts), 2) if w_counts and min(w_counts) > 0 else None
            ),
        }
    
    return all_results


# =========================================================================
# 4. N×M Evaluation Imbalance
# =========================================================================

def analyze_eval_imbalance(split_info: Dict, max_entities: int = 30) -> Dict:
    """Analyze how N×M comparisons distribute across entities in evaluation."""
    results = {}
    for split_name in ["val", "test"]:
        entities = split_info[f"{split_name}_entities"]
        split_results = {}
        for etype, ent_dict in entities.items():
            n_a = len(ent_dict["asian"])
            n_w = len(ent_dict["western"])
            n_a_used = min(n_a, max_entities)
            n_w_used = min(n_w, max_entities)
            total_comparisons = n_a_used * n_w_used
            
            # Each Asian entity is compared with n_w_used Western entities
            # Each Western entity is compared with n_a_used Asian entities
            split_results[etype] = {
                "n_asian": n_a,
                "n_western": n_w,
                "n_asian_used": n_a_used,
                "n_western_used": n_w_used,
                "total_NxM_comparisons": total_comparisons,
                "comparisons_per_asian_entity": n_w_used,
                "comparisons_per_western_entity": n_a_used,
                "asian_truncated": n_a > max_entities,
                "western_truncated": n_w > max_entities,
            }
        results[split_name] = split_results
    return results


# =========================================================================
# 5. Context Count Analysis
# =========================================================================

def analyze_contexts(data: CamelliaData, split_info: Dict) -> Dict:
    """Analyze context counts per entity type and split."""
    results = {}
    for ctx_type in ["grounded", "neutral"]:
        ctx_results = {}
        for split_name in ["train", "val", "test"]:
            df = split_info[f"{ctx_type}_{split_name}"]
            if len(df) > 0 and "entity_type" in df.columns:
                by_type = df["entity_type"].value_counts().to_dict()
            else:
                by_type = {}
            ctx_results[split_name] = {"total": len(df), "by_type": by_type}
        results[ctx_type] = ctx_results
    return results


# =========================================================================
# Pretty Printing
# =========================================================================

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_raw_entities(raw: Dict, culture: str, lang: str):
    print_section(f"Raw Entity Counts: {culture}/{lang}")
    print(f"{'Category':<18} {'Asian':>8} {'Western':>8} {'Ratio':>8} {'Diff':>8}")
    print("-" * 54)
    for etype, v in raw.items():
        if etype == "_total":
            print("-" * 54)
        print(f"{etype:<18} {v['n_asian']:>8} {v['n_western']:>8} "
              f"{v['ratio_a_w']:>8.2f} {v['diff']:>+8d}")


def print_pairing_analysis(pairing: Dict):
    print_section("Pairing Frequency Analysis (PairedDataset)")
    
    for cat_key, v in pairing.items():
        cap_str = f" *** CAPPED by {v['cap_reason']}" if v["capped_by_entity_count"] else ""
        print(f"\n  [{cat_key}] A={v['n_asian']}, W={v['n_western']} "
              f"→ {v['n_pairs_generated']} pairs (limit={v['pairs_per_category_limit']}){cap_str}")
        
        af = v["asian_freq"]
        wf = v["western_freq"]
        print(f"    Asian  freq: min={af['min']}, max={af['max']}, "
              f"mean={af['mean']:.1f}, std={af['std']:.1f}, "
              f"used={af['used']}/{af['total']}, unused={af['unused']}")
        print(f"    Western freq: min={wf['min']}, max={wf['max']}, "
              f"mean={wf['mean']:.1f}, std={wf['std']:.1f}, "
              f"used={wf['used']}/{wf['total']}, unused={wf['unused']}")
        
        if v["max_imbalance_ratio_asian"]:
            print(f"    Max imbalance: Asian={v['max_imbalance_ratio_asian']}x, "
                  f"Western={v['max_imbalance_ratio_western']}x")


def print_eval_imbalance(eval_imb: Dict):
    print_section("Evaluation N×M Imbalance")
    for split_name, etypes in eval_imb.items():
        print(f"\n  [{split_name}]")
        print(f"  {'Category':<18} {'A_used':>8} {'W_used':>8} {'NxM':>10} "
              f"{'per_A':>8} {'per_W':>8} {'Trunc':>8}")
        print(f"  {'-'*62}")
        for etype, v in etypes.items():
            trunc = ""
            if v["asian_truncated"]:
                trunc += "A"
            if v["western_truncated"]:
                trunc += "W"
            print(f"  {etype:<18} {v['n_asian_used']:>8} {v['n_western_used']:>8} "
                  f"{v['total_NxM_comparisons']:>10} "
                  f"{v['comparisons_per_asian_entity']:>8} "
                  f"{v['comparisons_per_western_entity']:>8} "
                  f"{trunc:>8}")


# =========================================================================
# Main
# =========================================================================

def run_single(culture, lang, data_root, seed, pairs_per_category, output_dir):
    """Run full analysis for one culture/lang."""
    print(f"\n{'#'*70}")
    print(f"# Culture: {culture}, Lang: {lang}, Seed: {seed}")
    print(f"{'#'*70}")
    
    # Load
    data = load_camellia_data(data_root, culture=culture, target_lang=lang)
    _, _, _, split_info = split_data(data, train_ratio=0.7, val_ratio=0.1, seed=seed)
    
    # 1. Raw entities
    raw = analyze_raw_entities(data, culture, lang)
    print_raw_entities(raw, culture, lang)
    
    # 2. Split entities
    split_ents = analyze_split_entities(split_info)
    print_section(f"Post-Split Entity Counts (seed={seed})")
    for split_name, etypes in split_ents.items():
        print(f"\n  [{split_name}]")
        for etype, v in etypes.items():
            print(f"    {etype:<18} A={v['n_asian']:>3}, W={v['n_western']:>3}, "
                  f"min(A,W)={v['min_nw']:>3}, ratio={v['ratio']:.2f}")
    
    # 3. Pairing frequency
    cat_data = build_category_data(
        split_info["grounded_train"],
        split_info["neutral_train"],
        split_info["train_entities"],
    )
    pairing = analyze_pairing_frequency(cat_data, pairs_per_category, seed)
    print_pairing_analysis(pairing)
    
    # 4. Eval imbalance
    eval_imb = analyze_eval_imbalance(split_info, max_entities=30)
    print_eval_imbalance(eval_imb)
    
    # 5. Context counts
    ctx_analysis = analyze_contexts(data, split_info)
    print_section("Context Counts")
    for ctx_type, splits in ctx_analysis.items():
        for split_name, v in splits.items():
            print(f"  {ctx_type}/{split_name}: {v['total']} total — {v['by_type']}")
    
    # Save
    result = {
        "culture": culture,
        "lang": lang,
        "seed": seed,
        "raw_entities": raw,
        "split_entities": split_ents,
        "pairing_frequency": pairing,
        "eval_imbalance": eval_imb,
        "context_counts": ctx_analysis,
    }
    
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fname = out_path / f"phase1_{culture}_{lang}_seed{seed}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved: {fname}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="COCOA Phase 1: Data Statistics")
    parser.add_argument("--cultures", nargs="+",
                        default=["ko", "zh", "ja", "vi", "hi", "ur", "gu", "mr", "ml", "ar"],
                        help="Cultures to analyze")
    parser.add_argument("--lang", default="cu", choices=["cu", "en"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", default="./dataset/camellia/raw")
    parser.add_argument("--pairs_per_category", type=int, default=200)
    parser.add_argument("--output_dir", default="./analysis/phase1")
    args = parser.parse_args()
    
    print(f"COCOA Phase 1: Data Statistics Analysis")
    print(f"Cultures: {args.cultures}")
    print(f"Language: {args.lang}")
    print(f"Seed: {args.seed}")
    
    all_results = {}
    for culture in args.cultures:
        try:
            result = run_single(
                culture, args.lang, args.data_root, args.seed,
                args.pairs_per_category, args.output_dir,
            )
            all_results[culture] = result
        except Exception as e:
            print(f"\n*** ERROR for {culture}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary table
    print_section("SUMMARY: Entity Count Imbalance Across Cultures")
    print(f"{'Culture':<12} {'Lang':<6} {'Total_A':>10} {'Total_W':>10} "
          f"{'Ratio':>8} {'Capped_Cats':>14} {'Min_Pairs':>12}")
    print("-" * 74)
    for culture, result in all_results.items():
        raw = result["raw_entities"]["_total"]
        pairing = result["pairing_frequency"]
        n_capped = sum(1 for v in pairing.values() if v["capped_by_entity_count"])
        min_pairs = min(v["n_pairs_generated"] for v in pairing.values()) if pairing else 0
        print(f"{culture:<12} {args.lang:<6} {raw['n_asian']:>10} {raw['n_western']:>10} "
              f"{raw['ratio_a_w']:>8.2f} {n_capped:>14} {min_pairs:>12}")
    
    # Save combined
    if args.output_dir:
        combined_path = Path(args.output_dir) / f"phase1_summary_{args.lang}.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nCombined results: {combined_path}")


if __name__ == "__main__":
    main()