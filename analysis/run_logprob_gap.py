#!/usr/bin/env python3
"""
Paired Logprob Gap Analysis — Context-Awareness Verification (v2)

Core question: Does COCOA teach models to behave differently
               based on context type, for the SAME entities?

Design:
    For each entity pair (asian, western) in the TEST set:
        - Compute gap = log P(asian) - log P(western) on grounded contexts
        - Compute gap = log P(asian) - log P(western) on neutral contexts
        - Context Effect = mean(grounded gaps) - mean(neutral gaps)

    Compare Context Effect before vs after COCOA.
    Same entity pairs × same contexts → purely paired comparison.
    Uses the experiment's seed to ensure test split matches training.

    Entity pairs are drawn from the TEST split's entity pool,
    guaranteeing zero overlap with training data.

Usage:
    CUDA_VISIBLE_DEVICES=2 python analysis/run_logprob_gap_paired.py \
        --summary experiments/summary.json \
        --models llama

    # Specific cultures only:
    CUDA_VISIBLE_DEVICES=2 python analysis/run_logprob_gap_paired.py \
        --summary experiments/summary.json \
        --cultures ko ja zh
"""

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from itertools import product

import torch
import numpy as np
from scipy import stats
from tqdm import tqdm
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_model, ModelConfig, MODEL_SHORTCUTS
from src.data import load_camellia_data, split_data
from src.evaluate import compute_log_probs_for_entities_batched


# ============================================================================
# Constants
# ============================================================================

MODEL_NORMALIZE = {
    "llama3_8b": "llama", "llama3-8b": "llama",
    "meta-llama/llama-3.1-8b": "llama",
    "qwen3_8b": "qwen", "qwen3-8b": "qwen",
    "qwen/qwen3-8b": "qwen",
}

MODEL_SHORTCUT_REVERSE = {
    "llama": "llama3_8b",
    "qwen": "qwen3_8b",
}

# Context entity type matching (entity file name → context Entity Type values)
REVERSE_ALIAS = {
    "locations": ["location", "locations"],
    "sports": ["sport", "sports"],
    "authors": ["author", "authors"],
    "names-male": ["names", "name", "names (m)", "name (m)", "names-male"],
    "names-female": ["names", "name", "names (f)", "name (f)", "names-female"],
    "beverage": ["beverage"],
    "food": ["food"],
    "sports-clubs": ["sport", "sports", "sports-clubs"],
}


def norm_model(m):
    m = m.lower().strip()
    return MODEL_NORMALIZE.get(m, "llama" if "llama" in m else "qwen" if "qwen" in m else m)


# ============================================================================
# Find best experiments from summary.json
# ============================================================================

def find_best_experiments(summary_path, cultures=None, models=None):
    with open(summary_path) as f:
        data = json.load(f)

    best = {}
    for e in data["experiments"]:
        if e.get("lang") == "en":
            continue
        culture = e["culture"]
        model_key = norm_model(e.get("model", ""))

        if cultures and culture not in cultures:
            continue
        if culture in ("indian", "indian_combined"):
            continue
        if models and model_key not in models:
            continue

        key = (culture, model_key)
        score = e["final"]["score"]
        if key not in best or score < best[key]["score"]:
            best[key] = {
                "name": e["name"],
                "seed": e.get("seed"),
                "score": score,
                "cbs_g": e["final"]["cbs_g"],
                "cbs_n": e["final"]["cbs_n"],
            }
    return best


# ============================================================================
# Paired gap computation
# ============================================================================

def compute_paired_gaps(
    model, tokenizer, split_info, entities, device,
    split="test",
    max_contexts_per_type=30,
    max_pairs_per_type=50,
    seed=42,
    batch_size=32,
):
    """
    Compute paired logprob gaps: same entity pairs on grounded vs neutral contexts.

    For each entity_type:
        - Sample entity pairs from TEST split entities
        - Sample grounded & neutral contexts from TEST split
        - For each pair × each context: compute gap = log P(asian) - log P(western)

    Returns:
        List of per-pair results:
        {
            "entity_type": str,
            "asian_entity": str,
            "western_entity": str,
            "grounded_gaps": [float, ...],   # one per grounded context
            "neutral_gaps": [float, ...],    # one per neutral context
            "mean_grounded_gap": float,
            "mean_neutral_gap": float,
            "context_effect": float,         # mean_grounded - mean_neutral
        }
    """
    rng = random.Random(seed)
    model.eval()
    results = []

    grounded_df = split_info[f"grounded_{split}"]
    neutral_df = split_info[f"neutral_{split}"]

    # Get test entities from split_info
    test_entities = split_info.get(f"entities_{split}", entities)

    for etype in entities.keys():
        # --- Get test entities for this type ---
        if isinstance(test_entities, dict) and etype in test_entities:
            asian_ents = test_entities[etype].get("asian", [])
            western_ents = test_entities[etype].get("western", [])
        else:
            asian_ents = entities[etype].get("asian", [])
            western_ents = entities[etype].get("western", [])

        if not asian_ents or not western_ents:
            continue

        # --- Get contexts for this entity type ---
        possible_names = REVERSE_ALIAS.get(etype, [etype])

        g_mask = grounded_df["entity_type"].str.lower().isin([n.lower() for n in possible_names])
        g_contexts = grounded_df[g_mask]["context"].tolist()

        n_mask = neutral_df["entity_type"].str.lower().isin([n.lower() for n in possible_names])
        n_contexts = neutral_df[n_mask]["context"].tolist()

        if not g_contexts or not n_contexts:
            continue

        # Sample contexts
        if len(g_contexts) > max_contexts_per_type:
            g_contexts = rng.sample(g_contexts, max_contexts_per_type)
        if len(n_contexts) > max_contexts_per_type:
            n_contexts = rng.sample(n_contexts, max_contexts_per_type)

        # Create entity pairs and sample
        all_pairs = list(product(asian_ents, western_ents))
        if len(all_pairs) > max_pairs_per_type:
            all_pairs = rng.sample(all_pairs, max_pairs_per_type)

        print(f"  [{etype}] {len(all_pairs)} pairs × ({len(g_contexts)}G + {len(n_contexts)}N) contexts")

        # --- Compute gaps for each pair ---
        for asian_ent, western_ent in tqdm(all_pairs, desc=f"    {etype}", leave=False):
            grounded_gaps = []
            neutral_gaps = []

            # Grounded contexts
            for ctx in g_contexts:
                lps = compute_log_probs_for_entities_batched(
                    model, tokenizer, ctx, [asian_ent, western_ent],
                    device, batch_size=batch_size,
                )
                if len(lps) == 2:
                    grounded_gaps.append(lps[0] - lps[1])

            # Neutral contexts
            for ctx in n_contexts:
                lps = compute_log_probs_for_entities_batched(
                    model, tokenizer, ctx, [asian_ent, western_ent],
                    device, batch_size=batch_size,
                )
                if len(lps) == 2:
                    neutral_gaps.append(lps[0] - lps[1])

            if grounded_gaps and neutral_gaps:
                mean_g = float(np.mean(grounded_gaps))
                mean_n = float(np.mean(neutral_gaps))
                results.append({
                    "entity_type": etype,
                    "asian_entity": asian_ent,
                    "western_entity": western_ent,
                    "grounded_gaps": grounded_gaps,
                    "neutral_gaps": neutral_gaps,
                    "mean_grounded_gap": mean_g,
                    "mean_neutral_gap": mean_n,
                    "context_effect": mean_g - mean_n,
                })

    return results


def analyze_results(before_pairs, after_pairs):
    """
    Statistical analysis of paired results.

    Matches entity pairs between before/after, computes paired statistics.
    """
    # Build lookup by (asian_entity, western_entity)
    before_map = {(p["asian_entity"], p["western_entity"]): p for p in before_pairs}
    after_map = {(p["asian_entity"], p["western_entity"]): p for p in after_pairs}

    common_keys = set(before_map.keys()) & set(after_map.keys())

    before_effects = []
    after_effects = []
    before_g_gaps = []
    after_g_gaps = []
    before_n_gaps = []
    after_n_gaps = []

    for key in common_keys:
        b = before_map[key]
        a = after_map[key]
        before_effects.append(b["context_effect"])
        after_effects.append(a["context_effect"])
        before_g_gaps.append(b["mean_grounded_gap"])
        after_g_gaps.append(a["mean_grounded_gap"])
        before_n_gaps.append(b["mean_neutral_gap"])
        after_n_gaps.append(a["mean_neutral_gap"])

    before_effects = np.array(before_effects)
    after_effects = np.array(after_effects)

    # Paired t-test on context effect
    if len(before_effects) >= 2:
        t_stat, p_value = stats.ttest_rel(after_effects, before_effects)
    else:
        t_stat, p_value = 0.0, 1.0

    # Cohen's d for effect size
    diff = after_effects - before_effects
    cohens_d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0

    analysis = {
        "n_paired": len(common_keys),
        "before": {
            "mean_context_effect": float(np.mean(before_effects)),
            "std_context_effect": float(np.std(before_effects)),
            "mean_grounded_gap": float(np.mean(before_g_gaps)),
            "mean_neutral_gap": float(np.mean(before_n_gaps)),
        },
        "after": {
            "mean_context_effect": float(np.mean(after_effects)),
            "std_context_effect": float(np.std(after_effects)),
            "mean_grounded_gap": float(np.mean(after_g_gaps)),
            "mean_neutral_gap": float(np.mean(after_n_gaps)),
        },
        "delta_context_effect": float(np.mean(after_effects) - np.mean(before_effects)),
        "paired_ttest": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_005": bool(p_value < 0.05),
            "significant_001": bool(p_value < 0.01),
        },
        "cohens_d": cohens_d,
    }

    return analysis


def print_analysis(culture, model, analysis):
    """Print formatted analysis for one culture."""
    b = analysis["before"]
    a = analysis["after"]
    t = analysis["paired_ttest"]

    print(f"\n  {'─'*70}")
    print(f"  {culture.upper()} ({model})  |  {analysis['n_paired']} paired entity comparisons")
    print(f"  {'─'*70}")
    print(f"  {'':25} {'Grounded Gap':>14} {'Neutral Gap':>14} {'Context Effect':>16}")
    print(f"  {'Before COCOA':25} {b['mean_grounded_gap']:>+14.2f} {b['mean_neutral_gap']:>+14.2f} {b['mean_context_effect']:>+16.2f}")
    print(f"  {'After COCOA':25} {a['mean_grounded_gap']:>+14.2f} {a['mean_neutral_gap']:>+14.2f} {a['mean_context_effect']:>+16.2f}")
    print(f"  {'Δ (After - Before)':25} {a['mean_grounded_gap']-b['mean_grounded_gap']:>+14.2f} {a['mean_neutral_gap']-b['mean_neutral_gap']:>+14.2f} {analysis['delta_context_effect']:>+16.2f}")
    print()
    print(f"  Paired t-test:  t={t['t_statistic']:.3f},  p={t['p_value']:.2e}  {'***' if t['significant_001'] else '**' if t['significant_005'] else 'n.s.'}")
    print(f"  Cohen's d:      {analysis['cohens_d']:.3f}")
    print(f"  Verdict:        {'✓ Significant context-awareness gain' if t['significant_005'] else '✗ Not significant'}")


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Paired Logprob Gap Analysis")
    p.add_argument("--summary", default="experiments/summary.json")
    p.add_argument("--exp_base", default="experiments")
    p.add_argument("--data_root", default="./dataset/camellia/raw")
    p.add_argument("--output", default="analysis/logprob_gap_paired")
    p.add_argument("--cultures", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=["llama"])
    p.add_argument("--max_contexts", type=int, default=30)
    p.add_argument("--max_pairs", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Find best experiments
    # =========================================================================
    print("Finding best experiments...")
    best_exps = find_best_experiments(args.summary, args.cultures, args.models)

    print(f"Found {len(best_exps)} experiments:")
    for (c, m), info in sorted(best_exps.items()):
        print(f"  {c:>5} {m:>6}: Score={info['score']:5.1f}  seed={info['seed']}  {info['name']}")
    print()

    # =========================================================================
    # 2. Process each culture×model
    # =========================================================================
    all_analyses = {}

    by_model = defaultdict(list)
    for (culture, model_key), info in best_exps.items():
        by_model[model_key].append((culture, info))

    for model_key, culture_list in by_model.items():
        model_shortcut = MODEL_SHORTCUT_REVERSE.get(model_key, model_key)
        full_model_name = MODEL_SHORTCUTS.get(model_shortcut, model_shortcut)

        print(f"{'='*60}")
        print(f"Loading base model: {full_model_name}")
        print(f"{'='*60}")

        model_config = ModelConfig(name=full_model_name, use_lora=False)
        base_model, tokenizer = load_model(full_model_name, model_config, for_distributed=False)
        base_model = base_model.to(device)

        for culture, info in sorted(culture_list):
            exp_name = info["name"]
            seed = info["seed"]
            exp_dir = Path(args.exp_base) / exp_name

            print(f"\n{'='*60}")
            print(f"{culture.upper()} (seed={seed}, score={info['score']:.1f})")
            print(f"{'='*60}")

            # Skip if already done
            out_file = output_dir / f"paired_{culture}_{model_key}_seed{seed}.json"
            if out_file.exists():
                print(f"  ✓ Already done, skipping")
                # Load for summary table
                with open(out_file) as f:
                    saved = json.load(f)
                all_analyses[(culture, model_key)] = saved["analysis"]
                continue

            # Check checkpoint
            adapter_path = exp_dir / "checkpoints" / "best"
            if not adapter_path.exists():
                adapter_path = exp_dir / "checkpoints" / "final"
            if not adapter_path.exists():
                print(f"  ✗ No checkpoint at {exp_dir}/checkpoints/")
                continue

            # Load data with SAME seed as training
            print("  Loading data (seed={} for correct test split)...".format(seed))
            data = load_camellia_data(args.data_root, culture=culture, target_lang="cu")
            _, _, _, split_info = split_data(
                data, train_ratio=0.7, val_ratio=0.1, seed=seed,
                max_pairs_per_context=10, min_entities_per_split=2,
            )

            # --- Before (base model) ---
            print("\n  [BEFORE] Base model gaps...")
            before_pairs = compute_paired_gaps(
                base_model, tokenizer, split_info, data.entities, device,
                split="test", max_contexts_per_type=args.max_contexts,
                max_pairs_per_type=args.max_pairs, seed=seed,
            )
            print(f"  → {len(before_pairs)} paired measurements")

            # --- After (COCOA model) ---
            print(f"\n  [AFTER] Loading adapter: {adapter_path}")
            cocoa_model = PeftModel.from_pretrained(base_model, str(adapter_path))
            cocoa_model = cocoa_model.merge_and_unload()
            cocoa_model = cocoa_model.to(device)

            print("  [AFTER] COCOA model gaps...")
            after_pairs = compute_paired_gaps(
                cocoa_model, tokenizer, split_info, data.entities, device,
                split="test", max_contexts_per_type=args.max_contexts,
                max_pairs_per_type=args.max_pairs, seed=seed,
            )
            print(f"  → {len(after_pairs)} paired measurements")

            # --- Analysis ---
            analysis = analyze_results(before_pairs, after_pairs)
            print_analysis(culture, model_key, analysis)

            all_analyses[(culture, model_key)] = analysis

            # Save
            result = {
                "culture": culture,
                "model": model_key,
                "seed": seed,
                "exp_name": exp_name,
                "score": info["score"],
                "analysis": analysis,
                "before_pairs": before_pairs,
                "after_pairs": after_pairs,
            }
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {out_file}")

            # Cleanup
            del cocoa_model
            torch.cuda.empty_cache()
            del base_model
            torch.cuda.empty_cache()
            base_model, tokenizer = load_model(full_model_name, model_config, for_distributed=False)
            base_model = base_model.to(device)

        del base_model
        torch.cuda.empty_cache()

    # =========================================================================
    # 3. Final summary
    # =========================================================================
    print(f"\n{'='*90}")
    print("  FINAL SUMMARY: Paired Context-Awareness Analysis")
    print(f"{'='*90}")
    print(f"  {'Culture':>8} {'Model':>6} │ {'N':>5} │ {'Before CE':>10} {'After CE':>10} {'Δ CE':>8} │ {'t-stat':>7} {'p-value':>10} │ {'d':>6} │ {'Sig':>4}")
    print(f"  {'─'*85}")

    for (culture, model_key), a in sorted(all_analyses.items()):
        t = a["paired_ttest"]
        sig = "***" if t["significant_001"] else "**" if t["significant_005"] else "n.s."
        print(
            f"  {culture:>8} {model_key:>6} │"
            f" {a['n_paired']:>5} │"
            f" {a['before']['mean_context_effect']:>+10.2f}"
            f" {a['after']['mean_context_effect']:>+10.2f}"
            f" {a['delta_context_effect']:>+8.2f} │"
            f" {t['t_statistic']:>7.2f} {t['p_value']:>10.2e} │"
            f" {a['cohens_d']:>6.2f} │"
            f" {sig:>4}"
        )

    # Save summary
    summary = {
        "analyses": {
            f"{c}_{m}": a for (c, m), a in all_analyses.items()
        }
    }
    summary_file = output_dir / "paired_summary_qwen.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {summary_file}")


if __name__ == "__main__":
    main()