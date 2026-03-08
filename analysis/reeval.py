#!/usr/bin/env python3
"""
Re-evaluation Script — Fix REVERSE_ALIAS Category Matching

Affected cultures: hi, mr, ml, gu, vi, ur, ar
Issue: names-male/names-female categories were not matched in evaluation
       due to missing aliases in REVERSE_ALIAS.
Fix: evaluate.py REVERSE_ALIAS updated to include "names-male", "names-female"
     and "sports-clubs" as valid context entity type names.

This script re-evaluates existing checkpoints WITHOUT re-training.

Supports:
  - COCOA (LoRA adapters from experiments/)
  - BiasEdit (full model from baselines/results/biasedit/)
  - BiasUnlearn (LoRA adapters from baselines/results/biasunlearn/)

Usage:
    # COCOA only (most important)
    CUDA_VISIBLE_DEVICES=2 python analysis/reeval.py --method cocoa

    # All methods
    CUDA_VISIBLE_DEVICES=2 python analysis/reeval.py --method all

    # Specific cultures
    CUDA_VISIBLE_DEVICES=2 python analysis/reeval.py --method cocoa --cultures hi mr ml

    # Dry run (just check what would be re-evaluated)
    python analysis/reeval.py --method cocoa --dry_run
"""

import argparse
import json
import sys
import shutil
from pathlib import Path
from collections import defaultdict

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_model, ModelConfig, MODEL_SHORTCUTS
from src.data import load_camellia_data, split_data
from src.evaluate import evaluate_robust_fair

# ============================================================================
# Constants
# ============================================================================

# Cultures affected by the REVERSE_ALIAS bug
AFFECTED_CULTURES = ["hi", "mr", "ml", "gu", "vi", "ur", "ar"]

# ALL cultures (for --all_cultures mode)
ALL_CULTURES = ["ko", "ja", "zh", "ar", "hi", "mr", "ml", "gu", "ur", "vi"]

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

MODEL_FOLDER = {"llama": "llama3-8b", "qwen": "qwen3-8b"}

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


def norm_model(m):
    m = m.lower().strip()
    return MODEL_NORMALIZE.get(m, "llama" if "llama" in m else "qwen" if "qwen" in m else m)


# ============================================================================
# Evaluation Core
# ============================================================================

def run_evaluation(model, tokenizer, split_info, entities, device, split="test"):
    """Run evaluate_robust_fair and return CBS results."""
    model.eval()
    with torch.no_grad():
        results = evaluate_robust_fair(
            model=model,
            tokenizer=tokenizer,
            split_info=split_info,
            entities=entities,
            device=device,
            split=split,
            max_entities=30,
            show_progress=True,
        )

    cbs_g = results["grounded"]["overall"] or 0.0
    cbs_n = results["neutral"]["overall"] or 0.0
    score = cbs_g + abs(cbs_n - 50)

    return {
        "cbs_grounded": round(cbs_g, 4),
        "cbs_neutral": round(cbs_n, 4),
        "combined_score": round(score, 4),
        "split": split,
        "by_category": results,
    }


# ============================================================================
# COCOA Re-evaluation
# ============================================================================

def find_cocoa_experiments(exp_dir, cultures, models):
    """Find COCOA experiment folders for affected cultures."""
    exp_dir = Path(exp_dir)
    experiments = []

    for (culture, model), seed in BEST_SEEDS.items():
        if culture not in cultures:
            continue
        if model not in models:
            continue

        mf = MODEL_FOLDER[model]
        # goal_aware_pcgrad has no suffix
        exp_name = f"{culture}_cu_{mf}_mse_wg1.0_wn2.0_tau1.0_r16_seed{seed}"
        exp_path = exp_dir / exp_name

        if exp_path.exists():
            adapter_path = exp_path / "checkpoints" / "best"
            if not adapter_path.exists():
                adapter_path = exp_path / "checkpoints" / "final"

            experiments.append({
                "culture": culture,
                "model": model,
                "seed": seed,
                "exp_name": exp_name,
                "exp_path": exp_path,
                "adapter_path": adapter_path,
                "results_file": exp_path / "results.json",
            })

    return experiments


def reeval_cocoa(args):
    """Re-evaluate COCOA experiments."""
    device = torch.device(args.device)
    exp_dir = Path(args.exp_dir)
    cultures = args.cultures or AFFECTED_CULTURES
    models = args.models

    experiments = find_cocoa_experiments(exp_dir, cultures, models)
    print(f"\nFound {len(experiments)} COCOA experiments to re-evaluate:")
    for e in experiments:
        print(f"  {e['culture']:>3} {e['model']:>6} seed={e['seed']} → {e['exp_name']}")

    if args.dry_run:
        return

    # Group by model to reuse base model
    by_model = defaultdict(list)
    for e in experiments:
        by_model[e["model"]].append(e)

    for model_key, exp_list in by_model.items():
        model_shortcut = MODEL_SHORTCUT_REVERSE[model_key]
        full_model_name = MODEL_SHORTCUTS.get(model_shortcut, model_shortcut)

        print(f"\n{'='*60}")
        print(f"Loading base model: {full_model_name}")
        print(f"{'='*60}")

        model_config = ModelConfig(name=full_model_name, use_lora=False)
        base_model, tokenizer = load_model(full_model_name, model_config, for_distributed=False)
        base_model = base_model.to(device)

        for exp in exp_list:
            culture = exp["culture"]
            seed = exp["seed"]
            adapter_path = exp["adapter_path"]
            results_file = exp["results_file"]

            print(f"\n--- {culture.upper()} {model_key} (seed={seed}) ---")

            if not adapter_path.exists():
                print(f"  ✗ No adapter at {adapter_path}")
                continue

            # Load data with correct seed
            print("  Loading data...")
            data = load_camellia_data(args.data_root, culture=culture, target_lang="cu")
            _, _, _, split_info = split_data(
                data, train_ratio=0.7, val_ratio=0.1, seed=seed,
                max_pairs_per_context=10, min_entities_per_split=2,
            )

            # --- Baseline (base model) ---
            print("  [Baseline] Evaluating base model...")
            test_entities = split_info.get("test_entities", data.entities)
            baseline_test = run_evaluation(base_model, tokenizer, split_info, test_entities, device, "test")
            baseline_val = run_evaluation(base_model, tokenizer, split_info, test_entities, device, "val")

            print(f"    Baseline test: CBS_g={baseline_test['cbs_grounded']:.1f} CBS_n={baseline_test['cbs_neutral']:.1f}")

            # --- COCOA (base + LoRA adapter) ---
            print(f"  [COCOA] Loading adapter from {adapter_path}...")
            cocoa_model = PeftModel.from_pretrained(base_model, str(adapter_path))
            cocoa_model = cocoa_model.merge_and_unload()
            cocoa_model = cocoa_model.to(device)

            print("  [COCOA] Evaluating...")
            final_test = run_evaluation(cocoa_model, tokenizer, split_info, test_entities, device, "test")
            best_val = run_evaluation(cocoa_model, tokenizer, split_info, test_entities, device, "val")

            print(f"    Final test:  CBS_g={final_test['cbs_grounded']:.1f} CBS_n={final_test['cbs_neutral']:.1f} Score={final_test['combined_score']:.1f}")

            # --- Update results.json ---
            if results_file.exists():
                # Backup original
                backup = results_file.parent / "results_pre_reeval.json"
                if not backup.exists():
                    shutil.copy(results_file, backup)
                    print(f"  Backed up original → {backup.name}")

                with open(results_file) as f:
                    orig = json.load(f)

                # Compare old vs new
                old_score = orig.get("final", {}).get("combined_score", "?")
                new_score = final_test["combined_score"]
                print(f"  Score: {old_score} → {new_score}")

                # Update
                orig["baseline"] = baseline_val
                orig["baseline_test"] = baseline_test
                orig["final"] = final_test
                orig["best"] = {**best_val, "step": orig.get("best", {}).get("step", "?")}

                with open(results_file, "w") as f:
                    json.dump(orig, f, indent=2, ensure_ascii=False)
                print(f"  ✓ Updated {results_file}")
            else:
                print(f"  ✗ No results.json to update")

            # Cleanup
            del cocoa_model
            torch.cuda.empty_cache()

            # Reload base model (merge_and_unload modified it)
            del base_model
            torch.cuda.empty_cache()
            base_model, tokenizer = load_model(full_model_name, model_config, for_distributed=False)
            base_model = base_model.to(device)

        del base_model
        torch.cuda.empty_cache()


# ============================================================================
# BiasEdit Re-evaluation
# ============================================================================

def find_biasedit_experiments(base_dir, cultures, models):
    """Find BiasEdit result folders."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    experiments = []
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        rfile = folder / "results.json"
        if not rfile.exists():
            continue

        with open(rfile) as f:
            d = json.load(f)

        cfg = d.get("config", {})
        culture = cfg.get("culture", "")
        model = norm_model(cfg.get("model", cfg.get("model_full", "")))

        if culture not in cultures or model not in models:
            continue

        # Check if model checkpoint exists
        model_path = folder / "edited_model"
        if not model_path.exists():
            model_path = folder / "model"

        experiments.append({
            "culture": culture,
            "model": model,
            "seed": cfg.get("seed"),
            "folder": folder,
            "results_file": rfile,
            "model_path": model_path,
        })

    return experiments


def reeval_biasedit(args):
    """Re-evaluate BiasEdit experiments."""
    device = torch.device(args.device)
    cultures = args.cultures or AFFECTED_CULTURES
    base_dir = Path(args.biasedit_dir)

    experiments = find_biasedit_experiments(base_dir, cultures, args.models)
    print(f"\nFound {len(experiments)} BiasEdit experiments to re-evaluate:")
    for e in experiments:
        print(f"  {e['culture']:>3} {e['model']:>6} seed={e['seed']} → {e['folder'].name}")

    if args.dry_run:
        return

    print("\n⚠ BiasEdit re-evaluation requires saved model weights.")
    print("  Checking which experiments have saved models...")

    for exp in experiments:
        if exp["model_path"].exists():
            print(f"  ✓ {exp['folder'].name}: model found at {exp['model_path']}")
        else:
            print(f"  ✗ {exp['folder'].name}: no saved model — must re-run BiasEdit")


# ============================================================================
# BiasUnlearn Re-evaluation
# ============================================================================

def find_biasunlearn_experiments(base_dir, cultures, models):
    """Find BiasUnlearn result folders."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    experiments = []
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        rfile = folder / "results.json"
        if not rfile.exists():
            continue

        with open(rfile) as f:
            d = json.load(f)

        cfg = d.get("config", {})
        culture = cfg.get("culture", "")
        model = norm_model(cfg.get("model", cfg.get("model_full", "")))

        if culture not in cultures or model not in models:
            continue

        # Check for LoRA adapter
        adapter_path = folder / "best_adapter"
        if not adapter_path.exists():
            adapter_path = folder / "adapter"
        if not adapter_path.exists():
            adapter_path = folder / "checkpoints" / "best"

        experiments.append({
            "culture": culture,
            "model": model,
            "seed": cfg.get("seed"),
            "folder": folder,
            "results_file": rfile,
            "adapter_path": adapter_path,
        })

    return experiments


def reeval_biasunlearn(args):
    """Re-evaluate BiasUnlearn experiments."""
    device = torch.device(args.device)
    cultures = args.cultures or AFFECTED_CULTURES
    base_dir = Path(args.biasunlearn_dir)

    experiments = find_biasunlearn_experiments(base_dir, cultures, args.models)
    print(f"\nFound {len(experiments)} BiasUnlearn experiments to re-evaluate:")
    for e in experiments:
        has_adapter = "✓" if e["adapter_path"].exists() else "✗"
        print(f"  {has_adapter} {e['culture']:>3} {e['model']:>6} seed={e['seed']} → {e['folder'].name}")

    if args.dry_run:
        return

    # Group by model
    by_model = defaultdict(list)
    for e in experiments:
        if e["adapter_path"].exists():
            by_model[e["model"]].append(e)

    for model_key, exp_list in by_model.items():
        model_shortcut = MODEL_SHORTCUT_REVERSE[model_key]
        full_model_name = MODEL_SHORTCUTS.get(model_shortcut, model_shortcut)

        print(f"\n{'='*60}")
        print(f"Loading base model: {full_model_name}")
        print(f"{'='*60}")

        model_config = ModelConfig(name=full_model_name, use_lora=False)
        base_model, tokenizer = load_model(full_model_name, model_config, for_distributed=False)
        base_model = base_model.to(device)

        for exp in exp_list:
            culture = exp["culture"]
            seed = exp["seed"]
            adapter_path = exp["adapter_path"]
            results_file = exp["results_file"]

            print(f"\n--- BiasUnlearn: {culture.upper()} {model_key} (seed={seed}) ---")

            # Load data
            print("  Loading data...")
            data = load_camellia_data(args.data_root, culture=culture, target_lang="cu")
            _, _, _, split_info = split_data(
                data, train_ratio=0.7, val_ratio=0.1, seed=seed,
                max_pairs_per_context=10, min_entities_per_split=2,
            )

            test_entities = split_info.get("test_entities", data.entities)

            # Baseline
            print("  [Baseline]...")
            baseline = run_evaluation(base_model, tokenizer, split_info, test_entities, device, "test")

            # Load adapter
            print(f"  [BiasUnlearn] Loading adapter from {adapter_path}...")
            trained_model = PeftModel.from_pretrained(base_model, str(adapter_path))
            trained_model = trained_model.merge_and_unload()
            trained_model = trained_model.to(device)

            trained_result = run_evaluation(trained_model, tokenizer, split_info, test_entities, device, "test")
            print(f"    CBS_g={trained_result['cbs_grounded']:.1f} CBS_n={trained_result['cbs_neutral']:.1f} Score={trained_result['combined_score']:.1f}")

            # Update results.json
            backup = results_file.parent / "results_pre_reeval.json"
            if not backup.exists():
                shutil.copy(results_file, backup)

            with open(results_file) as f:
                orig = json.load(f)

            old_score = orig.get("trained", {}).get("score", "?")
            new_score = round(trained_result["combined_score"], 2)
            print(f"  Score: {old_score} → {new_score}")

            orig["baseline"] = {
                "cbs_g": round(baseline["cbs_grounded"], 2),
                "cbs_n": round(baseline["cbs_neutral"], 2),
                "score": round(baseline["combined_score"], 2),
            }
            orig["trained"] = {
                "cbs_g": round(trained_result["cbs_grounded"], 2),
                "cbs_n": round(trained_result["cbs_neutral"], 2),
                "score": new_score,
            }
            orig["delta"] = {
                "cbs_g": round(orig["trained"]["cbs_g"] - orig["baseline"]["cbs_g"], 2),
                "cbs_n": round(orig["trained"]["cbs_n"] - orig["baseline"]["cbs_n"], 2),
                "score": round(orig["trained"]["score"] - orig["baseline"]["score"], 2),
            }

            with open(results_file, "w") as f:
                json.dump(orig, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Updated {results_file}")

            # Cleanup
            del trained_model
            torch.cuda.empty_cache()
            del base_model
            torch.cuda.empty_cache()
            base_model, tokenizer = load_model(full_model_name, model_config, for_distributed=False)
            base_model = base_model.to(device)

        del base_model
        torch.cuda.empty_cache()


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Re-evaluate with fixed REVERSE_ALIAS")
    p.add_argument("--method", default="cocoa",
                   choices=["cocoa", "biasedit", "biasunlearn", "all"],
                   help="Which method to re-evaluate")
    p.add_argument("--exp_dir", default="experiments")
    p.add_argument("--biasedit_dir", default="baselines/results/biasedit")
    p.add_argument("--biasunlearn_dir", default="baselines/results/biasunlearn")
    p.add_argument("--data_root", default="./dataset/camellia/raw")
    p.add_argument("--cultures", nargs="*", default=None,
                   help="Cultures to re-evaluate (default: affected ones)")
    p.add_argument("--all_cultures", action="store_true",
                   help="Re-evaluate ALL cultures, not just affected")
    p.add_argument("--models", nargs="*", default=["llama", "qwen"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry_run", action="store_true",
                   help="Just show what would be re-evaluated")
    args = p.parse_args()

    if args.all_cultures:
        args.cultures = ALL_CULTURES

    print("=" * 60)
    print("  Re-evaluation with Fixed REVERSE_ALIAS")
    print("=" * 60)
    print(f"  Method:   {args.method}")
    print(f"  Cultures: {args.cultures or AFFECTED_CULTURES}")
    print(f"  Models:   {args.models}")
    print(f"  Dry run:  {args.dry_run}")
    print()

    if args.method in ["cocoa", "all"]:
        print("\n>>> COCOA Re-evaluation")
        reeval_cocoa(args)

    if args.method in ["biasedit", "all"]:
        print("\n>>> BiasEdit Re-evaluation")
        reeval_biasedit(args)

    if args.method in ["biasunlearn", "all"]:
        print("\n>>> BiasUnlearn Re-evaluation")
        reeval_biasunlearn(args)

    print("\n✓ Re-evaluation complete!")
    print("  Remember to re-run: python make_main_table.py (to update main table)")
    print("  Remember to re-run: python analysis/run_logprob_gap_paired.py (if needed)")


if __name__ == "__main__":
    main()