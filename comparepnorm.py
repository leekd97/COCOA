"""
CoCoA: Standard vs Prior-Normalized CBS Evaluation (Base Model)

Runs evaluate_robust_fair on base model with and without prior normalization.
No training — pure evaluation to quantify prior normalization's effect.

Usage:
    # Single culture
    python eval_pnorm_compare.py --culture ko --model llama3_8b --device cuda:0

    # All cultures
    python eval_pnorm_compare.py --model llama3_8b --device cuda:0

    # With K-Fold (evaluates on each fold's test split)
    python eval_pnorm_compare.py --culture ko --model llama3_8b --fold 0 --device cuda:0

    # Full dataset (no fold, uses 70/10/20 split)
    python eval_pnorm_compare.py --culture ko --model llama3_8b --seed 42 --device cuda:0
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_camellia_data, split_data
from src.evaluate import evaluate_robust_fair
from src.model import MODEL_SHORTCUTS
from src.prior_utils import load_entity_priors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

ALL_CULTURES = ["ko", "ja", "zh", "vi", "hi", "ur", "gu", "mr", "ml", "ar"]


def evaluate_single(
    model, tokenizer, split_info, entities, device,
    split, entity_priors, show_progress,
):
    """Run evaluate_robust_fair with optional priors."""
    model.eval()
    with torch.no_grad():
        results = evaluate_robust_fair(
            model, tokenizer, split_info, entities, device,
            split=split, max_contexts=None, max_entities=30,
            show_progress=show_progress,
            entity_priors=entity_priors,
        )
    cbs_g = results["grounded"]["overall"] or 50.0
    cbs_n = results["neutral"]["overall"] or 50.0
    score = cbs_g + abs(cbs_n - 50)
    return {
        "cbs_g": round(cbs_g, 2),
        "cbs_n": round(cbs_n, 2),
        "score": round(score, 2),
        "by_category": results,
    }


def run_culture(culture, lang, model_key, model, tokenizer, device, args):
    """Run standard + normalized evaluation for one culture."""
    LOG.info(f"\n{'='*60}")
    LOG.info(f"  {culture}/{lang} — {model_key}")
    LOG.info(f"{'='*60}")

    # Load data
    data = load_camellia_data(args.data_root, culture=culture, target_lang=lang)

    # Use FULL dataset (no split — base model has no leakage concern)
    split_info = {
        "grounded_test": data.grounded_contexts,
        "neutral_test": data.neutral_contexts,
        "test_entities": data.entities,
    }
    split = "test"
    entities = data.entities

    n_g = len(data.grounded_contexts)
    n_n = len(data.neutral_contexts)
    n_a = sum(len(v["asian"]) for v in data.entities.values())
    n_w = sum(len(v["western"]) for v in data.entities.values())
    LOG.info(f"  Full dataset: {n_g}G + {n_n}N contexts, {n_a}A + {n_w}W entities")

    # Load priors
    priors = load_entity_priors(args.priors_root, model_key, culture, lang)
    LOG.info(f"  Loaded {len(priors)} entity priors")

    # Standard CBS
    LOG.info(f"  Evaluating STANDARD CBS...")
    std_result = evaluate_single(
        model, tokenizer, split_info, entities, device,
        split, entity_priors=None, show_progress=args.show_progress,
    )

    # Normalized CBS
    LOG.info(f"  Evaluating NORMALIZED CBS...")
    norm_result = evaluate_single(
        model, tokenizer, split_info, entities, device,
        split, entity_priors=priors, show_progress=args.show_progress,
    )

    # Print comparison
    delta_g = norm_result["cbs_g"] - std_result["cbs_g"]
    delta_n = norm_result["cbs_n"] - std_result["cbs_n"]
    delta_sc = norm_result["score"] - std_result["score"]

    LOG.info(f"  Standard:   CBS_g={std_result['cbs_g']:>6.1f}  CBS_n={std_result['cbs_n']:>6.1f}  Score={std_result['score']:>6.1f}")
    LOG.info(f"  Normalized: CBS_g={norm_result['cbs_g']:>6.1f}  CBS_n={norm_result['cbs_n']:>6.1f}  Score={norm_result['score']:>6.1f}")
    LOG.info(f"  Delta:      CBS_g={delta_g:>+6.1f}  CBS_n={delta_n:>+6.1f}  Score={delta_sc:>+6.1f}")

    # Category detail
    std_by_cat = std_result["by_category"]
    norm_by_cat = norm_result["by_category"]

    for ctx_type in ["grounded", "neutral"]:
        std_cats = std_by_cat.get(ctx_type, {}).get("by_category", {})
        norm_cats = norm_by_cat.get(ctx_type, {}).get("by_category", {})
        if std_cats:
            LOG.info(f"  [{ctx_type}] per category:")
            for cat in sorted(std_cats.keys()):
                s = std_cats.get(cat, 50.0)
                n = norm_cats.get(cat, 50.0)
                LOG.info(f"    {cat:<18} Std={s:>6.1f}  Norm={n:>6.1f}  Δ={n-s:>+6.1f}")

    return {
        "culture": culture,
        "lang": lang,
        "model": model_key,
        "standard": std_result,
        "normalized": norm_result,
        "delta": {
            "cbs_g": round(delta_g, 2),
            "cbs_n": round(delta_n, 2),
            "score": round(delta_sc, 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="CoCoA: Standard vs Normalized CBS Comparison")
    parser.add_argument("--cultures", nargs="+", default=None,
                        help=f"Cultures (default: all)")
    parser.add_argument("--culture", default=None, help="Single culture shortcut")
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--lang", default="cu", choices=["cu", "en"])
    parser.add_argument("--seed", type=int, default=42, help="Split seed (if not using fold)")
    parser.add_argument("--fold", type=int, default=None, help="Fold index")
    parser.add_argument("--fold_seed", type=int, default=42, help="Seed used to generate folds")
    parser.add_argument("--folds_root", default="./dataset/folds")
    parser.add_argument("--priors_root", default="./dataset/priors")
    parser.add_argument("--data_root", default="./dataset/camellia/raw")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--show_progress", action="store_true", default=True)
    parser.add_argument("--no_progress", dest="show_progress", action="store_false")
    args = parser.parse_args()

    cultures = [args.culture] if args.culture else (args.cultures or ALL_CULTURES)

    model_name = MODEL_SHORTCUTS.get(args.model, args.model)
    device = torch.device(args.device)

    LOG.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}
    for culture in cultures:
        try:
            result = run_culture(culture, args.lang, args.model, model, tokenizer, device, args)
            all_results[culture] = result
        except Exception as e:
            LOG.error(f"Failed for {culture}: {e}")
            import traceback; traceback.print_exc()

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*90}")
        print(f"  SUMMARY: Standard vs Normalized CBS (Base Model, {args.model})")
        print(f"{'='*90}")
        print(f"  {'Culture':<8} │ {'Std_g':>7} {'Nrm_g':>7} {'Δg':>7} │ "
              f"{'Std_n':>7} {'Nrm_n':>7} {'Δn':>7} │ {'Std_Sc':>7} {'Nrm_Sc':>7} {'ΔSc':>7}")
        print(f"  {'─'*82}")

        for culture, r in all_results.items():
            s, n, d = r["standard"], r["normalized"], r["delta"]
            print(f"  {culture.upper():<8} │ {s['cbs_g']:>7.1f} {n['cbs_g']:>7.1f} {d['cbs_g']:>+7.1f} │ "
                  f"{s['cbs_n']:>7.1f} {n['cbs_n']:>7.1f} {d['cbs_n']:>+7.1f} │ "
                  f"{s['score']:>7.1f} {n['score']:>7.1f} {d['score']:>+7.1f}")

    # Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        LOG.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()