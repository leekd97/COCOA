#!/usr/bin/env python3
"""
CoCoA: Prior Drift Verification

Checks if trained model's entity priors have drifted from base model,
and re-evaluates CBS with updated priors.

Usage:
    python analysis/verify_prior_drift.py \
        --checkpoint experiments/kfold_nxn_scaled_0.3/ko_cu_llama3-8b_.../checkpoints/best \
        --model llama3_8b \
        --culture ko --lang cu \
        --device cuda:0
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import load_camellia_data
from src.model import MODEL_SHORTCUTS
from src.evaluate import evaluate_robust_fair
from src.prior_utils import load_entity_priors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
LOG = logging.getLogger(__name__)


@torch.no_grad()
def compute_priors_batched(model, tokenizer, entities, device, bos, batch_size=32):
    """Compute log P(entity|BOS) for a list of entities."""
    ctx_enc = tokenizer(bos, return_tensors="pt", add_special_tokens=True)
    ctx_len = ctx_enc["input_ids"].size(1)

    results = {}
    texts = [bos + ent for ent in entities]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ents = entities[i:i+batch_size]

        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits, dim=-1)

        for j in range(len(batch_ents)):
            seq_len = attention_mask[j].sum().item()
            total_lp = 0.0
            for pos in range(ctx_len - 1, int(seq_len) - 1):
                next_token = input_ids[j, pos + 1]
                if next_token != tokenizer.pad_token_id:
                    total_lp += log_probs[j, pos, next_token].item()
            results[batch_ents[j]] = round(total_lp, 6)

    return results


def compute_all_priors(model, tokenizer, data, device):
    """Compute priors for all entities in a CamelliaData."""
    bos = tokenizer.bos_token or ""
    all_priors = {}
    for etype, ent_dict in data.entities.items():
        for side in ["asian", "western"]:
            batch_priors = compute_priors_batched(
                model, tokenizer, ent_dict[side], device, bos
            )
            all_priors.update(batch_priors)
    return all_priors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained LoRA checkpoint (e.g., .../checkpoints/best)")
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--culture", default="ko")
    parser.add_argument("--lang", default="cu")
    parser.add_argument("--data_root", default="./dataset/camellia/raw")
    parser.add_argument("--priors_root", default="./dataset/priors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--folds_root", default="./dataset/folds")
    args = parser.parse_args()

    device = torch.device(args.device)
    base_model_name = MODEL_SHORTCUTS.get(args.model, args.model)

    # =====================================================================
    # 1. Load base model priors (pre-computed)
    # =====================================================================
    LOG.info("Loading base model priors...")
    base_priors = load_entity_priors(args.priors_root, args.model, args.culture, args.lang)
    LOG.info(f"  {len(base_priors)} entities")

    # =====================================================================
    # 2. Load trained model (base + LoRA)
    # =====================================================================
    LOG.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOG.info(f"Loading LoRA from: {args.checkpoint}")
    trained_model = PeftModel.from_pretrained(base_model, args.checkpoint)
    trained_model = trained_model.eval()

    # =====================================================================
    # 3. Compute trained model priors
    # =====================================================================
    LOG.info("Computing trained model priors...")
    data = load_camellia_data(args.data_root, culture=args.culture, target_lang=args.lang)
    trained_priors = compute_all_priors(trained_model, tokenizer, data, device)
    LOG.info(f"  {len(trained_priors)} entities computed")

    # =====================================================================
    # 4. Compare prior drift
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"  Prior Drift Analysis: {args.culture.upper()} / {args.model}")
    print(f"{'='*80}")

    drifts = []
    asian_drifts = []
    western_drifts = []

    for etype, ent_dict in data.entities.items():
        for side in ["asian", "western"]:
            for ent in ent_dict[side]:
                bp = base_priors.get(ent, 0)
                tp = trained_priors.get(ent, 0)
                drift = tp - bp
                drifts.append(drift)
                if side == "asian":
                    asian_drifts.append(drift)
                else:
                    western_drifts.append(drift)

    drifts = np.array(drifts)
    asian_drifts = np.array(asian_drifts)
    western_drifts = np.array(western_drifts)

    print(f"\n  Overall drift (trained - base):")
    print(f"    Mean: {drifts.mean():+.3f}")
    print(f"    Std:  {drifts.std():.3f}")
    print(f"    Min:  {drifts.min():+.3f}")
    print(f"    Max:  {drifts.max():+.3f}")

    print(f"\n  Asian entity drift:")
    print(f"    Mean: {asian_drifts.mean():+.3f}")
    print(f"    Std:  {asian_drifts.std():.3f}")

    print(f"\n  Western entity drift:")
    print(f"    Mean: {western_drifts.mean():+.3f}")
    print(f"    Std:  {western_drifts.std():.3f}")

    print(f"\n  ★ Asymmetric drift: {asian_drifts.mean() - western_drifts.mean():+.3f}")
    print(f"    (positive = Asian priors rose more than Western)")

    # =====================================================================
    # 5. Re-evaluate CBS with both priors
    # =====================================================================
    LOG.info("\nRe-evaluating CBS...")

    from src.fold_utils import load_fold
    split_info = load_fold(args.folds_root, args.culture, args.lang, args.fold, args.seed)
    test_entities = split_info["test_entities"]

    # Eval with base priors
    LOG.info("  Eval with BASE priors...")
    result_base = evaluate_robust_fair(
        trained_model, tokenizer, split_info, test_entities,
        device=device, split="test", entity_priors=base_priors,
    )

    # Eval with trained priors
    LOG.info("  Eval with TRAINED priors...")
    result_trained = evaluate_robust_fair(
        trained_model, tokenizer, split_info, test_entities,
        device=device, split="test", entity_priors=trained_priors,
    )

    # Eval without priors (standard CBS)
    LOG.info("  Eval with NO priors (standard CBS)...")
    result_std = evaluate_robust_fair(
        trained_model, tokenizer, split_info, test_entities,
        device=device, split="test", entity_priors=None,
    )

    print(f"\n{'='*80}")
    print(f"  CBS Comparison")
    print(f"{'='*80}")
    print(f"  {'Metric':<12} │ {'Base Prior':>12} {'Trained Prior':>14} {'No Prior':>12}")
    print(f"  {'─'*55}")

    def get_cbs(r, key):
        return r.get(key, {}).get("overall", 0) or 0

    for ctx_type, short in [("grounded", "CBS_g"), ("neutral", "CBS_n")]:
        vb = get_cbs(result_base, ctx_type)
        vt = get_cbs(result_trained, ctx_type)
        vs = get_cbs(result_std, ctx_type)
        print(f"  {short:<12} │ {vb:>12.1f} {vt:>14.1f} {vs:>12.1f}")

    # Score
    def score(r):
        g = get_cbs(r, "grounded")
        n = get_cbs(r, "neutral")
        return g + abs(n - 50)

    print(f"  {'Score':<12} │ {score(result_base):>12.1f} {score(result_trained):>14.1f} {score(result_std):>12.1f}")
    print(f"\n  ★ If 'Trained Prior' CBS_n ≈ 50 → prior drift was the cause!")
    print(f"{'='*80}")

    # Save
    out = {
        "culture": args.culture,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "drift": {
            "overall_mean": float(drifts.mean()),
            "overall_std": float(drifts.std()),
            "asian_mean": float(asian_drifts.mean()),
            "western_mean": float(western_drifts.mean()),
            "asymmetric": float(asian_drifts.mean() - western_drifts.mean()),
        },
        "cbs_base_prior": {
            "cbs_g": get_cbs(result_base, "grounded"),
            "cbs_n": get_cbs(result_base, "neutral"),
            "score": score(result_base),
        },
        "cbs_trained_prior": {
            "cbs_g": get_cbs(result_trained, "grounded"),
            "cbs_n": get_cbs(result_trained, "neutral"),
            "score": score(result_trained),
        },
        "cbs_no_prior": {
            "cbs_g": get_cbs(result_std, "grounded"),
            "cbs_n": get_cbs(result_std, "neutral"),
            "score": score(result_std),
        },
    }

    out_path = Path(args.checkpoint).parent.parent / "prior_drift.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    LOG.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()