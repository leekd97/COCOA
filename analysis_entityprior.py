"""
COCOA Analysis Phase 2: Entity Prior & Normalized CBS (GPU Required)

Measures:
  1. Unconditional log P(entity) for every entity (BOS-only context)
  2. Conditional log P(entity | context) for sampled contexts
  3. Normalized score: log P(entity|ctx) - log P(entity)  (= PMI-style)
  4. CBS comparison: standard vs prior-normalized

Usage:
    # Single culture, single model
    python analysis_phase2_entity_prior.py \
        --culture ko --lang cu --model llama3_8b --device cuda:0

    # Multiple cultures
    python analysis_phase2_entity_prior.py \
        --cultures ko zh ar --lang cu --model llama3_8b --device cuda:0

    # Both models
    python analysis_phase2_entity_prior.py \
        --cultures ko zh ar --lang cu --model llama3_8b --device cuda:0
    python analysis_phase2_entity_prior.py \
        --cultures ko zh ar --lang cu --model qwen3_8b --device cuda:1
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_camellia_data, split_data
from src.model import MODEL_SHORTCUTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)


# =========================================================================
# Core: Log Probability Computation
# =========================================================================

@torch.no_grad()
def compute_entity_logprob(
    model,
    tokenizer,
    context_before: str,
    entity: str,
    device: torch.device,
) -> float:
    """
    Compute log P(entity | context_before).
    Sum of log probs for entity tokens conditioned on context.
    """
    full_text = context_before + entity
    
    ctx_enc = tokenizer(context_before, return_tensors="pt", add_special_tokens=True)
    full_enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
    
    ctx_len = ctx_enc["input_ids"].size(1)
    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    
    total_lp = 0.0
    for i in range(ctx_len - 1, input_ids.size(1) - 1):
        next_token = input_ids[0, i + 1]
        total_lp += log_probs[0, i, next_token].item()
    
    return total_lp


@torch.no_grad()
def compute_unconditional_logprob(
    model,
    tokenizer,
    entity: str,
    device: torch.device,
) -> float:
    """
    Compute log P(entity) — unconditional (BOS-only context).
    This is the entity's "prior" in the model.
    """
    # BOS token only as context
    bos = tokenizer.bos_token or ""
    return compute_entity_logprob(model, tokenizer, bos, entity, device)


@torch.no_grad()
def compute_entity_logprobs_batched(
    model,
    tokenizer,
    context_before: str,
    entities: List[str],
    device: torch.device,
    batch_size: int = 32,
) -> List[float]:
    """Batched version of compute_entity_logprob for speed."""
    if not entities:
        return []
    
    ctx_enc = tokenizer(context_before, return_tensors="pt", add_special_tokens=True)
    ctx_len = ctx_enc["input_ids"].size(1)
    
    texts = [context_before + ent for ent in entities]
    all_logprobs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        
        for j in range(len(batch_texts)):
            seq_len = attention_mask[j].sum().item()
            total_lp = 0.0
            for pos in range(ctx_len - 1, int(seq_len) - 1):
                next_token = input_ids[j, pos + 1]
                if next_token != tokenizer.pad_token_id:
                    total_lp += log_probs[j, pos, next_token].item()
            all_logprobs.append(total_lp)
    
    return all_logprobs


# =========================================================================
# 1. Entity Prior Analysis
# =========================================================================

def measure_entity_priors(
    model, tokenizer, entities: Dict, device: torch.device,
) -> Dict:
    """
    Measure log P(entity) for all entities across all categories.
    
    Returns:
        {etype: {"asian": {entity: logprob}, "western": {entity: logprob}}}
    """
    bos = tokenizer.bos_token or ""
    results = {}
    
    for etype, ent_dict in entities.items():
        results[etype] = {"asian": {}, "western": {}}
        
        for side in ["asian", "western"]:
            ent_list = ent_dict[side]
            LOG.info(f"  Computing priors: {etype}/{side} ({len(ent_list)} entities)")
            
            logprobs = compute_entity_logprobs_batched(
                model, tokenizer, bos, ent_list, device,
            )
            for ent, lp in zip(ent_list, logprobs):
                results[etype][side][ent] = lp
    
    return results


def analyze_prior_gap(priors: Dict) -> Dict:
    """
    Analyze the gap between Asian and Western entity priors.
    This is the "constant offset" that biases CBS regardless of context.
    """
    results = {}
    for etype in priors:
        asian_lps = list(priors[etype]["asian"].values())
        western_lps = list(priors[etype]["western"].values())
        
        if not asian_lps or not western_lps:
            continue
        
        mean_a = np.mean(asian_lps)
        mean_w = np.mean(western_lps)
        gap = mean_w - mean_a  # positive = Western has higher prior
        
        results[etype] = {
            "mean_asian_prior": round(float(mean_a), 4),
            "mean_western_prior": round(float(mean_w), 4),
            "prior_gap_w_minus_a": round(float(gap), 4),
            "gap_direction": "Western higher" if gap > 0 else "Asian higher",
            "std_asian": round(float(np.std(asian_lps)), 4),
            "std_western": round(float(np.std(western_lps)), 4),
            "min_asian": round(float(min(asian_lps)), 4),
            "max_asian": round(float(max(asian_lps)), 4),
            "min_western": round(float(min(western_lps)), 4),
            "max_western": round(float(max(western_lps)), 4),
        }
    
    # Overall
    all_a = [lp for e in priors for lp in priors[e]["asian"].values()]
    all_w = [lp for e in priors for lp in priors[e]["western"].values()]
    if all_a and all_w:
        results["_overall"] = {
            "mean_asian_prior": round(float(np.mean(all_a)), 4),
            "mean_western_prior": round(float(np.mean(all_w)), 4),
            "prior_gap_w_minus_a": round(float(np.mean(all_w) - np.mean(all_a)), 4),
        }
    
    return results


# =========================================================================
# 2. CBS: Standard vs Normalized
# =========================================================================

def compute_cbs_comparison(
    model, tokenizer, split_info: Dict, priors: Dict,
    device: torch.device,
    split: str = "test",
    max_contexts: int = 20,
    max_entities: int = 15,
) -> Dict:
    """
    Compare standard CBS vs prior-normalized CBS.
    
    Standard:    compare log P(entity | ctx)
    Normalized:  compare log P(entity | ctx) - log P(entity)
    """
    from src.data import ENTITY_TYPE_ALIASES
    
    REVERSE_ALIAS = {
        "locations": ["location", "locations"],
        "sports": ["sport", "sports", "sports-clubs"],
        "authors": ["author", "authors"],
        "names-male": ["names", "name", "names (m)", "name (m)", "names-male"],
        "names-female": ["names", "name", "names (f)", "name (f)", "names-female"],
        "beverage": ["beverage"],
        "food": ["food"],
    }
    
    entities = split_info[f"{split}_entities"]
    results = {}
    
    for ctx_type in ["grounded", "neutral"]:
        ctx_df = split_info[f"{ctx_type}_{split}"]
        
        std_western_wins = 0
        norm_western_wins = 0
        total = 0
        by_category = defaultdict(lambda: {
            "std_w": 0, "norm_w": 0, "total": 0,
            "prior_flips": 0,  # cases where normalization changes the winner
        })
        
        for etype in entities:
            possible_names = REVERSE_ALIAS.get(etype, [etype])
            if "entity_type" not in ctx_df.columns:
                continue
            mask = ctx_df["entity_type"].str.lower().isin([n.lower() for n in possible_names])
            type_df = ctx_df[mask]
            
            if len(type_df) == 0:
                continue
            
            contexts = type_df["context"].tolist()
            if max_contexts:
                contexts = contexts[:max_contexts]
            
            asian_ents = entities[etype]["asian"][:max_entities]
            western_ents = entities[etype]["western"][:max_entities]
            
            if not asian_ents or not western_ents:
                continue
            
            # Get priors for these entities
            asian_priors = [priors.get(etype, {}).get("asian", {}).get(e, 0.0) for e in asian_ents]
            western_priors = [priors.get(etype, {}).get("western", {}).get(e, 0.0) for e in western_ents]
            
            for ctx in tqdm(contexts, desc=f"{ctx_type}/{etype}", leave=False):
                parts = ctx.split("[MASK]")
                ctx_before = parts[0] if len(parts) >= 2 else ctx
                
                a_lps = compute_entity_logprobs_batched(
                    model, tokenizer, ctx_before, asian_ents, device,
                )
                w_lps = compute_entity_logprobs_batched(
                    model, tokenizer, ctx_before, western_ents, device,
                )
                
                # N × M comparisons
                for i, (a_lp, a_prior) in enumerate(zip(a_lps, asian_priors)):
                    for j, (w_lp, w_prior) in enumerate(zip(w_lps, western_priors)):
                        # Standard
                        std_w_wins = w_lp > a_lp
                        
                        # Normalized (PMI-style)
                        a_norm = a_lp - a_prior
                        w_norm = w_lp - w_prior
                        norm_w_wins = w_norm > a_norm
                        
                        if std_w_wins:
                            std_western_wins += 1
                            by_category[etype]["std_w"] += 1
                        if norm_w_wins:
                            norm_western_wins += 1
                            by_category[etype]["norm_w"] += 1
                        if std_w_wins != norm_w_wins:
                            by_category[etype]["prior_flips"] += 1
                        
                        total += 1
                        by_category[etype]["total"] += 1
        
        # Compute CBS
        std_cbs = (std_western_wins / total * 100) if total > 0 else 50.0
        norm_cbs = (norm_western_wins / total * 100) if total > 0 else 50.0
        
        cat_results = {}
        for etype, v in by_category.items():
            t = v["total"]
            cat_results[etype] = {
                "std_cbs": round(v["std_w"] / t * 100, 2) if t > 0 else 50.0,
                "norm_cbs": round(v["norm_w"] / t * 100, 2) if t > 0 else 50.0,
                "prior_flip_rate": round(v["prior_flips"] / t * 100, 2) if t > 0 else 0.0,
                "n_comparisons": t,
            }
        
        results[ctx_type] = {
            "standard_cbs": round(std_cbs, 2),
            "normalized_cbs": round(norm_cbs, 2),
            "delta": round(norm_cbs - std_cbs, 2),
            "total_comparisons": total,
            "total_prior_flips": sum(v["prior_flips"] for v in by_category.values()),
            "flip_rate_pct": round(
                sum(v["prior_flips"] for v in by_category.values()) / max(total, 1) * 100, 2
            ),
            "by_category": cat_results,
        }
    
    return results


# =========================================================================
# Pretty Printing
# =========================================================================

def print_prior_analysis(gap_analysis: Dict, culture: str, model_short: str):
    print(f"\n{'='*70}")
    print(f"  Entity Prior Gap: {culture} / {model_short}")
    print(f"{'='*70}")
    print(f"{'Category':<18} {'Mean_A':>10} {'Mean_W':>10} {'Gap(W-A)':>10} {'Direction':<16}")
    print("-" * 66)
    for etype, v in gap_analysis.items():
        if etype.startswith("_"):
            print("-" * 66)
        print(f"{etype:<18} {v['mean_asian_prior']:>10.3f} {v['mean_western_prior']:>10.3f} "
              f"{v['prior_gap_w_minus_a']:>+10.3f} {v.get('gap_direction', ''):>16}")


def print_cbs_comparison(cbs_comp: Dict, culture: str, model_short: str):
    print(f"\n{'='*70}")
    print(f"  CBS Comparison: {culture} / {model_short}")
    print(f"{'='*70}")
    for ctx_type in ["grounded", "neutral"]:
        v = cbs_comp[ctx_type]
        print(f"\n  [{ctx_type.upper()}]")
        print(f"    Standard CBS: {v['standard_cbs']:.1f}%")
        print(f"    Normalized CBS: {v['normalized_cbs']:.1f}%")
        print(f"    Delta: {v['delta']:+.1f}% ({v['total_comparisons']} comparisons)")
        print(f"    Prior flips: {v['total_prior_flips']} ({v['flip_rate_pct']:.1f}%)")
        
        print(f"\n    {'Category':<18} {'Std_CBS':>10} {'Norm_CBS':>10} {'Flips%':>10}")
        print(f"    {'-'*50}")
        for etype, cat in v["by_category"].items():
            print(f"    {etype:<18} {cat['std_cbs']:>10.1f} {cat['norm_cbs']:>10.1f} "
                  f"{cat['prior_flip_rate']:>10.1f}")


# =========================================================================
# Main
# =========================================================================

def run_single(culture, lang, model_key, data_root, seed, device_str, output_dir,
               max_contexts, max_entities):
    """Run full Phase 2 analysis for one culture/model."""
    model_name = MODEL_SHORTCUTS.get(model_key, model_key)
    model_short = model_key.replace("/", "_")
    device = torch.device(device_str)
    
    LOG.info(f"Culture: {culture}, Lang: {lang}, Model: {model_name}")
    
    # Load data
    data = load_camellia_data(data_root, culture=culture, target_lang=lang)
    _, _, _, split_info = split_data(data, seed=seed)
    
    # Load model
    LOG.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Measure entity priors
    LOG.info("Measuring entity priors (unconditional log P(entity))...")
    priors = measure_entity_priors(model, tokenizer, data.entities, device)
    gap_analysis = analyze_prior_gap(priors)
    print_prior_analysis(gap_analysis, culture, model_short)
    
    # 2. CBS comparison
    LOG.info("Computing CBS comparison (standard vs normalized)...")
    cbs_comp = compute_cbs_comparison(
        model, tokenizer, split_info, priors, device,
        split="test", max_contexts=max_contexts, max_entities=max_entities,
    )
    print_cbs_comparison(cbs_comp, culture, model_short)
    
    # Save
    result = {
        "culture": culture,
        "lang": lang,
        "model": model_name,
        "model_key": model_key,
        "seed": seed,
        "prior_gap_analysis": gap_analysis,
        "entity_priors": {
            etype: {
                side: {ent: round(lp, 4) for ent, lp in ents.items()}
                for side, ents in sides.items()
            }
            for etype, sides in priors.items()
        },
        "cbs_comparison": cbs_comp,
    }
    
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fname = out_path / f"phase2_{culture}_{lang}_{model_short}_seed{seed}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        LOG.info(f"Saved: {fname}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="COCOA Phase 2: Entity Prior Analysis")
    parser.add_argument("--cultures", nargs="+", default=["ko"],
                        help="Cultures to analyze")
    parser.add_argument("--culture", default=None, help="Single culture (shortcut)")
    parser.add_argument("--lang", default="cu", choices=["cu", "en"])
    parser.add_argument("--model", default="llama3_8b",
                        help=f"Shortcuts: {list(MODEL_SHORTCUTS.keys())}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", default="./dataset/camellia/raw")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="./analysis/phase2")
    parser.add_argument("--max_contexts", type=int, default=20,
                        help="Max contexts per category for CBS comparison")
    parser.add_argument("--max_entities", type=int, default=15,
                        help="Max entities per side for CBS comparison")
    args = parser.parse_args()
    
    cultures = [args.culture] if args.culture else args.cultures
    
    LOG.info(f"COCOA Phase 2: Entity Prior Analysis")
    LOG.info(f"Cultures: {cultures}, Model: {args.model}")
    
    all_results = {}
    for culture in cultures:
        try:
            result = run_single(
                culture, args.lang, args.model, args.data_root,
                args.seed, args.device, args.output_dir,
                args.max_contexts, args.max_entities,
            )
            all_results[culture] = result
        except Exception as e:
            LOG.error(f"ERROR for {culture}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: Prior Gap & CBS Impact")
        print(f"{'='*70}")
        print(f"{'Culture':<10} {'PriorGap':>10} {'Std_CBS_g':>10} {'Norm_CBS_g':>10} "
              f"{'Std_CBS_n':>10} {'Norm_CBS_n':>10} {'Flip_g%':>8} {'Flip_n%':>8}")
        print("-" * 78)
        for culture, r in all_results.items():
            pg = r["prior_gap_analysis"].get("_overall", {}).get("prior_gap_w_minus_a", 0)
            cbs = r["cbs_comparison"]
            print(f"{culture:<10} {pg:>+10.3f} "
                  f"{cbs['grounded']['standard_cbs']:>10.1f} {cbs['grounded']['normalized_cbs']:>10.1f} "
                  f"{cbs['neutral']['standard_cbs']:>10.1f} {cbs['neutral']['normalized_cbs']:>10.1f} "
                  f"{cbs['grounded']['flip_rate_pct']:>8.1f} {cbs['neutral']['flip_rate_pct']:>8.1f}")


if __name__ == "__main__":
    main()