"""
CoCoA Analysis Phase 3: Cross-Lingual Entity Prior Analysis

Measures feasibility of cross-cultural negative training by analyzing:
  A. Unconditional prior: log P(entity|BOS) for entities from ALL cultures
  B. In-context prior: log P(entity|target_culture_context) for all culture entities
  C. Context penalty: B - A (how much does target-language context suppress foreign entities)

Pilot: ko context × all 10 cultures' entities (expandable to other ctx cultures)

Usage:
    # Pilot: ko contexts, Llama
    python analysis_phase3_cross_cultural.py \
        --ctx_culture ko --model llama3_8b --device cuda:0

    # Multiple context cultures
    python analysis_phase3_cross_cultural.py \
        --ctx_culture ko ja zh --model llama3_8b --device cuda:0

    # Both models
    python analysis_phase3_cross_cultural.py \
        --ctx_culture ko --model llama3_8b --device cuda:0
    python analysis_phase3_cross_cultural.py \
        --ctx_culture ko --model qwen3_8b --device cuda:1
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_camellia_data, LANGUAGE_NAME_MAP, LANGUAGE_CODE_MAP
from src.model import MODEL_SHORTCUTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

ALL_CULTURES = ["ko", "ja", "zh", "vi", "hi", "ur", "gu", "mr", "ml", "ar"]
CATEGORIES = ["authors", "beverage", "food", "locations", "names-female", "names-male", "sports"]


# =========================================================================
# Log Prob Computation (reused from Phase 2)
# =========================================================================

@torch.no_grad()
def compute_entity_logprobs_batched(
    model, tokenizer, context_before: str, entities: List[str],
    device: torch.device, batch_size: int = 32,
) -> List[float]:
    """Compute log P(entity | context_before) for a list of entities."""
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
            n_tokens = 0
            for pos in range(ctx_len - 1, int(seq_len) - 1):
                next_token = input_ids[j, pos + 1]
                if next_token != tokenizer.pad_token_id:
                    total_lp += log_probs[j, pos, next_token].item()
                    n_tokens += 1
            all_logprobs.append(total_lp)

    return all_logprobs


# =========================================================================
# Data Loading: Multi-Culture Entities
# =========================================================================

def load_all_culture_entities(
    data_root: str,
    cultures: List[str],
    max_entities_per_side: int = 15,
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Load native-language entities for multiple cultures.

    Returns:
        {culture: {category: {"asian": [entities], "western": [entities]}}}
        Western entities are in the CULTURE's native language.
    """
    all_entities = {}

    for culture in cultures:
        LOG.info(f"  Loading entities: {culture}")
        try:
            data = load_camellia_data(data_root, culture=culture, target_lang="cu")
            culture_ents = {}
            for cat, ent_dict in data.entities.items():
                asian = ent_dict["asian"][:max_entities_per_side]
                western = ent_dict["western"][:max_entities_per_side]
                if asian:  # only need asian for cross-cultural negative
                    culture_ents[cat] = {
                        "asian": asian,
                        "western": western,
                    }
            all_entities[culture] = culture_ents
            total_a = sum(len(v["asian"]) for v in culture_ents.values())
            LOG.info(f"    {len(culture_ents)} categories, {total_a} Asian entities")
        except Exception as e:
            LOG.warning(f"    Failed: {e}")
            continue

    return all_entities


# =========================================================================
# A. Unconditional Prior Matrix
# =========================================================================

def measure_unconditional_priors(
    model, tokenizer, all_entities: Dict, device: torch.device,
) -> Dict:
    """
    Measure log P(entity|BOS) for all cultures' Asian entities.
    Returns: {culture: {category: {"mean": float, "std": float, "values": [...]}}}
    """
    bos = tokenizer.bos_token or ""
    results = {}

    for culture, cats in all_entities.items():
        results[culture] = {}
        for cat, ent_dict in cats.items():
            entities = ent_dict["asian"]
            if not entities:
                continue
            lps = compute_entity_logprobs_batched(model, tokenizer, bos, entities, device)
            results[culture][cat] = {
                "mean": float(np.mean(lps)),
                "std": float(np.std(lps)),
                "n": len(lps),
            }

    return results


# =========================================================================
# B. In-Context Prior (target culture context × all culture entities)
# =========================================================================

def measure_in_context_priors(
    model, tokenizer,
    ctx_culture: str,
    all_entities: Dict,
    data_root: str,
    device: torch.device,
    max_contexts: int = 10,
) -> Dict:
    """
    Measure log P(entity|target_culture_context) for entities from ALL cultures.

    Returns: {entity_culture: {category: {"mean": float, "std": float}}}
    """
    # Load contexts from target culture
    ctx_data = load_camellia_data(data_root, culture=ctx_culture, target_lang="cu")

    results = {}

    for ent_culture, cats in all_entities.items():
        results[ent_culture] = {}

        for cat, ent_dict in cats.items():
            entities = ent_dict["asian"]
            if not entities:
                continue

            # Find matching contexts (same category)
            # Try to match entity category to context entity_type
            grounded_df = ctx_data.grounded_contexts
            if "entity_type" not in grounded_df.columns:
                continue

            # Fuzzy match category names
            cat_aliases = {
                "authors": ["author", "authors"],
                "beverage": ["beverage"],
                "food": ["food"],
                "locations": ["location", "locations"],
                "names-female": ["names", "name", "names-female", "names (f)", "name (f)"],
                "names-male": ["names", "name", "names-male", "names (m)", "name (m)"],
                "sports": ["sport", "sports", "sports clubs", "sports-clubs"],
            }
            possible = cat_aliases.get(cat, [cat])
            mask = grounded_df["entity_type"].str.lower().isin([n.lower() for n in possible])
            type_df = grounded_df[mask]

            if len(type_df) == 0:
                continue

            contexts = type_df["context"].tolist()[:max_contexts]

            # Compute log probs for each context
            all_lps = []
            for ctx in contexts:
                parts = ctx.split("[MASK]")
                ctx_before = parts[0] if len(parts) >= 2 else ctx

                lps = compute_entity_logprobs_batched(
                    model, tokenizer, ctx_before, entities, device
                )
                all_lps.extend(lps)

            if all_lps:
                results[ent_culture][cat] = {
                    "mean": float(np.mean(all_lps)),
                    "std": float(np.std(all_lps)),
                    "n_entities": len(entities),
                    "n_contexts": len(contexts),
                    "n_measurements": len(all_lps),
                }

    return results


# =========================================================================
# C. Context Penalty Computation
# =========================================================================

def compute_context_penalty(
    unconditional: Dict, in_context: Dict,
) -> Dict:
    """
    Context penalty = in_context_mean - unconditional_mean

    Negative = context suppresses the entity (expected for foreign-language entities)
    Near zero = context doesn't additionally suppress beyond prior
    """
    penalties = {}

    for ent_culture in in_context:
        penalties[ent_culture] = {}
        for cat in in_context[ent_culture]:
            ic = in_context[ent_culture][cat]["mean"]
            uc = unconditional.get(ent_culture, {}).get(cat, {}).get("mean")

            if uc is not None:
                penalty = ic - uc
                penalties[ent_culture][cat] = {
                    "unconditional": round(uc, 3),
                    "in_context": round(ic, 3),
                    "penalty": round(penalty, 3),
                }

    return penalties


# =========================================================================
# Pretty Printing
# =========================================================================

def print_results(ctx_culture, unconditional, in_context, penalties, all_entities):
    # A. Unconditional prior summary
    print(f"\n{'='*80}")
    print(f"  A. Unconditional Prior: mean log P(entity|BOS)")
    print(f"     (all entities in their native language)")
    print(f"{'='*80}")
    print(f"{'Culture':<8}", end="")
    for cat in CATEGORIES:
        print(f" {cat[:6]:>8}", end="")
    print(f" {'OVERALL':>10}")
    print("-" * 80)

    for culture in ALL_CULTURES:
        if culture not in unconditional:
            continue
        print(f"{culture.upper():<8}", end="")
        all_vals = []
        for cat in CATEGORIES:
            v = unconditional[culture].get(cat, {}).get("mean")
            if v is not None:
                print(f" {v:>8.2f}", end="")
                all_vals.append(v)
            else:
                print(f" {'—':>8}", end="")
        if all_vals:
            print(f" {np.mean(all_vals):>10.2f}", end="")
        print()

    # B. In-context prior
    print(f"\n{'='*80}")
    print(f"  B. In-Context Prior: mean log P(entity|{ctx_culture.upper()} context)")
    print(f"     (entities from each culture, inserted into {ctx_culture.upper()} grounded contexts)")
    print(f"{'='*80}")
    print(f"{'EntCult':<8}", end="")
    for cat in CATEGORIES:
        print(f" {cat[:6]:>8}", end="")
    print(f" {'OVERALL':>10}")
    print("-" * 80)

    for culture in ALL_CULTURES:
        if culture not in in_context:
            continue
        print(f"{culture.upper():<8}", end="")
        all_vals = []
        for cat in CATEGORIES:
            v = in_context[culture].get(cat, {}).get("mean")
            if v is not None:
                print(f" {v:>8.2f}", end="")
                all_vals.append(v)
            else:
                print(f" {'—':>8}", end="")
        if all_vals:
            print(f" {np.mean(all_vals):>10.2f}", end="")
        print()

    # C. Context penalty
    print(f"\n{'='*80}")
    print(f"  C. Context Penalty: in_context - unconditional")
    print(f"     (negative = {ctx_culture.upper()} context suppresses foreign entities)")
    print(f"     (near zero = no additional suppression beyond prior)")
    print(f"{'='*80}")
    print(f"{'EntCult':<8}", end="")
    for cat in CATEGORIES:
        print(f" {cat[:6]:>8}", end="")
    print(f" {'OVERALL':>10}")
    print("-" * 80)

    penalty_matrix = {}
    for culture in ALL_CULTURES:
        if culture not in penalties:
            continue
        print(f"{culture.upper():<8}", end="")
        all_vals = []
        for cat in CATEGORIES:
            v = penalties[culture].get(cat, {}).get("penalty")
            if v is not None:
                print(f" {v:>+8.2f}", end="")
                all_vals.append(v)
            else:
                print(f" {'—':>8}", end="")
        avg = np.mean(all_vals) if all_vals else None
        if avg is not None:
            print(f" {avg:>+10.2f}", end="")
            penalty_matrix[culture] = avg
        print()

    # D. Summary: which cultures are feasible as cross-cultural negatives?
    print(f"\n{'='*80}")
    print(f"  D. Feasibility Summary for Cross-Cultural Negatives")
    print(f"     (relative to {ctx_culture.upper()} as target culture)")
    print(f"{'='*80}")

    if ctx_culture in penalty_matrix:
        self_penalty = penalty_matrix[ctx_culture]
        print(f"  {ctx_culture.upper()} (self): penalty = {self_penalty:+.2f} (reference)")
        print()
        print(f"  {'Culture':<8} {'Penalty':>10} {'Δ from self':>12} {'Feasibility':>14}")
        print(f"  {'-'*46}")

        for culture in ALL_CULTURES:
            if culture == ctx_culture or culture not in penalty_matrix:
                continue
            p = penalty_matrix[culture]
            delta = p - self_penalty
            # Feasibility: if additional penalty (beyond self) is small
            if abs(delta) < 3:
                feasibility = "✓ GOOD"
            elif abs(delta) < 6:
                feasibility = "△ MARGINAL"
            else:
                feasibility = "✗ RISKY"
            print(f"  {culture.upper():<8} {p:>+10.2f} {delta:>+12.2f} {feasibility:>14}")


# =========================================================================
# Main
# =========================================================================

def run_single(ctx_culture, model_key, data_root, device_str, output_dir,
               max_contexts, max_entities):
    model_name = MODEL_SHORTCUTS.get(model_key, model_key)
    model_short = model_key.replace("/", "_")
    device = torch.device(device_str)

    LOG.info(f"Context culture: {ctx_culture}, Model: {model_name}")

    # Load model
    LOG.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load entities from ALL cultures (native language)
    LOG.info("Loading entities from all cultures...")
    all_entities = load_all_culture_entities(
        data_root, ALL_CULTURES, max_entities_per_side=max_entities,
    )

    # A. Unconditional priors
    LOG.info("A. Measuring unconditional priors (BOS)...")
    unconditional = measure_unconditional_priors(model, tokenizer, all_entities, device)

    # B. In-context priors
    LOG.info(f"B. Measuring in-context priors ({ctx_culture} contexts)...")
    in_context = measure_in_context_priors(
        model, tokenizer, ctx_culture, all_entities,
        data_root, device, max_contexts,
    )

    # C. Context penalty
    LOG.info("C. Computing context penalties...")
    penalties = compute_context_penalty(unconditional, in_context)

    # Print results
    print_results(ctx_culture, unconditional, in_context, penalties, all_entities)

    # Save
    result = {
        "ctx_culture": ctx_culture,
        "model": model_name,
        "model_key": model_key,
        "unconditional": unconditional,
        "in_context": in_context,
        "penalties": {
            ent_c: {
                cat: v for cat, v in cats.items()
            } for ent_c, cats in penalties.items()
        },
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fname = out_path / f"phase3_{ctx_culture}_{model_short}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        LOG.info(f"Saved: {fname}")

    del model
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="CoCoA Phase 3: Cross-Cultural Entity Analysis")
    parser.add_argument("--ctx_culture", nargs="+", default=["ko"],
                        help="Context culture(s) to analyze")
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--data_root", default="./dataset/camellia/raw")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="./analysis/phase3")
    parser.add_argument("--max_contexts", type=int, default=10)
    parser.add_argument("--max_entities", type=int, default=15)
    args = parser.parse_args()

    for ctx_culture in args.ctx_culture:
        run_single(
            ctx_culture, args.model, args.data_root,
            args.device, args.output_dir,
            args.max_contexts, args.max_entities,
        )


if __name__ == "__main__":
    main()