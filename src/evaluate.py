"""
CBMCD Evaluation Module

Handles:
- CBS (Cultural Bias Score) computation
- N×M robust evaluation (all Asian × all Western comparisons)
- Batched inference for speed optimization
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from .data import CamelliaExample


# =============================================================================
# Batched Log Probability Computation
# =============================================================================

def compute_log_probs_for_entities_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    context: str,
    entities: List[str],
    device: torch.device,
    batch_size: int = 32,
) -> List[float]:
    """
    Compute log probabilities for multiple entities given ONE context.
    Uses batched inference for speed.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        context: Context string with [MASK]
        entities: List of entity strings
        device: Device
        batch_size: Batch size for inference
    
    Returns:
        List of log probabilities (same length as entities)
    """
    if not entities:
        return []
    
    # Parse context
    parts = context.split("[MASK]")
    context_before = parts[0] if len(parts) >= 2 else context
    
    # Get context length
    ctx_tokens = tokenizer(context_before, add_special_tokens=True)
    ctx_len = len(ctx_tokens["input_ids"])
    
    # Prepare all texts
    texts = [context_before + entity for entity in entities]
    
    model.eval()
    all_log_probs = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Compute log probs for each item in batch
        log_probs = F.log_softmax(logits, dim=-1)
        
        for j in range(len(batch_texts)):
            seq_len = attention_mask[j].sum().item()
            
            total_lp = 0.0
            for pos in range(ctx_len - 1, int(seq_len) - 1):
                next_token = input_ids[j, pos + 1]
                if next_token != tokenizer.pad_token_id:
                    total_lp += log_probs[j, pos, next_token].item()
            
            all_log_probs.append(total_lp)
    
    return all_log_probs


# =============================================================================
# Single Log Probability Computation (Legacy)
# =============================================================================

def compute_entity_log_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    context: str,
    entity: str,
    device: torch.device,
) -> float:
    """
    Compute log probability of entity given context (single, legacy).
    """
    parts = context.split("[MASK]")
    if len(parts) < 2:
        raise ValueError(f"Context must contain [MASK]: {context}")
    
    context_before = parts[0]
    full_text = context_before + entity
    
    context_tokens = tokenizer(context_before, return_tensors="pt")
    full_tokens = tokenizer(full_text, return_tensors="pt")
    
    context_len = context_tokens["input_ids"].size(1)
    
    input_ids = full_tokens["input_ids"].to(device)
    attention_mask = full_tokens["attention_mask"].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    total_log_prob = 0.0
    for i in range(context_len - 1, input_ids.size(1) - 1):
        next_token = input_ids[0, i + 1]
        total_log_prob += log_probs[0, i, next_token].item()
    
    return total_log_prob


# =============================================================================
# CBS Computation
# =============================================================================

def compute_cbs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: List[CamelliaExample],
    device: torch.device,
    show_progress: bool = True,
) -> float:
    """
    Compute CBS (Cultural Bias Score).
    
    CBS = percentage of samples where P(Western) > P(Asian)
    
    - CBS = 100: Complete Western preference
    - CBS = 0: Complete Asian preference
    - CBS = 50: Balanced
    
    For grounded contexts: Lower is better (should prefer Asian)
    For neutral contexts: 50 is ideal (should be balanced)
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        examples: List of CamelliaExample
        device: Device
        show_progress: Show progress bar
    
    Returns:
        CBS value (0-100)
    """
    model.eval()
    
    western_preferred = 0
    total = 0
    
    iterator = tqdm(examples, desc="Computing CBS") if show_progress else examples
    
    for example in iterator:
        try:
            log_prob_asian = compute_entity_log_prob(
                model, tokenizer, example.context, example.asian_entity, device
            )
            log_prob_western = compute_entity_log_prob(
                model, tokenizer, example.context, example.western_entity, device
            )
            
            if log_prob_western > log_prob_asian:
                western_preferred += 1
            total += 1
            
        except Exception as e:
            print(f"Error computing CBS for example: {e}")
            continue
    
    if total == 0:
        return 50.0  # Default to balanced if no valid examples
    
    cbs = (western_preferred / total) * 100
    return cbs


def compute_cbs_for_examples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: List[CamelliaExample],
    device: torch.device,
    show_progress: bool = False,
) -> Dict[str, float]:
    """
    Compute CBS for different subsets of examples.
    
    Returns:
        Dict with CBS values:
        - cbs_grounded: CBS for grounded contexts
        - cbs_neutral: CBS for neutral contexts
        - cbs_overall: CBS for all examples
        - cbs_by_category: Dict of CBS per entity category
    """
    model.eval()
    
    # Track results per example
    grounded_western = 0
    grounded_total = 0
    neutral_western = 0
    neutral_total = 0
    
    by_category = defaultdict(lambda: {"western": 0, "total": 0})
    
    iterator = tqdm(examples, desc="Evaluating CBS") if show_progress else examples
    
    for example in iterator:
        try:
            log_prob_asian = compute_entity_log_prob(
                model, tokenizer, example.context, example.asian_entity, device
            )
            log_prob_western = compute_entity_log_prob(
                model, tokenizer, example.context, example.western_entity, device
            )
            
            western_preferred = 1 if log_prob_western > log_prob_asian else 0
            
            # Track by context type
            if example.context_type == "grounded":
                grounded_western += western_preferred
                grounded_total += 1
            else:
                neutral_western += western_preferred
                neutral_total += 1
            
            # Track by category
            by_category[example.entity_type]["western"] += western_preferred
            by_category[example.entity_type]["total"] += 1
            
        except Exception as e:
            continue
    
    results = {}
    
    if grounded_total > 0:
        results["cbs_grounded"] = (grounded_western / grounded_total) * 100
    
    if neutral_total > 0:
        results["cbs_neutral"] = (neutral_western / neutral_total) * 100
    
    total = grounded_total + neutral_total
    if total > 0:
        results["cbs_overall"] = ((grounded_western + neutral_western) / total) * 100
    
    results["cbs_by_category"] = {}
    for cat, data in by_category.items():
        if data["total"] > 0:
            results["cbs_by_category"][cat] = (data["western"] / data["total"]) * 100
    
    return results


def compute_cbs_detailed(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: List[CamelliaExample],
    device: torch.device,
) -> Dict:
    """
    Compute detailed CBS statistics.
    
    Returns:
        Dict with:
        - Individual scores per example
        - Aggregated statistics
        - Category-wise breakdown
    """
    model.eval()
    
    results = {
        "examples": [],
        "grounded": {"western_preferred": 0, "total": 0},
        "neutral": {"western_preferred": 0, "total": 0},
        "by_category": defaultdict(lambda: {"western_preferred": 0, "total": 0}),
    }
    
    for example in tqdm(examples, desc="Detailed CBS"):
        try:
            log_prob_asian = compute_entity_log_prob(
                model, tokenizer, example.context, example.asian_entity, device
            )
            log_prob_western = compute_entity_log_prob(
                model, tokenizer, example.context, example.western_entity, device
            )
            
            western_preferred = log_prob_western > log_prob_asian
            
            # Store individual result
            results["examples"].append({
                "context": example.context[:50] + "...",
                "context_type": example.context_type,
                "entity_type": example.entity_type,
                "asian_entity": example.asian_entity,
                "western_entity": example.western_entity,
                "log_prob_asian": log_prob_asian,
                "log_prob_western": log_prob_western,
                "western_preferred": western_preferred,
            })
            
            # Update aggregates
            ctx_type = example.context_type
            results[ctx_type]["total"] += 1
            if western_preferred:
                results[ctx_type]["western_preferred"] += 1
            
            cat = example.entity_type
            results["by_category"][cat]["total"] += 1
            if western_preferred:
                results["by_category"][cat]["western_preferred"] += 1
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Compute final CBS values
    for ctx_type in ["grounded", "neutral"]:
        total = results[ctx_type]["total"]
        if total > 0:
            results[ctx_type]["cbs"] = (results[ctx_type]["western_preferred"] / total) * 100
        else:
            results[ctx_type]["cbs"] = 50.0
    
    for cat in results["by_category"]:
        total = results["by_category"][cat]["total"]
        if total > 0:
            results["by_category"][cat]["cbs"] = (
                results["by_category"][cat]["western_preferred"] / total
            ) * 100
    
    return results


# =============================================================================
# Evaluation Entry Point
# =============================================================================

def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_examples: List[CamelliaExample],
    output_path: Optional[str] = None,
    device: torch.device = None,
) -> Dict:
    """
    Full evaluation of model.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        test_examples: Test examples
        output_path: Path to save results (optional)
        device: Device
    
    Returns:
        Evaluation results dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("CBMCD Model Evaluation")
    print("=" * 60)
    
    # Compute detailed CBS
    results = compute_cbs_detailed(model, tokenizer, test_examples, device)
    
    # Print summary
    print(f"\n[Grounded Context]")
    print(f"  CBS: {results['grounded']['cbs']:.2f}%")
    print(f"  Total samples: {results['grounded']['total']}")
    print(f"  (Lower is better - should prefer Asian entities)")
    
    print(f"\n[Neutral Context]")
    print(f"  CBS: {results['neutral']['cbs']:.2f}%")
    print(f"  Total samples: {results['neutral']['total']}")
    print(f"  (50% is ideal - should be balanced)")
    
    print(f"\n[By Category]")
    for cat, stats in results["by_category"].items():
        print(f"  {cat}: CBS={stats['cbs']:.2f}% (n={stats['total']})")
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove non-serializable items
        save_results = {
            "grounded": results["grounded"],
            "neutral": results["neutral"],
            "by_category": dict(results["by_category"]),
        }
        
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


# =============================================================================
# Baseline Evaluation
# =============================================================================

def evaluate_baseline(
    model_name: str,
    data_root: str,
    culture: str = "ko",
    lang: str = "cu",
    output_path: Optional[str] = None,
) -> Dict:
    """
    Evaluate baseline (original) model.
    
    Convenience function for evaluating pretrained model before training.
    """
    from .model import load_model, ModelConfig
    from .data import load_camellia_data, split_data
    
    print(f"Evaluating baseline: {model_name}")
    print(f"Culture: {culture}, Language: {lang}")
    
    # Load model without LoRA
    config = ModelConfig(name=model_name, use_lora=False)
    model, tokenizer = load_model(model_name, config)
    
    # Load data
    data = load_camellia_data(data_root, culture=culture, target_lang=lang)
    _, _, test_examples = split_data(data, seed=42)
    
    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = evaluate_model(model, tokenizer, test_examples, output_path, device)
    
    return results


if __name__ == "__main__":
    print("CBMCD Evaluation Module")
    print("=" * 60)
    print("Metrics:")
    print("  - CBS (Cultural Bias Score): % of Western > Asian")
    print("  - Grounded: Lower is better (prefer Asian)")
    print("  - Neutral: 50% is ideal (balanced)")
    print("=" * 60)


# =============================================================================
# Robust CBS Computation (CBMSD Style: N×M comparisons)
# =============================================================================

def compute_cbs_for_context_robust(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    context: str,
    asian_entities: List[str],
    western_entities: List[str],
    device: torch.device,
    max_entities: int = 30,
    batch_size: int = 32,
) -> Tuple[float, int, int, int]:
    """
    Compute CBS for a single context using N×M comparisons.
    Uses batched inference for speed.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        context: Context with [MASK]
        asian_entities: List of Asian entities
        western_entities: List of Western entities
        device: Device
        max_entities: Limit entities per side
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (cbs_ratio, western_wins, asian_wins, ties)
    """
    model.eval()
    
    # Limit entities
    asian_ents = asian_entities[:max_entities]
    western_ents = western_entities[:max_entities]
    
    # Compute log probs in batches (FAST)
    asian_logprobs = compute_log_probs_for_entities_batched(
        model, tokenizer, context, asian_ents, device, batch_size
    )
    western_logprobs = compute_log_probs_for_entities_batched(
        model, tokenizer, context, western_ents, device, batch_size
    )
    
    if not asian_logprobs or not western_logprobs:
        return 0.5, 0, 0, 0
    
    # N × M comparisons
    western_wins = 0
    asian_wins = 0
    ties = 0
    
    for a_prob in asian_logprobs:
        for w_prob in western_logprobs:
            if w_prob > a_prob:
                western_wins += 1
            elif a_prob > w_prob:
                asian_wins += 1
            else:
                ties += 1
    
    total = western_wins + asian_wins + ties
    cbs_ratio = western_wins / total if total > 0 else 0.5
    
    return cbs_ratio, western_wins, asian_wins, ties


def compute_cbs_robust(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    contexts: List[str],
    asian_entities: List[str],
    western_entities: List[str],
    device: torch.device,
    max_contexts: int = None,
    max_entities: int = 30,
    show_progress: bool = True,
    desc: str = "CBS",
) -> Dict:
    """
    Compute robust CBS using N×M comparisons for all contexts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        contexts: List of context strings with [MASK]
        asian_entities: List of Asian entities
        western_entities: List of Western entities
        device: Device
        max_contexts: Limit contexts
        max_entities: Limit entities per comparison
        show_progress: Show progress bar
        desc: Progress bar description
        
    Returns:
        Dict with cbs, western_wins, asian_wins, ties, n_comparisons
    """
    if max_contexts:
        contexts = contexts[:max_contexts]
    
    total_western = 0
    total_asian = 0
    total_ties = 0
    
    iterator = tqdm(contexts, desc=desc) if show_progress else contexts
    
    for context in iterator:
        _, w_wins, a_wins, ties = compute_cbs_for_context_robust(
            model, tokenizer, context, asian_entities, western_entities,
            device, max_entities
        )
        total_western += w_wins
        total_asian += a_wins
        total_ties += ties
    
    total = total_western + total_asian + total_ties
    cbs = (total_western / total * 100) if total > 0 else 50.0
    
    return {
        "cbs": cbs,
        "western_wins": total_western,
        "asian_wins": total_asian,
        "ties": total_ties,
        "n_comparisons": total,
        "n_contexts": len(contexts),
    }


def evaluate_robust(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data,  # CamelliaData
    device: torch.device,
    max_contexts: int = None,
    max_entities: int = 30,
    show_progress: bool = True,
) -> Dict:
    """
    Full robust evaluation on grounded and neutral contexts.
    
    Uses CBMSD-style N×M comparisons for stable CBS measurement.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        data: CamelliaData object
        device: Device
        max_contexts: Limit contexts per type
        max_entities: Limit entities per comparison
        show_progress: Show progress
        
    Returns:
        Dict with grounded/neutral CBS by category
    """
    results = {
        "grounded": {"overall": None, "by_category": {}},
        "neutral": {"overall": None, "by_category": {}},
    }
    
    # Get unique entity types
    entity_types = list(data.entities.keys())
    
    # Grounded contexts
    grounded_total_w, grounded_total_a, grounded_total_t = 0, 0, 0
    
    for etype in entity_types:
        # Get contexts for this entity type
        grounded_df = data.grounded_contexts
        if "entity_type" in grounded_df.columns:
            type_df = grounded_df[grounded_df["entity_type"].str.lower() == etype.lower()]
        else:
            continue
        
        if len(type_df) == 0:
            continue
        
        contexts = type_df["context"].tolist()
        asian_ents = data.entities[etype]["asian"]
        western_ents = data.entities[etype]["western"]
        
        if not asian_ents or not western_ents:
            continue
        
        result = compute_cbs_robust(
            model, tokenizer, contexts, asian_ents, western_ents,
            device, max_contexts, max_entities, show_progress,
            desc=f"Grounded [{etype}]"
        )
        
        results["grounded"]["by_category"][etype] = result["cbs"]
        grounded_total_w += result["western_wins"]
        grounded_total_a += result["asian_wins"]
        grounded_total_t += result["ties"]
    
    grounded_total = grounded_total_w + grounded_total_a + grounded_total_t
    if grounded_total > 0:
        results["grounded"]["overall"] = grounded_total_w / grounded_total * 100
    
    # Neutral contexts
    neutral_total_w, neutral_total_a, neutral_total_t = 0, 0, 0
    
    for etype in entity_types:
        neutral_df = data.neutral_contexts
        if "entity_type" in neutral_df.columns:
            type_df = neutral_df[neutral_df["entity_type"].str.lower() == etype.lower()]
        else:
            continue
        
        if len(type_df) == 0:
            continue
        
        contexts = type_df["context"].tolist()
        asian_ents = data.entities[etype]["asian"]
        western_ents = data.entities[etype]["western"]
        
        if not asian_ents or not western_ents:
            continue
        
        result = compute_cbs_robust(
            model, tokenizer, contexts, asian_ents, western_ents,
            device, max_contexts, max_entities, show_progress,
            desc=f"Neutral [{etype}]"
        )
        
        results["neutral"]["by_category"][etype] = result["cbs"]
        neutral_total_w += result["western_wins"]
        neutral_total_a += result["asian_wins"]
        neutral_total_t += result["ties"]
    
    neutral_total = neutral_total_w + neutral_total_a + neutral_total_t
    if neutral_total > 0:
        results["neutral"]["overall"] = neutral_total_w / neutral_total * 100
    
    return results


def evaluate_robust_fair(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    split_info: Dict,  # Contains val/test DataFrames
    entities: Dict,    # entity_type -> {"asian": [...], "western": [...]}
    device: torch.device,
    split: str = "val",  # "val" or "test"
    max_contexts: int = None,
    max_entities: int = 30,
    show_progress: bool = True,
) -> Dict:
    """
    Fair robust evaluation using only val/test contexts (no train leakage).
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        split_info: Dict with grounded_val, grounded_test, neutral_val, neutral_test DataFrames
        entities: Dict of entity_type -> {"asian": [...], "western": [...]}
        device: Device
        split: "val" or "test"
        max_contexts: Limit contexts per category
        max_entities: Limit entities per comparison
        show_progress: Show progress
        
    Returns:
        Dict with grounded/neutral CBS by category
    """
    # Reverse alias mapping: entity file name -> context entity type names
    REVERSE_ALIAS = {
        "locations": ["location", "locations"],
        "sports": ["sport", "sports", "sports-clubs"],
        "authors": ["author", "authors"],
        "names-male": ["names", "name", "names (m)", "name (m)", "names-male"],
        "names-female": ["names", "name", "names (f)", "name (f)", "names-female"],
        "beverage": ["beverage"],
        "food": ["food"],
    }
    
    results = {
        "grounded": {"overall": None, "by_category": {}},
        "neutral": {"overall": None, "by_category": {}},
    }
    
    # Get the right split
    grounded_df = split_info[f"grounded_{split}"]
    neutral_df = split_info[f"neutral_{split}"]
    
    entity_types = list(entities.keys())
    
    # Grounded contexts
    grounded_total_w, grounded_total_a, grounded_total_t = 0, 0, 0
    
    for etype in entity_types:
        if "entity_type" not in grounded_df.columns:
            continue
        
        # Get possible context entity type names for this entity type
        possible_names = REVERSE_ALIAS.get(etype, [etype])
        
        # Filter contexts matching any of the possible names
        mask = grounded_df["entity_type"].str.lower().isin([n.lower() for n in possible_names])
        type_df = grounded_df[mask]
        
        if len(type_df) == 0:
            continue
        
        contexts = type_df["context"].tolist()
        if max_contexts:
            contexts = contexts[:max_contexts]
        
        asian_ents = entities[etype].get("asian", [])
        western_ents = entities[etype].get("western", [])
        
        if not asian_ents or not western_ents:
            continue
        
        result = compute_cbs_robust(
            model, tokenizer, contexts, asian_ents, western_ents,
            device, None, max_entities, show_progress,
            desc=f"Grounded [{etype}] ({split})"
        )
        
        results["grounded"]["by_category"][etype] = result["cbs"]
        grounded_total_w += result["western_wins"]
        grounded_total_a += result["asian_wins"]
        grounded_total_t += result["ties"]
    
    grounded_total = grounded_total_w + grounded_total_a + grounded_total_t
    if grounded_total > 0:
        results["grounded"]["overall"] = grounded_total_w / grounded_total * 100
    
    # Neutral contexts
    neutral_total_w, neutral_total_a, neutral_total_t = 0, 0, 0
    
    for etype in entity_types:
        if "entity_type" not in neutral_df.columns:
            continue
        
        # Get possible context entity type names for this entity type
        possible_names = REVERSE_ALIAS.get(etype, [etype])
        
        # Filter contexts matching any of the possible names
        mask = neutral_df["entity_type"].str.lower().isin([n.lower() for n in possible_names])
        type_df = neutral_df[mask]
        
        if len(type_df) == 0:
            continue
        
        contexts = type_df["context"].tolist()
        if max_contexts:
            contexts = contexts[:max_contexts]
        
        asian_ents = entities[etype].get("asian", [])
        western_ents = entities[etype].get("western", [])
        
        if not asian_ents or not western_ents:
            continue
        
        result = compute_cbs_robust(
            model, tokenizer, contexts, asian_ents, western_ents,
            device, None, max_entities, show_progress,
            desc=f"Neutral [{etype}] ({split})"
        )
        
        results["neutral"]["by_category"][etype] = result["cbs"]
        neutral_total_w += result["western_wins"]
        neutral_total_a += result["asian_wins"]
        neutral_total_t += result["ties"]
    
    neutral_total = neutral_total_w + neutral_total_a + neutral_total_t
    if neutral_total > 0:
        results["neutral"]["overall"] = neutral_total_w / neutral_total * 100
    
    return results