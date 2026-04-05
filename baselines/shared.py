"""
COCOA Baselines - Shared Data Loading & Evaluation

All baselines use:
  1. COCOA's load_camellia_data() + split_data() for identical splits
  2. COCOA's evaluate_robust_fair() for CBS_g + CBS_n measurement
  3. Standardized result saving under baselines/results/{method}/{exp_name}/
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

# Import COCOA's core functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import load_camellia_data, split_data, CamelliaData
from src.evaluate import evaluate_robust_fair
from src.model import MODEL_SHORTCUTS

LOG = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_and_split_kfold(
    data_root: str,
    culture: str,
    lang: str,
    seed: int,
    fold: int,
    folds_root: str = "./dataset/folds",
) -> Tuple[CamelliaData, Dict]:
    """
    Load Camellia data with K-Fold split (same as CoCoA).
    """
    from src.fold_utils import load_fold
    
    LOG.info(f"Loading data: culture={culture}, lang={lang}, fold={fold}, seed={seed}")
    data = load_camellia_data(data_root, culture=culture, target_lang=lang)
    split_info = load_fold(folds_root, culture, lang, fold, seed)
    
    for split_name in ["train", "val", "test"]:
        n_g = len(split_info[f"grounded_{split_name}"])
        n_n = len(split_info[f"neutral_{split_name}"])
        ents = split_info[f"{split_name}_entities"]
        n_a = sum(len(v["asian"]) for v in ents.values())
        n_w = sum(len(v["western"]) for v in ents.values())
        LOG.info(f"  {split_name}: {n_g}G + {n_n}N contexts, {n_a}A + {n_w}W entities")
    
    return data, split_info

# Consistent short names for experiment naming
MODEL_SHORT = {
    "llama3_8b": "llama3-8b",
    "meta-llama/Llama-3.1-8B": "llama3-8b",
    "qwen3_8b": "qwen3-8b",
    "Qwen/Qwen3-8B": "qwen3-8b",
    "qwen25_7b": "qwen25-7b",
    "Qwen/Qwen2.5-7B": "qwen25-7b",
    "gemma3_12b": "gemma3-12b",
    "google/gemma-3-12b-pt": "gemma3-12b",
    "aya_8b": "aya-8b",
    "CohereForAI/aya-expanse-8b": "aya-8b",
}


# =============================================================================
# Data Loading
# =============================================================================

def load_and_split(
    data_root: str = "./dataset/camellia/raw",
    culture: str = "ko",
    lang: str = "cu",
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Tuple[CamelliaData, Dict]:
    """
    Load Camellia data and split using COCOA's exact logic.
    Same seed → same split across all methods.
    """
    LOG.info(f"Loading data: culture={culture}, lang={lang}, seed={seed}")
    
    data = load_camellia_data(data_root, culture=culture, target_lang=lang)
    
    train_examples, val_examples, test_examples, split_info = split_data(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    
    for split_name in ["train", "val", "test"]:
        n_g = len(split_info[f"grounded_{split_name}"])
        n_n = len(split_info[f"neutral_{split_name}"])
        ents = split_info[f"{split_name}_entities"]
        n_a = sum(len(v["asian"]) for v in ents.values())
        n_w = sum(len(v["western"]) for v in ents.values())
        LOG.info(f"  {split_name}: {n_g}G + {n_n}N contexts, {n_a}A + {n_w}W entities")
    
    return data, split_info


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    split_info: Dict,
    split: str = "test",
    max_entities: int = 30,
    show_progress: bool = True,
    entity_priors: Dict = None,
) -> Dict:
    """
    Evaluate CBS_g and CBS_n using COCOA's evaluate_robust_fair.
    Identical function for all methods → identical baseline scores guaranteed.
    
    Args:
        entity_priors: If provided, applies PMI normalization (same as CoCoA eval).
    """
    entities = split_info[f"{split}_entities"]
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        results = evaluate_robust_fair(
            model=model,
            tokenizer=tokenizer,
            split_info=split_info,
            entities=entities,
            device=device,
            split=split,
            max_entities=max_entities,
            show_progress=show_progress,
            entity_priors=entity_priors,
        )
    
    cbs_g = results["grounded"]["overall"] or 0.0
    cbs_n = results["neutral"]["overall"] or 0.0
    score = cbs_g + abs(cbs_n - 50)
    
    return {
        "cbs_g": round(cbs_g, 2),
        "cbs_n": round(cbs_n, 2),
        "score": round(score, 2),
        "detail": results,
    }


# =============================================================================
# Naming
# =============================================================================

def resolve_model_name(model_key: str) -> str:
    return MODEL_SHORTCUTS.get(model_key, model_key)

def get_model_short(model_key: str) -> str:
    if model_key in MODEL_SHORT:
        return MODEL_SHORT[model_key]
    full_name = resolve_model_name(model_key)
    if full_name in MODEL_SHORT:
        return MODEL_SHORT[full_name]
    return model_key.split("/")[-1].lower().replace(".", "-")

def build_biasedit_exp_name(
    culture, lang, model, seed,
    k, n_edits, meta_lr, rank, n_epochs,
) -> str:
    """Example: ko_cu_llama3-8b_k15_ne4_mlr1e-04_rank1920_ep10_seed42"""
    m = get_model_short(model)
    return f"{culture}_{lang}_{m}_k{k}_ne{n_edits}_mlr{meta_lr:.0e}_rank{rank}_ep{n_epochs}_seed{seed}"

def build_biasunlearn_exp_name(
    culture, lang, model, seed,
    lr, beta, ster_weight, anti_weight, kl_weight, lora_r, max_steps,
) -> str:
    """Example: ko_cu_llama3-8b_lr5e-05_beta0.1_ws1.0_wa1.0_wk0.2_r8_steps500_seed42"""
    m = get_model_short(model)
    return f"{culture}_{lang}_{m}_lr{lr:.0e}_beta{beta}_ws{ster_weight}_wa{anti_weight}_wk{kl_weight}_r{lora_r}_steps{max_steps}_seed{seed}"


# =============================================================================
# Output
# =============================================================================

def print_comparison(base_result: Dict, trained_result: Dict, method_name: str):
    d_g = trained_result['cbs_g'] - base_result['cbs_g']
    d_n = trained_result['cbs_n'] - base_result['cbs_n']
    d_s = trained_result['score'] - base_result['score']
    print(f"\n{'='*60}")
    print(f"  {method_name} Results")
    print(f"{'='*60}")
    print(f"  CBS_g: {base_result['cbs_g']:5.1f}% → {trained_result['cbs_g']:5.1f}% (Δ{d_g:+.1f})  goal: ↓0%")
    print(f"  CBS_n: {base_result['cbs_n']:5.1f}% → {trained_result['cbs_n']:5.1f}% (Δ{d_n:+.1f})  goal: →50%")
    print(f"  Score: {base_result['score']:5.1f}  → {trained_result['score']:5.1f}  (Δ{d_s:+.1f})")
    print(f"{'='*60}")


def save_results(
    output_dir: str,
    method: str,
    config_dict: Dict,
    base_result: Dict,
    trained_result: Dict,
    extra: Optional[Dict] = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "method": method,
        "config": config_dict,
        "baseline": {k: base_result[k] for k in ["cbs_g", "cbs_n", "score"]},
        "trained": {k: trained_result[k] for k in ["cbs_g", "cbs_n", "score"]},
        "delta": {
            "cbs_g": round(trained_result["cbs_g"] - base_result["cbs_g"], 2),
            "cbs_n": round(trained_result["cbs_n"] - base_result["cbs_n"], 2),
            "score": round(trained_result["score"] - base_result["score"], 2),
        },
    }
    if extra:
        results["extra"] = extra
    
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    LOG.info(f"Results saved to {output_path / 'results.json'}")
    return results