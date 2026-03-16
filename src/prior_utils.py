"""
CoCoA: Entity Prior Loading Utility

Loads pre-computed entity priors from JSON files.
Used by trainer and evaluator for prior normalization.

Usage:
    from src.prior_utils import load_entity_priors

    # Returns flat dict: {"막걸리": -25.3, "와인": -8.1, ...}
    priors = load_entity_priors("./dataset/priors", "llama3_8b", "ko", "cu")
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union


def load_entity_priors(
    priors_root: Union[str, Path],
    model_key: str,
    culture: str,
    lang: str,
) -> Dict[str, float]:
    """
    Load pre-computed entity priors as flat dict.

    Args:
        priors_root: Root directory (e.g., ./dataset/priors)
        model_key: Model shortcut (e.g., "llama3_8b")
        culture: Culture code (e.g., "ko")
        lang: Language code ("cu" or "en")

    Returns:
        Flat dict mapping entity string → log P(entity|BOS)
        e.g., {"막걸리": -25.3, "와인": -8.1, "김영하": -30.2, ...}
    """
    prior_path = Path(priors_root) / model_key / f"{culture}_{lang}" / "entity_priors.json"

    if not prior_path.exists():
        raise FileNotFoundError(
            f"Entity priors not found: {prior_path}\n"
            f"Run: python generate_priors.py --model {model_key} --cultures {culture} --lang {lang}"
        )

    with open(prior_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten: {etype: {side: {entity: value}}} → {entity: value}
    flat = {}
    for etype, sides in data["priors"].items():
        for side, entities in sides.items():
            for entity, value in entities.items():
                flat[entity] = value

    return flat


def get_prior(priors: Dict[str, float], entity: str, default: float = 0.0) -> float:
    """Look up prior for an entity, return default if not found."""
    return priors.get(entity, default)