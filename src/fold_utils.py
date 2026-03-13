"""
CoCoA: Fold Loading Utility

Add this to src/data.py, or import from src/fold_utils.py.

Loads pre-generated K-Fold split files and returns split_info
in the exact same format as split_data() — drop-in replacement.

Usage in code:
    from src.fold_utils import load_fold

    # Instead of:
    #   train, val, test, split_info = split_data(data, seed=42)
    # Use:
    split_info = load_fold("./dataset/folds", culture="ko", lang="cu", fold=0)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .data import (
    load_camellia_data, CamelliaData, CamelliaExample,
    ENTITY_TYPE_ALIASES,
)


def load_fold(
    folds_root: Union[str, Path],
    culture: str,
    lang: str,
    fold: int,
) -> Dict:
    """
    Load a pre-generated fold split.

    Args:
        folds_root: Root directory containing fold files (e.g., ./dataset/folds)
        culture: Culture code (e.g., "ko")
        lang: Language code ("cu" or "en")
        fold: Fold index (0 to K-1)

    Returns:
        split_info dict with same keys as split_data():
            grounded_train, grounded_val, grounded_test (DataFrames)
            neutral_train, neutral_val, neutral_test (DataFrames)
            train_entities, val_entities, test_entities (Dicts)
    """
    folds_root = Path(folds_root)
    fold_dir = folds_root / f"{culture}_{lang}"
    fold_path = fold_dir / f"fold_{fold}.json"

    if not fold_path.exists():
        raise FileNotFoundError(
            f"Fold file not found: {fold_path}\n"
            f"Run generate_folds.py first to create fold splits."
        )

    with open(fold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert context lists back to DataFrames
    split_info = {}
    for key, val in data.items():
        if key.startswith("grounded_") or key.startswith("neutral_"):
            split_info[key] = pd.DataFrame(val) if val else pd.DataFrame(columns=["context", "entity_type"])
        else:
            # entity dicts — already correct format
            split_info[key] = val

    return split_info


def load_fold_meta(
    folds_root: Union[str, Path],
    culture: str,
    lang: str,
) -> Dict:
    """Load fold metadata (K, seed, entity counts, etc.)."""
    meta_path = Path(folds_root) / f"{culture}_{lang}" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def create_examples_from_fold(
    split_info: Dict,
    culture: str,
    lang: str,
    split: str = "test",
    max_pairs_per_context: int = 10,
) -> List[CamelliaExample]:
    """
    Create CamelliaExample list from fold split_info.
    Used for legacy evaluation functions that expect example lists.
    """
    examples = []
    entities = split_info[f"{split}_entities"]

    for ctx_type in ["grounded", "neutral"]:
        df = split_info[f"{ctx_type}_{split}"]
        for _, row in df.iterrows():
            context = row["context"]
            entity_type_raw = row["entity_type"].strip().lower()
            entity_type = ENTITY_TYPE_ALIASES.get(entity_type_raw, entity_type_raw)

            if entity_type == "names-both":
                types_to_use = ["names-male", "names-female"]
            else:
                types_to_use = [entity_type]

            for etype in types_to_use:
                if etype not in entities:
                    continue

                asian_list = entities[etype].get("asian", [])
                western_list = entities[etype].get("western", [])
                if not asian_list or not western_list:
                    continue

                n_pairs = min(len(asian_list), len(western_list), max_pairs_per_context)
                for i in range(n_pairs):
                    examples.append(CamelliaExample(
                        context=context,
                        context_type=ctx_type,
                        entity_type=etype,
                        asian_entity=asian_list[i],
                        western_entity=western_list[i],
                        culture=culture,
                        lang=lang,
                    ))

    return examples


def get_available_folds(
    folds_root: Union[str, Path],
    culture: str,
    lang: str,
) -> List[int]:
    """List available fold indices for a culture/lang."""
    fold_dir = Path(folds_root) / f"{culture}_{lang}"
    if not fold_dir.exists():
        return []
    folds = []
    for f in sorted(fold_dir.glob("fold_*.json")):
        try:
            idx = int(f.stem.split("_")[1])
            folds.append(idx)
        except (IndexError, ValueError):
            continue
    return folds