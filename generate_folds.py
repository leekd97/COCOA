"""
CoCoA: K-Fold Cross-Validation Split Generator

Generates K fold splits for both contexts and entities (zero-leakage),
saves each fold as a JSON file for reproducible training/evaluation.

Fold rotation:
  fold i → test, fold (i+1)%K → val, remaining → train

Usage:
    # Generate folds for all cultures (cu)
    python generate_folds.py --lang cu

    # Specific cultures
    python generate_folds.py --cultures ko zh ja --lang cu

    # Custom K and seed
    python generate_folds.py --K 5 --seed 42 --lang cu

    # Both languages
    python generate_folds.py --lang cu
    python generate_folds.py --lang en
"""

import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.data import (
    load_camellia_data, CamelliaData, ENTITY_TYPE_ALIASES,
)


# =========================================================================
# K-Fold Splitting
# =========================================================================

def kfold_split_list(items: list, K: int, seed: int) -> List[list]:
    """Split a list into K roughly equal folds (shuffled)."""
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    folds = [[] for _ in range(K)]
    for i, item in enumerate(shuffled):
        folds[i % K].append(item)
    return folds


def kfold_split_df(df: pd.DataFrame, K: int, seed: int) -> List[pd.DataFrame]:
    """
    Split DataFrame into K folds, stratified by entity_type.
    Each fold is a DataFrame with same columns as input.
    """
    rng = random.Random(seed)
    fold_indices = [[] for _ in range(K)]

    for etype, group in df.groupby("entity_type"):
        indices = group.index.tolist()
        rng.shuffle(indices)
        for i, idx in enumerate(indices):
            fold_indices[i % K].append(idx)

    folds = []
    for fi in fold_indices:
        if fi:
            folds.append(df.loc[fi].reset_index(drop=True))
        else:
            folds.append(pd.DataFrame(columns=df.columns))
    return folds


def kfold_split_entities(
    entities: Dict[str, Dict[str, List[str]]],
    K: int,
    seed: int,
) -> List[Dict[str, Dict[str, List[str]]]]:
    """
    Split entities into K folds per entity_type, per side (asian/western).
    Returns list of K dicts, each with same structure as input.
    """
    rng_base = random.Random(seed)
    # Use different sub-seeds for each entity type to avoid correlation
    etype_seeds = {etype: rng_base.randint(0, 2**31) for etype in entities}

    fold_entities = [{} for _ in range(K)]

    for etype, ent_dict in entities.items():
        eseed = etype_seeds[etype]
        for side in ["asian", "western"]:
            ent_list = ent_dict[side]
            folds = kfold_split_list(ent_list, K, eseed + (0 if side == "asian" else 1))
            for k in range(K):
                if etype not in fold_entities[k]:
                    fold_entities[k][etype] = {}
                fold_entities[k][etype][side] = folds[k]

    return fold_entities


# =========================================================================
# Fold Assembly
# =========================================================================

def assemble_fold(
    grounded_folds: List[pd.DataFrame],
    neutral_folds: List[pd.DataFrame],
    entity_folds: List[Dict],
    fold_idx: int,
    K: int,
) -> Dict:
    """
    Assemble train/val/test for a given fold index.

    fold_idx → test
    (fold_idx + 1) % K → val
    remaining → train
    """
    test_idx = fold_idx
    val_idx = (fold_idx + 1) % K
    train_indices = [i for i in range(K) if i != test_idx and i != val_idx]

    # Contexts
    grounded_test = grounded_folds[test_idx]
    grounded_val = grounded_folds[val_idx]
    grounded_train = pd.concat(
        [grounded_folds[i] for i in train_indices], ignore_index=True
    )

    neutral_test = neutral_folds[test_idx]
    neutral_val = neutral_folds[val_idx]
    neutral_train = pd.concat(
        [neutral_folds[i] for i in train_indices], ignore_index=True
    )

    # Entities
    def merge_entity_folds(indices):
        merged = {}
        for i in indices:
            for etype, sides in entity_folds[i].items():
                if etype not in merged:
                    merged[etype] = {"asian": [], "western": []}
                merged[etype]["asian"].extend(sides["asian"])
                merged[etype]["western"].extend(sides["western"])
        return merged

    test_entities = entity_folds[test_idx]
    val_entities = entity_folds[val_idx]
    train_entities = merge_entity_folds(train_indices)

    return {
        "grounded_train": grounded_train,
        "grounded_val": grounded_val,
        "grounded_test": grounded_test,
        "neutral_train": neutral_train,
        "neutral_val": neutral_val,
        "neutral_test": neutral_test,
        "train_entities": train_entities,
        "val_entities": val_entities,
        "test_entities": test_entities,
    }


# =========================================================================
# Serialization
# =========================================================================

def df_to_records(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to list of dicts for JSON."""
    return df.to_dict(orient="records")


def split_info_to_json(split_info: Dict) -> Dict:
    """Convert split_info (with DataFrames) to JSON-serializable dict."""
    result = {}
    for key, val in split_info.items():
        if isinstance(val, pd.DataFrame):
            result[key] = df_to_records(val)
        elif isinstance(val, dict):
            # entities dict — already JSON-serializable
            result[key] = val
        else:
            result[key] = val
    return result


def json_to_split_info(data: Dict) -> Dict:
    """Convert loaded JSON back to split_info with DataFrames."""
    result = {}
    for key, val in data.items():
        if key.startswith("grounded_") or key.startswith("neutral_"):
            # Convert list of dicts back to DataFrame
            result[key] = pd.DataFrame(val)
        else:
            # entities dicts stay as-is
            result[key] = val
    return result


# =========================================================================
# Statistics
# =========================================================================

def print_fold_stats(split_info: Dict, fold_idx: int):
    """Print statistics for a single fold."""
    for split_name in ["train", "val", "test"]:
        n_g = len(split_info[f"grounded_{split_name}"])
        n_n = len(split_info[f"neutral_{split_name}"])
        ents = split_info[f"{split_name}_entities"]
        n_a = sum(len(v["asian"]) for v in ents.values())
        n_w = sum(len(v["western"]) for v in ents.values())
        print(f"    {split_name}: {n_g}G + {n_n}N contexts, {n_a}A + {n_w}W entities")


def check_leakage(split_info: Dict) -> bool:
    """Verify zero entity leakage across splits."""
    has_leak = False
    for etype in split_info["train_entities"]:
        for side in ["asian", "western"]:
            train_set = set(split_info["train_entities"][etype][side])
            val_set = set(split_info["val_entities"].get(etype, {}).get(side, []))
            test_set = set(split_info["test_entities"].get(etype, {}).get(side, []))

            leak = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
            if leak:
                print(f"    ⚠ LEAK in {etype}/{side}: {len(leak)} overlapping entities")
                has_leak = True
    return not has_leak


# =========================================================================
# Main
# =========================================================================

def generate_folds_for_culture(
    culture: str,
    lang: str,
    data_root: str,
    output_root: str,
    K: int = 5,
    seed: int = 42,
):
    """Generate and save K-fold splits for one culture/lang."""
    print(f"\n{'='*60}")
    print(f"  {culture}/{lang} — {K}-Fold CV (seed={seed})")
    print(f"{'='*60}")

    # Load raw data
    data = load_camellia_data(data_root, culture=culture, target_lang=lang)

    # Split into K folds
    print(f"  Splitting contexts into {K} folds (stratified by entity_type)...")
    grounded_folds = kfold_split_df(data.grounded_contexts, K, seed)
    neutral_folds = kfold_split_df(data.neutral_contexts, K, seed + 1)

    print(f"  Splitting entities into {K} folds...")
    entity_folds = kfold_split_entities(data.entities, K, seed + 2)

    # Print fold sizes
    for k in range(K):
        n_g = len(grounded_folds[k])
        n_n = len(neutral_folds[k])
        n_a = sum(len(v["asian"]) for v in entity_folds[k].values())
        n_w = sum(len(v["western"]) for v in entity_folds[k].values())
        print(f"    Fold {k}: {n_g}G + {n_n}N contexts, {n_a}A + {n_w}W entities")

    # Assemble and save each fold
    output_dir = Path(output_root) / f"seed{seed}" / f"{culture}_{lang}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx in range(K):
        print(f"\n  Fold {fold_idx} (test={fold_idx}, val={(fold_idx+1)%K}, train=rest):")
        split_info = assemble_fold(
            grounded_folds, neutral_folds, entity_folds, fold_idx, K
        )

        # Stats
        print_fold_stats(split_info, fold_idx)

        # Leakage check
        clean = check_leakage(split_info)
        if clean:
            print(f"    ✓ Zero leakage verified")

        # Save
        fold_data = split_info_to_json(split_info)
        fold_path = output_dir / f"fold_{fold_idx}.json"
        with open(fold_path, "w", encoding="utf-8") as f:
            json.dump(fold_data, f, ensure_ascii=False)
        print(f"    Saved: {fold_path} ({fold_path.stat().st_size / 1024:.0f} KB)")

    # Save metadata
    meta = {
        "culture": culture,
        "lang": lang,
        "K": K,
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
        "data_root": str(data_root),
        "n_grounded_contexts": len(data.grounded_contexts),
        "n_neutral_contexts": len(data.neutral_contexts),
        "entity_types": list(data.entities.keys()),
        "entity_counts": {
            etype: {"asian": len(v["asian"]), "western": len(v["western"])}
            for etype, v in data.entities.items()
        },
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  Meta: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="CoCoA: K-Fold CV Split Generator")
    parser.add_argument("--cultures", nargs="+",
                        default=["ko", "zh", "ja", "vi", "hi", "ur", "gu", "mr", "ml", "ar"],
                        help="Cultures to generate folds for")
    parser.add_argument("--lang", default="cu", choices=["cu", "en"])
    parser.add_argument("--K", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fold assignment")
    parser.add_argument("--data_root", default="./dataset/camellia/raw")
    parser.add_argument("--output_root", default="./dataset/folds")
    args = parser.parse_args()

    print(f"CoCoA K-Fold Split Generator")
    print(f"K={args.K}, seed={args.seed}, lang={args.lang}")
    print(f"Cultures: {args.cultures}")
    print(f"Output: {args.output_root}")

    for culture in args.cultures:
        try:
            generate_folds_for_culture(
                culture, args.lang, args.data_root, args.output_root,
                args.K, args.seed,
            )
        except Exception as e:
            print(f"\n*** ERROR for {culture}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Done! All folds saved to {args.output_root}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()