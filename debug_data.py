"""
CBMCD Data Diagnostic Script

Checks:
1. Language handling: Are Western entities loaded in Korean (cu) or English?
2. Grounded names: Why are names missing from grounded evaluation?
3. Entity type alias mapping coverage
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.data import (
    load_camellia_data, split_data,
    ENTITY_TYPE_ALIASES, WESTERN_FILE_MAPPING,
    load_entities, load_contexts,
    LANGUAGE_NAME_MAP,
)

DATA_ROOT = "./dataset/camellia/raw"


def check_1_language():
    """Check: Are entities loaded in the correct language?"""
    print("=" * 60)
    print("CHECK 1: Language Handling")
    print("=" * 60)
    
    data_root = Path(DATA_ROOT)
    entity_dir = data_root / "entities"
    
    # Load Korean entities (target_lang=cu → Korean text)
    for etype in ["food", "authors", "names-male", "locations"]:
        try:
            asian_cu = load_entities(entity_dir, etype, "korean", language="korean", target_lang="cu")
            asian_en = load_entities(entity_dir, etype, "korean", language="korean", target_lang="en")
            print(f"\n[{etype}] Asian (Korean culture):")
            print(f"  cu (native): {asian_cu[:3]}")
            print(f"  en (english): {asian_en[:3]}")
        except Exception as e:
            print(f"\n[{etype}] Asian: ERROR - {e}")
    
    # Load Western entities (target_lang=cu → should be Korean translated!)
    for etype in ["food", "authors", "names-male", "locations"]:
        try:
            western_cu = load_entities(entity_dir, etype, "western", language="korean", target_lang="cu")
            western_en = load_entities(entity_dir, etype, "western", language="korean", target_lang="en")
            print(f"\n[{etype}] Western:")
            print(f"  cu (korean col): {western_cu[:3]}")
            print(f"  en (english col): {western_en[:3]}")
        except Exception as e:
            print(f"\n[{etype}] Western: ERROR - {e}")
    
    # Check what columns exist in western entity files
    print(f"\n--- Western entity file columns ---")
    import pandas as pd
    for etype in ["food", "authors", "names-male", "locations", "sports"]:
        actual = WESTERN_FILE_MAPPING.get(etype, etype)
        fpath = entity_dir / "western" / f"{actual}.xlsx"
        if fpath.exists():
            df = pd.read_excel(fpath)
            print(f"  {etype} ({fpath.name}): columns = {list(df.columns)}")
            print(f"    first row: {df.iloc[0].to_dict()}")
        else:
            print(f"  {etype}: FILE NOT FOUND at {fpath}")


def check_2_grounded_names():
    """Check: Why are names missing from grounded contexts?"""
    print("\n" + "=" * 60)
    print("CHECK 2: Grounded Names Issue")
    print("=" * 60)
    
    data_root = Path(DATA_ROOT)
    context_dir = data_root / "contexts"
    
    # Load grounded contexts
    grounded_df = load_contexts(context_dir, "grounded", "korean", "cu")
    neutral_df = load_contexts(context_dir, "neutral", "korean", "cu")
    
    print(f"\nGrounded contexts: {len(grounded_df)} total")
    print(f"Entity types in grounded:")
    for etype, count in grounded_df["entity_type"].value_counts().items():
        print(f"  '{etype}' → {count}")
    
    print(f"\nNeutral contexts: {len(neutral_df)} total")
    print(f"Entity types in neutral:")
    for etype, count in neutral_df["entity_type"].value_counts().items():
        print(f"  '{etype}' → {count}")
    
    # Check alias mapping
    print(f"\n--- Alias Mapping ---")
    print(f"ENTITY_TYPE_ALIASES: {ENTITY_TYPE_ALIASES}")
    
    # Check which grounded entity types map to our entity keys
    entity_keys = ["authors", "beverage", "food", "locations", "names-female", "names-male", "sports"]
    
    print(f"\n--- Matching: grounded entity_type → entity key ---")
    for raw_type in grounded_df["entity_type"].unique():
        raw_lower = raw_type.strip().lower()
        mapped = ENTITY_TYPE_ALIASES.get(raw_lower, raw_lower)
        match = mapped in entity_keys
        print(f"  '{raw_type}' → alias: '{mapped}' → {'✓ MATCH' if match else '✗ NO MATCH'}")
    
    print(f"\n--- Matching: neutral entity_type → entity key ---")
    for raw_type in neutral_df["entity_type"].unique():
        raw_lower = raw_type.strip().lower()
        mapped = ENTITY_TYPE_ALIASES.get(raw_lower, raw_lower)
        match = mapped in entity_keys
        print(f"  '{raw_type}' → alias: '{mapped}' → {'✓ MATCH' if match else '✗ NO MATCH'}")


def check_3_evaluate_reverse_alias():
    """Check: evaluate.py's REVERSE_ALIAS coverage"""
    print("\n" + "=" * 60)
    print("CHECK 3: evaluate.py REVERSE_ALIAS Coverage")
    print("=" * 60)
    
    # From evaluate.py
    REVERSE_ALIAS = {
        "locations": ["location", "locations"],
        "sports": ["sport", "sports"],
        "authors": ["author", "authors"],
        "names-male": ["names", "name", "names (m)", "name (m)"],
        "names-female": ["names", "name", "names (f)", "name (f)"],
        "beverage": ["beverage"],
        "food": ["food"],
    }
    
    data_root = Path(DATA_ROOT)
    context_dir = data_root / "contexts"
    grounded_df = load_contexts(context_dir, "grounded", "korean", "cu")
    
    print(f"\nChecking if all grounded entity types are covered by REVERSE_ALIAS:")
    all_reverse = set()
    for aliases in REVERSE_ALIAS.values():
        all_reverse.update(a.lower() for a in aliases)
    
    for raw_type in grounded_df["entity_type"].unique():
        raw_lower = raw_type.strip().lower()
        covered = raw_lower in all_reverse
        print(f"  '{raw_type}' → {'✓ covered' if covered else '✗ NOT COVERED'}")


def check_4_split_entity_pairing():
    """Check: What entity pairs are created after split?"""
    print("\n" + "=" * 60)
    print("CHECK 4: Split & Entity Pairing")
    print("=" * 60)
    
    data = load_camellia_data(DATA_ROOT, culture="ko", target_lang="cu")
    train_ex, val_ex, test_ex, split_info = split_data(data, seed=42)
    
    # Check grounded examples by entity type
    print(f"\n--- Train examples by type ---")
    from collections import Counter
    train_g = [ex for ex in train_ex if ex.context_type == "grounded"]
    train_n = [ex for ex in train_ex if ex.context_type == "neutral"]
    
    print(f"Grounded ({len(train_g)}):")
    for etype, cnt in Counter(ex.entity_type for ex in train_g).items():
        print(f"  {etype}: {cnt}")
    
    print(f"Neutral ({len(train_n)}):")
    for etype, cnt in Counter(ex.entity_type for ex in train_n).items():
        print(f"  {etype}: {cnt}")
    
    # Sample examples to verify language
    print(f"\n--- Sample examples (language check) ---")
    for ex in train_ex[:5]:
        print(f"  [{ex.context_type}] {ex.entity_type}")
        print(f"    context: {ex.context[:60]}...")
        print(f"    asian: {ex.asian_entity}")
        print(f"    western: {ex.western_entity}")
        print()


if __name__ == "__main__":
    check_1_language()
    check_2_grounded_names()
    check_3_evaluate_reverse_alias()
    check_4_split_entity_pairing()