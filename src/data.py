"""
CBMCD Data Module (v2)

Changes from v1:
- Both contexts AND entities are split into train/val/test (zero leakage)
- 3-way split: train/val/test (default 70/10/20)
- split_info carries per-split entity dictionaries for robust evaluation
- Minimum entity count enforcement per split
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from itertools import cycle

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


# =============================================================================
# Constants
# =============================================================================

LANGUAGE_CODE_MAP = {
    "korean": "ko",
    "english": "en",
    "japanese": "ja",
    "chinese": "zh",
    "hindi": "hi",
    "vietnamese": "vi",
    "gujarati": "gu",
    "marathi": "mr",
    "malayalam": "ml",
    "urdu": "ur",
    "arabic": "ar",
}

LANGUAGE_NAME_MAP = {
    "ko": "korean",
    "en": "english",
    "ja": "japanese",
    "zh": "chinese",
    "hi": "hindi",
    "vi": "vietnamese",
    "gu": "gujarati",
    "mr": "marathi",
    "ml": "malayalam",
    "ur": "urdu",
    "ar": "arabic",
}

ENTITY_TYPE_ALIASES = {
    "location": "locations",
    "sport": "sports",
    "author": "authors",
    "name": "names-male",
    "names": "names-both",
    "name (m)": "names-male",
    "name (f)": "names-female",
    "names (m)": "names-male",
    "names (f)": "names-female",
}

WESTERN_FILE_MAPPING = {
    "sports": "football-clubs",
}

# Cultures where western sports = cricket (South Asian)
CRICKET_CULTURES = {"urdu", "hindi", "malayalam", "marathi", "gujarati",
                    "ur", "hi", "ml", "mr", "gu"}

# Indian cultures: stored in a single "indian" file with per-language columns
INDIAN_CULTURES = {"hindi", "malayalam", "marathi", "gujarati",
                   "hi", "ml", "mr", "gu"}

# Pakistani culture: "urdu"/"ur" maps to "pakistani" filenames (standard format)
PAKISTANI_CULTURES = {"urdu", "ur"}

# Indian combined: pool all 4 Indian language contexts for shared cultural training
INDIAN_COMBINED = "indian_combined"
INDIAN_COMBINED_LANGUAGES = ["hi", "ml", "mr", "gu"]
INDIAN_COMBINED_ANCHOR = "hi"  # Entity language for combined mode

# Arab (CAMEL dataset): different directory structure from Camellia
ARAB_CULTURES = {"arabic", "arab", "ar"}

# CAMEL uses different entity type names than Camellia
CAMEL_ENTITY_TYPE_MAP = {
    "sports": "sports-clubs",
}

# CAMEL context file paths (relative to CAMEL data root)
CAMEL_CONTEXT_FILES = {
    "grounded": "prompts/camel-co/camelco-prompts-causal-lm.xlsx",
    "neutral": "prompts/camel-ag/camelag-prompts-causal-lms.xlsx",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CamelliaExample:
    """Single example from Camellia dataset"""
    context: str
    context_type: str       # "grounded" or "neutral"
    entity_type: str
    asian_entity: str
    western_entity: str
    culture: str
    lang: str


@dataclass
class CamelliaData:
    """Container for loaded Camellia data"""
    grounded_contexts: pd.DataFrame
    neutral_contexts: pd.DataFrame
    entities: Dict[str, Dict[str, List[str]]]
    culture: str
    lang: str


# =============================================================================
# Data Loading Functions (unchanged)
# =============================================================================

def load_contexts(
    context_dir: Path,
    context_type: str,
    language: str,
    target_lang: str = "cu",
) -> pd.DataFrame:
    # ---- Indian Combined: pool all 4 language contexts ----
    if language == INDIAN_COMBINED:
        subdir = f"camellia-{context_type}/causal-lms"
        filename = f"{context_type}-contexts-causal-lms-indian.xlsx"
        context_path = context_dir / subdir / filename

        if not context_path.exists():
            raise FileNotFoundError(f"Context file not found: {context_path}")

        df = pd.read_excel(context_path)
        df.columns = [col.strip() for col in df.columns]

        if target_lang == "en":
            for col_name in ["English Context", "en", "English Prompt"]:
                if col_name in df.columns:
                    sub = df[["Entity Type", col_name]].copy()
                    sub = sub.rename(columns={col_name: "context"})
                    sub = sub.dropna(subset=["context"])
                    sub["entity_type"] = sub["Entity Type"]
                    sub["context"] = sub["context"].str.replace("[Mask]", "[MASK]", regex=False)
                    sub["context"] = sub["context"].str.replace("[mask]", "[MASK]", regex=False)
                    print(f"  Indian combined (en): {len(sub)} contexts")
                    return sub[["context", "entity_type"]]
            raise ValueError(f"English column not found in {context_path}")

        frames = []
        for lang_code in INDIAN_COMBINED_LANGUAGES:
            if lang_code in df.columns:
                sub = df[["Entity Type", lang_code]].copy()
                sub = sub.rename(columns={lang_code: "context"})
                sub = sub.dropna(subset=["context"])
                sub["entity_type"] = sub["Entity Type"]
                frames.append(sub)

        combined = pd.concat(frames, ignore_index=True)
        combined["context"] = combined["context"].str.replace("[Mask]", "[MASK]", regex=False)
        combined["context"] = combined["context"].str.replace("[mask]", "[MASK]", regex=False)

        print(f"  Indian combined: {len(combined)} contexts from {len(frames)} languages")
        return combined[["context", "entity_type"]]

    # ---- Normal path ----
    language_full = LANGUAGE_NAME_MAP.get(language, language)
    language_short = LANGUAGE_CODE_MAP.get(language_full, language)
    subdir = f"camellia-{context_type}/causal-lms"

    # Resolve filename based on culture group
    if language_full.lower() in INDIAN_CULTURES or language_short.lower() in INDIAN_CULTURES:
        filename = f"{context_type}-contexts-causal-lms-indian.xlsx"
    elif language_full.lower() in PAKISTANI_CULTURES or language_short.lower() in PAKISTANI_CULTURES:
        filename = f"{context_type}-contexts-causal-lms-pakistani.xlsx"
    else:
        filename = f"{context_type}-contexts-causal-lms-{language_full}.xlsx"

    context_path = context_dir / subdir / filename

    if not context_path.exists():
        raise FileNotFoundError(f"Context file not found: {context_path}")

    df = pd.read_excel(context_path)
    df.columns = [col.strip() for col in df.columns]

    if target_lang == "en":
        # English column (varies by dataset: "English Context", "en", "English Prompt")
        for col_name in ["English Context", "en", "English Prompt"]:
            if col_name in df.columns:
                df["context"] = df[col_name]
                break
        else:
            raise ValueError(f"English column not found in {context_path}. Columns: {df.columns.tolist()}")
    elif language_full.lower() in INDIAN_CULTURES or language_short.lower() in INDIAN_CULTURES:
        # Indian file: use language code as column name (hi, mr, ml, gu)
        if language_short in df.columns:
            df["context"] = df[language_short]
        else:
            raise ValueError(
                f"Column '{language_short}' not found in {context_path}. "
                f"Available: {df.columns.tolist()}"
            )
    else:
        # Standard format: "Context" or "Prompt" column
        for col_name in ["Context", "Prompt"]:
            if col_name in df.columns:
                df["context"] = df[col_name]
                break
        else:
            raise ValueError(f"Context column not found in {context_path}. Columns: {df.columns.tolist()}")

    df["entity_type"] = df["Entity Type"]
    df["context"] = df["context"].str.replace("[Mask]", "[MASK]", regex=False)
    df["context"] = df["context"].str.replace("[mask]", "[MASK]", regex=False)

    # Drop rows where context is NaN (some languages may have missing entries)
    df = df.dropna(subset=["context"])

    return df[["context", "entity_type"]]


def load_entities(
    entity_dir: Path,
    entity_type: str,
    culture: str,
    language: str = "korean",
    target_lang: str = "cu",
) -> List[str]:
    # Resolve entity filename
    if culture == "western" and entity_type in WESTERN_FILE_MAPPING:
        # South Asian cultures use cricket instead of football
        if entity_type == "sports" and language.lower() in CRICKET_CULTURES:
            actual_filename = "cricket-clubs"
        else:
            actual_filename = WESTERN_FILE_MAPPING[entity_type]
    else:
        actual_filename = entity_type

    # Resolve entity directory: Indian/Pakistani cultures share group directories
    culture_dir = culture
    if culture.lower() in INDIAN_CULTURES:
        culture_dir = "indian"
    elif culture.lower() in PAKISTANI_CULTURES:
        culture_dir = "pakistani"

    entity_path = entity_dir / culture_dir / f"{actual_filename}.xlsx"
    if not entity_path.exists():
        raise FileNotFoundError(f"Entity file not found: {entity_path}")

    df = pd.read_excel(entity_path)
    df.columns = [col.strip() for col in df.columns]
    col_map = {col.lower(): col for col in df.columns}

    lang_code = LANGUAGE_CODE_MAP.get(language, language).lower()

    if culture == "western":
        # Western entities: use target language column
        if target_lang == "en":
            if "en" in col_map:
                entities = df[col_map["en"]].dropna().tolist()
            else:
                raise ValueError(f"'en' column not found in {entity_path}")
        else:
            if lang_code in col_map:
                entities = df[col_map[lang_code]].dropna().tolist()
            elif "en" in col_map:
                print(f"Warning: '{lang_code}' not found in {entity_path}, falling back to 'en'")
                entities = df[col_map["en"]].dropna().tolist()
            else:
                raise ValueError(f"Language '{lang_code}' not found in {entity_path}")
    elif culture.lower() in INDIAN_CULTURES:
        # Indian entities: multi-lang file with language code columns (hi, mr, ml, gu)
        if target_lang == "en":
            if "en" in col_map:
                entities = df[col_map["en"]].dropna().tolist()
            else:
                raise ValueError(f"'en' column not found in {entity_path}")
        else:
            if lang_code in col_map:
                entities = df[col_map[lang_code]].dropna().tolist()
            else:
                raise ValueError(
                    f"Column '{lang_code}' not found in {entity_path}. "
                    f"Available: {list(col_map.keys())}"
                )
    else:
        # Standard (ko, ja, zh, vi, ur/pakistani): "Entity" / "Translation" columns
        if target_lang == "en":
            if "Translation" in df.columns:
                entities = df["Translation"].dropna().tolist()
            else:
                raise ValueError(f"Translation column not found in {entity_path}")
        else:
            if "Entity" in df.columns:
                entities = df["Entity"].dropna().tolist()
            else:
                raise ValueError(f"Entity column not found in {entity_path}")

    return [str(e).strip() for e in entities if str(e).strip()]


# =============================================================================
# CAMEL (Arab) Loading Functions
# =============================================================================

def _resolve_camel_root(data_root: Path) -> Path:
    """
    Derive CAMEL data root from Camellia data root.
    dataset/camellia/raw → dataset/camel
    dataset/camel → dataset/camel (already correct)
    """
    if "camel" in data_root.name and "camellia" not in str(data_root):
        return data_root
    # Go up to dataset/ level and find camel/
    dataset_dir = data_root
    while dataset_dir.name != "dataset" and dataset_dir != dataset_dir.parent:
        dataset_dir = dataset_dir.parent
    camel_root = dataset_dir / "camel"
    if camel_root.exists():
        return camel_root
    raise FileNotFoundError(
        f"CAMEL dataset not found. Tried: {camel_root}. "
        f"Set --data_root to point to the CAMEL directory."
    )


def load_camel_contexts(
    camel_root: Path,
    context_type: str,
) -> pd.DataFrame:
    """Load CAMEL (Arab) contexts. context_type: 'grounded' or 'neutral'."""
    context_path = camel_root / CAMEL_CONTEXT_FILES[context_type]

    if not context_path.exists():
        raise FileNotFoundError(f"CAMEL context file not found: {context_path}")

    df = pd.read_excel(context_path)
    df.columns = [col.strip() for col in df.columns]

    df["context"] = df["Prompt"]
    df["entity_type"] = df["Entity Type"]
    df["context"] = df["context"].str.replace("[Mask]", "[MASK]", regex=False)
    df["context"] = df["context"].str.replace("[mask]", "[MASK]", regex=False)
    df = df.dropna(subset=["context"])

    return df[["context", "entity_type"]]


def load_camel_entities(
    camel_root: Path,
    entity_type: str,
) -> Tuple[List[str], List[str]]:
    """
    Load CAMEL entities (Arab + Western from same file).
    Returns: (arab_entities, western_entities)
    """
    actual_type = CAMEL_ENTITY_TYPE_MAP.get(entity_type, entity_type)
    entity_path = camel_root / "entities" / f"{actual_type}.xlsx"

    if not entity_path.exists():
        raise FileNotFoundError(f"CAMEL entity file not found: {entity_path}")

    df = pd.read_excel(entity_path)
    df.columns = [col.strip() for col in df.columns]

    arab = df[df["Culture"] == "Arab"]["Entity"].dropna().tolist()
    western = df[df["Culture"] == "Western"]["Entity"].dropna().tolist()

    arab = [str(e).strip() for e in arab if str(e).strip()]
    western = [str(e).strip() for e in western if str(e).strip()]

    return arab, western


def load_camellia_data(
    data_root: Union[str, Path],
    culture: str = "korean",
    target_lang: str = "cu",
    entity_types: Optional[List[str]] = None,
) -> CamelliaData:
    data_root = Path(data_root)

    # ---- CAMEL (Arab): completely different directory structure ----
    if culture.lower() in ARAB_CULTURES:
        camel_root = _resolve_camel_root(data_root)

        if entity_types is None:
            entity_types = [
                "authors", "beverage", "food", "locations",
                "names-female", "names-male", "sports"
            ]

        grounded_df = load_camel_contexts(camel_root, "grounded")
        neutral_df = load_camel_contexts(camel_root, "neutral")
        print(f"Loaded {len(grounded_df)} grounded, {len(neutral_df)} neutral contexts (CAMEL/Arab)")

        entities = {}
        for entity_type in entity_types:
            try:
                arab_ents, western_ents = load_camel_entities(camel_root, entity_type)
                entities[entity_type] = {
                    "asian": arab_ents,
                    "western": western_ents,
                }
                print(f"  {entity_type}: {len(arab_ents)} Arab, {len(western_ents)} Western")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue

        return CamelliaData(
            grounded_contexts=grounded_df,
            neutral_contexts=neutral_df,
            entities=entities,
            culture="ar",
            lang=target_lang,
        )

    context_dir = data_root / "contexts"
    entity_dir = data_root / "entities"

    # ---- Indian Combined: pool contexts, use hi as entity anchor ----
    if culture == INDIAN_COMBINED:
        if entity_types is None:
            entity_types = [
                "authors", "beverage", "food", "locations",
                "names-female", "names-male", "sports"
            ]

        grounded_df = load_contexts(context_dir, "grounded", INDIAN_COMBINED, target_lang)
        neutral_df = load_contexts(context_dir, "neutral", INDIAN_COMBINED, target_lang)
        print(f"Loaded {len(grounded_df)} grounded contexts, {len(neutral_df)} neutral contexts (indian_combined)")

        # Entities: use anchor language (hi) from indian directory
        anchor_lang = INDIAN_COMBINED_ANCHOR  # "hi"
        anchor_full = LANGUAGE_NAME_MAP.get(anchor_lang, anchor_lang)  # "hindi"

        entities = {}
        for entity_type in entity_types:
            try:
                asian_entities = load_entities(
                    entity_dir, entity_type, anchor_full,
                    language=anchor_full, target_lang=target_lang
                )
                western_entities = load_entities(
                    entity_dir, entity_type, "western",
                    language=anchor_full, target_lang=target_lang
                )
                entities[entity_type] = {
                    "asian": asian_entities,
                    "western": western_entities,
                }
                print(f"  {entity_type}: {len(asian_entities)} Asian, {len(western_entities)} Western")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue

        return CamelliaData(
            grounded_contexts=grounded_df,
            neutral_contexts=neutral_df,
            entities=entities,
            culture="indian_combined",
            lang=target_lang,
        )

    # ---- Normal path ----
    culture_full = LANGUAGE_NAME_MAP.get(culture, culture)
    culture_short = LANGUAGE_CODE_MAP.get(culture_full, culture)

    if entity_types is None:
        entity_types = [
            "authors", "beverage", "food", "locations",
            "names-female", "names-male", "sports"
        ]

    grounded_df = load_contexts(context_dir, "grounded", culture_full, target_lang)
    neutral_df = load_contexts(context_dir, "neutral", culture_full, target_lang)
    print(f"Loaded {len(grounded_df)} grounded contexts, {len(neutral_df)} neutral contexts")

    entities = {}
    for entity_type in entity_types:
        try:
            asian_entities = load_entities(
                entity_dir, entity_type, culture_full,
                language=culture_full, target_lang=target_lang
            )
            western_entities = load_entities(
                entity_dir, entity_type, "western",
                language=culture_full, target_lang=target_lang
            )
            entities[entity_type] = {
                "asian": asian_entities,
                "western": western_entities,
            }
            print(f"  {entity_type}: {len(asian_entities)} Asian, {len(western_entities)} Western")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    return CamelliaData(
        grounded_contexts=grounded_df,
        neutral_contexts=neutral_df,
        entities=entities,
        culture=culture_short,
        lang=target_lang,
    )


# =============================================================================
# 3-Way Split Utilities
# =============================================================================

def _split_list_three_way(
    items: list,
    train_ratio: float,
    val_ratio: float,
    min_per_split: int = 2,
) -> Tuple[list, list, list]:
    """
    Split a list into train/val/test with NO overlap.
    
    For very small lists (n < 3*min_per_split), reduces min_per_split to 1.
    Items are NEVER duplicated across splits.
    """
    n = len(items)
    
    if n == 0:
        return [], [], []
    
    shuffled = items[:]
    random.shuffle(shuffled)
    
    if n == 1:
        # Only 1 item: put in train, val/test empty
        return shuffled[:], [], []
    
    if n == 2:
        # 2 items: train gets 1, test gets 1, val empty
        return shuffled[:1], [], shuffled[1:]
    
    # For small lists, relax min_per_split
    effective_min = min(min_per_split, n // 3) if n < 3 * min_per_split else min_per_split
    effective_min = max(1, effective_min)
    
    # Calculate split sizes
    n_train = max(effective_min, int(n * train_ratio))
    n_val = max(effective_min, int(n * val_ratio))
    n_test = n - n_train - n_val
    
    # Ensure test gets at least effective_min
    if n_test < effective_min:
        deficit = effective_min - n_test
        n_train = max(effective_min, n_train - deficit)
        n_test = n - n_train - n_val
    
    # Final safety: ensure no negative
    if n_test < 0:
        n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
    
    assert n_train + n_val + n_test == n, f"Split error: {n_train}+{n_val}+{n_test} != {n}"
    assert n_train >= 1 and n_test >= 1, f"Empty split: train={n_train}, test={n_test}"
    
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    
    return train, val, test


def _split_df_three_way(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train/val/test, STRATIFIED by entity_type.
    
    Ensures every entity_type is represented in every split.
    """
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Group by entity_type, split each group
    for etype, group in df.groupby("entity_type"):
        indices = group.index.tolist()
        random.shuffle(indices)
        n = len(indices)
        
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val
        
        if n_test < 1:
            n_train = max(1, n_train - 1)
            n_test = n - n_train - n_val
        if n_test < 0:
            n_val = max(0, n - n_train)
            n_test = 0
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    return (
        df.loc[train_indices].reset_index(drop=True) if train_indices else pd.DataFrame(columns=df.columns),
        df.loc[val_indices].reset_index(drop=True) if val_indices else pd.DataFrame(columns=df.columns),
        df.loc[test_indices].reset_index(drop=True) if test_indices else pd.DataFrame(columns=df.columns),
    )


# =============================================================================
# Data Split (v2: context + entity split)
# =============================================================================

def split_data(
    data: CamelliaData,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_pairs_per_context: int = 10,
    min_entities_per_split: int = 2,
) -> Tuple[List[CamelliaExample], List[CamelliaExample], List[CamelliaExample], Dict]:
    """
    Split data into train/val/test sets.
    
    BOTH contexts AND entities are split to prevent any leakage.
    - Train examples use train_contexts × train_entities
    - Val examples use val_contexts × val_entities
    - Test examples use test_contexts × test_entities
    
    Args:
        data: CamelliaData object
        train_ratio: Ratio for training (default: 0.7)
        val_ratio: Ratio for validation (default: 0.1, test gets remainder)
        seed: Random seed for reproducibility
        max_pairs_per_context: Maximum entity pairs per context
        min_entities_per_split: Minimum entities per split per type
    
    Returns:
        (train_examples, val_examples, test_examples, split_info)
    """
    random.seed(seed)
    
    # =========================================================================
    # 1. Split contexts (train/val/test)
    # =========================================================================
    grounded_train, grounded_val, grounded_test = _split_df_three_way(
        data.grounded_contexts, train_ratio, val_ratio
    )
    neutral_train, neutral_val, neutral_test = _split_df_three_way(
        data.neutral_contexts, train_ratio, val_ratio
    )
    
    # =========================================================================
    # 2. Split entities (train/val/test) per entity type
    # =========================================================================
    train_entities = {}
    val_entities = {}
    test_entities = {}
    
    for etype, ent_dict in data.entities.items():
        a_train, a_val, a_test = _split_list_three_way(
            ent_dict["asian"], train_ratio, val_ratio, min_entities_per_split
        )
        w_train, w_val, w_test = _split_list_three_way(
            ent_dict["western"], train_ratio, val_ratio, min_entities_per_split
        )
        
        train_entities[etype] = {"asian": a_train, "western": w_train}
        val_entities[etype] = {"asian": a_val, "western": w_val}
        test_entities[etype] = {"asian": a_test, "western": w_test}
    
    # =========================================================================
    # 3. Create examples (each split uses its own entities)
    # =========================================================================
    def create_examples(
        contexts_df: pd.DataFrame,
        context_type: str,
        entities: Dict,
        culture: str,
        lang: str,
        max_pairs: int,
    ) -> List[CamelliaExample]:
        examples = []
        for _, row in contexts_df.iterrows():
            context = row["context"]
            entity_type_raw = row["entity_type"].strip().lower()
            entity_type = ENTITY_TYPE_ALIASES.get(entity_type_raw, entity_type_raw)

            if entity_type == "names-both":
                entity_types_to_use = ["names-male", "names-female"]
            else:
                entity_types_to_use = [entity_type]

            for etype in entity_types_to_use:
                if etype not in entities:
                    found = False
                    for key in entities.keys():
                        if key.lower() == etype or key.lower().startswith(etype.split("-")[0]):
                            etype = key
                            found = True
                            break
                    if not found:
                        continue

                asian_list = entities[etype]["asian"]
                western_list = entities[etype]["western"]
                if not asian_list or not western_list:
                    continue

                n_pairs = min(len(asian_list), len(western_list), max_pairs)
                asian_shuffled = asian_list[:]
                western_shuffled = western_list[:]
                random.shuffle(asian_shuffled)
                random.shuffle(western_shuffled)

                for i in range(n_pairs):
                    examples.append(CamelliaExample(
                        context=context,
                        context_type=context_type,
                        entity_type=etype,
                        asian_entity=asian_shuffled[i],
                        western_entity=western_shuffled[i],
                        culture=culture,
                        lang=lang,
                    ))
        return examples

    train_examples = (
        create_examples(grounded_train, "grounded", train_entities, data.culture, data.lang, max_pairs_per_context) +
        create_examples(neutral_train, "neutral", train_entities, data.culture, data.lang, max_pairs_per_context)
    )
    val_examples = (
        create_examples(grounded_val, "grounded", val_entities, data.culture, data.lang, max_pairs_per_context) +
        create_examples(neutral_val, "neutral", val_entities, data.culture, data.lang, max_pairs_per_context)
    )
    test_examples = (
        create_examples(grounded_test, "grounded", test_entities, data.culture, data.lang, max_pairs_per_context) +
        create_examples(neutral_test, "neutral", test_entities, data.culture, data.lang, max_pairs_per_context)
    )

    random.shuffle(train_examples)
    random.shuffle(val_examples)
    random.shuffle(test_examples)

    # =========================================================================
    # 4. Build split_info (for robust N×M evaluation)
    # =========================================================================
    split_info = {
        # Context DataFrames per split
        "grounded_train": grounded_train,
        "grounded_val": grounded_val,
        "grounded_test": grounded_test,
        "neutral_train": neutral_train,
        "neutral_val": neutral_val,
        "neutral_test": neutral_test,
        # Entity dictionaries per split
        "train_entities": train_entities,
        "val_entities": val_entities,
        "test_entities": test_entities,
    }

    # =========================================================================
    # 5. Print statistics
    # =========================================================================
    def print_split_stats(name, examples, ents):
        grounded = [ex for ex in examples if ex.context_type == "grounded"]
        neutral = [ex for ex in examples if ex.context_type == "neutral"]
        n_asian = sum(len(v["asian"]) for v in ents.values())
        n_western = sum(len(v["western"]) for v in ents.values())
        print(f"  {name}: {len(examples):,} examples "
              f"({len(grounded):,}G + {len(neutral):,}N), "
              f"entities: {n_asian}A + {n_western}W")

    print(f"\nData Split (seed={seed}, ratio={train_ratio}/{val_ratio}/{round(1-train_ratio-val_ratio,2)}):")
    print(f"  Contexts: G={len(grounded_train)}+{len(grounded_val)}+{len(grounded_test)}, "
          f"N={len(neutral_train)}+{len(neutral_val)}+{len(neutral_test)}")
    
    # Stratified context distribution
    for label, df in [("G_train", grounded_train), ("G_val", grounded_val), ("G_test", grounded_test)]:
        types = df["entity_type"].value_counts().to_dict() if len(df) > 0 else {}
        print(f"    {label}: {dict(types)}")
    
    print_split_stats("Train", train_examples, train_entities)
    print_split_stats("Val  ", val_examples, val_entities)
    print_split_stats("Test ", test_examples, test_entities)

    # Entity overlap check
    for etype in train_entities:
        train_a = set(train_entities[etype]["asian"])
        val_a = set(val_entities[etype]["asian"])
        test_a = set(test_entities[etype]["asian"])
        leak = (train_a & val_a) | (train_a & test_a) | (val_a & test_a)
        if leak:
            print(f"  ⚠ Entity leak in {etype} asian: {len(leak)} overlapping")
        
        train_w = set(train_entities[etype]["western"])
        val_w = set(val_entities[etype]["western"])
        test_w = set(test_entities[etype]["western"])
        leak = (train_w & val_w) | (train_w & test_w) | (val_w & test_w)
        if leak:
            print(f"  ⚠ Entity leak in {etype} western: {len(leak)} overlapping")

    print("  ✓ Context + Entity split complete (zero leakage)")

    return train_examples, val_examples, test_examples, split_info


# =============================================================================
# Paired Context Dataset (v3: same entity, different contexts)
# =============================================================================

@dataclass
class CategoryData:
    """Per-category data for paired training."""
    grounded_contexts: List[str]     # context strings with [MASK]
    neutral_contexts: List[str]
    asian_entities: List[str]
    western_entities: List[str]
    category: str


def build_category_data(
    grounded_df: pd.DataFrame,
    neutral_df: pd.DataFrame,
    entities: Dict[str, Dict[str, List[str]]],
) -> Dict[str, CategoryData]:
    """
    Organize split data by category for paired sampling.
    
    Maps context entity_type to entity keys via ENTITY_TYPE_ALIASES,
    handles 'Names' → names-male + names-female by merging entities.
    """
    categories = {}
    
    # Build context lists per raw entity_type
    grounded_by_type = {}
    for _, row in grounded_df.iterrows():
        raw = row["entity_type"].strip()
        grounded_by_type.setdefault(raw, []).append(row["context"])
    
    neutral_by_type = {}
    for _, row in neutral_df.iterrows():
        raw = row["entity_type"].strip()
        neutral_by_type.setdefault(raw, []).append(row["context"])
    
    # Map each raw type to entity keys
    for raw_type in set(list(grounded_by_type.keys()) + list(neutral_by_type.keys())):
        alias = ENTITY_TYPE_ALIASES.get(raw_type.lower(), raw_type.lower())
        
        if alias == "names-both":
            entity_types_to_use = ["names-male", "names-female"]
        else:
            entity_types_to_use = [alias]
        
        # Merge entities for names-both
        asian_all = []
        western_all = []
        for etype in entity_types_to_use:
            if etype in entities:
                asian_all.extend(entities[etype]["asian"])
                western_all.extend(entities[etype]["western"])
        
        if not asian_all or not western_all:
            continue
        
        g_contexts = grounded_by_type.get(raw_type, [])
        n_contexts = neutral_by_type.get(raw_type, [])
        
        if not g_contexts or not n_contexts:
            continue
        
        # Use raw_type as key (preserves original category name)
        cat_key = raw_type.lower()
        if cat_key in categories:
            # Merge (e.g., if names-male and names-female map to same contexts)
            categories[cat_key].asian_entities.extend(asian_all)
            categories[cat_key].western_entities.extend(western_all)
        else:
            categories[cat_key] = CategoryData(
                grounded_contexts=g_contexts,
                neutral_contexts=n_contexts,
                asian_entities=asian_all,
                western_entities=western_all,
                category=cat_key,
            )
    
    return categories


class PairedDataset(Dataset):
    """
    Paired Context Dataset: each item = (grounded_ctx, neutral_ctx, asian, western).
    
    Same entity pair is shown in both grounded and neutral contexts,
    forcing the model to learn context-dependent behavior.
    """
    
    def __init__(
        self,
        category_data: Dict[str, CategoryData],
        tokenizer: PreTrainedTokenizer,
        pairs_per_category: int = 200,
        max_length: int = 128,
        seed: int = 42,
    ):
        self.category_data = category_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-build pair index: (category_key, asian_idx, western_idx)
        rng = random.Random(seed)
        self.pairs = []
        
        for cat_key, cdata in category_data.items():
            n_asian = len(cdata.asian_entities)
            n_western = len(cdata.western_entities)
            n_pairs = min(n_asian, n_western, pairs_per_category)
            
            asian_indices = list(range(n_asian))
            western_indices = list(range(n_western))
            rng.shuffle(asian_indices)
            rng.shuffle(western_indices)
            
            for i in range(n_pairs):
                self.pairs.append((
                    cat_key,
                    asian_indices[i % n_asian],
                    western_indices[i % n_western],
                ))
        
        rng.shuffle(self.pairs)
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cat_key, a_idx, w_idx = self.pairs[idx]
        cdata = self.category_data[cat_key]
        
        asian = cdata.asian_entities[a_idx]
        western = cdata.western_entities[w_idx]
        
        # Randomly sample one grounded and one neutral context
        g_ctx = random.choice(cdata.grounded_contexts)
        n_ctx = random.choice(cdata.neutral_contexts)
        
        # Build 4 sequences
        g_asian_text = g_ctx.replace("[MASK]", asian)
        g_western_text = g_ctx.replace("[MASK]", western)
        n_asian_text = n_ctx.replace("[MASK]", asian)
        n_western_text = n_ctx.replace("[MASK]", western)
        
        # Context prefixes (for log prob offset)
        g_prefix = g_ctx.split("[MASK]")[0]
        n_prefix = n_ctx.split("[MASK]")[0]
        
        def tok(text):
            enc = self.tokenizer(
                text, max_length=self.max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        
        g_a_ids, g_a_mask = tok(g_asian_text)
        g_w_ids, g_w_mask = tok(g_western_text)
        n_a_ids, n_a_mask = tok(n_asian_text)
        n_w_ids, n_w_mask = tok(n_western_text)
        g_ctx_ids, g_ctx_mask = tok(g_prefix)
        n_ctx_ids, n_ctx_mask = tok(n_prefix)
        
        return {
            # Grounded (loss G)
            "g_asian_input_ids": g_a_ids,
            "g_asian_attention_mask": g_a_mask,
            "g_western_input_ids": g_w_ids,
            "g_western_attention_mask": g_w_mask,
            "g_context_input_ids": g_ctx_ids,
            # Neutral (loss N)
            "n_asian_input_ids": n_a_ids,
            "n_asian_attention_mask": n_a_mask,
            "n_western_input_ids": n_w_ids,
            "n_western_attention_mask": n_w_mask,
            "n_context_input_ids": n_ctx_ids,
        }


class PairedBatchSampler:
    """
    Samples batches with uniform category distribution.
    
    Each batch = pairs_per_batch pairs, spread across categories.
    """
    
    def __init__(
        self,
        dataset: PairedDataset,
        pairs_per_batch: int = 8,
        seed: int = 42,
        epoch: int = 0,
    ):
        self.pairs_per_batch = pairs_per_batch
        self.seed = seed
        self.epoch = epoch
        
        # Index pairs by category
        self.cat_indices = {}
        for i, (cat_key, _, _) in enumerate(dataset.pairs):
            self.cat_indices.setdefault(cat_key, []).append(i)
        
        self.categories = list(self.cat_indices.keys())
        self.total_pairs = len(dataset)
        self.total_batches = max(1, self.total_pairs // pairs_per_batch)
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        
        # Shuffle within each category
        cat_pools = {}
        for cat, indices in self.cat_indices.items():
            pool = indices[:]
            rng.shuffle(pool)
            cat_pools[cat] = pool
        
        cat_cycle_indices = {cat: 0 for cat in self.categories}
        
        for _ in range(self.total_batches):
            batch = []
            for _ in range(self.pairs_per_batch):
                # Pick category (round-robin with shuffle)
                cat = self.categories[len(batch) % len(self.categories)]
                pool = cat_pools[cat]
                ci = cat_cycle_indices[cat]
                
                if ci >= len(pool):
                    rng.shuffle(pool)
                    ci = 0
                
                batch.append(pool[ci])
                cat_cycle_indices[cat] = ci + 1
            
            yield batch
    
    def __len__(self) -> int:
        return self.total_batches


def create_paired_dataloader(
    grounded_df: pd.DataFrame,
    neutral_df: pd.DataFrame,
    entities: Dict[str, Dict[str, List[str]]],
    tokenizer: PreTrainedTokenizer,
    pairs_per_batch: int = 8,
    pairs_per_category: int = 200,
    max_length: int = 128,
    seed: int = 42,
    num_workers: int = 4,
) -> DataLoader:
    """Create paired context dataloader."""
    cat_data = build_category_data(grounded_df, neutral_df, entities)
    
    print(f"\nPaired Dataset (per category):")
    for cat_key, cdata in cat_data.items():
        print(f"  {cat_key}: {len(cdata.grounded_contexts)}G × {len(cdata.neutral_contexts)}N "
              f"× {len(cdata.asian_entities)}A × {len(cdata.western_entities)}W")
    
    dataset = PairedDataset(cat_data, tokenizer, pairs_per_category, max_length, seed)
    sampler = PairedBatchSampler(dataset, pairs_per_batch, seed)
    
    print(f"  Total pairs: {len(dataset)}, batches: {len(sampler)} "
          f"({pairs_per_batch} pairs/batch = {pairs_per_batch * 4} sequences)")
    
    return DataLoader(
        dataset, batch_sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )


# =============================================================================
# Legacy Dataset (kept for evaluation / backward compat)
# =============================================================================

class CBMCDDataset(Dataset):
    def __init__(self, examples: List[CamelliaExample], tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        asian_text = example.context.replace("[MASK]", example.asian_entity)
        western_text = example.context.replace("[MASK]", example.western_entity)

        asian_encoding = self.tokenizer(
            asian_text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        western_encoding = self.tokenizer(
            western_text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        context_before_mask = example.context.split("[MASK]")[0]
        context_encoding = self.tokenizer(
            context_before_mask, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )

        return {
            "asian_input_ids": asian_encoding["input_ids"].squeeze(0),
            "asian_attention_mask": asian_encoding["attention_mask"].squeeze(0),
            "western_input_ids": western_encoding["input_ids"].squeeze(0),
            "western_attention_mask": western_encoding["attention_mask"].squeeze(0),
            "context_input_ids": context_encoding["input_ids"].squeeze(0),
            "context_attention_mask": context_encoding["attention_mask"].squeeze(0),
            "context_type": 0 if example.context_type == "grounded" else 1,
            "entity_type": example.entity_type,
        }


class BalancedBatchSampler:
    def __init__(self, examples: List[CamelliaExample], samples_per_type: int = 16,
                 seed: int = 42, epoch: int = 0):
        self.samples_per_type = samples_per_type
        self.seed = seed
        self.epoch = epoch
        self.grounded_indices = [i for i, ex in enumerate(examples) if ex.context_type == "grounded"]
        self.neutral_indices = [i for i, ex in enumerate(examples) if ex.context_type == "neutral"]
        self.total_batches = max(
            len(self.grounded_indices) // samples_per_type,
            len(self.neutral_indices) // samples_per_type,
        ) + 1

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        random.seed(self.seed + self.epoch)
        g = self.grounded_indices[:]
        n = self.neutral_indices[:]
        random.shuffle(g)
        random.shuffle(n)
        gc = cycle(g)
        nc = cycle(n)
        for _ in range(self.total_batches):
            batch = [next(gc) for _ in range(self.samples_per_type)]
            batch += [next(nc) for _ in range(self.samples_per_type)]
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.total_batches


def create_balanced_dataloader(
    examples: List[CamelliaExample],
    tokenizer: PreTrainedTokenizer,
    samples_per_type: int = 16,
    max_length: int = 128,
    seed: int = 42,
    num_workers: int = 4,
) -> DataLoader:
    dataset = CBMCDDataset(examples, tokenizer, max_length)
    sampler = BalancedBatchSampler(examples, samples_per_type, seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)


def get_data_stats(examples: List[CamelliaExample]) -> Dict:
    grounded = [ex for ex in examples if ex.context_type == "grounded"]
    neutral = [ex for ex in examples if ex.context_type == "neutral"]
    entity_types = {}
    for ex in examples:
        entity_types[ex.entity_type] = entity_types.get(ex.entity_type, 0) + 1
    return {
        "total": len(examples),
        "grounded": len(grounded),
        "neutral": len(neutral),
        "entity_types": entity_types,
    }