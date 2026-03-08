"""
BiasUnlearn Data Adapter

Converts COCOA's split_info → BiasUnlearn format (ster/anti/unrelated).
Uses ONLY neutral contexts for training (social bias equalization).

Mapping:
  stereotype (ster)     → context + Western entity → NPO loss (forget)
  anti-stereotype (anti)→ context + Asian entity  → GD loss (learn)
  unrelated             → context only            → KL loss (preserve)
"""

import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

# Same aliases as COCOA
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


def prepare_unlearn_data(
    contexts_df,     # neutral_train DataFrame (context, entity_type)
    entities: Dict,  # train_entities: {etype: {"asian": [...], "western": [...]}}
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Convert COCOA split data to BiasUnlearn triplets.
    
    Returns:
        ster_data:  [{"context": ..., "text": context+western, "entity_type": ...}, ...]
        anti_data:  [{"context": ..., "text": context+asian, "entity_type": ...}, ...]
        unrelated:  [{"text": context, "entity_type": ...}, ...]
    """
    random.seed(seed)
    
    ster_data = []
    anti_data = []
    unrelated_data = []
    
    for _, row in contexts_df.iterrows():
        context = row["context"]
        entity_type_raw = row["entity_type"].strip().lower()
        entity_type = ENTITY_TYPE_ALIASES.get(entity_type_raw, entity_type_raw)
        
        if entity_type == "names-both":
            types_to_use = ["names-male", "names-female"]
        else:
            types_to_use = [entity_type]
        
        for etype in types_to_use:
            if etype not in entities:
                # Fuzzy match
                for key in entities:
                    if key.lower() == etype or key.lower().startswith(etype.split("-")[0]):
                        etype = key
                        break
                else:
                    continue
            
            asian_list = entities[etype].get("asian", [])
            western_list = entities[etype].get("western", [])
            
            if not asian_list or not western_list:
                continue
            
            # Sample one entity from each for this context
            asian_ent = random.choice(asian_list)
            western_ent = random.choice(western_list)
            
            # Replace [MASK] for completion-style training
            context_base = context.replace("[MASK]", "").strip()
            
            # Stereotype: Western entity (to forget/suppress)
            ster_data.append({
                "context": context_base,
                "text": context_base + " " + western_ent,
                "entity_type": etype,
            })
            
            # Anti-stereotype: Asian entity (to boost)
            anti_data.append({
                "context": context_base,
                "text": context_base + " " + asian_ent,
                "entity_type": etype,
            })
            
            # Unrelated: context only (for KL preservation)
            unrelated_data.append({
                "text": context_base,
                "entity_type": etype,
            })
    
    return ster_data, anti_data, unrelated_data


class UnlearnDataset(Dataset):
    """Simple dataset for BiasUnlearn that tokenizes on-the-fly."""
    
    def __init__(self, data: List[Dict], tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        context = item.get("context", text)  # For unrelated, context=text
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # Labels = input_ids (masked padding)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Find start_loc (where entity begins)
        context_encoded = self.tokenizer(
            context,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        start_loc = context_encoded["input_ids"].shape[1]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "start_locs": torch.tensor(start_loc, dtype=torch.long),
        }


def create_unlearn_dataloaders(
    contexts_df,
    entities: Dict,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 128,
    seed: int = 42,
    mix_anti: bool = False,
    mix_ratio: float = 0.25,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create all three dataloaders for BiasUnlearn.
    
    Returns:
        ster_loader, anti_loader, unrelated_loader
    """
    ster_data, anti_data, unrelated_data = prepare_unlearn_data(
        contexts_df, entities, seed
    )
    
    print(f"  Stereotype (Western): {len(ster_data)}")
    print(f"  Anti-stereotype (Asian): {len(anti_data)}")
    print(f"  Unrelated (context): {len(unrelated_data)}")
    
    # Adversarial Forget Set: mix anti into ster to prevent bias reversal
    if mix_anti and anti_data:
        n_mix = int(len(anti_data) * mix_ratio)
        random.seed(seed)
        mixed = random.sample(anti_data, min(n_mix, len(anti_data)))
        ster_data = ster_data + mixed
        random.shuffle(ster_data)
        print(f"  [mix_anti] Added {len(mixed)} anti samples to ster batch")
    
    ster_loader = DataLoader(
        UnlearnDataset(ster_data, tokenizer, max_length),
        batch_size=batch_size, shuffle=True,
    )
    anti_loader = DataLoader(
        UnlearnDataset(anti_data, tokenizer, max_length),
        batch_size=batch_size, shuffle=True,
    )
    unrelated_loader = DataLoader(
        UnlearnDataset(unrelated_data, tokenizer, max_length),
        batch_size=batch_size, shuffle=True,
    )
    
    return ster_loader, anti_loader, unrelated_loader