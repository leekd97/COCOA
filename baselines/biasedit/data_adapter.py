"""
BiasEdit Data Adapter

Converts COCOA's split_info → BiasEdit K:K format.
Uses ONLY neutral contexts (as per BiasEdit's original design).
Entities and contexts come from COCOA's split_data() for fair comparison.
"""

import copy
import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

# Entity type aliases (same as COCOA src/data.py)
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


class BiasEditDataset(Dataset):
    """
    K:K dataset for BiasEdit using COCOA's pre-split data.
    
    Each sample = 1 neutral context × (K Asian + K Western) entities.
    """
    
    def __init__(
        self,
        contexts_df,      # DataFrame with columns: context, entity_type
        entities: Dict,   # {etype: {"asian": [...], "western": [...]}}
        tokenizer,
        device: str = "cuda",
        max_length: int = 128,
        k_entities: int = 15,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.k_entities = k_entities
        
        # Detect model type
        model_name = tokenizer.name_or_path.lower()
        self.iscausal = any(m in model_name for m in ["gpt", "llama", "mistral", "gemma", "qwen"])
        
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create K:K examples
        self.data = self._create_examples(contexts_df, entities, seed)
        print(f"BiasEditDataset: {len(self.data)} examples (K={k_entities})")
    
    def _create_examples(self, contexts_df, entities, seed) -> List[Dict]:
        random.seed(seed)
        examples = []
        
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
                    # Try fuzzy match
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
                
                k = min(len(asian_list), len(western_list), self.k_entities)
                if k < 2:
                    continue
                
                # Shuffle and select
                a_shuffled = asian_list.copy()
                w_shuffled = western_list.copy()
                random.shuffle(a_shuffled)
                random.shuffle(w_shuffled)
                
                selected_asian = a_shuffled[:k]
                selected_western = w_shuffled[:k]
                
                asian_sentences = [context.replace("[MASK]", ent) for ent in selected_asian]
                western_sentences = [context.replace("[MASK]", ent) for ent in selected_western]
                
                examples.append({
                    "entity_type": etype,
                    "asian_sentences": asian_sentences,
                    "western_sentences": western_sentences,
                    "k": k,
                })
        
        random.shuffle(examples)
        return examples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]
    
    def collate_fn(self, batch):
        """
        Collate K:K batches. Each context kept as separate batch.
        Returns format compatible with BiasEdit editor.
        """
        edit_batches = []
        k_per_sample = []
        
        for b in batch:
            asian_sentences = b["asian_sentences"]
            western_sentences = b["western_sentences"]
            k = b["k"]
            
            # Asian first, then Western
            all_sentences = asian_sentences + western_sentences
            
            inputs = self.tokenizer(
                all_sentences,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            
            # Create labels (mask padding)
            inputs["labels"] = copy.deepcopy(inputs["input_ids"])
            for idx in range(len(inputs["labels"])):
                inputs["labels"][idx] = torch.where(
                    inputs["labels"][idx] != self.tokenizer.pad_token_id,
                    inputs["input_ids"][idx],
                    -100
                )
                inputs["labels"][idx][0] = -100  # Mask BOS
            
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            edit_batches.append(inputs)
            k_per_sample.append(k)
        
        return {
            "edit": edit_batches,
            "unrelated": None,
            "k_per_sample": k_per_sample,
        }