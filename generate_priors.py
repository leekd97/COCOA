"""
CoCoA: Entity Prior Generator

Pre-computes log P(entity|BOS) for all entities and saves to JSON.
Must be run once per model × culture × lang before training with --normalize_prior.

Usage:
    # All cultures, Llama
    python generate_priors.py --model llama3_8b --lang cu --device cuda:0

    # Specific culture
    python generate_priors.py --model llama3_8b --cultures ko --lang cu --device cuda:0

    # Both models (parallel on different GPUs)
    CUDA_VISIBLE_DEVICES=0 python generate_priors.py --model llama3_8b --lang cu --device cuda:0
    CUDA_VISIBLE_DEVICES=1 python generate_priors.py --model qwen3_8b --lang cu --device cuda:0

Output:
    dataset/priors/llama3_8b/ko_cu/entity_priors.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_camellia_data
from src.model import MODEL_SHORTCUTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

ALL_CULTURES = ["ko", "ja", "zh", "vi", "hi", "ur", "gu", "mr", "ml", "ar"]


@torch.no_grad()
def compute_entity_prior(model, tokenizer, entity: str, device, bos: str) -> float:
    """Compute log P(entity | BOS)."""
    full_text = bos + entity
    ctx_enc = tokenizer(bos, return_tensors="pt", add_special_tokens=True)
    full_enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)

    ctx_len = ctx_enc["input_ids"].size(1)
    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(outputs.logits, dim=-1)

    total_lp = 0.0
    for i in range(ctx_len - 1, input_ids.size(1) - 1):
        next_token = input_ids[0, i + 1]
        total_lp += log_probs[0, i, next_token].item()

    return total_lp


@torch.no_grad()
def compute_priors_batched(model, tokenizer, entities: List[str], device, bos: str, batch_size=32) -> Dict[str, float]:
    """Compute priors for a list of entities, batched."""
    ctx_enc = tokenizer(bos, return_tensors="pt", add_special_tokens=True)
    ctx_len = ctx_enc["input_ids"].size(1)

    results = {}
    texts = [bos + ent for ent in entities]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ents = entities[i:i+batch_size]

        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits, dim=-1)

        for j in range(len(batch_ents)):
            seq_len = attention_mask[j].sum().item()
            total_lp = 0.0
            for pos in range(ctx_len - 1, int(seq_len) - 1):
                next_token = input_ids[j, pos + 1]
                if next_token != tokenizer.pad_token_id:
                    total_lp += log_probs[j, pos, next_token].item()
            results[batch_ents[j]] = round(total_lp, 6)

    return results


def generate_priors_for_culture(
    model, tokenizer, culture, lang, data_root, output_root, model_key, device,
):
    LOG.info(f"Generating priors: {culture}/{lang}")

    data = load_camellia_data(data_root, culture=culture, target_lang=lang)
    bos = tokenizer.bos_token or ""

    priors = {}
    for etype, ent_dict in data.entities.items():
        priors[etype] = {}
        for side in ["asian", "western"]:
            ent_list = ent_dict[side]
            LOG.info(f"  {etype}/{side}: {len(ent_list)} entities")
            priors[etype][side] = compute_priors_batched(
                model, tokenizer, ent_list, device, bos
            )

    # Save
    output_dir = Path(output_root) / model_key / f"{culture}_{lang}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": MODEL_SHORTCUTS.get(model_key, model_key),
        "model_key": model_key,
        "culture": culture,
        "lang": lang,
        "priors": priors,
    }

    out_path = output_dir / "entity_priors.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for etype in priors for v in priors[etype].values())
    LOG.info(f"  Saved {total} entity priors → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="CoCoA: Entity Prior Generator")
    parser.add_argument("--cultures", nargs="+", default=ALL_CULTURES)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--lang", default="cu", choices=["cu", "en"])
    parser.add_argument("--data_root", default="./dataset/camellia/raw")
    parser.add_argument("--output_root", default="./dataset/priors")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_name = MODEL_SHORTCUTS.get(args.model, args.model)
    device = torch.device(args.device)

    LOG.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for culture in args.cultures:
        try:
            generate_priors_for_culture(
                model, tokenizer, culture, args.lang, args.data_root,
                args.output_root, args.model, device,
            )
        except Exception as e:
            LOG.error(f"Failed for {culture}: {e}")
            import traceback; traceback.print_exc()

    LOG.info("Done!")


if __name__ == "__main__":
    main()