"""
COCOA Baselines - Unified Entry Point

Results saved to:
    baselines/results/biasedit/{exp_name}/results.json
    baselines/results/biasunlearn/{exp_name}/results.json

Called by shell scripts (scripts/run_biasedit.sh, scripts/run_biasunlearn.sh).
All HPs passed via CLI.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
BASELINES_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.shared import (
    load_and_split, save_results, resolve_model_name,
    build_biasedit_exp_name, build_biasunlearn_exp_name,
)
from src.model import MODEL_SHORTCUTS, MODEL_NUM_LAYERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOG = logging.getLogger(__name__)


# =============================================================================
# BiasEdit
# =============================================================================

def run_biasedit(args, split_info):
    from baselines.biasedit.data_adapter import BiasEditDataset
    from baselines.biasedit.editor import BiasEditEditor, BiasEditConfig
    
    model_name = resolve_model_name(args.model)
    device = torch.device(args.device)
    
    LOG.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Dataset: neutral train only
    train_dataset = BiasEditDataset(
        contexts_df=split_info["neutral_train"],
        entities=split_info["train_entities"],
        tokenizer=tokenizer,
        device=str(device),
        max_length=args.max_length,
        k_entities=args.biasedit_k,
        seed=args.seed,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.biasedit_n_edits,
        shuffle=True, collate_fn=train_dataset.collate_fn,
    )
    
    # Edit target: last layer MLP (auto-detect)
    num_layers = MODEL_NUM_LAYERS.get(model_name, 32)
    edit_module = f"model.layers.{num_layers - 1}.mlp.down_proj"
    LOG.info(f"Edit target: {edit_module}")
    
    config = BiasEditConfig(
        edit_modules=[edit_module],
        rank=args.biasedit_rank,
        n_blocks=args.biasedit_n_blocks,
        lr=args.biasedit_lr,
        meta_lr=args.biasedit_meta_lr,
        max_grad_norm=args.biasedit_max_grad_norm,
        n_epochs=args.biasedit_epochs,
        n_edits=args.biasedit_n_edits,
        cache_batch_size=args.biasedit_cache_batch_size,
        early_stop_patience=args.biasedit_patience,
        eval_every=args.biasedit_eval_every,
        model_device=str(device),
        editor_device=str(device),
    )
    
    editor = BiasEditEditor(model, tokenizer, config)
    results = editor.run(train_loader, split_info)
    return results, config


# =============================================================================
# BiasUnlearn
# =============================================================================

def run_biasunlearn(args, split_info):
    from baselines.biasunlearn.data_adapter import create_unlearn_dataloaders
    from baselines.biasunlearn.trainer import BiasUnlearnTrainer, BiasUnlearnConfig
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
    
    model_name = resolve_model_name(args.model)
    device = torch.device(args.device)
    
    LOG.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA
    if args.unlearn_use_lora:
        LOG.info("Applying LoRA...")
        peft_config = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args.unlearn_lora_r,
            lora_alpha=args.unlearn_lora_alpha,
            lora_dropout=args.unlearn_lora_dropout,
            target_modules=list(args.unlearn_lora_targets),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Gradient checkpointing (use_reentrant=False required for LoRA compatibility)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    LOG.info("Gradient checkpointing enabled")
    model.to(device)
    
    # Reference model (frozen, stays on CPU to save VRAM)
    LOG.info("Loading reference model (CPU, loaded to GPU on-demand)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # Dataloaders: neutral train only
    LOG.info("Creating dataloaders (neutral context only)...")
    ster_loader, anti_loader, unrelated_loader = create_unlearn_dataloaders(
        contexts_df=split_info["neutral_train"],
        entities=split_info["train_entities"],
        tokenizer=tokenizer,
        batch_size=args.unlearn_batch_size,
        max_length=args.max_length,
        seed=args.seed,
        mix_anti=args.unlearn_mix_anti,
        mix_ratio=args.unlearn_mix_ratio,
    )
    
    config = BiasUnlearnConfig(
        max_steps=args.unlearn_max_steps,
        lr=args.unlearn_lr,
        warmup_steps=args.unlearn_warmup_steps,
        max_grad_norm=args.unlearn_max_grad_norm,
        gradient_accumulation_steps=args.unlearn_grad_accum,
        beta=args.unlearn_beta,
        ster_weight=args.unlearn_ster_weight,
        anti_weight=args.unlearn_anti_weight,
        kl_weight=args.unlearn_kl_weight,
        use_lora=args.unlearn_use_lora,
        lora_r=args.unlearn_lora_r,
        lora_alpha=args.unlearn_lora_alpha,
        lora_dropout=args.unlearn_lora_dropout,
        lora_target_modules=tuple(args.unlearn_lora_targets),
        cbs_target=args.unlearn_cbs_target,
        cbs_threshold=args.unlearn_cbs_threshold,
        eval_every=args.unlearn_eval_every,
        log_every=args.unlearn_log_every,
        save_every=args.unlearn_save_every,
        mix_anti=args.unlearn_mix_anti,
        mix_ratio=args.unlearn_mix_ratio,
    )
    
    trainer = BiasUnlearnTrainer(model, ref_model, tokenizer, config, device)
    results = trainer.run(ster_loader, anti_loader, unrelated_loader, split_info)
    
    del ref_model
    torch.cuda.empty_cache()
    return results, config


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="COCOA Baseline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # ---- Common ----
    g = parser.add_argument_group("Common")
    g.add_argument("--method", required=True, choices=["biasedit", "biasunlearn"])
    g.add_argument("--culture", default="ko")
    g.add_argument("--lang", default="cu")
    g.add_argument("--model", default="llama3_8b",
                   help=f"Shortcuts: {list(MODEL_SHORTCUTS.keys())}")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--data_root", default="./dataset/camellia/raw")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--max_length", type=int, default=128)
    g.add_argument("--exp_name", default=None, help="Override auto experiment name")
    
    # ---- BiasEdit ----
    g = parser.add_argument_group("BiasEdit")
    g.add_argument("--biasedit_k", type=int, default=15)
    g.add_argument("--biasedit_n_edits", type=int, default=4)
    g.add_argument("--biasedit_epochs", type=int, default=10)
    g.add_argument("--biasedit_rank", type=int, default=1920)
    g.add_argument("--biasedit_n_blocks", type=int, default=2)
    g.add_argument("--biasedit_lr", type=float, default=1e-6)
    g.add_argument("--biasedit_meta_lr", type=float, default=1e-4)
    g.add_argument("--biasedit_max_grad_norm", type=float, default=1.0)
    g.add_argument("--biasedit_cache_batch_size", type=int, default=128)
    g.add_argument("--biasedit_patience", type=int, default=5)
    g.add_argument("--biasedit_eval_every", type=int, default=2)
    
    # ---- BiasUnlearn ----
    g = parser.add_argument_group("BiasUnlearn")
    g.add_argument("--unlearn_max_steps", type=int, default=500)
    g.add_argument("--unlearn_batch_size", type=int, default=4)
    g.add_argument("--unlearn_lr", type=float, default=5e-5)
    g.add_argument("--unlearn_warmup_steps", type=int, default=10)
    g.add_argument("--unlearn_max_grad_norm", type=float, default=1.0)
    g.add_argument("--unlearn_grad_accum", type=int, default=4)
    g.add_argument("--unlearn_beta", type=float, default=0.1)
    g.add_argument("--unlearn_ster_weight", type=float, default=1.0)
    g.add_argument("--unlearn_anti_weight", type=float, default=1.0)
    g.add_argument("--unlearn_kl_weight", type=float, default=0.2)
    g.add_argument("--unlearn_use_lora", action="store_true", default=True)
    g.add_argument("--unlearn_no_lora", dest="unlearn_use_lora", action="store_false")
    g.add_argument("--unlearn_lora_r", type=int, default=8)
    g.add_argument("--unlearn_lora_alpha", type=int, default=16)
    g.add_argument("--unlearn_lora_dropout", type=float, default=0.1)
    g.add_argument("--unlearn_lora_targets", nargs="+",
                   default=["q_proj", "v_proj", "k_proj", "o_proj"])
    g.add_argument("--unlearn_cbs_target", type=float, default=50.0)
    g.add_argument("--unlearn_cbs_threshold", type=float, default=3.0)
    g.add_argument("--unlearn_eval_every", type=int, default=50)
    g.add_argument("--unlearn_log_every", type=int, default=10)
    g.add_argument("--unlearn_save_every", type=int, default=100)
    g.add_argument("--unlearn_mix_anti", action="store_true", default=False)
    g.add_argument("--unlearn_mix_ratio", type=float, default=0.25)
    
    args = parser.parse_args()
    
    # =========================================================================
    # 1. Data
    # =========================================================================
    LOG.info("=" * 60)
    LOG.info(f"COCOA Baseline: {args.method}")
    LOG.info(f"  culture={args.culture}, lang={args.lang}, "
             f"model={args.model}, seed={args.seed}")
    LOG.info("=" * 60)
    
    data, split_info = load_and_split(
        data_root=args.data_root, culture=args.culture,
        lang=args.lang, seed=args.seed,
    )
    
    # =========================================================================
    # 2. Experiment name
    # =========================================================================
    if args.exp_name is None:
        if args.method == "biasedit":
            exp_name = build_biasedit_exp_name(
                args.culture, args.lang, args.model, args.seed,
                args.biasedit_k, args.biasedit_n_edits,
                args.biasedit_meta_lr, args.biasedit_rank, args.biasedit_epochs,
            )
        else:
            exp_name = build_biasunlearn_exp_name(
                args.culture, args.lang, args.model, args.seed,
                args.unlearn_lr, args.unlearn_beta,
                args.unlearn_ster_weight, args.unlearn_anti_weight,
                args.unlearn_kl_weight, args.unlearn_lora_r, args.unlearn_max_steps,
            )
    else:
        exp_name = args.exp_name
    
    exp_dir = BASELINES_DIR / "results" / args.method / exp_name
    LOG.info(f"Experiment: {exp_name}")
    LOG.info(f"Output: {exp_dir}")
    
    # =========================================================================
    # 3. Setup output directory & logging (before training)
    # =========================================================================
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # File logger → train.log
    fh = logging.FileHandler(exp_dir / "train.log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    LOG.info(f"Logging to {exp_dir / 'train.log'}")
    
    # Save config immediately (reproducibility even if crash)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    # =========================================================================
    # 4. Train
    # =========================================================================
    if args.method == "biasedit":
        results, config = run_biasedit(args, split_info)
    else:
        results, config = run_biasunlearn(args, split_info)
    
    # =========================================================================
    # 5. Save final results
    # =========================================================================
    config_dict = {
        "method": args.method,
        "culture": args.culture,
        "lang": args.lang,
        "model": args.model,
        "model_full": resolve_model_name(args.model),
        "seed": args.seed,
        "max_length": args.max_length,
        **asdict(config),
    }
    
    save_results(
        output_dir=str(exp_dir),
        method=args.method,
        config_dict=config_dict,
        base_result=results["base"],
        trained_result=results["trained"],
        extra={"history": results.get("history", [])},
    )
    
    LOG.info(f"\nDone! → {exp_dir}/results.json")


if __name__ == "__main__":
    main()