"""
CoCoA Main

Trains one model on one culture with the published CoCoA method:
paired-context training (same entity pair under a grounded and a neutral
context), with grounded alignment (soft contrastive) + neutral calibration
(MSE) + drift regularization, reconciled by goal-aware PCGrad. PMI prior
normalization is applied to the scores.

All hyperparameters are exposed as CLI arguments. Defaults match the paper.

Usage:
    # K-Fold run (recommended), priors must be pre-generated
    python main.py --model llama3_8b --culture ko --lang cu --fold 0 --seed 45

    # Per-culture override (see paper Table 4), e.g. Llama / KO uses w_drift=20
    python main.py --model llama3_8b --culture ko --fold 0 --w_drift 20
"""

import argparse
import sys
from pathlib import Path

import torch

from src.utils import set_seed
from src.model import load_model, ModelConfig, MODEL_SHORTCUTS
from src.data import load_camellia_data, split_data, create_paired_dataloader
from src.trainer import TrainingConfig, train_cocoa


def parse_args():
    p = argparse.ArgumentParser(description="CoCoA Training")

    
    # Data
    p.add_argument("--data_root", type=str, default="./dataset/camellia/raw",
                   help="Path to Camellia dataset root")
    p.add_argument("--culture", type=str, default="ko",
                   choices=["ko", "zh", "ja", "hi", "vi", "ur", "gu", "mr", "ml", "ar",
                            "korean", "chinese", "japanese", "hindi", "vietnamese",
                            "urdu", "gujarati", "marathi", "malayalam", "arabic"],
                   help="Target culture")
    p.add_argument("--lang", type=str, default="cu", choices=["cu", "en"],
                   help="Language: cu=native, en=English")

    
    # Data Split (legacy seed split; used only when --fold is not set)
    p.add_argument("--seed", type=int, default=45,
                   help="Random seed (controls data split + training)")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_pairs", type=int, default=10,
                   help="Max entity pairs per context (legacy split only)")
    p.add_argument("--min_entities_per_split", type=int, default=2)

    
    # K-Fold CV
    p.add_argument("--fold", type=int, default=None,
                   help="Fold index (0 to K-1). If set, loads a pre-generated fold "
                        "from --folds_root instead of using split_data().")
    p.add_argument("--folds_root", type=str, default="./dataset/folds",
                   help="Root directory for pre-generated fold files")

    
    # Prior Normalization (PMI) — on by default
    p.add_argument("--no_prior_norm", action="store_true", default=False,
                   help="Disable PMI prior normalization. On by default; "
                        "requires pre-generated priors from generate_priors.py")
    p.add_argument("--priors_root", type=str, default="./dataset/priors",
                   help="Root directory for pre-generated entity priors")
    p.add_argument("--prior_alpha_g", type=float, default=1.0,
                   help="PMI scaling s for grounded scores (1.0=full, 0.0=none)")
    p.add_argument("--prior_alpha_n", type=float, default=0.3,
                   help="PMI scaling s for neutral scores (1.0=full, 0.0=none)")

    
    # Entity Pairing
    p.add_argument("--pairing", type=str, default="nxn",
                   choices=["1to1", "nxn", "nxm"],
                   help="Entity pairing: nxn (equal-sided all combos, default), "
                        "1to1, or nxm (all combos)")

    
    # Experiment Organization
    p.add_argument("--exp_subdir", type=str, default=None,
                   help="Subdirectory under output_dir")

    
    # Model
    p.add_argument("--model", type=str, default="llama3_8b",
                   help=f"Model name or shortcut: {list(MODEL_SHORTCUTS.keys())}")
    p.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    p.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")

    
    # LoRA
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--target_layer_start", type=int, default=0,
                   help="Start layer for LoRA (inclusive)")
    p.add_argument("--target_layer_end", type=int, default=-1,
                   help="End layer for LoRA (exclusive, -1=auto from model)")
    p.add_argument("--target_modules_type", type=str, default="attention",
                   choices=["attention", "mlp", "both"],
                   help="Which module types to target")

    
    # Training
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--pairs_per_batch", type=int, default=16,
                   help="Entity pairs per batch (each pair = 4 forward sequences)")
    p.add_argument("--pairs_per_category", type=int, default=200,
                   help="Max entity pairs per category in dataset")
    p.add_argument("--no_fp16", action="store_true", help="Disable FP16 (FP16 on by default)")

    
    # Objectives - grounded alignment (L_g)
    p.add_argument("--w_grounded", type=float, default=1.0,
                   help="Weight for grounded alignment loss L_g")
    p.add_argument("--contrastive_temperature", type=float, default=1.0,
                   help="Temperature tau for the soft contrastive loss")

    
    # Objectives - neutral calibration (L_n) + drift (L_d)
    p.add_argument("--w_neutral", type=float, default=2.0,
                   help="Weight for neutral calibration loss L_n")
    p.add_argument("--w_drift", type=float, default=1.0,
                   help="Weight for drift regularization L_d (0=off). "
                        "Penalizes the neutral gap diverging from the base model.")
    p.add_argument("--k", type=float, default=10.0,
                   help="kappa: gap scale for neutral calibration L_n")
    p.add_argument("--lam", type=float, default=10.0,
                   help="lambda: gap scale for drift regularization L_d")

    
    # CBS Tracking (goal-aware PCGrad)
    p.add_argument("--cbs_ema_alpha", type=float, default=0.1,
                   help="EMA smoothing for running CBS distance (0.1=slow, 0.3=fast)")

    
    # Logging & Checkpoints
    p.add_argument("--output_dir", type=str, default="./experiments", help="Output directory")
    p.add_argument("--exp_name", type=str, default=None,
                   help="Experiment name (auto-generated if not set)")
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=500)

    return p.parse_args()


def build_exp_name(args) -> str:

    # Auto-generate an experiment name from the key parameters.
    MODEL_SHORT = {
        "llama3_8b": "llama3-8b",
        "qwen3_8b": "qwen3-8b",
        "gemma3_12b": "gemma3-12b",
        "mistral_7b": "mistral-7b",
    }
    model_key = args.model.split("/")[-1] if "/" in args.model else args.model
    model_short = MODEL_SHORT.get(model_key, model_key)

    parts = [args.culture, args.lang, model_short, "mse"]

    parts.append(f"wg{args.w_grounded}")
    parts.append(f"wn{args.w_neutral}")
    parts.append(f"tau{args.contrastive_temperature}")
    parts.append(f"r{args.lora_r}")

    if args.pairs_per_category != 200:
        parts.append(f"ppc{args.pairs_per_category}")

    # Prior normalization
    if args.normalize_prior:
        ag, an = args.prior_alpha_g, args.prior_alpha_n
        if ag == 1.0 and an == 1.0:
            parts.append("pnorm")
        elif an == 0.0:
            parts.append("pnorm-g")
        else:
            parts.append(f"pnorm-g{ag}-n{an}")

    # Entity pairing (only if non-default)
    if args.pairing != "1to1":
        parts.append(args.pairing)

    if args.fold is not None:
        parts.append(f"fold{args.fold}")

    parts.append(f"seed{args.seed}")
    return "_".join(parts)


def main():
    args = parse_args()

    # Derived flags
    args.normalize_prior = not args.no_prior_norm
    fp16 = not args.no_fp16

    set_seed(args.seed)

    if args.exp_name is None:
        args.exp_name = build_exp_name(args)

    print(f"Experiment: {args.exp_name}")
    print(f"Output: {args.output_dir}/{args.exp_name}")

    
    # 1. Load Data
    print("\n[1/4] Loading data...")
    data = load_camellia_data(args.data_root, culture=args.culture, target_lang=args.lang)

    if args.fold is not None:
        # K-Fold CV: load pre-generated split
        from src.fold_utils import load_fold
        print(f"  Loading fold {args.fold} from {args.folds_root}/seed{args.seed}")
        split_info = load_fold(args.folds_root, args.culture, args.lang, args.fold, args.seed)

        for sn in ["train", "val", "test"]:
            ng = len(split_info[f"grounded_{sn}"])
            nn = len(split_info[f"neutral_{sn}"])
            ents = split_info[f"{sn}_entities"]
            na = sum(len(v["asian"]) for v in ents.values())
            nw = sum(len(v["western"]) for v in ents.values())
            print(f"  {sn} (cu): {ng}G + {nn}N contexts, {na}A + {nw}W entities")
    else:
        # seed-based random split
        _, _, _, split_info = split_data(
            data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_pairs_per_context=args.max_pairs,
            min_entities_per_split=args.min_entities_per_split,
        )

    
    # 2. Load Model
    print("\n[2/4] Loading model...")
    full_model_name = args.model if "/" in args.model else MODEL_SHORTCUTS.get(args.model, args.model)

    layer_end = args.target_layer_end
    if layer_end < 0:
        from src.model import MODEL_NUM_LAYERS
        layer_end = MODEL_NUM_LAYERS.get(full_model_name, 32)
        print(f"  Auto-detected {layer_end} layers for {full_model_name}")

    model_config = ModelConfig(
        name=full_model_name,
        use_lora=True,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_layer_start=args.target_layer_start,
        target_layer_end=layer_end,
        target_modules_type=args.target_modules_type,
    )
    model, tokenizer = load_model(model_config.name, model_config, for_distributed=True)

    
    # 3. Create Paired DataLoader
    print("\n[3/4] Creating paired dataloader...")
    train_dataloader = create_paired_dataloader(
        grounded_df=split_info["grounded_train"],
        neutral_df=split_info["neutral_train"],
        entities=split_info["train_entities"],
        tokenizer=tokenizer,
        pairs_per_batch=args.pairs_per_batch,
        pairs_per_category=args.pairs_per_category,
        seed=args.seed,
        pairing=args.pairing,
        paired=True,
    )

    
    # 4. Load Entity Priors (PMI) - on by default
    prior_config = None
    if args.normalize_prior:
        from src.prior_utils import load_entity_priors
        entity_priors = load_entity_priors(
            args.priors_root, args.model, args.culture, args.lang
        )
        print(f"  Prior norm: {len(entity_priors)} entities, "
              f"alpha_g={args.prior_alpha_g}, alpha_n={args.prior_alpha_n}")
        prior_config = {
            "priors": entity_priors,
            "alpha_g": args.prior_alpha_g,
            "alpha_n": args.prior_alpha_n,
        }

    
    # 5. Train
    print("\n[4/4] Starting training...")
    output_dir = args.output_dir
    if args.exp_subdir:
        output_dir = f"{args.output_dir}/{args.exp_subdir}"

    train_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        fp16=fp16,

        w_grounded=args.w_grounded,
        contrastive_temperature=args.contrastive_temperature,

        w_neutral=args.w_neutral,
        w_drift=args.w_drift,
        k=args.k,
        lam=args.lam,

        cbs_ema_alpha=args.cbs_ema_alpha,

        log_steps=args.log_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=output_dir,
        exp_name=args.exp_name,
    )

    trainer = train_cocoa(
        model, tokenizer, train_dataloader, train_config,
        camellia_data=data, split_info=split_info,
        prior_config=prior_config,
    )

    print(f"\nDone! Results saved to: {train_config.output_dir}/{train_config.exp_name}")


if __name__ == "__main__":
    main()