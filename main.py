"""
CBMCD Main Entry Point (v2)

All hyperparameters are exposed as CLI arguments for automated sweeps.

Usage:
    python -m src.main --model llama3_8b --culture ko --lang cu --seed 42
    
    # With custom loss settings
    python -m src.main \
        --grounded_loss soft_contrastive --contrastive_temperature 2.0 \
        --neutral_loss npo --npo_beta 0.1 \
        --gradient_method goal_aware_pcgrad

See run_sweep.sh for full hyperparameter sweep examples.
"""

import argparse
import sys
from pathlib import Path

import torch

from src.utils import set_seed
from src.model import load_model, ModelConfig, MODEL_SHORTCUTS
from src.data import load_camellia_data, split_data, create_paired_dataloader
from src.trainer import TrainingConfig, train_cbmcd


def parse_args():
    p = argparse.ArgumentParser(description="CBMCD Training (v2)")
    
    # ==========================================================================
    # Data
    # ==========================================================================
    p.add_argument("--data_root", type=str, default="./dataset/camellia/raw",
                   help="Path to Camellia dataset root")
    p.add_argument("--culture", type=str, default="ko",
                choices=["ko", "zh", "ja", "hi", "vi", "ur", "gu", "mr", "ml",
                            "korean", "chinese", "japanese", "hindi", "vietnamese",
                            "urdu", "gujarati", "marathi", "malayalam",
                            "indian_combined", "ar", "arabic"],
                help="Target culture")
    p.add_argument("--lang", type=str, default="cu",
                   choices=["cu", "en"],
                   help="Language: cu=native, en=English")
    
    # ==========================================================================
    # Data Split
    # ==========================================================================
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (controls data split + training)")
    p.add_argument("--train_ratio", type=float, default=0.7,
                   help="Train split ratio")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Validation split ratio (test = 1 - train - val)")
    p.add_argument("--max_pairs", type=int, default=10,
                   help="Max entity pairs per context")
    p.add_argument("--min_entities_per_split", type=int, default=2,
                   help="Minimum entities per split per type")
    
    # ==========================================================================
    # K-Fold CV
    # ==========================================================================
    p.add_argument("--fold", type=int, default=None,
                   help="Fold index (0 to K-1). If set, loads pre-generated fold "
                        "from --folds_root instead of using split_data().")
    p.add_argument("--folds_root", type=str, default="./dataset/folds",
                   help="Root directory for pre-generated fold files")
    
    # ==========================================================================
    # Prior Normalization
    # ==========================================================================
    p.add_argument("--normalize_prior", action="store_true", default=False,
                   help="Subtract entity unconditional prior from log probs "
                        "(requires pre-generated priors from generate_priors.py)")
    p.add_argument("--priors_root", type=str, default="./dataset/priors",
                   help="Root directory for pre-generated entity priors")
    
    # ==========================================================================
    # Model
    # ==========================================================================
    p.add_argument("--model", type=str, default="llama3_8b",
                   help=f"Model name or shortcut: {list(MODEL_SHORTCUTS.keys())}")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Use 4-bit quantization")
    p.add_argument("--load_in_8bit", action="store_true",
                   help="Use 8-bit quantization")
    
    # ==========================================================================
    # LoRA
    # ==========================================================================
    p.add_argument("--lora_r", type=int, default=16,
                   help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout")
    p.add_argument("--target_layer_start", type=int, default=0,
                   help="Start layer for LoRA (inclusive)")
    p.add_argument("--target_layer_end", type=int, default=-1,
                   help="End layer for LoRA (exclusive, -1=auto from model)")
    p.add_argument("--target_modules_type", type=str, default="attention",
                   choices=["attention", "mlp", "both"],
                   help="Which module types to target")
    
    # ==========================================================================
    # Training
    # ==========================================================================
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--pairs_per_batch", type=int, default=8,
                   help="Entity pairs per batch (each pair = 4 forward sequences)")
    p.add_argument("--pairs_per_category", type=int, default=200,
                   help="Max entity pairs per category in dataset")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", action="store_true",
                   help="Disable FP16")
    
    # ==========================================================================
    # Grounded Loss
    # ==========================================================================
    p.add_argument("--grounded_loss", type=str, default="soft_contrastive",
                   choices=["soft_contrastive", "kl_target", "contrastive", "margin"],
                   help="Grounded context loss type")
    p.add_argument("--w_grounded", type=float, default=1.0,
                   help="Weight for grounded loss")
    p.add_argument("--contrastive_temperature", type=float, default=1.0,
                   help="Temperature for soft_contrastive (higher=softer)")
    p.add_argument("--kl_target_asian", type=float, default=0.8,
                   help="Target Asian prob for kl_target loss (0-1)")
    p.add_argument("--margin", type=float, default=1.0,
                   help="Margin for margin loss")
    
    # ==========================================================================
    # Neutral Loss
    # ==========================================================================
    p.add_argument("--neutral_loss", type=str, default="npo",
                   choices=["npo", "mse", "huber", "kl"],
                   help="Neutral context loss type")
    p.add_argument("--w_neutral", type=float, default=1.0,
                   help="Weight for neutral loss")
    p.add_argument("--npo_beta", type=float, default=0.1,
                   help="NPO temperature (lower=more aggressive unlearning)")
    p.add_argument("--npo_min_weight", type=float, default=0.05,
                   help="Min NPO weight threshold near CBS=50")
    p.add_argument("--mse_scale", type=float, default=10.0,
                   help="Scale for MSE/Huber neutral loss")
    p.add_argument("--huber_delta", type=float, default=5.0,
                   help="Delta for Huber neutral loss")
    
    # ==========================================================================
    # Gradient Method
    # ==========================================================================
    p.add_argument("--gradient_method", type=str, default="goal_aware_pcgrad",
                   choices=["goal_aware_pcgrad", "pcgrad", "weighted"],
                   help="Gradient conflict resolution method")
    
    # ==========================================================================
    # CBS Tracking
    # ==========================================================================
    p.add_argument("--cbs_ema_alpha", type=float, default=0.1,
                   help="EMA smoothing for running CBS (0.1=slow, 0.3=fast)")
    
    # ==========================================================================
    # Adaptive Reference (NPO)
    # ==========================================================================
    p.add_argument("--ref_update_steps", type=int, default=100,
                   help="Update NPO ref model every N steps (0=fixed ref, never update)")
    
    # ==========================================================================
    # Logging & Checkpoints
    # ==========================================================================
    p.add_argument("--output_dir", type=str, default="./experiments",
                   help="Output directory")
    p.add_argument("--exp_name", type=str, default=None,
                   help="Experiment name (auto-generated if not set)")
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=500)
    
    return p.parse_args()


def build_exp_name(args) -> str:
    """
    Auto-generate experiment name from key parameters.
    
    Format: {culture}_{lang}_{model}_{nloss}_wg{}_wn{}_tau{}_r{}_seed{seed}
    
    Examples:
        ko_cu_llama3-8b_mse_wg1.0_wn1.0_tau1.0_r16_seed42
        zh_en_qwen3-8b_mse_wg2.0_wn1.0_tau0.5_r32_seed123
        ja_cu_llama3-8b_npo_beta0.1_ref100_wg1.0_wn1.0_tau1.0_r16_seed42
    """
    # Model shortname
    MODEL_SHORT = {
        "llama3_8b": "llama3-8b",
        "qwen3_8b": "qwen3-8b",
        "qwen25_7b": "qwen25-7b",
        "gemma3_12b": "gemma3-12b",
    }
    model_key = args.model.split("/")[-1] if "/" in args.model else args.model
    model_short = MODEL_SHORT.get(model_key, model_key)
    
    # Core identifiers
    parts = [args.culture, args.lang, model_short, args.neutral_loss]
    
    # NPO-specific (only when using NPO)
    if args.neutral_loss == "npo":
        parts.append(f"beta{args.npo_beta}")
        if args.ref_update_steps > 0:
            parts.append(f"ref{args.ref_update_steps}")
    
    # Key HPs (always shown)
    parts.append(f"wg{args.w_grounded}")
    parts.append(f"wn{args.w_neutral}")
    parts.append(f"tau{args.contrastive_temperature}")
    parts.append(f"r{args.lora_r}")
    
    # Non-default HPs
    if args.pairs_per_category != 200:
        parts.append(f"ppc{args.pairs_per_category}")
    
    # Gradient method (only if non-default)
    if args.gradient_method != "goal_aware_pcgrad":
        parts.append(args.gradient_method)
    
    # Prior normalization flag
    if hasattr(args, "normalize_prior") and args.normalize_prior:
        parts.append("pnorm")
    
    # Fold (if using K-Fold CV)
    if hasattr(args, "fold") and args.fold is not None:
        parts.append(f"fold{args.fold}")
    
    # Seed
    parts.append(f"seed{args.seed}")
    
    return "_".join(parts)


def main():
    args = parse_args()
    
    # Seed
    set_seed(args.seed)
    
    # Experiment name
    if args.exp_name is None:
        args.exp_name = build_exp_name(args)
    
    print(f"Experiment: {args.exp_name}")
    print(f"Output: {args.output_dir}/{args.exp_name}")
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n[1/4] Loading data...")
    data = load_camellia_data(args.data_root, culture=args.culture, target_lang=args.lang)
    
    if args.fold is not None:
        # ---- K-Fold CV: load pre-generated split ----
        from src.fold_utils import load_fold, create_examples_from_fold
        print(f"  Loading fold {args.fold} from {args.folds_root}")
        split_info = load_fold(args.folds_root, args.culture, args.lang, args.fold)
        
        # Create example lists for evaluation (legacy compat)
        val_examples = create_examples_from_fold(
            split_info, args.culture, args.lang, "val", args.max_pairs
        )
        test_examples = create_examples_from_fold(
            split_info, args.culture, args.lang, "test", args.max_pairs
        )
        train_examples = []  # Not used in paired training
        
        for sn in ["train", "val", "test"]:
            ng = len(split_info[f"grounded_{sn}"])
            nn = len(split_info[f"neutral_{sn}"])
            ents = split_info[f"{sn}_entities"]
            na = sum(len(v["asian"]) for v in ents.values())
            nw = sum(len(v["western"]) for v in ents.values())
            print(f"  {sn}: {ng}G + {nn}N contexts, {na}A + {nw}W entities")
    else:
        # ---- Legacy: seed-based random split ----
        train_examples, val_examples, test_examples, split_info = split_data(
            data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_pairs_per_context=args.max_pairs,
            min_entities_per_split=args.min_entities_per_split,
        )
    
    # =========================================================================
    # 2. Load Model
    # =========================================================================
    print("\n[2/4] Loading model...")
    full_model_name = args.model if "/" in args.model else MODEL_SHORTCUTS.get(args.model, args.model)
    
    # Auto-detect layer end
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
    
    # =========================================================================
    # 3. Create Paired DataLoader
    # =========================================================================
    print("\n[3/4] Creating paired dataloader...")
    train_dataloader = create_paired_dataloader(
        grounded_df=split_info["grounded_train"],
        neutral_df=split_info["neutral_train"],
        entities=split_info["train_entities"],
        tokenizer=tokenizer,
        pairs_per_batch=args.pairs_per_batch,
        pairs_per_category=args.pairs_per_category,
        seed=args.seed,
    )
    
    # =========================================================================
    # 4. Load Entity Priors (if requested)
    # =========================================================================
    entity_priors = None
    if args.normalize_prior:
        from src.prior_utils import load_entity_priors
        entity_priors = load_entity_priors(
            args.priors_root, args.model, args.culture, args.lang
        )
        print(f"  Loaded {len(entity_priors)} entity priors for prior normalization")
    
    # =========================================================================
    # 5. Train
    # =========================================================================
    print("\n[4/4] Starting training...")
    train_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16 and not args.no_fp16,
        
        grounded_loss_type=args.grounded_loss,
        w_grounded=args.w_grounded,
        contrastive_temperature=args.contrastive_temperature,
        kl_target_asian=args.kl_target_asian,
        margin=args.margin,
        
        neutral_loss_type=args.neutral_loss,
        w_neutral=args.w_neutral,
        npo_beta=args.npo_beta,
        npo_min_weight=args.npo_min_weight,
        mse_scale=args.mse_scale,
        huber_delta=args.huber_delta,
        
        gradient_method=args.gradient_method,
        cbs_ema_alpha=args.cbs_ema_alpha,
        ref_update_steps=args.ref_update_steps,
        
        log_steps=args.log_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
    )
    
    trainer = train_cbmcd(
        model, tokenizer, train_dataloader,
        val_examples, test_examples, train_config,
        camellia_data=data, split_info=split_info,
        entity_priors=entity_priors,
    )
    
    print(f"\nDone! Results saved to: {train_config.output_dir}/{train_config.exp_name}")


if __name__ == "__main__":
    main()