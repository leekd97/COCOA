"""
BiasUnlearn Trainer - Adapted for COCOA Pipeline

Core logic preserved from original BiasUnlearn:
  - NPO loss on stereotype (Western) → forget bias
  - GD loss on anti-stereotype (Asian) → learn correct
  - KL loss on context → preserve language ability
  - Dynamic direction reversal based on CBS

Changes:
  - Data from COCOA's split (via data_adapter)
  - Evaluation via shared.evaluate_baseline() (CBS_g + CBS_n)
"""

import logging
from dataclasses import dataclass, field
from itertools import cycle
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)
from peft import LoraConfig, get_peft_model, TaskType

LOG = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclass
class BiasUnlearnConfig:
    """Configuration for BiasUnlearn."""
    # Training
    max_steps: int = 500
    lr: float = 5e-5
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    
    # Loss weights
    beta: float = 0.1       # NPO temperature
    ster_weight: float = 1.0
    anti_weight: float = 1.0
    kl_weight: float = 0.2
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj")
    
    # CBS convergence
    cbs_target: float = 50.0
    cbs_threshold: float = 3.0
    eval_every: int = 50     # evaluate CBS every N steps
    
    # Logging & saving
    log_every: int = 10
    save_every: int = 100
    
    # Adversarial Forget Set
    mix_anti: bool = False
    mix_ratio: float = 0.25
    
    # Misc
    log_every: int = 10
    save_every: int = 100
    mix_anti: bool = False
    mix_ratio: float = 0.25


# =============================================================================
# Loss Functions
# =============================================================================

def lm_loss(
    operation: str,
    batch: Dict,
    model: PreTrainedModel,
    device: torch.device,
) -> torch.Tensor:
    """
    Language modeling loss on entity part only.
    
    Args:
        operation: "ga" (gradient ascent, negate) or "gd" (gradient descent)
        batch: {input_ids, attention_mask, labels, start_locs}
        model: The model
        device: Device
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    start_locs = batch["start_locs"].to(device)
    
    # Mask out context part (only compute loss on entity tokens)
    seq_len = labels.size(1)
    col_indices = torch.arange(seq_len, device=device).expand_as(labels)
    mask = col_indices < start_locs.unsqueeze(1)
    modified_labels = labels.clone()
    modified_labels[mask] = -100
    
    outputs = model(input_ids, attention_mask=attention_mask)
    
    # Shift for next-token prediction
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = modified_labels[:, 1:]
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
    losses = []
    for bid in range(input_ids.shape[0]):
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":
            position_loss = -position_loss
        losses.append(position_loss)
    
    return torch.stack(losses).mean()


def npo_loss(
    batch: Dict,
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    device: torch.device,
    beta: float = 0.1,
) -> torch.Tensor:
    """NPO: encourage lower probability on biased outputs vs reference."""
    ster_loss = lm_loss("gd", batch, model, device)
    with torch.no_grad():
        ster_loss_ref = lm_loss("gd", batch, ref_model, device)
    
    neg_log_ratio = ster_loss - ster_loss_ref
    loss = -F.logsigmoid(beta * neg_log_ratio).mean() * 2 / beta
    return loss


def kl_loss(
    batch: Dict,
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    device: torch.device,
) -> torch.Tensor:
    """Forward KL divergence for utility preservation."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    with torch.no_grad():
        ref_outputs = ref_model(input_ids, attention_mask=attention_mask, labels=labels)
    
    prob_p = torch.softmax(ref_outputs.logits, dim=-1)
    prob_q = torch.softmax(outputs.logits, dim=-1)
    
    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()
    return loss


def compute_total_loss(
    ster_batch: Dict,
    anti_batch: Dict,
    unrelated_batch: Dict,
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    device: torch.device,
    config: BiasUnlearnConfig,
) -> Dict:
    """
    Total BiasUnlearn loss.
    
    L = ster_weight * L_npo + anti_weight * L_gd + kl_weight * L_kl
    
    ref_model is moved to GPU only during NPO/KL computation to save VRAM.
    """
    # Move ref to GPU for NPO + KL
    torch.cuda.empty_cache()
    ref_model.to(device)
    
    l_npo = npo_loss(ster_batch, model, ref_model, device, config.beta)
    l_anti = lm_loss("gd", anti_batch, model, device)
    l_kl = kl_loss(unrelated_batch, model, ref_model, device)
    
    # Move ref back to CPU to free VRAM for backward pass
    ref_model.to("cpu")
    torch.cuda.empty_cache()
    
    total = config.ster_weight * l_npo + config.anti_weight * l_anti + config.kl_weight * l_kl
    
    return {
        "total": total,
        "npo": l_npo.item(),
        "anti": l_anti.item(),
        "kl": l_kl.item(),
    }


# =============================================================================
# Trainer
# =============================================================================

class BiasUnlearnTrainer:
    """
    BiasUnlearn trainer adapted for COCOA pipeline.
    
    Training: NPO+GD+KL on neutral contexts only
    Evaluation: COCOA's evaluate_robust_fair (CBS_g + CBS_n)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: BiasUnlearnConfig,
        device: torch.device,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
    
    def run(
        self,
        ster_loader: DataLoader,
        anti_loader: DataLoader,
        unrelated_loader: DataLoader,
        split_info: Dict,
    ) -> Dict:
        """
        Full training loop with CBS monitoring.
        
        Returns:
            {"base": base_result, "trained": trained_result, "history": [...]}
        """
        from baselines.shared import evaluate_baseline, print_comparison
        
        config = self.config
        model = self.model
        ref_model = self.ref_model
        device = self.device
        
        # =================================================================
        # Baseline evaluation
        # =================================================================
        LOG.info("Evaluating baseline (before training)...")
        base_result = evaluate_baseline(model, self.tokenizer, split_info, split="test")
        LOG.info(f"Baseline — CBS_g: {base_result['cbs_g']:.1f}%, "
                 f"CBS_n: {base_result['cbs_n']:.1f}%, Score: {base_result['score']:.1f}")
        
        # =================================================================
        # Setup optimizer
        # =================================================================
        optimizer = AdamW(model.parameters(), lr=config.lr)
        num_steps = config.max_steps
        lr_scheduler = get_scheduler(
            "linear", optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_steps,
        )
        
        # =================================================================
        # Training loop
        # =================================================================
        global_step = 0
        reverse_mode = False
        history = []
        best_score = float("inf")
        best_state = None
        
        model.train()
        
        ster_iter = iter(ster_loader)
        anti_iter = cycle(anti_loader)
        unrelated_iter = cycle(unrelated_loader)
        
        LOG.info(f"Starting BiasUnlearn training (max {num_steps} steps)")
        
        while global_step < num_steps:
            try:
                ster_batch = next(ster_iter)
            except StopIteration:
                ster_iter = iter(ster_loader)
                ster_batch = next(ster_iter)
            
            anti_batch = next(anti_iter)
            unrelated_batch = next(unrelated_iter)
            
            # Swap ster/anti based on reverse_mode
            if reverse_mode:
                actual_ster, actual_anti = anti_batch, ster_batch
            else:
                actual_ster, actual_anti = ster_batch, anti_batch
            
            losses = compute_total_loss(
                actual_ster, actual_anti, unrelated_batch,
                model, ref_model, device, config,
            )
            
            loss = losses["total"] / config.gradient_accumulation_steps
            loss.backward()
            
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            # Logging
            if global_step % config.log_every == 0:
                direction = "REVERSE" if reverse_mode else "NORMAL"
                LOG.info(
                    f"[Step {global_step}] {direction} | "
                    f"NPO: {losses['npo']:.4f}, Anti: {losses['anti']:.4f}, "
                    f"KL: {losses['kl']:.4f}"
                )
            
            # Periodic evaluation on val split
            if config.eval_every > 0 and global_step % config.eval_every == 0:
                eval_result = evaluate_baseline(
                    model, self.tokenizer, split_info, split="val",
                    show_progress=False,
                )
                
                cbs_n = eval_result["cbs_n"]
                gap = abs(cbs_n - config.cbs_target)
                
                LOG.info(
                    f"[Step {global_step}] Val — CBS_g: {eval_result['cbs_g']:.1f}%, "
                    f"CBS_n: {cbs_n:.1f}%, Score: {eval_result['score']:.1f} "
                    f"(gap from 50%: {gap:.1f}%)"
                )
                
                history.append({
                    "step": global_step,
                    "val_cbs_g": eval_result["cbs_g"],
                    "val_cbs_n": eval_result["cbs_n"],
                    "val_score": eval_result["score"],
                    "reverse": reverse_mode,
                })
                
                # Track best
                if eval_result["score"] < best_score:
                    best_score = eval_result["score"]
                    # Save LoRA state
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    LOG.info(f"  ★ New best! (score={best_score:.1f})")
                
                # Early stopping
                if gap <= config.cbs_threshold:
                    LOG.info(f"  ✓ CBS_n converged to {cbs_n:.1f}% (within ±{config.cbs_threshold}%)")
                    break
                
                # Adjust direction
                old_reverse = reverse_mode
                if cbs_n < config.cbs_target - config.cbs_threshold:
                    reverse_mode = True   # Too Korean-biased → boost Western
                elif cbs_n > config.cbs_target + config.cbs_threshold:
                    reverse_mode = False  # Too Western-biased → boost Korean
                
                if old_reverse != reverse_mode:
                    direction = "REVERSE (boost Western)" if reverse_mode else "NORMAL (boost Korean)"
                    LOG.info(f"  → Switching to {direction}")
                
                model.train()
        
        # =================================================================
        # Load best state, final evaluation on test
        # =================================================================
        if best_state is not None:
            model.load_state_dict(best_state)
            LOG.info(f"Loaded best checkpoint (score={best_score:.1f})")
        
        LOG.info("\nFinal evaluation (test split)...")
        trained_result = evaluate_baseline(model, self.tokenizer, split_info, split="test")
        
        print_comparison(base_result, trained_result, "BiasUnlearn (NPO+GD+KL)")
        
        return {
            "base": base_result,
            "trained": trained_result,
            "history": history,
        }