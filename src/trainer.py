"""
CBMCD Trainer Module (v2)

Changes from v1:
1. Loss: SoftContrastive (grounded) + CBS-Guided NPO (neutral)
2. PCGrad: Goal-Aware variant using real-time CBS distance to goals
3. CBS tracking: EMA from training batches + val evaluation
4. KL regularization: Removed (NPO handles stability via ref model)
5. Split: Uses val for monitoring, test for final eval only
"""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import PreTrainedModel, PreTrainedTokenizer

from accelerate import Accelerator

from .loss import (
    build_grounded_loss, build_neutral_loss,
    CBSGuidedNPOLoss, compute_cbs_from_logprobs,
)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    # --- Optimization ---
    epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    
    # --- Grounded Loss ---
    grounded_loss_type: str = "soft_contrastive"  # soft_contrastive | kl_target | contrastive | margin
    w_grounded: float = 1.0
    contrastive_temperature: float = 1.0    # For soft_contrastive
    kl_target_asian: float = 0.8            # For kl_target
    margin: float = 1.0                     # For margin
    
    # --- Neutral Loss ---
    neutral_loss_type: str = "npo"  # npo | mse | huber | kl
    w_neutral: float = 1.0
    npo_beta: float = 0.1           # NPO temperature (lower = more aggressive)
    npo_min_weight: float = 0.05    # Min weight threshold near CBS=50
    mse_scale: float = 10.0         # For mse/huber
    huber_delta: float = 5.0        # For huber
    w_drift: float = 0.0            # Neutral drift regularization (0 = off)
    
    # --- Gradient Method ---
    gradient_method: str = "goal_aware_pcgrad"  # goal_aware_pcgrad | pcgrad | weighted
    
    # --- CBS Tracking ---
    cbs_ema_alpha: float = 0.1  # EMA smoothing (0.1 = slow, 0.3 = fast)
    
    # --- Adaptive Reference (NPO) ---
    ref_update_steps: int = 100  # Update ref model every N steps (0 = never, fixed ref)
    
    # --- Logging & Checkpoints ---
    log_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    output_dir: str = "./experiments"
    exp_name: str = "cbmcd_v2"


# =============================================================================
# Goal-Aware PCGrad
# =============================================================================

def goal_aware_pcgrad_backward(
    losses: List[torch.Tensor],
    model: nn.Module,
    accelerator,
    cbs_g: float,
    cbs_n: float,
    goal_g: float = 0.0,
    goal_n: float = 50.0,
) -> Dict:
    """
    Goal-Aware PCGrad: Asymmetric projection based on goal distance.
    
    Differences from standard PCGrad:
    - Uses CBS distance to goals for priority weighting
    - The task further from its goal gets its gradient protected more
    
    Args:
        losses: List of [loss_grounded, loss_neutral]
        model: Model
        accelerator: Accelerator instance
        cbs_g: Current CBS for grounded (goal: 0%)
        cbs_n: Current CBS for neutral (goal: 50%)
        goal_g: CBS goal for grounded (default 0)
        goal_n: CBS goal for neutral (default 50)
    
    Returns:
        Dict with conflict_ratio, priority info
    """
    # Filter zero losses
    valid_losses = [(i, l) for i, l in enumerate(losses) if l.requires_grad or l.item() > 0]
    n_valid = len(valid_losses)
    
    if n_valid == 0:
        return {"conflict_ratio": 0.0}
    
    if n_valid == 1:
        idx, loss = valid_losses[0]
        accelerator.backward(loss)
        return {"conflict_ratio": 0.0}
    
    # Collect gradients for each loss
    all_grads = []
    for idx, (i, loss) in enumerate(valid_losses):
        accelerator.backward(loss, retain_graph=(idx < n_valid - 1))
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()
                param.grad = None
        if grads:
            all_grads.append(grads)
    
    if len(all_grads) < 2:
        # Only one gradient collected, just apply it
        if all_grads:
            for name, param in model.named_parameters():
                if name in all_grads[0]:
                    param.grad = all_grads[0][name]
        return {"conflict_ratio": 0.0}
    
    # =========================================================================
    # Goal distance → priority
    # =========================================================================
    dist_g = abs(cbs_g - goal_g)
    dist_n = abs(cbs_n - goal_n)
    total_dist = dist_g + dist_n + 1e-8
    
    priority_g = dist_g / total_dist  # Higher = further from goal = protect more
    priority_n = dist_n / total_dist
    priorities = [priority_g, priority_n][:len(all_grads)]
    
    # =========================================================================
    # Asymmetric projection
    # =========================================================================
    param_names = list(all_grads[0].keys())
    num_conflicts = 0
    total_pairs = 0
    
    projected_grads = []
    for i, grads_i in enumerate(all_grads):
        proj = {name: grads_i[name].clone() for name in param_names if name in grads_i}
        
        for j, grads_j in enumerate(all_grads):
            if i == j:
                continue
            
            for name in proj:
                if name not in grads_j:
                    continue
                
                g_i = proj[name].view(-1)
                g_j = grads_j[name].view(-1)
                
                dot = torch.dot(g_i, g_j)
                total_pairs += 1
                
                if dot < 0:  # Conflict
                    num_conflicts += 1
                    
                    # Asymmetric: higher priority task loses less gradient
                    # projection_strength ∈ [0, 1], higher for lower-priority task
                    my_priority = priorities[i]
                    their_priority = priorities[j]
                    
                    # If I'm low priority and they're high priority,
                    # I should project away more (strength → 1)
                    # If I'm high priority, I keep more of my gradient (strength → 0)
                    if my_priority + their_priority > 0:
                        strength = their_priority / (my_priority + their_priority)
                    else:
                        strength = 0.5  # Equal
                    
                    projection = (dot / (torch.dot(g_j, g_j) + 1e-8)) * g_j
                    proj[name] = (g_i - strength * projection).view(proj[name].shape)
        
        projected_grads.append(proj)
    
    # Sum projected gradients
    for name, param in model.named_parameters():
        if name in param_names:
            param.grad = sum(pg.get(name, 0) for pg in projected_grads)
    
    return {
        "conflict_ratio": num_conflicts / max(total_pairs, 1),
        "priority_g": priorities[0] if len(priorities) > 0 else 0,
        "priority_n": priorities[1] if len(priorities) > 1 else 0,
    }


def pcgrad_backward_nway(losses: List[torch.Tensor], model, accelerator) -> Dict:
    """Standard PCGrad (legacy, for ablation)."""
    valid_losses = [(i, l) for i, l in enumerate(losses) if l.item() > 0]
    n_valid = len(valid_losses)
    
    if n_valid == 0:
        return {"conflict_ratio": 0.0}
    
    all_grads = []
    for idx, (i, loss) in enumerate(valid_losses):
        accelerator.backward(loss, retain_graph=(idx < n_valid - 1))
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()
                param.grad = None
        if grads:
            all_grads.append(grads)
    
    if len(all_grads) <= 1:
        if all_grads:
            for name, param in model.named_parameters():
                if name in all_grads[0]:
                    param.grad = all_grads[0][name]
        return {"conflict_ratio": 0.0}
    
    param_names = list(all_grads[0].keys())
    num_conflicts = 0
    total_pairs = 0
    
    projected_grads = []
    for i, grads_i in enumerate(all_grads):
        proj = {name: grads_i[name].clone() for name in param_names if name in grads_i}
        for j, grads_j in enumerate(all_grads):
            if i == j:
                continue
            for name in proj:
                if name not in grads_j:
                    continue
                g_i = proj[name].view(-1)
                g_j = grads_j[name].view(-1)
                dot = torch.dot(g_i, g_j)
                total_pairs += 1
                if dot < 0:
                    num_conflicts += 1
                    proj[name] = (g_i - (dot / (torch.dot(g_j, g_j) + 1e-8)) * g_j).view(proj[name].shape)
        projected_grads.append(proj)
    
    for name, param in model.named_parameters():
        if name in param_names:
            param.grad = sum(pg.get(name, 0) for pg in projected_grads)
    
    return {"conflict_ratio": num_conflicts / max(total_pairs, 1)}


def weighted_backward(losses: List[torch.Tensor], weights: List[float], accelerator) -> Dict:
    """Simple weighted sum backward."""
    total = sum(w * l for w, l in zip(weights, losses) if l.item() > 0)
    if isinstance(total, torch.Tensor) and total.item() > 0:
        accelerator.backward(total)
    return {}


# =============================================================================
# Trainer
# =============================================================================

class CBMCDTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        val_examples,    # For validation (CBS monitoring)
        test_examples,   # For final evaluation only
        config: TrainingConfig,
        camellia_data=None,
        split_info=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.val_examples = val_examples
        self.test_examples = test_examples
        self.camellia_data = camellia_data
        self.split_info = split_info
        self.entity_priors = None  # Set externally via set_prior_config()
        self.prior_alpha_g = 1.0
        self.prior_alpha_n = 1.0
        
        from datetime import timedelta
        from accelerate import InitProcessGroupKwargs
        
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
        self.accelerator = Accelerator(
            mixed_precision="fp16" if config.fp16 else "no",
            kwargs_handlers=[init_kwargs],
        )
        
        # =============================================================================
        # Reference Model (for NPO or drift regularization)
        # =============================================================================
        self.ref_model = None
        if config.neutral_loss_type == "npo" or config.w_drift > 0:
            reason = "NPO" if config.neutral_loss_type == "npo" else "drift reg"
            self.log_init(f"Creating reference model for {reason}...")
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # =============================================================================
        # Loss Functions (built from config)
        # =============================================================================
        self.grounded_loss_fn = build_grounded_loss(
            config.grounded_loss_type,
            temperature=config.contrastive_temperature,
            target_asian=config.kl_target_asian,
            margin=config.margin,
        )
        self.neutral_loss_fn = build_neutral_loss(
            config.neutral_loss_type,
            npo_beta=config.npo_beta,
            npo_min_weight=config.npo_min_weight,
            mse_scale=config.mse_scale,
            huber_delta=config.huber_delta,
        )
        self.is_npo = isinstance(self.neutral_loss_fn, CBSGuidedNPOLoss)
        
        # =============================================================================
        # Running CBS (EMA)
        # =============================================================================
        self.running_cbs_g = 50.0  # Will be initialized from baseline
        self.running_cbs_n = 50.0
        
        # =============================================================================
        # Optimizer & Scheduler
        # =============================================================================
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        total_steps = len(train_dataloader) * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        scheduler = SequentialLR(
            optimizer,
            [LinearLR(optimizer, 0.1, 1.0, warmup_steps),
             CosineAnnealingLR(optimizer, total_steps - warmup_steps)],
            [warmup_steps]
        )
        
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(model, optimizer, scheduler)
        self.train_dataloader = train_dataloader
        
        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.accelerator.device)
        
        self.output_dir = Path(config.output_dir) / config.exp_name
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Logger
        self.logger = None
        if self.accelerator.is_main_process:
            self.logger = logging.getLogger(f"cbmcd_{config.exp_name}")
            self.logger.setLevel(logging.INFO)
            self.logger.handlers.clear()
            fh = logging.FileHandler(self.output_dir / "train.log")
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
            self.logger.addHandler(fh)
        
        self.results = {
            "config": self._config_to_dict(),
            "baseline": None,
            "baseline_test": None,
            "best": None,
            "final": None,
            "history": [],
        }
        self.global_step = 0
        self.best_score = float("inf")

    def set_prior_config(self, prior_config: Dict):
        """Set entity priors and per-loss alpha scaling."""
        self.entity_priors = prior_config["priors"]
        self.prior_alpha_g = prior_config.get("alpha_g", 1.0)
        self.prior_alpha_n = prior_config.get("alpha_n", 1.0)
        self.log(f"Prior norm: {len(self.entity_priors)} entities, "
                 f"α_g={self.prior_alpha_g}, α_n={self.prior_alpha_n}")
    
    def set_entity_priors(self, priors: Dict[str, float]):
        """Backward compat."""
        self.set_prior_config({"priors": priors, "alpha_g": 1.0, "alpha_n": 1.0})

    def _config_to_dict(self) -> dict:
        return {
            "epochs": self.config.epochs,
            "lr": self.config.learning_rate,
            "gradient_method": self.config.gradient_method,
            "grounded_loss": self.config.grounded_loss_type,
            "neutral_loss": self.config.neutral_loss_type,
            "w_grounded": self.config.w_grounded,
            "w_neutral": self.config.w_neutral,
            "npo_beta": self.config.npo_beta,
            "contrastive_temp": self.config.contrastive_temperature,
            "cbs_ema_alpha": self.config.cbs_ema_alpha,
            "ref_update_steps": self.config.ref_update_steps,
        }
    
    def log_init(self, msg: str):
        print(msg)
    
    def log(self, msg: str):
        if self.accelerator.is_main_process:
            print(msg)
            if self.logger:
                self.logger.info(msg)
    
    @property
    def device(self):
        return self.accelerator.device
    
    # =========================================================================
    # Forward Pass
    # =========================================================================
    
    def _get_log_prob(self, logits, input_ids, context_len):
        """Extract entity-only log probs (after context prefix)."""
        B, S, _ = logits.shape
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        labels = input_ids[:, 1:]
        token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        pos = torch.arange(S-1, device=logits.device).unsqueeze(0).expand(B, -1)
        mask = (pos >= context_len.unsqueeze(1) - 1) & (labels != self.tokenizer.pad_token_id)
        return (token_lp * mask.float()).sum(dim=-1)
    
    def compute_log_probs_paired(
        self,
        asian_ids, asian_mask,
        western_ids, western_mask,
        context_ids,
        use_ref: bool = False,
    ):
        """
        Compute log probs for one context type (grounded or neutral).
        
        Args:
            asian_ids: [B, S] input ids for context + asian entity
            asian_mask: [B, S]
            western_ids: [B, S] input ids for context + western entity
            western_mask: [B, S]
            context_ids: [B, S] context prefix (for computing offset)
            use_ref: Use reference model
        
        Returns:
            (lp_asian, lp_western) each [B]
        """
        context_lens = (context_ids != self.tokenizer.pad_token_id).sum(dim=1)
        
        model = self.ref_model if use_ref else self.model
        ctx = torch.no_grad() if use_ref else torch.enable_grad()
        
        with ctx:
            asian_out = model(input_ids=asian_ids, attention_mask=asian_mask)
            western_out = model(input_ids=western_ids, attention_mask=western_mask)
        
        lp_asian = self._get_log_prob(asian_out.logits, asian_ids, context_lens)
        lp_western = self._get_log_prob(western_out.logits, western_ids, context_lens)
        
        return lp_asian, lp_western
    
    # =========================================================================
    # Training Step (v3: Paired Context)
    # =========================================================================
    
    def train_step(self, batch) -> Dict:
        """
        Paired context training step.
        
        Each batch item contains the SAME entity pair in both grounded and neutral contexts.
        This forces context-aware learning: LoRA can't shortcut via entity-level bias.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        g_a_ids = batch["g_asian_input_ids"].to(self.device)
        g_a_mask = batch["g_asian_attention_mask"].to(self.device)
        g_w_ids = batch["g_western_input_ids"].to(self.device)
        g_w_mask = batch["g_western_attention_mask"].to(self.device)
        g_ctx_ids = batch["g_context_input_ids"].to(self.device)
        
        n_a_ids = batch["n_asian_input_ids"].to(self.device)
        n_a_mask = batch["n_asian_attention_mask"].to(self.device)
        n_w_ids = batch["n_western_input_ids"].to(self.device)
        n_w_mask = batch["n_western_attention_mask"].to(self.device)
        n_ctx_ids = batch["n_context_input_ids"].to(self.device)
        
        # =====================================================================
        # 1. Forward: Grounded (same entity pair)
        # =====================================================================
        lp_g_a, lp_g_w = self.compute_log_probs_paired(
            g_a_ids, g_a_mask, g_w_ids, g_w_mask, g_ctx_ids
        )
        
        # =====================================================================
        # 2. Forward: Neutral (same entity pair, different context)
        # =====================================================================
        lp_n_a, lp_n_w = self.compute_log_probs_paired(
            n_a_ids, n_a_mask, n_w_ids, n_w_mask, n_ctx_ids
        )
        
        if self.entity_priors is not None:
            asian_entities = batch["asian_entity"]   # list of strings
            western_entities = batch["western_entity"]
            
            prior_a = torch.tensor(
                [self.entity_priors.get(e, 0.0) for e in asian_entities],
                device=self.device, dtype=lp_g_a.dtype
            )
            prior_w = torch.tensor(
                [self.entity_priors.get(e, 0.0) for e in western_entities],
                device=self.device, dtype=lp_g_w.dtype
            )
            
            # Grounded: scale by alpha_g
            lp_g_a = lp_g_a - self.prior_alpha_g * prior_a
            lp_g_w = lp_g_w - self.prior_alpha_g * prior_w
            # Neutral: scale by alpha_n (0.0 = no PMI on neutral)
            lp_n_a = lp_n_a - self.prior_alpha_n * prior_a
            lp_n_w = lp_n_w - self.prior_alpha_n * prior_w
        # =====================================================================
        # 3. Update running CBS (EMA)
        # =====================================================================
        alpha = self.config.cbs_ema_alpha
        with torch.no_grad():
            batch_cbs_g = compute_cbs_from_logprobs(lp_g_a, lp_g_w)
            batch_cbs_n = compute_cbs_from_logprobs(lp_n_a, lp_n_w)
            self.running_cbs_g = (1 - alpha) * self.running_cbs_g + alpha * batch_cbs_g
            self.running_cbs_n = (1 - alpha) * self.running_cbs_n + alpha * batch_cbs_n
        
        # =====================================================================
        # 4. Loss G: SoftContrastive on grounded
        # =====================================================================
        loss_g = self.grounded_loss_fn(lp_g_a, lp_g_w)
        
        # =====================================================================
        # 5. Loss N: CBS-Guided NPO on neutral (same entities!)
        # =====================================================================
        npo_info = {}
        if self.is_npo:
            ref_lp_n_a, ref_lp_n_w = self.compute_log_probs_paired(
                n_a_ids, n_a_mask, n_w_ids, n_w_mask, n_ctx_ids,
                use_ref=True,
            )
            loss_n, npo_info = self.neutral_loss_fn(
                lp_n_a, lp_n_w,
                ref_lp_n_a, ref_lp_n_w,
                self.running_cbs_n,
            )
        else:
            loss_n = self.neutral_loss_fn(lp_n_a, lp_n_w)
        
        # =====================================================================
        # 5b. Drift Regularization: penalize neutral gap diverging from ref
        # =====================================================================
        loss_drift = torch.tensor(0.0, device=self.device)
        if self.config.w_drift > 0 and self.ref_model is not None:
            if not self.is_npo:  # NPO already computed ref above
                ref_lp_n_a, ref_lp_n_w = self.compute_log_probs_paired(
                    n_a_ids, n_a_mask, n_w_ids, n_w_mask, n_ctx_ids,
                    use_ref=True,
                )
            # Apply same prior scaling as training
            if self.entity_priors is not None:
                ref_lp_n_a = ref_lp_n_a - self.prior_alpha_n * prior_a
                ref_lp_n_w = ref_lp_n_w - self.prior_alpha_n * prior_w
            
            current_gap = lp_n_a - lp_n_w
            ref_gap = ref_lp_n_a - ref_lp_n_w
            drift = (current_gap - ref_gap) / self.config.mse_scale
            loss_drift = (drift ** 2).mean()
        
        # =====================================================================
        # 6. Scale & Backward (PCGrad)
        # =====================================================================
        scaled_g = self.config.w_grounded * loss_g
        scaled_n = self.config.w_neutral * loss_n + self.config.w_drift * loss_drift
        
        if self.config.gradient_method == "goal_aware_pcgrad":
            stats = goal_aware_pcgrad_backward(
                [scaled_g, scaled_n],
                self.model, self.accelerator,
                cbs_g=self.running_cbs_g,
                cbs_n=self.running_cbs_n,
            )
        elif self.config.gradient_method == "pcgrad":
            stats = pcgrad_backward_nway([scaled_g, scaled_n], self.model, self.accelerator)
        else:
            stats = weighted_backward(
                [loss_g, loss_n],
                [self.config.w_grounded, self.config.w_neutral],
                self.accelerator,
            )
        
        if self.config.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "loss_g": loss_g.item(),
            "loss_n": loss_n.item(),
            "loss_drift": loss_drift.item() if isinstance(loss_drift, torch.Tensor) else 0.0,
            "cbs_g_ema": self.running_cbs_g,
            "cbs_n_ema": self.running_cbs_n,
            "lr": self.scheduler.get_last_lr()[0],
            **stats,
            **{f"npo_{k}": v for k, v in npo_info.items()},
        }
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    def evaluate(self, split: str = "val", show_progress: bool = False) -> Dict:
        """
        Evaluate using N×M robust CBS on the specified split.
        
        Args:
            split: "val" or "test"
            show_progress: Show progress bars
        """
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return {}
        
        self.model.eval()
        
        from .evaluate import evaluate_robust_fair
        
        # Use split-specific entities
        entities = self.split_info.get(f"{split}_entities", self.camellia_data.entities)
        
        results = evaluate_robust_fair(
            self.accelerator.unwrap_model(self.model),
            self.tokenizer,
            self.split_info,
            entities,
            self.device,
            split=split,
            max_contexts=None,
            max_entities=30,
            show_progress=show_progress,
            entity_priors=self.entity_priors,
        )
        
        cbs_g = results["grounded"]["overall"] if results["grounded"]["overall"] is not None else 50.0
        cbs_n = results["neutral"]["overall"] if results["neutral"]["overall"] is not None else 50.0
        
        eval_results = {
            "cbs_grounded": cbs_g,
            "cbs_neutral": cbs_n,
            "combined_score": cbs_g + abs(cbs_n - 50),
            "split": split,
            "by_category": results,
        }
        
        self.accelerator.wait_for_everyone()
        return eval_results
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    def train(self):
        self.log("=" * 60)
        self.log("CBMCD Training (v3: Paired Context)")
        self.log("=" * 60)
        self.log(f"Epochs: {self.config.epochs}")
        self.log(f"Gradient: {self.config.gradient_method}")
        self.log(f"Grounded loss: {self.config.grounded_loss_type} (w={self.config.w_grounded})")
        self.log(f"Neutral loss: {self.config.neutral_loss_type} (w={self.config.w_neutral})")
        if self.is_npo:
            self.log(f"  NPO beta={self.config.npo_beta}, min_weight={self.config.npo_min_weight}")
            if self.config.ref_update_steps > 0:
                self.log(f"  Adaptive ref: update every {self.config.ref_update_steps} steps")
            else:
                self.log(f"  Fixed ref: no updates")
        self.log(f"Training: Paired context (same entity in grounded + neutral)")
        self.log(f"Goals: CBS_g → 0% (Asian preferred), CBS_n → 50% (balanced)")
        self.log("=" * 60)
        
        # =====================================================================
        # Baseline evaluation (val + test)
        # =====================================================================
        self.log("\n[Baseline Evaluation] (val split, N×M robust)")
        baseline = self.evaluate(split="val", show_progress=True)
        if baseline:
            self.results["baseline"] = baseline
            self.log(f"  CBS_g: {baseline['cbs_grounded']:.1f}%")
            self.log(f"  CBS_n: {baseline['cbs_neutral']:.1f}%")
            self.log(f"  Score: {baseline['combined_score']:.1f}")
            
            # Initialize running CBS from baseline
            self.running_cbs_g = baseline["cbs_grounded"]
            self.running_cbs_n = baseline["cbs_neutral"]
            self.log(f"  → Running CBS initialized: g={self.running_cbs_g:.1f}, n={self.running_cbs_n:.1f}")
        
        self.log("\n[Baseline Evaluation] (test split, N×M robust)")
        baseline_test = self.evaluate(split="test", show_progress=True)
        if baseline_test:
            self.results["baseline_test"] = baseline_test
            self.log(f"  CBS_g: {baseline_test['cbs_grounded']:.1f}%")
            self.log(f"  CBS_n: {baseline_test['cbs_neutral']:.1f}%")
            self.log(f"  Score: {baseline_test['combined_score']:.1f}")
        
        self.log("\n" + "-" * 60)
        
        # =====================================================================
        # Training loop
        # =====================================================================
        for epoch in range(self.config.epochs):
            self.log(f"\n[Epoch {epoch+1}/{self.config.epochs}]")
            
            if hasattr(self.train_dataloader, 'batch_sampler') and \
               hasattr(self.train_dataloader.batch_sampler, 'set_epoch'):
                self.train_dataloader.batch_sampler.set_epoch(epoch)
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}") \
                   if self.accelerator.is_main_process else self.train_dataloader
            
            for batch in pbar:
                self.global_step += 1
                m = self.train_step(batch)
                
                if hasattr(pbar, 'set_postfix'):
                    postfix = {
                        "Lg": f"{m['loss_g']:.3f}",
                        "Ln": f"{m['loss_n']:.3f}",
                        "cbs_g": f"{m['cbs_g_ema']:.1f}",
                        "cbs_n": f"{m['cbs_n_ema']:.1f}",
                    }
                    if m.get("loss_drift", 0) > 0:
                        postfix["Ld"] = f"{m['loss_drift']:.3f}"
                    if "conflict_ratio" in m:
                        postfix["conf"] = f"{m['conflict_ratio']:.2f}"
                    pbar.set_postfix(**postfix)
                
                # Logging
                if self.global_step % self.config.log_steps == 0:
                    parts = [
                        f"L_g={m['loss_g']:.4f}",
                        f"L_n={m['loss_n']:.4f}",
                        f"cbs_g={m['cbs_g_ema']:.1f}%",
                        f"cbs_n={m['cbs_n_ema']:.1f}%",
                    ]
                    if m.get("loss_drift", 0) > 0:
                        parts.append(f"L_d={m['loss_drift']:.4f}")
                    if "conflict_ratio" in m:
                        parts.append(f"conf={m['conflict_ratio']:.2f}")
                    if "npo_bias_direction" in m:
                        parts.append(f"bias={m['npo_bias_direction']:.2f}")
                    parts.append(f"lr={m['lr']:.1e}")
                    self.log(f"  step {self.global_step}: {', '.join(parts)}")
                    
                    self.results["history"].append({
                        "step": self.global_step,
                        **{k: v for k, v in m.items() if isinstance(v, (int, float))},
                    })
                
                # Validation
                if self.global_step % self.config.eval_steps == 0:
                    ev = self.evaluate(split="val")
                    if ev:
                        self.log(f"  [Val@{self.global_step}] CBS_g={ev['cbs_grounded']:.1f}%, "
                                 f"CBS_n={ev['cbs_neutral']:.1f}%, score={ev['combined_score']:.1f}")
                        
                        if ev["combined_score"] < self.best_score:
                            self.best_score = ev["combined_score"]
                            self.results["best"] = {"step": self.global_step, **ev}
                            self.save_checkpoint("best")
                            self.log(f"  ★ New best! (score={ev['combined_score']:.1f})")
                
                # Save
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
                
                # Adaptive reference update (NPO)
                if (self.ref_model is not None
                    and self.config.ref_update_steps > 0
                    and self.global_step % self.config.ref_update_steps == 0):
                    self._update_reference_model()
            
            self.accelerator.wait_for_everyone()
        
        # =====================================================================
        # Final evaluation (on TEST split — first time test is used)
        # =====================================================================
        self.log("\n[Final Evaluation] (test split, N×M robust)")
        final = self.evaluate(split="test", show_progress=True)
        if final:
            self.results["final"] = final
            self.log(f"  CBS_g: {final['cbs_grounded']:.1f}%")
            self.log(f"  CBS_n: {final['cbs_neutral']:.1f}%")
            self.log(f"  Score: {final['combined_score']:.1f}")
        
        self.save_checkpoint("final")
        self._save_results()
        
        # Summary
        self.log("\n" + "=" * 60)
        self.log("Training Complete!")
        if self.results.get("baseline_test") and self.results["final"]:
            b, f = self.results["baseline_test"], self.results["final"]
            self.log(f"[Test] CBS_g: {b['cbs_grounded']:.1f}% → {f['cbs_grounded']:.1f}% (goal: ↓0%)")
            self.log(f"[Test] CBS_n: {b['cbs_neutral']:.1f}% → {f['cbs_neutral']:.1f}% (goal: →50%)")
            self.log(f"[Test] Score: {b['combined_score']:.1f} → {f['combined_score']:.1f}")
        elif self.results["baseline"] and self.results["final"]:
            b, f = self.results["baseline"], self.results["final"]
            self.log(f"CBS_g: {b['cbs_grounded']:.1f}% → {f['cbs_grounded']:.1f}% (goal: ↓0%)")
            self.log(f"CBS_n: {b['cbs_neutral']:.1f}% → {f['cbs_neutral']:.1f}% (goal: →50%)")
            self.log(f"Score: {b['combined_score']:.1f} → {f['combined_score']:.1f}")
        self.log("=" * 60)
    
    def _update_reference_model(self):
        """
        Update reference model to current model state (Adaptive Reference).
        
        This prevents NPO saturation: after grounded loss pushes Korean up,
        the new ref captures this state, so NPO can detect & unlearn the
        NEW bias in neutral contexts.
        """
        self.log(f"  🔄 Updating reference model (step {self.global_step})")
        
        # Copy current model weights to ref
        unwrapped = self.accelerator.unwrap_model(self.model)
        ref_state = unwrapped.state_dict()
        self.ref_model.load_state_dict(ref_state)
        self.ref_model.eval()
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def save_checkpoint(self, name: str):
        if not self.accelerator.is_main_process:
            return
        path = self.output_dir / "checkpoints" / name
        path.mkdir(exist_ok=True)
        self.accelerator.unwrap_model(self.model).save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def _save_results(self):
        if self.accelerator.is_main_process:
            with open(self.output_dir / "results.json", "w") as f:
                json.dump(self.results, f, indent=2, default=str)


# =============================================================================
# Entry Point
# =============================================================================

def train_cbmcd(model, tokenizer, train_dataloader, val_examples, test_examples,
                config, camellia_data=None, split_info=None,
                entity_priors=None, prior_config=None):
    trainer = CBMCDTrainer(
        model, tokenizer, train_dataloader,
        val_examples, test_examples, config,
        camellia_data, split_info,
    )
    if prior_config is not None:
        trainer.set_prior_config(prior_config)
    elif entity_priors is not None:
        trainer.set_entity_priors(entity_priors)
    trainer.train()
    return trainer