"""
BiasEdit Editor (MALMEN) - Adapted for COCOA Pipeline

Core MALMEN logic preserved. Changes:
  - Data comes from COCOA's split (via data_adapter)
  - Evaluation via shared.evaluate_baseline()
  - K:K loss only (no StereoSet)
  - Clean interface for baseline comparison
"""

import os
import math
import logging
from typing import Dict, List, Optional
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from .nets import MALMENNet, RunningMeanStd

LOG = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclass
class BiasEditConfig:
    """Configuration for BiasEdit (MALMEN)."""
    # Model
    edit_modules: List[str] = field(default_factory=lambda: ["model.layers.31.mlp.down_proj"])
    
    # MALMEN hyperparameters
    rank: int = 1920
    n_blocks: int = 2
    lr: float = 1e-6
    meta_lr: float = 1e-4
    max_grad_norm: float = 1.0
    
    # Training
    n_epochs: int = 10
    n_edits: int = 4        # contexts per batch
    cache_batch_size: int = 128
    early_stop_patience: int = 5
    
    # Eval during training
    eval_every: int = 2     # evaluate every N epochs
    
    # Device
    model_device: str = "cuda:0"
    editor_device: str = "cuda:0"


# =============================================================================
# Utility functions (from original util.py)
# =============================================================================

def get_module(module: nn.Module, module_name: str) -> nn.Module:
    for name in module_name.split("."):
        module = getattr(module, name)
    return module


def get_shape(module) -> tuple:
    shape = tuple(module.weight.shape)
    return shape[::-1] if isinstance(module, nn.Linear) else shape


class Tracer:
    """Capture forward keys and backward value gradients for a module."""
    
    def __init__(self, module: nn.Module, cache_mask: torch.LongTensor):
        cache_indices = torch.where(cache_mask)
        
        def forward_hook(module, inputs, outputs):
            self.keys = inputs[0][cache_indices].detach()
        
        def backward_hook(module, inputs_grad, outputs_grad):
            self.values_grad = outputs_grad[0][cache_indices].detach()
        
        self.handles = [
            module.register_forward_hook(forward_hook),
            module.register_full_backward_hook(backward_hook),
        ]
    
    def remove(self):
        for h in self.handles:
            h.remove()


# =============================================================================
# K:K Loss Function
# =============================================================================

def compute_kk_loss(logits_asian, labels_asian, logits_western, labels_western):
    """
    K:K loss: equalize average log-prob of Asian vs Western entities.
    
    Loss = (avg_log_prob_asian - avg_log_prob_western)²
    """
    # Shift for next-token prediction
    pred_a = logits_asian[:, :-1]
    targ_a = labels_asian[:, 1:]
    pred_w = logits_western[:, :-1]
    targ_w = labels_western[:, 1:]
    
    def get_score(p, t, log=True):
        mask = (t != -100)
        entity_tokens = t[mask]
        if len(entity_tokens) == 0:
            return torch.tensor(0.0, device=p.device)
        if log:
            probs = p.log_softmax(-1)[mask]
        else:
            probs = p.softmax(-1)[mask]
        scores = probs[torch.arange(len(entity_tokens)), entity_tokens]
        return scores.mean()
    
    # Compute scores
    asian_scores = torch.stack([get_score(p, t, log=False) for p, t in zip(pred_a, targ_a)])
    western_scores = torch.stack([get_score(p, t, log=False) for p, t in zip(pred_w, targ_w)])
    asian_log = torch.stack([get_score(p, t, log=True) for p, t in zip(pred_a, targ_a)])
    western_log = torch.stack([get_score(p, t, log=True) for p, t in zip(pred_w, targ_w)])
    
    # MSE loss on log-prob gap
    log_diff = asian_log.mean() - western_log.mean()
    loss = log_diff ** 2
    
    # CBS metric (Western preferred ratio)
    western_preferred = (western_scores > asian_scores[:len(western_scores)]).float().mean()
    
    return {
        "loss": loss,
        "cbs": western_preferred.item(),
        "asian_log_scores": asian_log,
        "western_log_scores": western_log,
    }


# =============================================================================
# BiasEdit Editor
# =============================================================================

class BiasEditEditor:
    """
    MALMEN-based bias editor adapted for COCOA pipeline.
    
    Training: meta-learn weight edits on neutral contexts
    Evaluation: COCOA's evaluate_robust_fair (CBS_g + CBS_n)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: BiasEditConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Module shape tracking
        shape_counter = Counter()
        self.name2idx = {}
        for module_name in config.edit_modules:
            shape = get_shape(get_module(model, module_name))
            self.name2idx[module_name] = shape_counter[shape]
            shape_counter[shape] += 1
        
        # Build MALMEN hypernet
        self.net = nn.ModuleDict({
            str(k): MALMENNet(
                *k,
                config.rank,
                config.n_blocks,
                v,
                config.lr,
            )
            for k, v in shape_counter.items()
        }).to(config.editor_device)
        
        self.opt = torch.optim.Adam(self.net.parameters(), config.meta_lr)
        
        # Cache directory
        self.cache_dir = os.path.join("cache", "biasedit")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    # =========================================================================
    # Core MALMEN operations
    # =========================================================================
    
    def edit_model(self, param_shifts: Dict[str, torch.Tensor], is_reverse: bool):
        """Apply or reverse weight edits."""
        for module_name, param_shift in param_shifts.items():
            module = get_module(self.model, module_name)
            if isinstance(module, nn.Linear):
                param_shift = param_shift.T
            if is_reverse:
                param_shift = -param_shift
            module.weight.data += param_shift.to(module.weight.data.dtype)
    
    def cache(self, tuples) -> List[Dict]:
        """Forward + backward pass, cache keys and value gradients."""
        os.makedirs(self.cache_dir, exist_ok=True)
        # Clear old cache
        for f in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, f))
        
        edit_dicts = []
        
        for idx in range(len(tuples["edit"])):
            batch = tuples["edit"][idx]
            k = tuples["k_per_sample"][idx]
            
            # Setup tracers
            cache_mask = batch["labels"] != -100
            tracers = {}
            for module_name in self.config.edit_modules:
                module = get_module(self.model, module_name)
                tracers[module_name] = Tracer(module, cache_mask)
            
            # Forward
            logits = self.model(**batch, return_dict=True).logits
            
            # K:K loss
            edit_dict = compute_kk_loss(
                logits[:k], batch["labels"][:k],
                logits[k:], batch["labels"][k:],
            )
            edit_dict["loss"].backward()
            edit_dicts.append(edit_dict)
            
            # Save cached activations
            for module_idx, module_name in enumerate(self.config.edit_modules):
                shape = get_shape(get_module(self.model, module_name))
                keys = tracers[module_name].keys.to(torch.float32).to(self.config.editor_device)
                values_grad = tracers[module_name].values_grad.to(torch.float32).to(self.config.editor_device)
                
                self.net[str(shape)].normalizer.update(torch.cat((keys, values_grad), -1))
                torch.save(keys, os.path.join(self.cache_dir, f"{module_idx}_{idx}_keys.pth"))
                torch.save(values_grad, os.path.join(self.cache_dir, f"{module_idx}_{idx}_values_grad.pth"))
            
            # Remove tracers
            for tr in tracers.values():
                tr.remove()
        
        return edit_dicts
    
    def predict_param_shifts(self) -> Dict[str, torch.Tensor]:
        """Predict weight edits using MALMEN hypernet."""
        n_samples = len([f for f in os.listdir(self.cache_dir) if f.endswith("_keys.pth")]) // len(self.config.edit_modules)
        
        param_shifts = {}
        for module_idx, module_name in enumerate(self.config.edit_modules):
            module = get_module(self.model, module_name)
            shape = get_shape(module)
            
            keys = torch.cat([
                torch.load(os.path.join(self.cache_dir, f"{module_idx}_{idx}_keys.pth"))
                for idx in range(n_samples)
            ], 0)
            values_grad = torch.cat([
                torch.load(os.path.join(self.cache_dir, f"{module_idx}_{idx}_values_grad.pth"))
                for idx in range(n_samples)
            ], 0)
            
            with torch.no_grad():
                net = self.net[str(shape)]
                layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
                
                pesudo_keys, pesudo_values_grad = net(keys, values_grad, layer_idx)
                coeffs = -net.lr(layer_idx) * (keys * pesudo_keys).sum(-1).unsqueeze(-1)
                value_diffs = coeffs * pesudo_values_grad
                
                mat = keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(
                    net.key_size, device=self.config.editor_device
                )
                param_shift = torch.linalg.solve(mat, keys.T @ value_diffs)
            
            param_shifts[module_name] = param_shift.to(self.config.model_device)
        
        return param_shifts
    
    def update_hypernet(self, param_shifts: Dict[str, torch.Tensor]):
        """Update MALMEN hypernet using meta-gradient."""
        self.opt.zero_grad()
        
        n_samples = len([f for f in os.listdir(self.cache_dir) if f.endswith("_keys.pth")]) // len(self.config.edit_modules)
        
        for module_idx, module_name in enumerate(self.config.edit_modules):
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            
            keys = torch.cat([
                torch.load(os.path.join(self.cache_dir, f"{module_idx}_{idx}_keys.pth"))
                for idx in range(n_samples)
            ], 0)
            values_grad = torch.cat([
                torch.load(os.path.join(self.cache_dir, f"{module_idx}_{idx}_values_grad.pth"))
                for idx in range(n_samples)
            ], 0)
            
            module = get_module(self.model, module_name)
            module_grad = module.weight.grad.to(torch.float32).to(self.config.editor_device)
            param_shift = param_shifts[module_name].to(self.config.editor_device)
            
            if isinstance(module, nn.Linear):
                module_grad = module_grad.T
            
            with torch.no_grad():
                mat = torch.linalg.solve(
                    keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(
                        net.key_size, device=self.config.editor_device
                    ),
                    module_grad
                )
                lamda_grad = -net.lamda(layer_idx).exp() * (mat * param_shift).sum()
            
            value_diffs_grad = keys @ mat
            (lamda_grad * net.lamda(layer_idx)).backward()
            
            for start_idx in range(0, keys.shape[0], self.config.cache_batch_size):
                end_idx = start_idx + self.config.cache_batch_size
                pesudo_keys, pesudo_values_grad = net(
                    keys[start_idx:end_idx],
                    values_grad[start_idx:end_idx],
                    layer_idx,
                )
                coeffs = -net.lr(layer_idx) * (keys[start_idx:end_idx] * pesudo_keys).sum(-1).unsqueeze(-1)
                value_diff = coeffs * pesudo_values_grad
                (value_diffs_grad[start_idx:end_idx] * value_diff).sum().backward()
        
        clip_grad_norm_(self.net.parameters(), self.config.max_grad_norm)
        self.opt.step()
    
    # =========================================================================
    # Training loop
    # =========================================================================
    
    def train_epoch(self, loader: DataLoader) -> Dict:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = []
        epoch_cbs = []
        
        for tuples in tqdm(loader, desc="Train", ncols=100):
            # Cache (forward + backward)
            pre_edit_dicts = self.cache(tuples)
            
            # Predict weight edits
            param_shifts = self.predict_param_shifts()
            self.model.zero_grad()
            
            # Apply edits and compute post-edit loss
            self.edit_model(param_shifts, False)
            
            for idx, batch in enumerate(tuples["edit"]):
                k = tuples["k_per_sample"][idx]
                logits = self.model(**batch).logits
                post_dict = compute_kk_loss(
                    logits[:k], batch["labels"][:k],
                    logits[k:], batch["labels"][k:],
                )
                post_dict["loss"].backward()
                epoch_losses.append(post_dict["loss"].item())
                epoch_cbs.append(post_dict["cbs"])
            
            # Reverse edits
            self.edit_model(param_shifts, True)
            
            # Update hypernet
            self.update_hypernet(param_shifts)
        
        return {
            "loss": np.mean(epoch_losses),
            "cbs": np.mean(epoch_cbs),
        }
    
    def run(
        self,
        train_loader: DataLoader,
        split_info: Dict,
        n_epochs: Optional[int] = None,
    ) -> Dict:
        """
        Full training loop with COCOA evaluation.
        
        Returns:
            {"base": base_result, "trained": trained_result, "history": [...]}
        """
        from baselines.shared import evaluate_baseline, print_comparison
        
        n_epochs = n_epochs or self.config.n_epochs
        
        # =====================================================================
        # Baseline evaluation (before training)
        # =====================================================================
        LOG.info("Evaluating baseline (before training)...")
        base_result = evaluate_baseline(
            self.model, self.tokenizer, split_info, split="test"
        )
        LOG.info(f"Baseline — CBS_g: {base_result['cbs_g']:.1f}%, "
                 f"CBS_n: {base_result['cbs_n']:.1f}%, Score: {base_result['score']:.1f}")
        
        # =====================================================================
        # Training
        # =====================================================================
        best_score = float("inf")
        patience_counter = 0
        history = []
        
        for epoch in range(n_epochs):
            LOG.info(f"\nEpoch {epoch + 1}/{n_epochs}")
            
            train_stats = self.train_epoch(train_loader)
            LOG.info(f"  Train loss: {train_stats['loss']:.4f}, CBS: {train_stats['cbs']*100:.1f}%")
            
            # Periodic evaluation
            if (epoch + 1) % self.config.eval_every == 0 or epoch == n_epochs - 1:
                # Apply best edit for evaluation
                for tuples in train_loader:
                    self.cache(tuples)
                    break
                
                param_shifts = self.predict_param_shifts()
                self.edit_model(param_shifts, False)
                
                eval_result = evaluate_baseline(
                    self.model, self.tokenizer, split_info, split="val",
                    show_progress=False,
                )
                
                self.edit_model(param_shifts, True)
                
                LOG.info(f"  Val — CBS_g: {eval_result['cbs_g']:.1f}%, "
                         f"CBS_n: {eval_result['cbs_n']:.1f}%, Score: {eval_result['score']:.1f}")
                
                history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_stats["loss"],
                    "val_cbs_g": eval_result["cbs_g"],
                    "val_cbs_n": eval_result["cbs_n"],
                    "val_score": eval_result["score"],
                })
                
                if eval_result["score"] < best_score:
                    best_score = eval_result["score"]
                    patience_counter = 0
                    # Save best hypernet
                    torch.save(self.net.state_dict(), os.path.join(self.cache_dir, "best_net.pth"))
                    LOG.info(f"  ★ New best! (score={best_score:.1f})")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        LOG.info(f"  Early stopping (patience={self.config.early_stop_patience})")
                        break
        
        # =====================================================================
        # Final evaluation (test split)
        # =====================================================================
        # Load best hypernet
        best_path = os.path.join(self.cache_dir, "best_net.pth")
        if os.path.exists(best_path):
            self.net.load_state_dict(torch.load(best_path))
        
        # Cache and apply edits
        for tuples in train_loader:
            self.cache(tuples)
            break
        
        param_shifts = self.predict_param_shifts()
        self.edit_model(param_shifts, False)
        
        LOG.info("\nFinal evaluation (test split)...")
        trained_result = evaluate_baseline(
            self.model, self.tokenizer, split_info, split="test"
        )
        
        # Reverse for clean state
        self.edit_model(param_shifts, True)
        
        print_comparison(base_result, trained_result, "BiasEdit (MALMEN)")
        
        return {
            "base": base_result,
            "trained": trained_result,
            "history": history,
        }