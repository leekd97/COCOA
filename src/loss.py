"""
CBMCD Loss Functions (v2)

New losses:
- Grounded: SoftContrastiveLoss (temperature-scaled) or KLToTargetLoss
- Neutral: CBSGuidedNPOLoss (dynamic unlearning based on real-time CBS)

Legacy (kept for backward compat):
- ContrastiveLoss, MarginLoss, NeutralMSELoss, NeutralHuberLoss, NeutralKLLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# =============================================================================
# Grounded Losses (v2)
# =============================================================================

class SoftContrastiveLoss(nn.Module):
    """
    Soft Contrastive loss for grounded contexts (NEW).
    
    L = -log(softmax([lp_K, lp_W] / tau)[0])
    
    Interpretation: "Korean이 더 자연스러운 선택이 되도록"
    temperature가 높을수록 soft (less aggressive), 낮을수록 hard.
    
    Args:
        temperature: Softmax temperature (default 1.0).
                     Higher = softer (gentler push), Lower = harder (stronger push).
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        log_prob_asian: torch.Tensor,
        log_prob_western: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = torch.stack([log_prob_asian, log_prob_western], dim=-1)
        log_probs = log_probs / self.temperature
        probs = F.softmax(log_probs, dim=-1)
        loss = -torch.log(probs[:, 0] + 1e-8)
        return loss.mean()


class KLToTargetLoss(nn.Module):
    """
    KL divergence to a target distribution for grounded contexts (NEW).
    
    L = KL(softmax([lp_K, lp_W]) || target)
    
    target = [target_asian, 1 - target_asian]
    e.g., target_asian=0.8 → "80% Asian / 20% Western" (not total domination)
    
    This is softer than pure contrastive — allows some Western probability.
    
    Args:
        target_asian: Target probability for Asian entity (default 0.8).
    """
    
    def __init__(self, target_asian: float = 0.8):
        super().__init__()
        self.target_asian = target_asian
    
    def forward(
        self,
        log_prob_asian: torch.Tensor,
        log_prob_western: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = torch.stack([log_prob_asian, log_prob_western], dim=-1)
        probs = F.softmax(log_probs, dim=-1)  # [batch, 2]
        
        target = torch.tensor(
            [self.target_asian, 1.0 - self.target_asian],
            device=probs.device, dtype=probs.dtype,
        ).unsqueeze(0).expand_as(probs)
        
        # KL(target || probs) — we want probs to match target
        loss = F.kl_div(probs.log(), target, reduction="batchmean", log_target=False)
        return loss


# =============================================================================
# Neutral Loss (v2): CBS-Guided NPO
# =============================================================================

def _npo_loss(lp: torch.Tensor, lp_ref: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """
    NPO (Negative Preference Optimization) loss for stable unlearning.
    
    L = -(2/beta) * log(sigmoid(-beta * (lp - lp_ref)))
    
    This pushes the current model's probability DOWN relative to reference.
    Stable alternative to gradient ascent for unlearning.
    
    Args:
        lp: Current model log probabilities [batch]
        lp_ref: Reference model log probabilities [batch]
        beta: Temperature (lower = more aggressive unlearning)
    
    Returns:
        Scalar loss
    """
    ratio = lp - lp_ref
    loss = -(2.0 / beta) * torch.log(torch.sigmoid(-beta * ratio) + 1e-8)
    return loss.mean()


class CBSGuidedNPOLoss(nn.Module):
    """
    CBS-Guided Dynamic Unlearning for neutral contexts (NEW).
    
    Inspired by BiasUnlearn's swap mechanism, extended to continuous:
    
    bias_direction = (CBS - 50) / 50    # [-1, 1]
    - If CBS > 50 (Western bias): NPO on Western probs (forget Western)
    - If CBS < 50 (Korean bias): NPO on Korean probs (forget Korean)
    - Strength proportional to |CBS - 50|
    
    When CBS ≈ 50 (balanced), both weights → 0, effectively no loss.
    
    Args:
        beta: NPO temperature (default 0.1, lower = more aggressive)
        min_weight: Minimum weight threshold to avoid noise when CBS ≈ 50
    """
    
    def __init__(self, beta: float = 0.1, min_weight: float = 0.05):
        super().__init__()
        self.beta = beta
        self.min_weight = min_weight
    
    def forward(
        self,
        log_prob_asian: torch.Tensor,
        log_prob_western: torch.Tensor,
        ref_log_prob_asian: torch.Tensor,
        ref_log_prob_western: torch.Tensor,
        running_cbs: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            log_prob_asian: Current model lp for Asian entities [batch]
            log_prob_western: Current model lp for Western entities [batch]
            ref_log_prob_asian: Reference model lp for Asian entities [batch]
            ref_log_prob_western: Reference model lp for Western entities [batch]
            running_cbs: Current CBS estimate (0-100, from EMA)
        
        Returns:
            (loss, info_dict)
        """
        bias_direction = (running_cbs - 50.0) / 50.0  # [-1, 1]
        
        # Smooth continuous weights
        w_forget_western = max(0.0, bias_direction)   # > 0 when Western biased
        w_forget_korean = max(0.0, -bias_direction)    # > 0 when Korean biased
        
        info = {
            "bias_direction": bias_direction,
            "w_forget_western": w_forget_western,
            "w_forget_korean": w_forget_korean,
        }
        
        device = log_prob_asian.device
        loss = torch.tensor(0.0, device=device)
        
        if w_forget_western > self.min_weight:
            npo_w = _npo_loss(log_prob_western, ref_log_prob_western, self.beta)
            loss = loss + w_forget_western * npo_w
            info["npo_western"] = npo_w.item()
        
        if w_forget_korean > self.min_weight:
            npo_a = _npo_loss(log_prob_asian, ref_log_prob_asian, self.beta)
            loss = loss + w_forget_korean * npo_a
            info["npo_korean"] = npo_a.item()
        
        return loss, info


# =============================================================================
# Legacy Losses (v1, kept for backward compat / ablation)
# =============================================================================

class ContrastiveLoss(nn.Module):
    """Legacy contrastive loss (same as SoftContrastiveLoss with temp=1)."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, log_prob_asian, log_prob_western):
        log_probs = torch.stack([log_prob_asian, log_prob_western], dim=-1)
        log_probs = log_probs / self.temperature
        probs = F.softmax(log_probs, dim=-1)
        loss = -torch.log(probs[:, 0] + 1e-8)
        return loss.mean()


class MarginLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, log_prob_asian, log_prob_western):
        diff = log_prob_asian - log_prob_western
        loss = F.relu(self.margin - diff)
        return loss.mean()


class NeutralMSELoss(nn.Module):
    def __init__(self, scale: float = 10.0):
        super().__init__()
        self.scale = scale

    def forward(self, log_prob_asian, log_prob_western):
        diff = log_prob_asian - log_prob_western
        normalized_diff = diff / self.scale
        loss = normalized_diff ** 2
        return loss.mean()


class NeutralHuberLoss(nn.Module):
    def __init__(self, delta: float = 5.0, scale: float = 10.0):
        super().__init__()
        self.delta = delta
        self.scale = scale

    def forward(self, log_prob_asian, log_prob_western):
        diff = log_prob_asian - log_prob_western
        normalized_diff = diff / self.scale
        abs_diff = torch.abs(normalized_diff)
        quadratic = 0.5 * normalized_diff ** 2
        linear = self.delta * (abs_diff - 0.5 * self.delta)
        loss = torch.where(abs_diff <= self.delta, quadratic, linear)
        return loss.mean()


class NeutralKLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_prob_asian, log_prob_western):
        log_probs = torch.stack([log_prob_asian, log_prob_western], dim=-1)
        probs = F.softmax(log_probs, dim=-1)
        uniform = torch.ones_like(probs) * 0.5
        loss = F.kl_div(probs.log(), uniform, reduction="batchmean")
        return loss


# =============================================================================
# Factory: Build loss from config string
# =============================================================================

def build_grounded_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Build grounded loss from config.
    
    Args:
        loss_type: "soft_contrastive", "kl_target", "contrastive", "margin"
        **kwargs: Loss-specific arguments
    """
    if loss_type == "soft_contrastive":
        return SoftContrastiveLoss(temperature=kwargs.get("temperature", 1.0))
    elif loss_type == "kl_target":
        return KLToTargetLoss(target_asian=kwargs.get("target_asian", 0.8))
    elif loss_type == "contrastive":
        return ContrastiveLoss(temperature=kwargs.get("temperature", 1.0))
    elif loss_type == "margin":
        return MarginLoss(margin=kwargs.get("margin", 1.0))
    else:
        raise ValueError(f"Unknown grounded loss: {loss_type}")


def build_neutral_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Build neutral loss from config.
    
    Args:
        loss_type: "npo", "mse", "huber", "kl"
        **kwargs: Loss-specific arguments
    """
    if loss_type == "npo":
        return CBSGuidedNPOLoss(
            beta=kwargs.get("npo_beta", 0.1),
            min_weight=kwargs.get("npo_min_weight", 0.05),
        )
    elif loss_type == "mse":
        return NeutralMSELoss(scale=kwargs.get("mse_scale", 10.0))
    elif loss_type == "huber":
        return NeutralHuberLoss(
            delta=kwargs.get("huber_delta", 5.0),
            scale=kwargs.get("mse_scale", 10.0),
        )
    elif loss_type == "kl":
        return NeutralKLLoss()
    else:
        raise ValueError(f"Unknown neutral loss: {loss_type}")


# =============================================================================
# Utility
# =============================================================================

def compute_cbs_from_logprobs(
    log_prob_asian: torch.Tensor,
    log_prob_western: torch.Tensor,
) -> float:
    """CBS = percentage where Western > Asian (0-100)."""
    western_preferred = (log_prob_western > log_prob_asian).float()
    return western_preferred.mean().item() * 100