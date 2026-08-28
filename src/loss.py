# CoCoA Loss Functions
"""
Two objectives, matching the paper:
- Grounded alignment (L_g): SoftContrastiveLoss  - Eq. 5
      L_g = -log sigmoid( (s(e_l|c_g) - s(e_o|c_g)) / tau )
  implemented as a 2-way (local, other) softmax cross-entropy, which is
  mathematically identical to the sigmoid form above.
- Neutral calibration (L_n): NeutralMSELoss       - Eq. 7
      L_n = ( (s(e_l|c_n) - s(e_o|c_n)) / kappa )^2

The drift regularizer (L_d, Eq. 8) is a gap-based MSE computed inline in the
trainer against the fixed vanilla reference, scaled by lambda; it is not a
separate loss module here.

Scores s(e|c) are the prior-adjusted (PMI) completion log-probs (Eq. 2);
this module operates on whatever scalar scores it is handed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# Grounded alignment loss (L_g, Eq. 5)
class SoftContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        score_local: torch.Tensor,
        score_other: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.stack([score_local, score_other], dim=-1)
        scores = scores / self.temperature
        probs = F.softmax(scores, dim=-1)
        loss = -torch.log(probs[:, 0] + 1e-8)
        return loss.mean()



# Neutral calibration loss (L_n, Eq. 7)
class NeutralMSELoss(nn.Module):
    def __init__(self, scale: float = 10.0):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        score_local: torch.Tensor,
        score_other: torch.Tensor,
    ) -> torch.Tensor:
        diff = (score_local - score_other) / self.scale
        return (diff ** 2).mean()



# Utility
def compute_cbs_from_logprobs(
    score_local: torch.Tensor,
    score_other: torch.Tensor,
) -> float:
    """CBS = percentage of pairs where the other-culture score exceeds the local score (0-100)."""
    other_preferred = (score_other > score_local).float()
    return other_preferred.mean().item() * 100