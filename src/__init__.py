"""
CBMCD - Cultural Bias Mitigation with Context-Dependent preference

A framework for mitigating cultural bias in LLMs using:
- Dual-context training (grounded + neutral)
- Goal-Aware PCGrad for gradient conflict resolution
- CBS-guided dynamic unlearning (NPO)
- Context-dependent preference learning
"""

from .data import (
    CamelliaExample,
    CamelliaData,
    load_camellia_data,
    split_data,
    CBMCDDataset,
    create_balanced_dataloader,
)

from .model import (
    ModelConfig,
    load_model,
)

from .loss import (
    # v2 new
    SoftContrastiveLoss,
    KLToTargetLoss,
    CBSGuidedNPOLoss,
    build_grounded_loss,
    build_neutral_loss,
    # v1 legacy (kept for ablation)
    ContrastiveLoss,
    MarginLoss,
    NeutralMSELoss,
    NeutralHuberLoss,
    NeutralKLLoss,
    compute_cbs_from_logprobs,
)

from .trainer import (
    TrainingConfig,
    CBMCDTrainer,
    train_cbmcd,
    goal_aware_pcgrad_backward,
    pcgrad_backward_nway,
    weighted_backward,
)

from .evaluate import (
    compute_entity_log_prob,
    compute_log_probs_for_entities_batched,
    compute_cbs,
    compute_cbs_for_examples,
    compute_cbs_for_context_robust,
    compute_cbs_robust,
    evaluate_robust,
    evaluate_robust_fair,
    evaluate_model,
)

from .utils import (
    set_seed,
    load_config,
    setup_logging,
    setup_distributed,
    generate_exp_name,
)

__version__ = "0.2.0"
__author__ = "CBMCD Team"

__all__ = [
    # Data
    "CamelliaExample",
    "CamelliaData",
    "load_camellia_data",
    "split_data",
    "CBMCDDataset",
    "create_balanced_dataloader",
    # Model
    "ModelConfig",
    "load_model",
    # Loss (v2)
    "SoftContrastiveLoss",
    "KLToTargetLoss",
    "CBSGuidedNPOLoss",
    "build_grounded_loss",
    "build_neutral_loss",
    # Loss (legacy)
    "ContrastiveLoss",
    "MarginLoss",
    "NeutralMSELoss",
    "NeutralHuberLoss",
    "NeutralKLLoss",
    "compute_cbs_from_logprobs",
    # Training
    "TrainingConfig",
    "CBMCDTrainer",
    "train_cbmcd",
    "goal_aware_pcgrad_backward",
    "pcgrad_backward_nway",
    "weighted_backward",
    # Evaluation
    "compute_entity_log_prob",
    "compute_log_probs_for_entities_batched",
    "compute_cbs",
    "compute_cbs_for_examples",
    "compute_cbs_for_context_robust",
    "compute_cbs_robust",
    "evaluate_robust",
    "evaluate_robust_fair",
    "evaluate_model",
    # Utils
    "set_seed",
    "load_config",
    "setup_logging",
    "setup_distributed",
    "generate_exp_name",
]