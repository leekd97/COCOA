"""
CoCoA - Context-Conditional Cultural Alignment

A framework for mitigating entity-centric cultural bias in LLMs via:
- Dual-context training (grounded + neutral) on the same entity pairs
- Grounded alignment (contrastive) + neutral calibration (MSE) objectives
- Drift regularization against a fixed vanilla reference
- Goal-aware PCGrad gradient reconciliation
"""

from .data import (
    CamelliaExample,
    CamelliaData,
    load_camellia_data,
    split_data,
    create_paired_dataloader,
    CoCoADataset,
    create_balanced_dataloader,
)

from .model import (
    ModelConfig,
    MODEL_SHORTCUTS,
    load_model,
)

from .loss import (
    SoftContrastiveLoss,
    NeutralMSELoss,
    compute_cbs_from_logprobs,
)

from .trainer import (
    TrainingConfig,
    CoCoATrainer,
    train_cocoa,
    goal_aware_pcgrad_backward,
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

from .prior_utils import (
    load_entity_priors,
)

from .fold_utils import (
    load_fold,
)

from .utils import (
    set_seed,
    load_config,
    setup_logging,
    setup_distributed,
    generate_exp_name,
)

__version__ = "1.0.0"
__author__ = "CoCoA Team"

__all__ = [
    # Data
    "CamelliaExample",
    "CamelliaData",
    "load_camellia_data",
    "split_data",
    "create_paired_dataloader",
    "CoCoADataset",
    "create_balanced_dataloader",
    # Model
    "ModelConfig",
    "MODEL_SHORTCUTS",
    "load_model",
    # Loss
    "SoftContrastiveLoss",
    "NeutralMSELoss",
    "compute_cbs_from_logprobs",
    # Training
    "TrainingConfig",
    "CoCoATrainer",
    "train_cocoa",
    "goal_aware_pcgrad_backward",
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
    # Priors / folds
    "load_entity_priors",
    "load_fold",
    # Utils
    "set_seed",
    "load_config",
    "setup_logging",
    "setup_distributed",
    "generate_exp_name",
]