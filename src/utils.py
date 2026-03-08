"""
CBMCD Utility Functions

Handles:
- Seed setting for reproducibility
- Configuration loading and merging
- Logging setup
- Distributed training utilities
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.distributed as dist


# =============================================================================
# Seed Management
# =============================================================================

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Controls:
    - Python random
    - NumPy random
    - PyTorch random (CPU and CUDA)
    - CUDNN deterministic behavior
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # For Python hash
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"Random seed set to {seed}")


# =============================================================================
# Configuration Management
# =============================================================================

def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Recursively merge two configurations.
    
    Override values take precedence over base values.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
    
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def config_from_args(args) -> Dict:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed arguments (argparse namespace)
    
    Returns:
        Configuration dictionary
    """
    # Load base config
    config = {}
    if hasattr(args, "config") and args.config:
        config = load_config(args.config)
    
    # Override with command line args
    arg_dict = vars(args)
    
    # Map CLI args to config structure
    if arg_dict.get("model"):
        config.setdefault("model", {})["name"] = arg_dict["model"]
    
    if arg_dict.get("culture"):
        config.setdefault("data", {})["culture"] = arg_dict["culture"]
    
    if arg_dict.get("lang"):
        config.setdefault("data", {})["lang"] = arg_dict["lang"]
    
    if arg_dict.get("seed"):
        config.setdefault("split", {})["seed"] = arg_dict["seed"]
    
    if arg_dict.get("epochs"):
        config.setdefault("training", {})["epochs"] = arg_dict["epochs"]
    
    if arg_dict.get("batch_size"):
        config.setdefault("training", {})["batch_size"] = arg_dict["batch_size"]
    
    if arg_dict.get("learning_rate"):
        config.setdefault("training", {})["learning_rate"] = arg_dict["learning_rate"]
    
    if arg_dict.get("gradient_method"):
        config.setdefault("training", {})["gradient_method"] = arg_dict["gradient_method"]
    
    if arg_dict.get("exp_name"):
        config.setdefault("logging", {})["exp_name"] = arg_dict["exp_name"]
    
    if arg_dict.get("output_dir"):
        config.setdefault("logging", {})["log_dir"] = arg_dict["output_dir"]
    
    return config


def save_config(config: Dict, path: Union[str, Path]):
    """Save configuration to YAML file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    output_dir: Optional[Union[str, Path]] = None,
    exp_name: str = "cbmcd",
    rank: int = 0,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        output_dir: Directory for log files
        exp_name: Experiment name
        rank: Process rank (for distributed)
    
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger("cbmcd")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler (only rank 0)
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if output_dir and rank == 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"{exp_name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger() -> logging.Logger:
    """Get the cbmcd logger"""
    return logging.getLogger("cbmcd")


# =============================================================================
# Distributed Training Utilities
# =============================================================================

def setup_distributed() -> tuple:
    """
    Setup distributed training environment.
    
    Returns:
        (rank, world_size, local_rank)
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    
    return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier():
    """Synchronization barrier for distributed training"""
    if dist.is_initialized():
        dist.barrier()


# =============================================================================
# Experiment Utilities
# =============================================================================

def generate_exp_name(
    model: str,
    culture: str,
    lang: str,
    seed: int,
    gradient_method: str = "pcgrad",
) -> str:
    """
    Generate experiment name from parameters.
    
    Format: {culture}_{model}_{lang}_{method}_seed{seed}
    
    Args:
        model: Model name
        culture: Culture (ko, zh, ja)
        lang: Language (cu, en)
        seed: Random seed
        gradient_method: Gradient method
    
    Returns:
        Experiment name string
    """
    # Shorten model name
    model_short = model.split("/")[-1].lower()
    model_short = model_short.replace("-", "_").replace(".", "")
    model_short = model_short[:20]  # Truncate if too long
    
    exp_name = f"{culture}_{model_short}_{lang}_{gradient_method}_seed{seed}"
    return exp_name


def get_device() -> torch.device:
    """Get the appropriate device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dict with total, trainable, and frozen counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": (trainable / total) * 100 if total > 0 else 0,
    }


# =============================================================================
# File Utilities
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Union[str, Path]):
    """Save data to JSON file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("CBMCD Utilities Module")
    print("=" * 60)
    
    # Test seed setting
    set_seed(42)
    print(f"Random int: {random.randint(0, 100)}")
    print(f"NumPy random: {np.random.rand():.4f}")
    print(f"Torch random: {torch.rand(1).item():.4f}")
    
    # Reset and verify reproducibility
    set_seed(42)
    print(f"Random int (same seed): {random.randint(0, 100)}")
    print(f"NumPy random (same seed): {np.random.rand():.4f}")
    print(f"Torch random (same seed): {torch.rand(1).item():.4f}")
