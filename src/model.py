"""
CBMCD Model Module

Handles:
- Loading pretrained models (Llama, Qwen, Gemma, etc.)
- LoRA configuration and application
- Layer-wise LoRA targeting for bias localization
- Multi-GPU setup
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration with layer-wise LoRA support"""
    name: str = "meta-llama/Llama-3.1-8B"
    use_lora: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # ==========================================================================
    # Layer-wise LoRA Targeting (NEW)
    # ==========================================================================
    # Which layers to apply LoRA (0-indexed, exclusive end)
    # For Llama-3.1-8B: 32 layers (0-31)
    target_layer_start: int = 0
    target_layer_end: int = 32  # exclusive
    
    # Which module types to target
    # Options: "attention", "mlp", "both"
    target_modules_type: str = "attention"
    
    # Legacy: used if target_modules_type is "custom"
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")
    # ==========================================================================


# Model name shortcuts
MODEL_SHORTCUTS = {
    "llama3_8b": "meta-llama/Llama-3.1-8B",
    "llama3_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen25_7b": "Qwen/Qwen2.5-7B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "gemma3_12b": "google/gemma-3-12b-pt",
    "gemma3_12b_it": "google/gemma-3-12b-it",
    "aya_8b": "CohereForAI/aya-expanse-8b",
    "gpt_oss_20b": "EleutherAI/gpt-neox-20b",
}

# Number of layers per model (for validation)
MODEL_NUM_LAYERS = {
    "meta-llama/Llama-3.1-8B": 32,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "Qwen/Qwen2.5-7B": 28,
    "Qwen/Qwen3-8B": 36,
    "google/gemma-3-12b-pt": 48,
    "google/gemma-3-12b-it": 48,
    "CohereForAI/aya-expanse-8b": 32,
}


# =============================================================================
# Model Loading
# =============================================================================

def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def load_model(
    model_name: str,
    config: Optional[ModelConfig] = None,
    device_map: str = None,
    for_distributed: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a pretrained model with optional LoRA.
    
    Args:
        model_name: Model name or shortcut (e.g., "llama3_8b")
        config: ModelConfig object
        device_map: Device mapping strategy (None for distributed training)
        for_distributed: If True, don't use device_map (for accelerate/DDP)
    
    Returns:
        (model, tokenizer)
    """
    if config is None:
        config = ModelConfig()
    
    # Resolve shortcut
    full_name = MODEL_SHORTCUTS.get(model_name, model_name)
    
    print(f"Loading model: {full_name}")
    
    # Quantization config
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=get_torch_dtype(config.torch_dtype),
            bnb_4bit_use_double_quant=True,
        )
    elif config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        full_name,
        trust_remote_code=config.trust_remote_code,
    )
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
    }
    
    # For distributed training, don't use device_map
    if not for_distributed and device_map is not None:
        model_kwargs["device_map"] = device_map
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = get_torch_dtype(config.torch_dtype)
    
    model = AutoModelForCausalLM.from_pretrained(full_name, **model_kwargs)
    
    # Prepare for training
    if config.load_in_4bit or config.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    if config.use_lora:
        model = apply_lora(model, config, full_name)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    print(f"Model loaded. Trainable params: {count_trainable_params(model):,}")
    
    return model, tokenizer


def get_target_modules(config: ModelConfig, model_name: str) -> List[str]:
    """
    Generate target module names based on configuration.
    
    Args:
        config: ModelConfig with layer targeting settings
        model_name: Full model name for architecture detection
    
    Returns:
        List of module names to apply LoRA
    """
    # Determine architecture
    is_llama = "llama" in model_name.lower()
    is_qwen = "qwen" in model_name.lower()
    is_gemma3 = "gemma-3" in model_name.lower() or "gemma3" in model_name.lower()
    is_gemma = "gemma" in model_name.lower() and not is_gemma3
    is_cohere = "cohere" in model_name.lower() or "aya" in model_name.lower()
    
    # Define module names per architecture
    if is_llama or is_qwen or is_cohere:
        attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        mlp_modules = ["gate_proj", "up_proj", "down_proj"]
        layer_prefix = "model.layers"
        attn_prefix = "self_attn"
        mlp_prefix = "mlp"
    elif is_gemma3:
        # Gemma 3 is multimodal — text layers under language_model
        attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        mlp_modules = ["gate_proj", "up_proj", "down_proj"]
        layer_prefix = "model.language_model.layers"
        attn_prefix = "self_attn"
        mlp_prefix = "mlp"
    elif is_gemma:
        attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        mlp_modules = ["gate_proj", "up_proj", "down_proj"]
        layer_prefix = "model.layers"
        attn_prefix = "self_attn"
        mlp_prefix = "mlp"
    else:
        # Fallback to generic
        print(f"Warning: Unknown architecture for {model_name}, using generic config")
        return list(config.lora_target_modules)
    
    # Select modules based on type
    if config.target_modules_type == "attention":
        base_modules = [(attn_prefix, m) for m in attn_modules]
    elif config.target_modules_type == "mlp":
        base_modules = [(mlp_prefix, m) for m in mlp_modules]
    elif config.target_modules_type == "both":
        base_modules = (
            [(attn_prefix, m) for m in attn_modules] +
            [(mlp_prefix, m) for m in mlp_modules]
        )
    elif config.target_modules_type == "custom":
        # Use legacy config
        return list(config.lora_target_modules)
    else:
        raise ValueError(f"Unknown target_modules_type: {config.target_modules_type}")
    
    # Generate full module paths for specified layer range
    target_modules = []
    for layer_idx in range(config.target_layer_start, config.target_layer_end):
        for prefix, module in base_modules:
            target_modules.append(f"{layer_prefix}.{layer_idx}.{prefix}.{module}")
    
    return target_modules


def apply_lora(model: PreTrainedModel, config: ModelConfig, model_name: str = None) -> PreTrainedModel:
    """Apply LoRA to model with layer-wise targeting"""
    
    # Get target modules
    if model_name:
        target_modules = get_target_modules(config, model_name)
    else:
        # Fallback to legacy
        target_modules = list(config.lora_target_modules)
    
    # Print targeting info
    layer_range = f"[{config.target_layer_start}, {config.target_layer_end})"
    print(f"LoRA targeting:")
    print(f"  - Layer range: {layer_range}")
    print(f"  - Module type: {config.target_modules_type}")
    print(f"  - Total target modules: {len(target_modules)}")
    
    # Sample of modules (first 3 and last 3)
    if len(target_modules) > 6:
        sample = target_modules[:3] + ["..."] + target_modules[-3:]
    else:
        sample = target_modules
    print(f"  - Modules: {sample}")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def count_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Log Probability Computation
# =============================================================================

def compute_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_start_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute log probabilities for sequences.
    
    Args:
        model: Language model
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        target_start_idx: Starting index for target tokens (entity part)
                         If None, compute for entire sequence
    
    Returns:
        Log probabilities [batch]
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab]
    
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    
    # Compute log probs
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask padding
    token_log_probs = token_log_probs * shift_mask
    
    # Sum log probs (or mean, depending on preference)
    if target_start_idx is not None:
        # Only sum from target_start_idx onwards
        batch_log_probs = []
        for i in range(input_ids.size(0)):
            start = target_start_idx[i].item() if torch.is_tensor(target_start_idx[i]) else target_start_idx[i]
            batch_log_probs.append(token_log_probs[i, start:].sum())
        return torch.stack(batch_log_probs)
    else:
        return token_log_probs.sum(dim=-1)


def compute_entity_log_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    context: str,
    entity: str,
    device: torch.device,
) -> float:
    """
    Compute log probability of entity given context.
    
    The context should contain [MASK] which will be replaced with entity.
    We compute P(entity | context_before_mask).
    """
    # Split context at [MASK]
    parts = context.split("[MASK]")
    if len(parts) < 2:
        raise ValueError(f"Context must contain [MASK]: {context}")
    
    context_before = parts[0]
    full_text = context_before + entity
    
    # Tokenize
    context_tokens = tokenizer(context_before, return_tensors="pt")
    full_tokens = tokenizer(full_text, return_tensors="pt")
    
    context_len = context_tokens["input_ids"].size(1)
    
    # Move to device
    input_ids = full_tokens["input_ids"].to(device)
    attention_mask = full_tokens["attention_mask"].to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Compute log prob for entity tokens only
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    total_log_prob = 0.0
    for i in range(context_len - 1, input_ids.size(1) - 1):
        next_token = input_ids[0, i + 1]
        total_log_prob += log_probs[0, i, next_token].item()
    
    return total_log_prob


# =============================================================================
# Multi-GPU Utilities
# =============================================================================

def setup_distributed():
    """Setup distributed training"""
    import os
    import torch.distributed as dist
    
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    
    return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Utility: Layer Analysis
# =============================================================================

def get_layer_info(model_name: str) -> Dict:
    """Get layer information for a model"""
    full_name = MODEL_SHORTCUTS.get(model_name, model_name)
    num_layers = MODEL_NUM_LAYERS.get(full_name, 32)
    
    return {
        "model_name": full_name,
        "num_layers": num_layers,
        "suggested_ranges": {
            "early": (0, num_layers // 3),
            "middle": (num_layers // 3, 2 * num_layers // 3),
            "later": (2 * num_layers // 3, num_layers),
            "all": (0, num_layers),
        }
    }


if __name__ == "__main__":
    # Test layer targeting
    print("Testing layer-wise LoRA configuration...")
    
    config = ModelConfig(
        name="meta-llama/Llama-3.1-8B",
        use_lora=True,
        lora_r=8,
        target_layer_start=20,
        target_layer_end=32,
        target_modules_type="attention",
    )
    
    modules = get_target_modules(config, config.name)
    print(f"\nGenerated {len(modules)} target modules:")
    for m in modules[:5]:
        print(f"  {m}")
    print("  ...")
    for m in modules[-3:]:
        print(f"  {m}")
    
    # Layer info
    print("\nLayer info for Llama-3.1-8B:")
    info = get_layer_info("llama3_8b")
    print(f"  Total layers: {info['num_layers']}")
    print(f"  Suggested ranges: {info['suggested_ranges']}")