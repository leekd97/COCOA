# CBMCD: Cultural Bias Mitigation with Context-Dependent Preference

A framework for mitigating cultural bias in Large Language Models using context-dependent preference learning.

## Overview

### Problem
LLMs exhibit Western cultural bias even in Asian cultural contexts. Existing bias mitigation methods assume context-agnostic fairness, which leads to:
- **Bias Reversal**: Training on grounded contexts overcorrects, suppressing Western entities even in neutral contexts
- **Performance Degradation**: Training on neutral contexts fails to correct bias in grounded contexts

### Solution
CBMCD uses **dual-context training** with **gradient conflict resolution**:
- **Grounded contexts**: Learn to prefer Asian entities when cultural context is present
- **Neutral contexts**: Learn to maintain balanced preferences when no cultural context exists
- **PCGrad**: Resolve gradient conflicts between the two objectives

## Key Features

- **Context-Dependent Preference Learning**: Different objectives for grounded vs neutral contexts
- **PCGrad Integration**: Automatic gradient conflict resolution
- **Balanced Sampling**: Equal grounded/neutral samples per batch
- **Multi-GPU Support**: Distributed training with torchrun
- **Flexible Configuration**: YAML configs + CLI overrides
- **Reproducibility**: Seed control for data splits, sampling, and initialization

## Installation

```bash
# Create environment
conda create -n cbmcd python=3.10 -y
conda activate cbmcd

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# (Optional) Flash Attention for faster training
pip install flash-attn --no-build-isolation
```

## Data Setup

Download the Camellia dataset and place it in `dataset/camellia/raw/`:

```
dataset/camellia/raw/
├── contexts/
│   ├── camellia-grounded/
│   │   └── causal-lms/
│   │       ├── grounded-contexts-causal-lms-korean.xlsx
│   │       ├── grounded-contexts-causal-lms-chinese.xlsx
│   │       └── ...
│   └── camellia-neutral/
│       └── causal-lms/
│           ├── neutral-contexts-causal-lms-korean.xlsx
│           └── ...
└── entities/
    ├── korean/
    │   ├── authors.xlsx
    │   ├── food.xlsx
    │   └── ...
    ├── chinese/
    ├── japanese/
    └── western/
```

## Usage

### Training

```bash
# Single experiment
bash scripts/train.sh \
    --culture ko \
    --model llama3_8b \
    --lang cu \
    --seed 42 \
    --gpus 4

# Multiple experiments
bash scripts/train_multi.sh
```

### CLI Options

```bash
python main.py train \
    --config configs/default.yaml \
    --model llama3_8b \          # Model: llama3_8b, qwen3_8b, gemma3_12b
    --culture ko \               # Culture: ko, zh, ja, vi
    --lang cu \                  # Language: cu (native), en (English)
    --seed 42 \                  # Random seed
    --epochs 10 \
    --gradient_method pcgrad \   # Gradient: weighted, pcgrad
    --output_dir ./experiments
```

### Evaluation

```bash
# Evaluate trained model
python main.py eval \
    --checkpoint experiments/exp_name/checkpoints/best \
    --culture ko

# Evaluate baseline (before training)
python main.py baseline \
    --model llama3_8b \
    --culture ko
```

## Project Structure

```
CBMCD/
├── configs/
│   └── default.yaml          # Default configuration
├── dataset/
│   └── camellia/raw/         # Camellia dataset
├── src/
│   ├── data.py               # Data loading, splitting, DataLoader
│   ├── model.py              # Model loading, LoRA
│   ├── loss.py               # Contrastive & MSE losses
│   ├── trainer.py            # Training loop, PCGrad
│   ├── evaluate.py           # CBS evaluation
│   └── utils.py              # Utilities
├── scripts/
│   ├── train.sh              # Single experiment script
│   └── train_multi.sh        # Multi-experiment script
├── experiments/              # Output directory
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

## Method

### Loss Functions

**Grounded Context (Contrastive Loss)**:
```
L_grounded = -log(P(Asian) / (P(Asian) + P(Western)))
```
Encourages the model to prefer Asian entities when cultural context is present.

**Neutral Context (MSE Loss)**:
```
L_neutral = (log P(Asian) - log P(Western))²
```
Encourages balanced preferences when no cultural context is present.

### PCGrad (Projecting Conflicting Gradients)

When gradients from grounded and neutral losses conflict:
```
if g_grounded · g_neutral < 0:
    g_grounded' = g_grounded - proj(g_grounded, g_neutral)
```

This removes conflicting components, allowing both objectives to improve.

### Evaluation Metric: CBS (Cultural Bias Score)

```
CBS = % of samples where P(Western) > P(Asian)
```

- **Grounded contexts**: Lower CBS is better (should prefer Asian)
- **Neutral contexts**: CBS ≈ 50% is ideal (balanced)

## Results

| Model | Context | Baseline CBS | CBMCD CBS |
|-------|---------|--------------|-----------|
| Llama-3.1-8B | Grounded | ~70% | ~30% |
| Llama-3.1-8B | Neutral | ~65% | ~50% |

## Citation

```bibtex
@article{cbmcd2024,
  title={Cultural Bias Mitigation with Context-Dependent Preference},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License
