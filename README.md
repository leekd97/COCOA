# CoCoA

Code for our EMNLP 2026 Findings paper **"CoCoA: Context-Conditional Cultural Alignment for Large Language Models"**.

## Installation

```bash
pip install -r requirements.txt
```

## Prepare Data

Place the preprocessed data under:

```text
dataset/
├── folds/
└── priors/
```

## Train CoCoA

Example for Llama-3 8B on Korean:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --culture ko \
    --lang cu \
    --model llama3_8b \
    --fold 0 \
    --seed 45 \
    --folds_root ./dataset/folds \
    --priors_root ./dataset/priors \
    --prior_alpha_g 1.0 \
    --prior_alpha_n 0.3 \
    --pairing nxn \
    --k 10.0 \
    --lam 10.0 \
    --contrastive_temperature 1.0 \
    --w_grounded 1.0 \
    --w_neutral 2.0 \
    --w_drift 1.0 \
    --epochs 15 \
    --pairs_per_batch 16 \
    --pairs_per_category 200 \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir ./experiments \
    --exp_subdir main_table
```
