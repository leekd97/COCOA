#!/bin/bash
# GPU 3: English (en) language experiments
# Usage: bash scripts/run_gpu3.sh
# Run in tmux/screen: tmux new -s gpu3 'bash scripts/run_gpu3.sh'

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3

OUTPUT="./experiments/sweep_en"
mkdir -p "$OUTPUT"

echo "=== GPU 3: en (English) experiments ==="
echo "Output: $OUTPUT"
echo ""

# --- 1. Llama3 + en ---
echo "[1/2] llama3_8b_en_mse"
python main.py \
    --culture ko --lang en --model llama3_8b \
    --seed 42 --epochs 10 \
    --pairs_per_batch 8 --pairs_per_category 200 \
    --grounded_loss soft_contrastive --contrastive_temperature 1.0 \
    --neutral_loss mse \
    --gradient_method goal_aware_pcgrad \
    --ref_update_steps 0 \
    --output_dir "$OUTPUT" \
    --exp_name llama3_8b_en_mse \
    --eval_steps 200 --log_steps 50 \
    2>&1 | tee "$OUTPUT/llama3_8b_en_mse.log"

echo ""

# --- 2. Qwen3 + en ---
echo "[2/2] qwen3_8b_en_mse"
python main.py \
    --culture ko --lang en --model qwen3_8b \
    --seed 42 --epochs 10 \
    --pairs_per_batch 8 --pairs_per_category 200 \
    --grounded_loss soft_contrastive --contrastive_temperature 1.0 \
    --neutral_loss mse \
    --gradient_method goal_aware_pcgrad \
    --ref_update_steps 0 \
    --output_dir "$OUTPUT" \
    --exp_name qwen3_8b_en_mse \
    --eval_steps 200 --log_steps 50 \
    2>&1 | tee "$OUTPUT/qwen3_8b_en_mse.log"

echo ""
echo "=== GPU 3 done! ==="