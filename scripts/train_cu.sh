#!/bin/bash
# GPU 2: Korean (cu) language experiments
# Usage: bash scripts/run_gpu2.sh
# Run in tmux/screen: tmux new -s gpu2 'bash scripts/run_gpu2.sh'

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=2

OUTPUT="./experiments/sweep_cu"
mkdir -p "$OUTPUT"

echo "=== GPU 2: cu (Korean) experiments ==="
echo "Output: $OUTPUT"
echo ""

# --- 1. Llama3 + cu ---
python main.py \
    --culture ko --lang cu --model llama3_8b \
    --seed 45 --epochs 10 \
    --pairs_per_batch 8 --pairs_per_category 200 \
    --grounded_loss soft_contrastive --contrastive_temperature 1.0 \
    --neutral_loss mse \
    --w_grounded 1.0 --w_neutral 2.0 \
    --gradient_method goal_aware_pcgrad \
    --ref_update_steps 0 \
    --output_dir "$OUTPUT" \
    --eval_steps 200 --log_steps 50 \
    2>&1 | tee "$OUTPUT/llama3_8b_cu_mse.log"

# --- 2. Qwen3 + cu ---
python main.py \
    --culture ko --lang cu --model qwen3_8b \
    --seed 45 --epochs 10 \
    --pairs_per_batch 8 --pairs_per_category 200 \
    --grounded_loss soft_contrastive --contrastive_temperature 1.0 \
    --neutral_loss mse \
    --w_grounded 1.0 --w_neutral 2.0 \
    --gradient_method goal_aware_pcgrad \
    --ref_update_steps 0 \
    --output_dir "$OUTPUT" \
    --eval_steps 200 --log_steps 50 \
    2>&1 | tee "$OUTPUT/qwen3_8b_cu_mse.log"
echo ""
echo "=== GPU 2 done! ==="