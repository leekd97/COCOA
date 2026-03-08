#!/bin/bash
# BiasEdit Re-run — affected cultures only (no saved models, must re-run)
# Fast: weight editing only, ~5min per run
# GPU 2

cd "$(dirname "$0")/../.."

GPU=2
export CUDA_VISIBLE_DEVICES=$GPU

# Best seeds per culture×model (from main experiments)
# Only affected cultures: hi, mr, ml, gu, vi, ur, ar
RUNS=(
    "hi llama3_8b 45"
    "hi qwen3_8b 45"
    "mr llama3_8b 42"
    "mr qwen3_8b 42"
    "ml llama3_8b 42"
    "ml qwen3_8b 42"
    "gu llama3_8b 42"
    "gu qwen3_8b 42"
    "vi llama3_8b 48"
    "vi qwen3_8b 48"
    "ur llama3_8b 42"
    "ur qwen3_8b 42"
    "ar llama3_8b 45"
    "ar qwen3_8b 45"
)

TOTAL=${#RUNS[@]}
COUNT=0

echo "=== BiasEdit Re-run (fixed REVERSE_ALIAS) ==="
echo "GPU: $GPU"
echo "Total runs: $TOTAL"
echo "=============================================="

for RUN in "${RUNS[@]}"; do
    read -r CULTURE MODEL SEED <<< "$RUN"
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] BiasEdit: ${CULTURE}_${MODEL}_seed${SEED}"

    python -m baselines.run_baseline \
        --method biasedit \
        --culture "$CULTURE" \
        --lang cu \
        --model "$MODEL" \
        --seed "$SEED" \
        --data_root ./dataset/camellia/raw \
        --device "cuda:0" \
        --max_length 128 \
        --edit_k 5 \
        --edit_n_edits 2 \
        --edit_meta_lr 1e-4 \
        --edit_rank 1920 \
        --edit_n_epochs 10 \
        --edit_cache_batch_size 128 \
        --edit_early_stop_patience 5 \
        --edit_eval_every 2

done

echo ""
echo "=== All $TOTAL BiasEdit runs complete! ==="