#!/bin/bash
# BiasEdit Re-run — only failed Qwen runs (OOM on first attempt)
# Fix: cache_batch_size 128→64, expandable_segments enabled
# GPU 2

cd "$(dirname "$0")/../.."

GPU=2
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNS=(
    "mr qwen3_8b 42"
    "ml qwen3_8b 42"
    "ar qwen3_8b 45"
)

TOTAL=${#RUNS[@]}
COUNT=0

echo "=== BiasEdit Re-run (failed Qwen runs) ==="
echo "GPU: $GPU"
echo "Total runs: $TOTAL"
echo "============================================"

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
        --biasedit_k 5 \
        --biasedit_n_edits 2 \
        --biasedit_meta_lr 1e-4 \
        --biasedit_rank 1920 \
        --biasedit_epochs 10 \
        --biasedit_cache_batch_size 64 \
        --biasedit_patience 5 \
        --biasedit_eval_every 2

done

echo ""
echo "=== All $TOTAL BiasEdit runs complete! ==="