#!/bin/bash
# Baseline Sweep: Llama Remaining Group 1 — ko, vi
# BiasEdit + BiasUnlearn, K=5, PMI eval
cd "$(dirname "$0")/../.."

GPU=0
MODEL="llama3_8b"
CULTURES=("ko" "vi")
LANG="cu"; SEED=45; K=5

export CUDA_VISIBLE_DEVICES=$GPU

TOTAL=$(( 2 * ${#CULTURES[@]} * K ))
COUNT=0

echo "=== Baseline Sweep: Llama Remain1 (${CULTURES[*]}) ==="
echo "GPU=$GPU, Model=$MODEL, K=$K, PMI=true"
echo "Total runs: $TOTAL"
echo "========================================================="

# -----------------------------------------------------------------
# BiasEdit
# -----------------------------------------------------------------
for CULTURE in "${CULTURES[@]}"; do
for FOLD in $(seq 0 $((K-1))); do
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] BiasEdit: ${CULTURE} fold${FOLD} ($MODEL)"

    python -m baselines.run_baseline \
        --method biasedit \
        --culture "$CULTURE" --lang "$LANG" --model "$MODEL" \
        --seed $SEED --device cuda:0 \
        --fold $FOLD --folds_root ./dataset/folds \
        --normalize_prior --priors_root ./dataset/priors \
        --max_length 128 \
        --biasedit_k 15 \
        --biasedit_n_edits 4 \
        --biasedit_epochs 10 \
        --biasedit_rank 1920 \
        --biasedit_n_blocks 2 \
        --biasedit_lr 1e-6 \
        --biasedit_meta_lr 1e-4 \
        --biasedit_max_grad_norm 1.0 \
        --biasedit_cache_batch_size 128 \
        --biasedit_patience 5 \
        --biasedit_eval_every 2

    [ $? -eq 0 ] && echo "  OK" || echo "  FAILED ($?)"
done; done

# -----------------------------------------------------------------
# BiasUnlearn
# -----------------------------------------------------------------
for CULTURE in "${CULTURES[@]}"; do
for FOLD in $(seq 0 $((K-1))); do
    COUNT=$((COUNT + 1))
    MAX_LEN=128; GRAD_ACCUM=16

    echo ""
    echo "[$COUNT/$TOTAL] BiasUnlearn: ${CULTURE} fold${FOLD} ($MODEL)"

    python -m baselines.run_baseline \
        --method biasunlearn \
        --culture "$CULTURE" --lang "$LANG" --model "$MODEL" \
        --seed $SEED --device cuda:0 \
        --fold $FOLD --folds_root ./dataset/folds \
        --normalize_prior --priors_root ./dataset/priors \
        --max_length $MAX_LEN \
        --unlearn_max_steps 500 \
        --unlearn_batch_size 1 \
        --unlearn_lr 5e-5 \
        --unlearn_warmup_steps 10 \
        --unlearn_max_grad_norm 1.0 \
        --unlearn_grad_accum $GRAD_ACCUM \
        --unlearn_beta 0.1 \
        --unlearn_ster_weight 1.0 \
        --unlearn_anti_weight 1.0 \
        --unlearn_kl_weight 0.2 \
        --unlearn_use_lora \
        --unlearn_lora_r 8 \
        --unlearn_lora_alpha 16 \
        --unlearn_lora_dropout 0.1 \
        --unlearn_cbs_target 50.0 \
        --unlearn_cbs_threshold 3.0 \
        --unlearn_eval_every 50 \
        --unlearn_log_every 10 \
        --unlearn_save_every 100 \
        --unlearn_mix_ratio 0.25

    [ $? -eq 0 ] && echo "  OK" || echo "  FAILED ($?)"
done; done

echo ""
echo "=== All $TOTAL runs complete! ==="