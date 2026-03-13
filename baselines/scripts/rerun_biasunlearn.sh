
#!/bin/bash
# BiasUnlearn Re-run — affected cultures only (no saved adapters, must re-run)
# Slower: LoRA fine-tuning, ~30min per run
# GPU 3

cd "$(dirname "$0")/../.."

GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

# Best seeds per culture×model (from main experiments)
# Only affected cultures: hi, mr, ml, gu, vi, ur, ar
# NOTE: ur uses max_length=96 and grad_accum=8 to avoid OOM

TOTAL=14
COUNT=0

echo "=== BiasUnlearn Re-run (fixed REVERSE_ALIAS) ==="
echo "GPU: $GPU"
echo "Total runs: $TOTAL"
echo "================================================="

run_unlearn() {
    local CULTURE=$1
    local MODEL=$2
    local SEED=$3
    local MAX_LEN=${4:-128}
    local GRAD_ACCUM=${5:-16}

    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] BiasUnlearn: ${CULTURE}_${MODEL}_seed${SEED} (maxlen=${MAX_LEN}, accum=${GRAD_ACCUM})"

    python -m baselines.run_baseline \
        --method biasunlearn \
        --culture "$CULTURE" \
        --lang cu \
        --model "$MODEL" \
        --seed "$SEED" \
        --data_root ./dataset/camellia/raw \
        --device "cuda:0" \
        --max_length "$MAX_LEN" \
        --unlearn_max_steps 500 \
        --unlearn_batch_size 1 \
        --unlearn_lr 5e-5 \
        --unlearn_warmup_steps 10 \
        --unlearn_max_grad_norm 1.0 \
        --unlearn_grad_accum "$GRAD_ACCUM" \
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
}

# Standard runs (max_length=128, grad_accum=16)
run_unlearn hi llama3_8b 45
run_unlearn hi qwen3_8b 45
run_unlearn mr llama3_8b 42
run_unlearn mr qwen3_8b 42
run_unlearn ml llama3_8b 42
run_unlearn ml qwen3_8b 42
run_unlearn gu llama3_8b 42
run_unlearn gu qwen3_8b 42
run_unlearn vi llama3_8b 48
run_unlearn vi qwen3_8b 48
run_unlearn ar llama3_8b 45
run_unlearn ar qwen3_8b 45

# UR: reduced max_length and grad_accum to avoid OOM
run_unlearn ur llama3_8b 42 96 8
run_unlearn ur qwen3_8b 42 96 8

echo ""
echo "=== All $TOTAL BiasUnlearn runs complete! ==="