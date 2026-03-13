#!/bin/bash
# Resume-capable CoCoA Sweep (cu, Llama)
# Skips runs that already have results.json
#
# Usage:
#   tmux new -s cu_llama 'bash scripts/sweep_cu_llama_resume.sh'

cd "$(dirname "$0")/.."

# =============================================
# Crash logging
# =============================================
SWEEP_LOG="./experiments/_sweep_cu_llama.log"
exec > >(tee -a "$SWEEP_LOG") 2>&1

trap 'echo "[$(date)] CRASH: sweep terminated with signal $? (culture=$CULTURE seed=$SEED)" >> "$SWEEP_LOG"' EXIT ERR TERM

echo "[$(date)] Sweep started (PID=$$)"

GPU=0
CULTURES=("ko" "ja" "zh" "vi" "ur" "hi" "ml" "mr" "gu" "ar")
MODELS=("llama3_8b")
SEEDS=(41 43 44)
LANG="cu"

# Hyperparameters
W_GROUNDED=1.0
W_NEUTRAL=2.0
TAU=1.0
LORA_R=16
NEUTRAL_LOSS="mse"
GRADIENT="goal_aware_pcgrad"
EPOCHS=15
PAIRS_PER_BATCH=16
REF_UPDATE=0

export CUDA_VISIBLE_DEVICES=$GPU
OUTPUT="./experiments"
mkdir -p "$OUTPUT"

# Model short name (must match build_exp_name in main.py)
MODEL_SHORT="llama3-8b"

# Count total and already done
TOTAL=0
DONE=0
for c in "${CULTURES[@]}"; do
for s in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    # Expected folder name from build_exp_name():
    # {culture}_{lang}_{model}_{nloss}_wg{}_wn{}_tau{}_r{}_seed{}
    EXP_NAME="${c}_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${W_GROUNDED}_wn${W_NEUTRAL}_tau${TAU}_r${LORA_R}_seed${s}"
    if [ -f "$OUTPUT/$EXP_NAME/results.json" ]; then
        DONE=$((DONE + 1))
    fi
done; done

echo "=== COCOA Sweep RESUME (${LANG}, ${MODEL_SHORT}) ==="
echo "GPU: $GPU"
echo "Cultures: ${CULTURES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "HP: wg=${W_GROUNDED} wn=${W_NEUTRAL} tau=${TAU} r=${LORA_R} loss=${NEUTRAL_LOSS}"
echo "Total: $TOTAL, Already done: $DONE, Remaining: $((TOTAL - DONE))"
echo "=========================================="
echo ""

COUNT=0
SKIP=0
for CULTURE in "${CULTURES[@]}"; do
for SEED in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))

    EXP_NAME="${CULTURE}_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${W_GROUNDED}_wn${W_NEUTRAL}_tau${TAU}_r${LORA_R}_seed${SEED}"

    # Skip if results.json exists
    if [ -f "$OUTPUT/$EXP_NAME/results.json" ]; then
        SKIP=$((SKIP + 1))
        echo "[$COUNT/$TOTAL] SKIP (done) $EXP_NAME"
        continue
    fi

    echo "[$COUNT/$TOTAL] RUN  ${CULTURE}_${LANG}_llama3_8b_seed${SEED}"

    START_TIME=$(date +%s)
    python main.py \
        --culture "$CULTURE" \
        --lang "$LANG" \
        --model "llama3_8b" \
        --seed "$SEED" \
        --epochs $EPOCHS \
        --pairs_per_batch $PAIRS_PER_BATCH \
        --pairs_per_category 200 \
        --grounded_loss soft_contrastive \
        --contrastive_temperature $TAU \
        --neutral_loss "$NEUTRAL_LOSS" \
        --w_grounded $W_GROUNDED \
        --w_neutral $W_NEUTRAL \
        --lora_r $LORA_R \
        --lora_alpha $((LORA_R * 2)) \
        --gradient_method "$GRADIENT" \
        --ref_update_steps $REF_UPDATE \
        --output_dir "$OUTPUT" \
        --eval_steps 200 \
        --log_steps 50 \
        2>&1 | tee "$OUTPUT/${CULTURE}_${LANG}_llama3_8b_seed${SEED}.log"
    EXIT_CODE=${PIPESTATUS[0]}
    ELAPSED=$(( $(date +%s) - START_TIME ))

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date)] FAILED: ${CULTURE}_seed${SEED} exit_code=$EXIT_CODE elapsed=${ELAPSED}s" | tee -a "$SWEEP_LOG"
        echo "  Check: $OUTPUT/${CULTURE}_${LANG}_llama3_8b_seed${SEED}.log"
    else
        echo "[$(date)] OK: ${CULTURE}_seed${SEED} elapsed=${ELAPSED}s"
    fi
done; done

echo "=== Done! Skipped: $SKIP, Ran: $((TOTAL - SKIP)) ==="
echo "[$(date)] Sweep completed normally" >> "$SWEEP_LOG"
trap - EXIT  # clear trap on clean exit