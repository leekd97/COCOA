#!/bin/bash
# Drift Regularization Sweep — all cultures, fold 0
cd "$(dirname "$0")/.."
source ~/.bashrc && conda activate cocoa

# ==========================================================================
GPU=1
MODEL="qwen3_8b"
MODEL_SHORT="qwen3-8b"
# GPU=1; MODEL="qwen3_8b"; MODEL_SHORT="qwen3-8b"
# ==========================================================================

CULTURES=("ko" "ja" "zh" "vi" "ur" "hi" "ml" "mr" "gu" "ar")
LANG="cu"; K=1; SEED=45
AG=1.0; AN=0.3
W_GROUNDED=1.0; W_NEUTRAL=2.0; TAU=1.0; LORA_R=16
NEUTRAL_LOSS="mse"; GRADIENT="goal_aware_pcgrad"
EPOCHS=15; PAIRS_PER_BATCH=16; REF_UPDATE=0; MSE_SCALE=10.0

export CUDA_VISIBLE_DEVICES=$GPU
OUTPUT="./experiments"; FOLDS_ROOT="./dataset/folds"; PRIORS_ROOT="./dataset/priors"

# ★ Drift weights to sweep
DRIFT_LIST=(1.0 3.0 5.0)

for W_DRIFT in "${DRIFT_LIST[@]}"; do
    EXP_SUBDIR="sweep_drift_${W_DRIFT}"
    SWEEP_LOG="$OUTPUT/${EXP_SUBDIR}/_sweep_${MODEL_SHORT}.log"
    mkdir -p "$OUTPUT/$EXP_SUBDIR"

    echo "============================================================" >> "$SWEEP_LOG"
    echo "[$(date)] START: drift=$W_DRIFT / $MODEL_SHORT / GPU=$GPU" >> "$SWEEP_LOG"
    echo "  ag=$AG an=$AN wg=$W_GROUNDED wn=$W_NEUTRAL w_drift=$W_DRIFT" >> "$SWEEP_LOG"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null >> "$SWEEP_LOG"

    exec > >(tee -a "$SWEEP_LOG") 2>&1
    trap 'CODE=$?; echo "[$(date)] TERMINATED sig=$CODE w_drift=$W_DRIFT culture=$CULTURE" >> "$SWEEP_LOG"' EXIT ERR TERM INT HUP

    echo ""
    echo "============================================================"
    echo "  [drift=$W_DRIFT] ag=$AG an=$AN wg=$W_GROUNDED wn=$W_NEUTRAL ($MODEL_SHORT)"
    echo "============================================================"

    TOTAL=${#CULTURES[@]}
    COUNT=0; SKIP=0; FAIL=0

    for CULTURE in "${CULTURES[@]}"; do
        FOLD=0
        COUNT=$((COUNT + 1))
        EXP_NAME="${CULTURE}_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${W_GROUNDED}_wn${W_NEUTRAL}_tau${TAU}_r${LORA_R}_pnorm-g${AG}-n${AN}_nxn_fold${FOLD}_seed${SEED}"

        [ -f "$OUTPUT/$EXP_SUBDIR/$EXP_NAME/results.json" ] && { SKIP=$((SKIP+1)); echo "[$COUNT/$TOTAL] SKIP $CULTURE"; continue; }

        echo "[$COUNT/$TOTAL] RUN $CULTURE [drift=$W_DRIFT]"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  gpu_mem: /'
        START_TIME=$(date +%s)

        python main.py \
            --culture "$CULTURE" --lang "$LANG" --model "$MODEL" \
            --seed $SEED --fold $FOLD --folds_root "$FOLDS_ROOT" \
            --normalize_prior --priors_root "$PRIORS_ROOT" \
            --prior_alpha_g $AG --prior_alpha_n $AN \
            --pairing nxn \
            --mse_scale $MSE_SCALE \
            --contrastive_temperature $TAU \
            --w_drift $W_DRIFT \
            --exp_subdir "$EXP_SUBDIR" \
            --epochs $EPOCHS --pairs_per_batch $PAIRS_PER_BATCH --pairs_per_category 200 \
            --grounded_loss soft_contrastive \
            --neutral_loss "$NEUTRAL_LOSS" --w_grounded $W_GROUNDED --w_neutral $W_NEUTRAL \
            --lora_r $LORA_R --lora_alpha $((LORA_R * 2)) \
            --gradient_method "$GRADIENT" --ref_update_steps $REF_UPDATE \
            --output_dir "$OUTPUT" --eval_steps 200 --log_steps 50 \
            2>&1 | tee "$OUTPUT/$EXP_SUBDIR/${CULTURE}_${MODEL_SHORT}_fold${FOLD}.log"
        EXIT_CODE=${PIPESTATUS[0]}; ELAPSED=$(( $(date +%s) - START_TIME ))

        if [ $EXIT_CODE -eq 0 ]; then echo "[$(date)] OK: $CULTURE ${ELAPSED}s"
        elif [ $EXIT_CODE -eq 137 ]; then FAIL=$((FAIL+1)); echo "[$(date)] ★ OOM (137): $CULTURE"
        elif [ $EXIT_CODE -eq 139 ]; then FAIL=$((FAIL+1)); echo "[$(date)] ★ SEGFAULT (139): $CULTURE"
        else FAIL=$((FAIL+1)); echo "[$(date)] FAILED ($EXIT_CODE): $CULTURE ${ELAPSED}s"; fi
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  gpu_mem_after: /'
    done

    echo "[$(date)] DONE drift=$W_DRIFT: Total=$TOTAL Skip=$SKIP Ran=$((TOTAL-SKIP)) Fail=$FAIL"
    echo ""
done

trap - EXIT
echo "[$(date)] ALL DRIFT CONFIGS COMPLETE"