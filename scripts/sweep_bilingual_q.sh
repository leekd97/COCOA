#!/bin/bash
# Bilingual (cu+en) training — Ar excluded (no en data)
# Fold 0 only for quick exploration
# ==========================================================================
GPU=1
MODEL="qwen3_8b"
MODEL_SHORT="qwen3-8b"
# GPU=1; MODEL="qwen3_8b"; MODEL_SHORT="qwen3-8b"
# ==========================================================================

# ★ Ar excluded — no English translation available
CULTURES=("ko" "ja" "zh" "vi" "ur" "hi" "ml" "mr" "gu")
LANG="cu"; K=1; SEED=45
EXP_SUBDIR="kfold_bilingual"

AG=0.5; AN=0.3
W_GROUNDED=1.0; W_NEUTRAL=2.0; TAU=1.0; LORA_R=16
NEUTRAL_LOSS="mse"; GRADIENT="goal_aware_pcgrad"
EPOCHS=15; PAIRS_PER_BATCH=16; REF_UPDATE=0; MSE_SCALE=10.0

export CUDA_VISIBLE_DEVICES=$GPU
OUTPUT="./experiments"; FOLDS_ROOT="./dataset/folds"; PRIORS_ROOT="./dataset/priors"

SWEEP_LOG="$OUTPUT/${EXP_SUBDIR}/_sweep_${MODEL_SHORT}.log"
mkdir -p "$OUTPUT/$EXP_SUBDIR"

echo "============================================================" >> "$SWEEP_LOG"
echo "[$(date)] START: $EXP_SUBDIR / $MODEL_SHORT / GPU=$GPU / PID=$$" >> "$SWEEP_LOG"
echo "  HOST=$(hostname)  ag=$AG an=$AN wg=$W_GROUNDED wn=$W_NEUTRAL bilingual=true" >> "$SWEEP_LOG"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null >> "$SWEEP_LOG"

exec > >(tee -a "$SWEEP_LOG") 2>&1
trap 'CODE=$?; echo "[$(date)] TERMINATED sig=$CODE culture=$CULTURE fold=$FOLD" >> "$SWEEP_LOG"' EXIT ERR TERM INT HUP
echo "[$(date)] Sweep: $EXP_SUBDIR / $MODEL_SHORT (PID=$$)"

TOTAL=$(( ${#CULTURES[@]} * K ))
COUNT=0; SKIP=0; FAIL=0

for CULTURE in "${CULTURES[@]}"; do
for FOLD in $(seq 0 $((K-1))); do
    COUNT=$((COUNT + 1))
    EXP_NAME="${CULTURE}_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${W_GROUNDED}_wn${W_NEUTRAL}_tau${TAU}_r${LORA_R}_pnorm-g${AG}-n${AN}_nxn_bilingual_fold${FOLD}_seed${SEED}"

    [ -f "$OUTPUT/$EXP_SUBDIR/$EXP_NAME/results.json" ] && { SKIP=$((SKIP+1)); echo "[$COUNT/$TOTAL] SKIP"; continue; }

    echo "[$COUNT/$TOTAL] RUN ${CULTURE} fold${FOLD} (${MODEL_SHORT}, bilingual)"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  gpu_mem: /'
    START_TIME=$(date +%s)

    python main.py \
        --culture "$CULTURE" --lang "$LANG" --model "$MODEL" \
        --seed $SEED --fold $FOLD --folds_root "$FOLDS_ROOT" \
        --bilingual \
        --normalize_prior --priors_root "$PRIORS_ROOT" \
        --prior_alpha_g $AG --prior_alpha_n $AN \
        --pairing nxn \
        --mse_scale $MSE_SCALE \
        --exp_subdir "$EXP_SUBDIR" \
        --epochs $EPOCHS --pairs_per_batch $PAIRS_PER_BATCH --pairs_per_category 200 \
        --grounded_loss soft_contrastive --contrastive_temperature $TAU \
        --neutral_loss "$NEUTRAL_LOSS" --w_grounded $W_GROUNDED --w_neutral $W_NEUTRAL \
        --lora_r $LORA_R --lora_alpha $((LORA_R * 2)) \
        --gradient_method "$GRADIENT" --ref_update_steps $REF_UPDATE \
        --output_dir "$OUTPUT" --eval_steps 200 --log_steps 50 \
        2>&1 | tee "$OUTPUT/$EXP_SUBDIR/${CULTURE}_${MODEL_SHORT}_fold${FOLD}.log"
    EXIT_CODE=${PIPESTATUS[0]}; ELAPSED=$(( $(date +%s) - START_TIME ))

    if [ $EXIT_CODE -eq 0 ]; then echo "[$(date)] OK: ${CULTURE} fold${FOLD} ${ELAPSED}s"
    elif [ $EXIT_CODE -eq 137 ]; then FAIL=$((FAIL+1)); echo "[$(date)] ★ OOM (137): ${CULTURE} fold${FOLD}"
    elif [ $EXIT_CODE -eq 139 ]; then FAIL=$((FAIL+1)); echo "[$(date)] ★ SEGFAULT (139): ${CULTURE} fold${FOLD}"
    else FAIL=$((FAIL+1)); echo "[$(date)] FAILED ($EXIT_CODE): ${CULTURE} fold${FOLD} ${ELAPSED}s"; fi
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  gpu_mem_after: /'
done; done

echo "[$(date)] DONE: Total=$TOTAL Skip=$SKIP Ran=$((TOTAL-SKIP)) Fail=$FAIL"
trap - EXIT