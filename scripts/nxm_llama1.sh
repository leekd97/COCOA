#!/bin/bash
# Exp 3: N×M entity pairing (no PMI)
cd "$(dirname "$0")/.."
source ~/.bashrc && conda activate cocoa

GPU=0
CULTURES=("ko" "ja" "zh" "vi" "ur")
MODEL="llama3_8b"
MODEL_SHORT="llama3-8b"
LANG="cu"
K=5
SEED=45
EXP_SUBDIR="kfold_nxm"

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
FOLDS_ROOT="./dataset/folds"

SWEEP_LOG="$OUTPUT/${EXP_SUBDIR}/_sweep.log"
mkdir -p "$OUTPUT/$EXP_SUBDIR"
exec > >(tee -a "$SWEEP_LOG") 2>&1
trap 'echo "[$(date)] CRASH: culture=$CULTURE fold=$FOLD" >> "$SWEEP_LOG"' EXIT ERR TERM
echo "[$(date)] Sweep started: $EXP_SUBDIR (PID=$$)"

TOTAL=$(( ${#CULTURES[@]} * K ))
COUNT=0
SKIP=0

for CULTURE in "${CULTURES[@]}"; do
for FOLD in $(seq 0 $((K-1))); do
    COUNT=$((COUNT + 1))

    EXP_NAME="${CULTURE}_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${W_GROUNDED}_wn${W_NEUTRAL}_tau${TAU}_r${LORA_R}_nxm_fold${FOLD}_seed${SEED}"

    if [ -f "$OUTPUT/$EXP_SUBDIR/$EXP_NAME/results.json" ]; then
        SKIP=$((SKIP + 1))
        echo "[$COUNT/$TOTAL] SKIP $EXP_NAME"
        continue
    fi

    echo "[$COUNT/$TOTAL] RUN ${CULTURE} fold${FOLD} (N×M)"
    START_TIME=$(date +%s)

    python main.py \
        --culture "$CULTURE" --lang "$LANG" --model "$MODEL" \
        --seed $SEED --fold $FOLD --folds_root "$FOLDS_ROOT" \
        --pairing nxm \
        --exp_subdir "$EXP_SUBDIR" \
        --epochs $EPOCHS --pairs_per_batch $PAIRS_PER_BATCH \
        --pairs_per_category 200 \
        --grounded_loss soft_contrastive --contrastive_temperature $TAU \
        --neutral_loss "$NEUTRAL_LOSS" \
        --w_grounded $W_GROUNDED --w_neutral $W_NEUTRAL \
        --lora_r $LORA_R --lora_alpha $((LORA_R * 2)) \
        --gradient_method "$GRADIENT" --ref_update_steps $REF_UPDATE \
        --output_dir "$OUTPUT" --eval_steps 200 --log_steps 50 \
        2>&1 | tee "$OUTPUT/$EXP_SUBDIR/${CULTURE}_fold${FOLD}.log"
    EXIT_CODE=${PIPESTATUS[0]}
    ELAPSED=$(( $(date +%s) - START_TIME ))

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date)] FAILED: ${CULTURE} fold${FOLD} exit=$EXIT_CODE elapsed=${ELAPSED}s"
    else
        echo "[$(date)] OK: ${CULTURE} fold${FOLD} elapsed=${ELAPSED}s"
    fi
done; done

echo "=== Done! Skipped=$SKIP, Ran=$((TOTAL-SKIP)) ==="
trap - EXIT