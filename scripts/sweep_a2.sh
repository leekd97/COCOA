#!/bin/bash
# Aya Expanse 8B: Group 2 — hi,ml,mr,gu,ar, 5-fold
cd "$(dirname "$0")/.."
source ~/.bashrc && conda activate cocoa

GPU=1
MODEL="aya_8b"
MODEL_SHORT="aya-8b"

export CUDA_VISIBLE_DEVICES=$GPU

CULTURES=("hi" "ml" "mr" "gu" "ar")
LANG="cu"; SEED=45; K=5
AG=1.0; AN=0.3; W_DRIFT=1.0
W_GROUNDED=1.0; W_NEUTRAL=2.0; TAU=1.0; LORA_R=16
NEUTRAL_LOSS="mse"; GRADIENT="goal_aware_pcgrad"
EPOCHS=15; PAIRS_PER_BATCH=16; REF_UPDATE=0; MSE_SCALE=10.0
OUTPUT="./experiments"; FOLDS_ROOT="./dataset/folds"; PRIORS_ROOT="./dataset/priors"
EXP_SUBDIR="sweep_drift_1.0"

TOTAL=$(( ${#CULTURES[@]} * K ))
COUNT=0

echo "[$(date)] Aya Group2: ${CULTURES[*]}, K=$K"

for CULTURE in "${CULTURES[@]}"; do
for FOLD in $(seq 0 $((K-1))); do
    COUNT=$((COUNT + 1))
    EXP_NAME="${CULTURE}_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${W_GROUNDED}_wn${W_NEUTRAL}_tau${TAU}_r${LORA_R}_pnorm-g${AG}-n${AN}_nxn_fold${FOLD}_seed${SEED}"

    if [ -f "$OUTPUT/$EXP_SUBDIR/$EXP_NAME/results.json" ]; then
        echo "[$COUNT/$TOTAL] SKIP $CULTURE fold$FOLD"
        continue
    fi

    echo "[$COUNT/$TOTAL] RUN $CULTURE fold$FOLD ($MODEL_SHORT)"
    START_TIME=$(date +%s)

    python main.py \
        --culture "$CULTURE" --lang "$LANG" --model "$MODEL" \
        --seed $SEED --fold $FOLD --folds_root "$FOLDS_ROOT" \
        --normalize_prior --priors_root "$PRIORS_ROOT" \
        --prior_alpha_g $AG --prior_alpha_n $AN \
        --pairing nxn --mse_scale $MSE_SCALE \
        --contrastive_temperature $TAU --w_drift $W_DRIFT \
        --exp_subdir "$EXP_SUBDIR" \
        --epochs $EPOCHS --pairs_per_batch $PAIRS_PER_BATCH --pairs_per_category 200 \
        --grounded_loss soft_contrastive \
        --neutral_loss "$NEUTRAL_LOSS" --w_grounded $W_GROUNDED --w_neutral $W_NEUTRAL \
        --lora_r $LORA_R --lora_alpha $((LORA_R * 2)) \
        --gradient_method "$GRADIENT" --ref_update_steps $REF_UPDATE \
        --output_dir "$OUTPUT" --eval_steps 200 --log_steps 50 \
        2>&1 | tee "$OUTPUT/$EXP_SUBDIR/${CULTURE}_${MODEL_SHORT}_fold${FOLD}.log"
    ELAPSED=$(( $(date +%s) - START_TIME ))

    [ $? -eq 0 ] && echo "[$(date)] OK: $CULTURE fold$FOLD ${ELAPSED}s" || echo "[$(date)] FAILED: $CULTURE fold$FOLD"
done; done

echo "[$(date)] DONE: $TOTAL runs"