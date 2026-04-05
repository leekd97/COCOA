#!/bin/bash
# Qwen KO: alpha_n=1.0, drift=3, 5-fold
cd "$(dirname "$0")/.."
source ~/.bashrc && conda activate cocoa

GPU=0
MODEL="qwen3_8b"
MODEL_SHORT="qwen3-8b"

export CUDA_VISIBLE_DEVICES=$GPU

LANG="cu"; SEED=45; K=5
AG=1.0; AN=1.0; W_DRIFT=3
W_GROUNDED=1.0; W_NEUTRAL=2.0; TAU=1.0; LORA_R=16; MSE_SCALE=10.0
NEUTRAL_LOSS="mse"; GRADIENT="goal_aware_pcgrad"
EPOCHS=15; PAIRS_PER_BATCH=16; REF_UPDATE=0
OUTPUT="./experiments"; FOLDS_ROOT="./dataset/folds"; PRIORS_ROOT="./dataset/priors"

EXP_SUBDIR="explore_ko_an1.0_d3"
mkdir -p "$OUTPUT/$EXP_SUBDIR"

echo "[$(date)] Qwen KO: an=1.0 drift=3, 5-fold"

for FOLD in $(seq 0 $((K-1))); do
    EXP_NAME="ko_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${W_GROUNDED}_wn${W_NEUTRAL}_tau${TAU}_r${LORA_R}_pnorm-g${AG}-n${AN}_nxn_fold${FOLD}_seed${SEED}"

    if [ -f "$OUTPUT/$EXP_SUBDIR/$EXP_NAME/results.json" ]; then
        echo "SKIP fold$FOLD"
        continue
    fi

    echo "[$(date)] RUN ko fold$FOLD (an=1.0, drift=3)"
    START_TIME=$(date +%s)

    python main.py \
        --culture ko --lang "$LANG" --model "$MODEL" \
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
        2>&1 | tee "$OUTPUT/$EXP_SUBDIR/ko_${MODEL_SHORT}_fold${FOLD}.log"
    ELAPSED=$(( $(date +%s) - START_TIME ))

    [ $? -eq 0 ] && echo "[$(date)] OK: fold$FOLD ${ELAPSED}s" || echo "[$(date)] FAILED: fold$FOLD"
done

echo "[$(date)] DONE"