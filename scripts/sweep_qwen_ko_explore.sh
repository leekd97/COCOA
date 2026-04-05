#!/bin/bash
# Qwen KO: alpha_n=1.0 + drift tuning, fold 0
cd "$(dirname "$0")/.."
source ~/.bashrc && conda activate cocoa

GPU=0
MODEL="qwen3_8b"
MODEL_SHORT="qwen3-8b"

export CUDA_VISIBLE_DEVICES=$GPU

LANG="cu"; SEED=45; FOLD=0; TAU=1.0; LORA_R=16
NEUTRAL_LOSS="mse"; GRADIENT="goal_aware_pcgrad"
EPOCHS=15; PAIRS_PER_BATCH=16; REF_UPDATE=0
OUTPUT="./experiments"; FOLDS_ROOT="./dataset/folds"; PRIORS_ROOT="./dataset/priors"

# (tag, alpha_g, alpha_n, w_drift, w_g, w_n, mse_scale)
CONFIGS=(
    "ko_an1.0_d3:1.0:1.0:3:1.0:2.0:10.0"
    "ko_an1.0_d10:1.0:1.0:10:1.0:2.0:10.0"
)

TOTAL=${#CONFIGS[@]}
COUNT=0

echo "[$(date)] Qwen KO an=1.0 drift tuning: $TOTAL configs"

for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r TAG AG AN W_DRIFT W_GROUNDED W_NEUTRAL MSE_SCALE <<< "$CFG"
    COUNT=$((COUNT + 1))

    EXP_SUBDIR="explore_${TAG}"
    mkdir -p "$OUTPUT/$EXP_SUBDIR"

    EXP_NAME="ko_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${W_GROUNDED}_wn${W_NEUTRAL}_tau${TAU}_r${LORA_R}_pnorm-g${AG}-n${AN}_nxn_fold${FOLD}_seed${SEED}"

    if [ -f "$OUTPUT/$EXP_SUBDIR/$EXP_NAME/results.json" ]; then
        echo "[$COUNT/$TOTAL] SKIP $TAG"
        continue
    fi

    echo "[$COUNT/$TOTAL] $TAG: ag=$AG an=$AN drift=$W_DRIFT wg=$W_GROUNDED wn=$W_NEUTRAL scale=$MSE_SCALE"
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

    [ $? -eq 0 ] && echo "[$(date)] OK: $TAG ${ELAPSED}s" || echo "[$(date)] FAILED: $TAG"
done

echo "[$(date)] DONE"