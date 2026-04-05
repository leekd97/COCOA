#!/bin/bash
# High Drift Sweep: overshooting cultures, fold 0
# KO: drift 10/15/20, HI+MR: drift 3/5
cd "$(dirname "$0")/.."
source ~/.bashrc && conda activate cocoa

# ==========================================================================
GPU=0
MODEL="llama3_8b"
MODEL_SHORT="llama3-8b"
# GPU=1; MODEL="qwen3_8b"; MODEL_SHORT="qwen3-8b"
# ==========================================================================

LANG="cu"; SEED=45; FOLD=0
AG=1.0; AN=0.3
W_GROUNDED=1.0; W_NEUTRAL=2.0; TAU=1.0; LORA_R=16
NEUTRAL_LOSS="mse"; GRADIENT="goal_aware_pcgrad"
EPOCHS=15; PAIRS_PER_BATCH=16; REF_UPDATE=0; MSE_SCALE=10.0

export CUDA_VISIBLE_DEVICES=$GPU
OUTPUT="./experiments"; FOLDS_ROOT="./dataset/folds"; PRIORS_ROOT="./dataset/priors"

# ★ Config: (culture, drift_value)
CONFIGS=(
    "ko:10"
    "ko:15"
    "ko:20"
    "hi:3"
    "hi:5"
    # "mr:3"
    # "mr:5"
    # "vi:0.3"
    # "vi:0.5"
)

TOTAL=${#CONFIGS[@]}
COUNT=0

echo "[$(date)] High Drift Sweep: ${MODEL_SHORT}, ${#CONFIGS[@]} configs"

for CFG in "${CONFIGS[@]}"; do
    CULTURE="${CFG%%:*}"
    W_DRIFT="${CFG##*:}"
    COUNT=$((COUNT + 1))

    EXP_SUBDIR="sweep_drift_${W_DRIFT}"
    mkdir -p "$OUTPUT/$EXP_SUBDIR"

    EXP_NAME="${CULTURE}_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${W_GROUNDED}_wn${W_NEUTRAL}_tau${TAU}_r${LORA_R}_pnorm-g${AG}-n${AN}_nxn_fold${FOLD}_seed${SEED}"

    if [ -f "$OUTPUT/$EXP_SUBDIR/$EXP_NAME/results.json" ]; then
        echo "[$COUNT/$TOTAL] SKIP $CULTURE drift=$W_DRIFT"
        continue
    fi

    echo ""
    echo "[$COUNT/$TOTAL] RUN $CULTURE drift=$W_DRIFT ($MODEL_SHORT)"
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

    if [ $EXIT_CODE -eq 0 ]; then echo "[$(date)] OK: $CULTURE drift=$W_DRIFT ${ELAPSED}s"
    else echo "[$(date)] FAILED ($EXIT_CODE): $CULTURE drift=$W_DRIFT"; fi
done

echo ""
echo "[$(date)] DONE: $TOTAL configs"