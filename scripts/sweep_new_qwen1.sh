#!/bin/bash
# CoCoA K-Fold Sweep with Prior Normalization (Llama)
# Usage: tmux new -s kfold_pnorm 'bash scripts/sweep_kfold_pnorm_llama.sh'

cd "$(dirname "$0")/.."

source ~/.bashrc
conda activate cocoa

GPU=1
CULTURES=("ko" "ja" "zh" "vi" "ur")
MODEL="qwen3_8b"
MODEL_SHORT="qwen3-8b"
LANG="cu"
K=5
SEED=45

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
PRIORS_ROOT="./dataset/priors"
mkdir -p "$OUTPUT"

# Generate priors if not already done
if [ ! -f "$PRIORS_ROOT/${MODEL}/${CULTURES[0]}_${LANG}/entity_priors.json" ]; then
    echo "Generating entity priors for ${MODEL}..."
    python generate_priors.py --model $MODEL --lang $LANG --device cuda:0 --output_root $PRIORS_ROOT
fi

# Generate folds if not already done
if [ ! -d "$FOLDS_ROOT/seed${SEED}" ]; then
    echo "Generating ${K}-fold splits (seed=${SEED})..."
    python generate_folds.py --K $K --seed $SEED --lang $LANG --output_root $FOLDS_ROOT
fi

# Logging
SWEEP_LOG="$OUTPUT/_sweep_kfold_pnorm_${MODEL_SHORT}.log"
exec > >(tee -a "$SWEEP_LOG") 2>&1
trap 'echo "[$(date)] CRASH: culture=$CULTURE fold=$FOLD" >> "$SWEEP_LOG"' EXIT ERR TERM
echo "[$(date)] K-Fold + Prior Norm sweep started (PID=$$)"

TOTAL=$(( ${#CULTURES[@]} * K ))
COUNT=0
SKIP=0

echo "=== CoCoA K-Fold + PriorNorm Sweep (${MODEL_SHORT}, ${LANG}) ==="
echo "GPU: $GPU, K=$K, Seed=$SEED"
echo "Cultures: ${CULTURES[*]}"
echo "Total runs: $TOTAL"
echo "=========================================="

for CULTURE in "${CULTURES[@]}"; do
for FOLD in $(seq 0 $((K-1))); do
    COUNT=$((COUNT + 1))

    # pnorm in exp name
    EXP_NAME="${CULTURE}_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${W_GROUNDED}_wn${W_NEUTRAL}_tau${TAU}_r${LORA_R}_pnorm_fold${FOLD}_seed${SEED}"

    if [ -f "$OUTPUT/$EXP_NAME/results.json" ]; then
        SKIP=$((SKIP + 1))
        echo "[$COUNT/$TOTAL] SKIP (done) $EXP_NAME"
        continue
    fi

    echo "[$COUNT/$TOTAL] RUN ${CULTURE} fold${FOLD} (prior normalized)"
    START_TIME=$(date +%s)

    python main.py \
        --culture "$CULTURE" \
        --lang "$LANG" \
        --model "$MODEL" \
        --seed $SEED \
        --fold $FOLD \
        --folds_root "$FOLDS_ROOT" \
        --normalize_prior \
        --priors_root "$PRIORS_ROOT" \
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
        2>&1 | tee "$OUTPUT/${CULTURE}_${LANG}_${MODEL_SHORT}_pnorm_fold${FOLD}.log"
    EXIT_CODE=${PIPESTATUS[0]}
    ELAPSED=$(( $(date +%s) - START_TIME ))

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date)] FAILED: ${CULTURE} fold${FOLD} exit=$EXIT_CODE elapsed=${ELAPSED}s"
    else
        echo "[$(date)] OK: ${CULTURE} fold${FOLD} elapsed=${ELAPSED}s"
    fi
done; done

echo "=== Done! Total=$TOTAL, Skipped=$SKIP, Ran=$((TOTAL - SKIP)) ==="
echo "[$(date)] Sweep completed normally"
trap - EXIT