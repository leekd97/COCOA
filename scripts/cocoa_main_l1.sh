#!/bin/bash
cd "$(dirname "$0")/.."
source ~/.bashrc && conda activate cocoa


GPU=0
MODEL="llama3_8b"
MODEL_SHORT="llama3-8b"
# GPU=1; MODEL="qwen3_8b"; MODEL_SHORT="qwen3-8b"

CULTURES=("ko" "ja" "zh" "vi" "ur")
LANG="cu"; K=5; SEED=45

# Objective weights / scales
AG=1.0; AN=0.3                       # PMI scaling s (grounded / neutral)
W_GROUNDED=1.0; W_NEUTRAL=2.0; W_DRIFT=1.0
TAU=1.0; KAPPA=10.0; LAMBDA=10.0
LORA_R=16
EPOCHS=15; PAIRS_PER_BATCH=16
NEUTRAL_LOSS="mse"                   # label only: for folder name / skip check
                                     # (main.py fixes the neutral loss to MSE)

# Quick smoke test
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
    CULTURES=("ko"); K=1; EPOCHS=2
    echo "[SMOKE] ko / fold0 / ${EPOCHS} epochs"
fi

export CUDA_VISIBLE_DEVICES=$GPU
OUTPUT="./experiments"; FOLDS_ROOT="./dataset/folds"; PRIORS_ROOT="./dataset/priors"
EXP_SUBDIR="main_table"

SWEEP_LOG="$OUTPUT/${EXP_SUBDIR}/_run_${MODEL_SHORT}.log"
mkdir -p "$OUTPUT/$EXP_SUBDIR"

echo "============================================================" >> "$SWEEP_LOG"
echo "[$(date)] START: $EXP_SUBDIR / $MODEL_SHORT / K=$K / GPU=$GPU / PID=$$" >> "$SWEEP_LOG"
echo "  ag=$AG an=$AN wg=$W_GROUNDED wn=$W_NEUTRAL w_drift=$W_DRIFT k=$KAPPA lam=$LAMBDA" >> "$SWEEP_LOG"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null >> "$SWEEP_LOG"

exec > >(tee -a "$SWEEP_LOG") 2>&1
trap 'CODE=$?; echo "[$(date)] TERMINATED sig=$CODE culture=$CULTURE fold=$FOLD" >> "$SWEEP_LOG"' EXIT ERR TERM INT HUP
echo "[$(date)] Main Table Run: $EXP_SUBDIR / $MODEL_SHORT / K=$K"

TOTAL=$(( ${#CULTURES[@]} * K ))
COUNT=0; SKIP=0; FAIL=0

for CULTURE in "${CULTURES[@]}"; do
for FOLD in $(seq 0 $((K-1))); do
    COUNT=$((COUNT + 1))
    EXP_NAME="${CULTURE}_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${W_GROUNDED}_wn${W_NEUTRAL}_tau${TAU}_r${LORA_R}_pnorm-g${AG}-n${AN}_nxn_fold${FOLD}_seed${SEED}"

    [ -f "$OUTPUT/$EXP_SUBDIR/$EXP_NAME/results.json" ] && { SKIP=$((SKIP+1)); echo "[$COUNT/$TOTAL] SKIP $CULTURE fold$FOLD"; continue; }

    echo "[$COUNT/$TOTAL] RUN $CULTURE fold$FOLD ($MODEL_SHORT)"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  gpu_mem: /'
    START_TIME=$(date +%s)

    python main.py \
        --culture "$CULTURE" --lang "$LANG" --model "$MODEL" \
        --seed $SEED --fold $FOLD --folds_root "$FOLDS_ROOT" \
        --priors_root "$PRIORS_ROOT" \
        --prior_alpha_g $AG --prior_alpha_n $AN \
        --pairing nxn \
        --k $KAPPA --lam $LAMBDA \
        --contrastive_temperature $TAU \
        --w_grounded $W_GROUNDED --w_neutral $W_NEUTRAL --w_drift $W_DRIFT \
        --exp_subdir "$EXP_SUBDIR" \
        --epochs $EPOCHS --pairs_per_batch $PAIRS_PER_BATCH --pairs_per_category 200 \
        --lora_r $LORA_R --lora_alpha $((LORA_R * 2)) \
        --output_dir "$OUTPUT" --eval_steps 200 --log_steps 50 \
        2>&1 | tee "$OUTPUT/$EXP_SUBDIR/${CULTURE}_${MODEL_SHORT}_fold${FOLD}.log"
    EXIT_CODE=${PIPESTATUS[0]}; ELAPSED=$(( $(date +%s) - START_TIME ))

    if [ $EXIT_CODE -eq 0 ]; then echo "[$(date)] OK: $CULTURE fold$FOLD ${ELAPSED}s"
    elif [ $EXIT_CODE -eq 137 ]; then FAIL=$((FAIL+1)); echo "[$(date)] ★ OOM (137): $CULTURE fold$FOLD"
    elif [ $EXIT_CODE -eq 139 ]; then FAIL=$((FAIL+1)); echo "[$(date)] ★ SEGFAULT (139): $CULTURE fold$FOLD"
    else FAIL=$((FAIL+1)); echo "[$(date)] FAILED ($EXIT_CODE): $CULTURE fold$FOLD ${ELAPSED}s"; fi
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  gpu_mem_after: /'
done; done

echo "[$(date)] DONE: Total=$TOTAL Skip=$SKIP Ran=$((TOTAL-SKIP)) Fail=$FAIL"
trap - EXIT