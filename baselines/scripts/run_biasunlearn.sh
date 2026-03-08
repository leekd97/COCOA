#!/bin/bash
# =============================================================================
# BiasUnlearn Baseline — All cultures, COCOA best seeds
# =============================================================================
# Best seed per culture×model from COCOA results:
#   ko:45  ja:45  zh:42  hi:45  mr:42  ml:42  gu:42  vi:48  ur:42
# =============================================================================

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

# Culture-seed pairs (from COCOA best runs)
CULTURE_SEEDS=(
    "ar 45"
)

LANGS=("cu")
MODELS=("llama3_8b" "qwen3_8b")

# BiasUnlearn HP (paper defaults)
LR=5e-5
BETA=0.1
STER_WEIGHT=1.0
ANTI_WEIGHT=1.0
KL_WEIGHT=0.2
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
MAX_STEPS=500
BATCH_SIZE=1
WARMUP_STEPS=10
MAX_GRAD_NORM=1.0
GRAD_ACCUM=16
CBS_TARGET=50.0
CBS_THRESHOLD=3.0
EVAL_EVERY=50
LOG_EVERY=10
SAVE_EVERY=100
MAX_LENGTH=128
MIX_ANTI=false
MIX_RATIO=0.25

DATA_ROOT="./dataset/camellia/raw"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TOTAL=$((${#CULTURE_SEEDS[@]} * ${#MODELS[@]} * ${#LANGS[@]}))
DONE=0

echo "=============================================="
echo " BiasUnlearn Baseline: ${TOTAL} experiments"
echo "=============================================="

MIX_ANTI_FLAG=""
if [ "${MIX_ANTI}" = true ]; then
    MIX_ANTI_FLAG="--unlearn_mix_anti"
fi

for lang in "${LANGS[@]}"; do
for model in "${MODELS[@]}"; do
for pair in "${CULTURE_SEEDS[@]}"; do
    culture=$(echo $pair | cut -d' ' -f1)
    seed=$(echo $pair | cut -d' ' -f2)

    DONE=$((DONE + 1))
    echo ""
    echo "[${DONE}/${TOTAL}] culture=${culture} lang=${lang} model=${model} seed=${seed}"

    python -m baselines.run_baseline \
        --method biasunlearn \
        --culture "${culture}" \
        --lang "${lang}" \
        --model "${model}" \
        --seed "${seed}" \
        --data_root "${DATA_ROOT}" \
        --device "cuda:0" \
        --max_length "${MAX_LENGTH}" \
        --unlearn_max_steps "${MAX_STEPS}" \
        --unlearn_batch_size "${BATCH_SIZE}" \
        --unlearn_lr "${LR}" \
        --unlearn_warmup_steps "${WARMUP_STEPS}" \
        --unlearn_max_grad_norm "${MAX_GRAD_NORM}" \
        --unlearn_grad_accum "${GRAD_ACCUM}" \
        --unlearn_beta "${BETA}" \
        --unlearn_ster_weight "${STER_WEIGHT}" \
        --unlearn_anti_weight "${ANTI_WEIGHT}" \
        --unlearn_kl_weight "${KL_WEIGHT}" \
        --unlearn_use_lora \
        --unlearn_lora_r "${LORA_R}" \
        --unlearn_lora_alpha "${LORA_ALPHA}" \
        --unlearn_lora_dropout "${LORA_DROPOUT}" \
        --unlearn_cbs_target "${CBS_TARGET}" \
        --unlearn_cbs_threshold "${CBS_THRESHOLD}" \
        --unlearn_eval_every "${EVAL_EVERY}" \
        --unlearn_log_every "${LOG_EVERY}" \
        --unlearn_save_every "${SAVE_EVERY}" \
        --unlearn_mix_ratio "${MIX_RATIO}" \
        ${MIX_ANTI_FLAG}

done; done; done

echo ""
echo "=============================================="
echo " BiasUnlearn Complete (${TOTAL} experiments)"
echo " Results: baselines/results/biasunlearn/"
echo "=============================================="