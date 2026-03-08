#!/bin/bash
# =============================================================================
# BiasEdit Baseline — All cultures, COCOA best seeds
# =============================================================================
# Best seed per culture×model from COCOA results:
#   ko:45  ja:45  zh:42  hi:45  mr:42  ml:42  gu:42  vi:48  ur:42
# =============================================================================

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2

# Culture-seed pairs (from COCOA best runs)
CULTURE_SEEDS=(
    "ar 45"
)

LANGS=("cu")
MODELS=("llama3_8b" "qwen3_8b")

# BiasEdit HP (paper defaults)
K=5
N_EDITS=2
META_LR=1e-4
RANK=1920
EPOCHS=10
LR=1e-6
N_BLOCKS=2
MAX_GRAD_NORM=1.0
CACHE_BATCH_SIZE=128
PATIENCE=5
EVAL_EVERY=2
MAX_LENGTH=128

DATA_ROOT="./dataset/camellia/raw"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TOTAL=$((${#CULTURE_SEEDS[@]} * ${#MODELS[@]} * ${#LANGS[@]}))
DONE=0

echo "=============================================="
echo " BiasEdit Baseline: ${TOTAL} experiments"
echo "=============================================="

for lang in "${LANGS[@]}"; do
for model in "${MODELS[@]}"; do
for pair in "${CULTURE_SEEDS[@]}"; do
    culture=$(echo $pair | cut -d' ' -f1)
    seed=$(echo $pair | cut -d' ' -f2)

    DONE=$((DONE + 1))
    echo ""
    echo "[${DONE}/${TOTAL}] culture=${culture} lang=${lang} model=${model} seed=${seed}"

    python -m baselines.run_baseline \
        --method biasedit \
        --culture "${culture}" \
        --lang "${lang}" \
        --model "${model}" \
        --seed "${seed}" \
        --data_root "${DATA_ROOT}" \
        --device "cuda:0" \
        --max_length "${MAX_LENGTH}" \
        --biasedit_k "${K}" \
        --biasedit_n_edits "${N_EDITS}" \
        --biasedit_epochs "${EPOCHS}" \
        --biasedit_rank "${RANK}" \
        --biasedit_n_blocks "${N_BLOCKS}" \
        --biasedit_lr "${LR}" \
        --biasedit_meta_lr "${META_LR}" \
        --biasedit_max_grad_norm "${MAX_GRAD_NORM}" \
        --biasedit_cache_batch_size "${CACHE_BATCH_SIZE}" \
        --biasedit_patience "${PATIENCE}" \
        --biasedit_eval_every "${EVAL_EVERY}"

done; done; done

echo ""
echo "=============================================="
echo " BiasEdit Complete (${TOTAL} experiments)"
echo " Results: baselines/results/biasedit/"
echo "=============================================="