#!/bin/bash
# English (en) Sweep — Qwen on GPU 3
# 2 seeds (42, 45) × 9 cultures = 18 runs
# ar excluded: no English translation in CAMEL

cd "$(dirname "$0")/.."

GPU=3
LANG="en"
MODEL="qwen3_8b"

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
mkdir -p "$OUTPUT"

CULTURES=(ko ja zh hi mr ml gu ur vi)
SEEDS=(42 45)

TOTAL=$(( ${#CULTURES[@]} * ${#SEEDS[@]} ))
COUNT=0

echo "=== COCOA English Sweep — Qwen (GPU $GPU) ==="
echo "Cultures: ${CULTURES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $TOTAL"
echo "=============================================="

for CULTURE in "${CULTURES[@]}"; do
for SEED in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] ${CULTURE}_${LANG}_${MODEL}_seed${SEED}"

    python main.py \
        --culture "$CULTURE" \
        --lang "$LANG" \
        --model "$MODEL" \
        --seed "$SEED" \
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
        2>&1 | tee "$OUTPUT/${CULTURE}_${LANG}_${MODEL}_seed${SEED}.log"

    echo ""
done; done

echo "=== All $TOTAL runs complete! ==="