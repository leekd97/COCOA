#!/bin/bash
# PCGrad Ablation Study
# Runs weighted and pcgrad methods using same best seeds as COCOA (goal_aware_pcgrad)
# goal_aware_pcgrad results already exist, so only 2 methods × 10 cultures × 2 models = 40 runs

cd "$(dirname "$0")/.."

GPU=2
LANG="cu"
METHODS=("pcgrad")   # goal_aware_pcgrad already done

# Hyperparameters (identical to main COCOA experiments)
W_GROUNDED=1.0
W_NEUTRAL=2.0
TAU=1.0
LORA_R=16
NEUTRAL_LOSS="mse"
EPOCHS=15
PAIRS_PER_BATCH=16
REF_UPDATE=0

export CUDA_VISIBLE_DEVICES=$GPU
OUTPUT="./experiments"
mkdir -p "$OUTPUT"

# Best seeds per culture×model (from main experiments)
# Format: run CULTURE MODEL SEED
RUNS=(
    "ar llama3_8b 45"
    "ar qwen3_8b 45"
    "gu llama3_8b 42"
    "gu qwen3_8b 42"
    "hi llama3_8b 45"
    "hi qwen3_8b 45"
    "ja llama3_8b 45"
    "ja qwen3_8b 45"
    "ko llama3_8b 45"
    "ko qwen3_8b 45"
    "ml llama3_8b 42"
    "ml qwen3_8b 42"
    "mr llama3_8b 42"
    "mr qwen3_8b 42"
    "ur llama3_8b 42"
    "ur qwen3_8b 42"
    "vi llama3_8b 48"
    "vi qwen3_8b 48"
    "zh llama3_8b 42"
    "zh qwen3_8b 42"
)

TOTAL=$(( ${#RUNS[@]} * ${#METHODS[@]} ))
COUNT=0

echo "=== PCGrad Ablation Study ==="
echo "GPU: $GPU"
echo "Methods: ${METHODS[*]}"
echo "HP: wg=${W_GROUNDED} wn=${W_NEUTRAL} tau=${TAU} r=${LORA_R} loss=${NEUTRAL_LOSS}"
echo "Total runs: $TOTAL"
echo "=========================================="
echo ""

for METHOD in "${METHODS[@]}"; do
    echo ">>> Method: $METHOD"
    for RUN in "${RUNS[@]}"; do
        read -r CULTURE MODEL SEED <<< "$RUN"
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] ${CULTURE}_${MODEL}_${METHOD}_seed${SEED}"

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
            --gradient_method "$METHOD" \
            --ref_update_steps $REF_UPDATE \
            --output_dir "$OUTPUT" \
            --eval_steps 200 \
            --log_steps 50 \
            2>&1 | tee "$OUTPUT/ablation_${CULTURE}_${LANG}_${MODEL}_${METHOD}_seed${SEED}.log"

        echo ""
    done
done

echo "=== All $TOTAL runs complete! ==="