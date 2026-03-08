# Usage:
#   tmux new -s en 'bash scripts/sweep_en.sh'

cd "$(dirname "$0")/.."

GPU=3
CULTURES=("ko")                     # ko, zh, ja
MODELS=("llama3_8b" "qwen3_8b")    # llama3_8b, qwen3_8b
SEEDS=(42 43 45)                          # 42 123 456
LANG="en"

# Hyperparameters
W_GROUNDED=2.0
W_NEUTRAL=1.0
TAU=1.0             # contrastive temperature
LORA_R=16
NEUTRAL_LOSS="mse"
GRADIENT="goal_aware_pcgrad"
EPOCHS=15
PAIRS_PER_BATCH=16
REF_UPDATE=0         # 0=fixed ref

export CUDA_VISIBLE_DEVICES=$GPU
OUTPUT="./experiments"
mkdir -p "$OUTPUT"

# Count total
TOTAL=0
for c in "${CULTURES[@]}"; do
for m in "${MODELS[@]}"; do
for s in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
done; done; done

echo "=== CBMCD Sweep (${LANG}) ==="
echo "GPU: $GPU"
echo "Cultures: ${CULTURES[*]}"
echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "HP: wg=${W_GROUNDED} wn=${W_NEUTRAL} tau=${TAU} r=${LORA_R} loss=${NEUTRAL_LOSS}"
echo "Total runs: $TOTAL"
echo "=========================================="
echo ""

COUNT=0
for CULTURE in "${CULTURES[@]}"; do
for MODEL in "${MODELS[@]}"; do
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
done; done; done

echo "=== All $TOTAL runs complete! ==="