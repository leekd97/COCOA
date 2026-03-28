
#!/bin/bash
# Quick HP test: D/E/F configs — KO only, fold 0, sequential
# ==========================================================================
# ★ CONFIGURE
# ==========================================================================
GPU=0
MODEL="llama3_8b"
MODEL_SHORT="llama3-8b"
# GPU=1; MODEL="qwen3_8b"; MODEL_SHORT="qwen3-8b"
# ==========================================================================

CULTURES=("ko")
LANG="cu"; K=1; SEED=45
LORA_R=16; TAU=1.0
NEUTRAL_LOSS="mse"; PAIRS_PER_BATCH=16; REF_UPDATE=0

export CUDA_VISIBLE_DEVICES=$GPU
OUTPUT="./experiments"; FOLDS_ROOT="./dataset/folds"; PRIORS_ROOT="./dataset/priors"

# ==========================================================================
# ★ Three configs: D, E, F
#   NAME       α_g  α_n  w_g  w_n  epochs  gradient
# ==========================================================================
NAMES=(       "D_wn4"          "E_ep8"          "F_weighted"     )
AG_LIST=(     0.5              0.5              0.5              )
AN_LIST=(     0.3              0.3              0.3              )
WG_LIST=(     1.0              1.0              1.0              )
WN_LIST=(     4.0              2.0              2.0              )
EP_LIST=(     15               8                15               )
GRAD_LIST=(   "goal_aware_pcgrad" "goal_aware_pcgrad" "weighted" )

for IDX in "${!NAMES[@]}"; do
    TAG="${NAMES[$IDX]}"
    AG="${AG_LIST[$IDX]}"
    AN="${AN_LIST[$IDX]}"
    WG="${WG_LIST[$IDX]}"
    WN="${WN_LIST[$IDX]}"
    EPOCHS="${EP_LIST[$IDX]}"
    GRADIENT="${GRAD_LIST[$IDX]}"

    EXP_SUBDIR="quick_${TAG}"

    SWEEP_LOG="$OUTPUT/${EXP_SUBDIR}/_sweep_${MODEL_SHORT}.log"
    mkdir -p "$OUTPUT/$EXP_SUBDIR"

    echo "============================================================" >> "$SWEEP_LOG"
    echo "[$(date)] START: $TAG / $MODEL_SHORT / GPU=$GPU" >> "$SWEEP_LOG"
    echo "  α_g=$AG α_n=$AN w_g=$WG w_n=$WN epochs=$EPOCHS grad=$GRADIENT" >> "$SWEEP_LOG"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null >> "$SWEEP_LOG"

    exec > >(tee -a "$SWEEP_LOG") 2>&1
    trap 'CODE=$?; echo "[$(date)] TERMINATED sig=$CODE config=$TAG culture=$CULTURE fold=$FOLD" >> "$SWEEP_LOG"' EXIT ERR TERM INT HUP

    echo ""
    echo "============================================================"
    echo "  [$TAG] α_g=$AG α_n=$AN w_g=$WG w_n=$WN ep=$EPOCHS grad=$GRADIENT ($MODEL_SHORT)"
    echo "============================================================"

    for CULTURE in "${CULTURES[@]}"; do
    for FOLD in $(seq 0 $((K-1))); do

        # Build exp name based on gradient method
        if [ "$GRADIENT" == "goal_aware_pcgrad" ]; then
            GRAD_TAG=""
        else
            GRAD_TAG="_${GRADIENT}"
        fi
        EXP_NAME="${CULTURE}_${LANG}_${MODEL_SHORT}_${NEUTRAL_LOSS}_wg${WG}_wn${WN}_tau${TAU}_r${LORA_R}_pnorm-g${AG}-n${AN}_nxn${GRAD_TAG}_fold${FOLD}_seed${SEED}"

        [ -f "$OUTPUT/$EXP_SUBDIR/$EXP_NAME/results.json" ] && { echo "SKIP $EXP_NAME"; continue; }

        echo "RUN ${CULTURE} fold${FOLD} [$TAG]"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  gpu_mem: /'
        START_TIME=$(date +%s)

        python main.py \
            --culture "$CULTURE" --lang "$LANG" --model "$MODEL" \
            --seed $SEED --fold $FOLD --folds_root "$FOLDS_ROOT" \
            --normalize_prior --priors_root "$PRIORS_ROOT" \
            --prior_alpha_g $AG --prior_alpha_n $AN \
            --pairing nxn \
            --exp_subdir "$EXP_SUBDIR" \
            --epochs $EPOCHS --pairs_per_batch $PAIRS_PER_BATCH --pairs_per_category 200 \
            --grounded_loss soft_contrastive --contrastive_temperature $TAU \
            --neutral_loss "$NEUTRAL_LOSS" --w_grounded $WG --w_neutral $WN \
            --lora_r $LORA_R --lora_alpha $((LORA_R * 2)) \
            --gradient_method "$GRADIENT" --ref_update_steps $REF_UPDATE \
            --output_dir "$OUTPUT" --eval_steps 200 --log_steps 50 \
            2>&1 | tee "$OUTPUT/$EXP_SUBDIR/${CULTURE}_${MODEL_SHORT}_fold${FOLD}.log"
        EXIT_CODE=${PIPESTATUS[0]}; ELAPSED=$(( $(date +%s) - START_TIME ))

        if [ $EXIT_CODE -eq 0 ]; then echo "[$(date)] OK: [$TAG] ${CULTURE} fold${FOLD} ${ELAPSED}s"
        elif [ $EXIT_CODE -eq 137 ]; then echo "[$(date)] ★ OOM (137): [$TAG] ${CULTURE} fold${FOLD}"
        elif [ $EXIT_CODE -eq 139 ]; then echo "[$(date)] ★ SEGFAULT (139): [$TAG] ${CULTURE} fold${FOLD}"
        else echo "[$(date)] FAILED ($EXIT_CODE): [$TAG] ${CULTURE} fold${FOLD} ${ELAPSED}s"; fi
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  gpu_mem_after: /'
    done; done

    echo "[$(date)] DONE: $TAG"
    echo ""
done

trap - EXIT
echo "[$(date)] ALL CONFIGS (D/E/F) COMPLETE"