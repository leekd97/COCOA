#!/bin/bash
# COCOA Re-evaluation — affected cultures only
# Checkpoints exist, just re-run evaluation with fixed REVERSE_ALIAS
# GPU 2

cd "$(dirname "$0")/.."

GPU=2
export CUDA_VISIBLE_DEVICES=$GPU

echo "=== COCOA Re-evaluation (fixed REVERSE_ALIAS) ==="
echo "GPU: $GPU"
echo "Affected cultures: hi mr ml gu vi ur ar"
echo "=================================================="

python analysis/reeval.py \
    --method cocoa \
    --exp_dir experiments \
    --data_root ./dataset/camellia/raw \
    --cultures hi mr ml gu vi ur ar \
    --models llama qwen \
    --device cuda:0

echo "=== COCOA re-evaluation complete! ==="
echo "Run: python make_main_table.py to update table"