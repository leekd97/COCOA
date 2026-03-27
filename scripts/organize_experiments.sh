#!/bin/bash
# Organize experiments/ into subdirectories
# Usage:
#   bash scripts/organize_experiments.sh          # dry run (preview only)
#   bash scripts/organize_experiments.sh --move   # actually move

cd "$(dirname "$0")/.."

DRY_RUN=true
if [ "$1" == "--move" ]; then
    DRY_RUN=false
fi

EXP_DIR="./experiments"
STD_DIR="$EXP_DIR/kfold_std"
PNORM_DIR="$EXP_DIR/kfold_pnorm"

STD_COUNT=0
PNORM_COUNT=0
SKIP_COUNT=0

for dir in "$EXP_DIR"/*/; do
    # Skip if it's already a subdirectory container (not an experiment)
    dirname=$(basename "$dir")
    
    # Skip known subdirectories
    case "$dirname" in
        kfold_std|kfold_pnorm|kfold_nxm|kfold_pnorm_g_only|kfold_pnorm_scaled_*)
            continue
            ;;
    esac
    
    # Only process fold experiments
    if [[ ! "$dirname" =~ fold[0-9] ]]; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi
    
    # Determine: pnorm or std?
    if [[ "$dirname" =~ pnorm ]]; then
        TARGET="$PNORM_DIR"
        PNORM_COUNT=$((PNORM_COUNT + 1))
    else
        TARGET="$STD_DIR"
        STD_COUNT=$((STD_COUNT + 1))
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY] $dirname → $TARGET/"
    else
        mkdir -p "$TARGET"
        mv "$dir" "$TARGET/"
        echo "[MOVED] $dirname → $TARGET/"
    fi
done

# Also move associated log files
for logfile in "$EXP_DIR"/*.log; do
    [ -f "$logfile" ] || continue
    logname=$(basename "$logfile")
    
    # Skip sweep-level logs
    if [[ "$logname" =~ ^_sweep ]]; then
        continue
    fi
    
    if [[ "$logname" =~ pnorm ]]; then
        TARGET="$PNORM_DIR"
    elif [[ "$logname" =~ fold ]]; then
        TARGET="$STD_DIR"
    else
        continue
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY] $logname → $TARGET/"
    else
        mkdir -p "$TARGET"
        mv "$logfile" "$TARGET/"
        echo "[MOVED] $logname → $TARGET/"
    fi
done

echo ""
echo "Summary: std=$STD_COUNT, pnorm=$PNORM_COUNT, skipped=$SKIP_COUNT"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "This was a DRY RUN. To actually move, run:"
    echo "  bash scripts/organize_experiments.sh --move"
fi