#!/bin/bash

# ./example_batch.sh [NOTEBOOK] [MODEL] [DEVICE]
NOTEBOOK="${1:-example}"
MODEL="${2:-rcnn}"
DEVICE="${3:-0}"

DATASETS=("shift" "city")
SCENARIOS=("continual_tta" "gradual_tta")
METHOD="gita_engine"

ERRORS=()

uv run --no-sync jupyter nbconvert --to script ${NOTEBOOK}.ipynb

TOTAL=$((${#DATASETS[@]} * ${#SCENARIOS[@]}))
IDX=0

for DATASET in "${DATASETS[@]}"; do
    for SCENARIO in "${SCENARIOS[@]}"; do
        IDX=$((IDX + 1))
        echo "▶ [$IDX/$TOTAL] $METHOD | dataset=$DATASET | scenario=$SCENARIO"
        echo "----------------------------------------"
        if ! uv run --no-sync ${NOTEBOOK}.py --disable-datalog --dataset $DATASET --model $MODEL --method $METHOD --device $DEVICE --scenario $SCENARIO; then
            ERRORS+=("FAILED: method=$METHOD dataset=$DATASET scenario=$SCENARIO")
        fi
        echo "----------------------------------------"
    done
done

# Error summary
if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo "========================================"
    echo "FAILED RUNS (${#ERRORS[@]} total)"
    echo "========================================"
    for ERR in "${ERRORS[@]}"; do
        echo "  - $ERR"
    done
else
    echo ""
    echo "========================================"
    echo "All runs completed successfully!"
    echo "========================================"
fi
