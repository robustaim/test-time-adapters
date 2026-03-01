#!/bin/bash

# ./example_batch.sh [NOTEBOOK] [DATASET] [MODEL] [DEVICE]
NOTEBOOK="${1:-example}"
DATASET="${2:-shift}"
MODEL="${3:-rcnn}"
DEVICE="${4:-0}"

METHODS=("dua_engine" "actmad_engine" "mean_teacher_engine" "test_engine" "whw_engine" "pit_engine")

ERRORS=()

uv run --no-sync jupyter nbconvert --to script ${NOTEBOOK}.ipynb

echo "Running norm_engine with --adapt-batch 4 --total-rounds 1"
if ! uv run --no-sync ${NOTEBOOK}.py --disable-datalog --dataset $DATASET --model $MODEL --method norm_engine --device $DEVICE --adapt-batch 4 --total-rounds 1; then
    ERRORS+=("FAILED: method=norm_engine --adapt-batch 4 --total-rounds 1")
fi

echo "Running norm_engine with --adapt-batch 1 --total-rounds 1"
if ! uv run --no-sync ${NOTEBOOK}.py --disable-datalog --dataset $DATASET --model $MODEL --method norm_engine --device $DEVICE --adapt-batch 1; then
    ERRORS+=("FAILED: method=norm_engine --adapt-batch 1 --total-rounds 1")
fi

for METHOD in "${METHODS[@]}"; do
    echo "Running method: $METHOD"
    if ! uv run --no-sync ${NOTEBOOK}.py --disable-datalog --dataset $DATASET --model $MODEL --method $METHOD --device $DEVICE; then
        ERRORS+=("FAILED: method=$METHOD")
    fi
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
    echo "All runs completed successfully!"
fi
