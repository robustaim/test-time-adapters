#!/bin/bash

uv run jupyter nbconvert --to script example.ipynb

DATASET="shift"
MODEL="rcnn"
DEVICE=0

METHODS=("norm_engine" "dua_engine" "actmad_engine" "mean_teacher_engine" "test_engine" "whw_engine" "pit_engine")

ERRORS=()

echo "Running norm_engine with --adapt-batch 5"
if ! uv run example.py --dataset $DATASET --model $MODEL --method norm_engine --device $DEVICE --adapt-batch 5; then
    ERRORS+=("FAILED: method=norm_engine --adapt-batch 5")
fi

for METHOD in "${METHODS[@]}"; do
    echo "Running method: $METHOD"
    if ! uv run example.py --dataset $DATASET --model $MODEL --method $METHOD --device $DEVICE; then
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
