#!/bin/bash

# ./example_batch.sh [NOTEBOOK] [DATASET] [MODEL] [DEVICE]
NOTEBOOK="${1:-example}"
DATASET="${2:-shift}"
MODEL="${3:-rcnn}"
DEVICE="${4:-0}"

METHODS=("dua_engine" "act_mad_engine" "mean_teacher_engine" "te_st_engine" "whw_engine" "pit_engine")

ERRORS=()

uv run --no-sync jupyter nbconvert --to script ${NOTEBOOK}.ipynb

# ----------------------------------------
# norm_engine: --adapt-batch 4 --total-rounds 1
# ----------------------------------------
echo "▶ [1/8] norm_engine | adapt-batch=4  total-rounds=1"
echo "----------------------------------------"
if ! uv run --no-sync ${NOTEBOOK}.py --disable-datalog --dataset $DATASET --model $MODEL --method norm_engine --device $DEVICE --adapt-batch 4 --total-rounds 1; then
    ERRORS+=("FAILED: method=norm_engine --adapt-batch 4 --total-rounds 1")
fi
echo "----------------------------------------"

# ----------------------------------------
# norm_engine: --adapt-batch 1
# ----------------------------------------
echo "▶ [2/8] norm_engine | adapt-batch=1"
echo "----------------------------------------"
if ! uv run --no-sync ${NOTEBOOK}.py --disable-datalog --dataset $DATASET --model $MODEL --method norm_engine --device $DEVICE --adapt-batch 1 --total-rounds 1; then
    ERRORS+=("FAILED: method=norm_engine --adapt-batch 1 --total-rounds 1")
fi
echo "----------------------------------------"

TOTAL=${#METHODS[@]}
for i in "${!METHODS[@]}"; do
    METHOD="${METHODS[$i]}"
    IDX=$((i + 3))
    echo "▶ [$IDX/$((TOTAL + 2))] $METHOD"
    echo "----------------------------------------"
    if ! uv run --no-sync ${NOTEBOOK}.py --disable-datalog --dataset $DATASET --model $MODEL --method $METHOD --device $DEVICE; then
        ERRORS+=("FAILED: method=$METHOD")
    fi
    echo "----------------------------------------"
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
