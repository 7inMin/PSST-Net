#!/usr/bin/env bash

set -Eeuo pipefail

# Real-only PSST-Net training launcher. Paths and hyperparameters can be
# overridden with environment variables; additional train.py arguments may be
# appended directly to this script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/datasets}"
CAVE_PATH="${CAVE_PATH:-${DATA_ROOT}/CAVE_512_28}"
KAIST_PATH="${KAIST_PATH:-${DATA_ROOT}/KAIST_CVPR2021}"
MASK_PATH="${MASK_PATH:-${DATA_ROOT}/TSA_real_data/mask.mat}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/real/train_code/checkpoints}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-499}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-1250}"
WORKERS="${WORKERS:-4}"
CACHE_CUBES="${CACHE_CUBES:-1}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
SAVE_EVERY="${SAVE_EVERY:-10}"

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Missing dataset directory: $1" >&2
        exit 1
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing file: $1" >&2
        exit 1
    fi
}

require_dir "$CAVE_PATH"
require_dir "$KAIST_PATH"
require_file "$MASK_PATH"
mkdir -p "$OUTPUT_DIR"

echo "Repository:        $SCRIPT_DIR"
echo "Python:            $PYTHON_BIN"
echo "CAVE:              $CAVE_PATH"
echo "KAIST:             $KAIST_PATH"
echo "Real mask:         $MASK_PATH"
echo "Output:            $OUTPUT_DIR"
echo "GPU:               $GPU"
echo "Epochs:            $EPOCHS"
echo "Samples per epoch: $SAMPLES_PER_EPOCH"

export PYTHONUNBUFFERED=1
"$PYTHON_BIN" "$SCRIPT_DIR/real/train_code/train.py" \
    --cave-path "$CAVE_PATH" \
    --kaist-path "$KAIST_PATH" \
    --mask-path "$MASK_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --gpu "$GPU" \
    --epochs "$EPOCHS" \
    --samples-per-epoch "$SAMPLES_PER_EPOCH" \
    --batch-size 1 \
    --workers "$WORKERS" \
    --learning-rate "$LEARNING_RATE" \
    --patch-size 384 \
    --save-every "$SAVE_EVERY" \
    --cache-cubes "$CACHE_CUBES" \
    "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
