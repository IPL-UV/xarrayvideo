#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=terracodec_suite_node10
#SBATCH --output=terracodec_suite_node10_%j.out
#SBATCH --error=terracodec_suite_node10_%j.err
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:1
#SBATCH --time=6-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --ntasks=1

set -euo pipefail

REPO_ROOT="/home/oscar/xarrayvideo"
ENV_PREFIX="/home/oscar/.conda/envs/xv_terracodec"
PYTHON_BIN="$ENV_PREFIX/bin/python"
SUITE_RUNNER="$REPO_ROOT/scripts/run_terracodec_suite.py"

export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set +u
source /opt/ohpc/pub/miniforge3/bin/activate "$ENV_PREFIX"
set -u
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

SUITE_DATASETS="${SUITE_DATASETS:-custom,deepextremes}"
SUITE_OUTPUT_ROOT="${SUITE_OUTPUT_ROOT:-$REPO_ROOT/results/terracodec_suite}"
SUITE_COMPRESSION_VALUES="${SUITE_COMPRESSION_VALUES:-0.4,1,5,20,100,200,700}"
SUITE_MODEL="${SUITE_MODEL:-terracodec_v1_tt_s2l2a}"
SUITE_QUALITY="${SUITE_QUALITY:-8}"
SUITE_DEVICE="${SUITE_DEVICE:-auto}"
SUITE_TILE_SIZE="${SUITE_TILE_SIZE:-256}"
SUITE_TIME_WINDOW="${SUITE_TIME_WINDOW:-4}"
SUITE_METRICS_VALUE_RANGE="${SUITE_METRICS_VALUE_RANGE:-auto}"
SUITE_INPUT_SCALE="${SUITE_INPUT_SCALE:-auto}"
SUITE_REFLECTANCE_SCALE_FACTOR="${SUITE_REFLECTANCE_SCALE_FACTOR:-10000}"
SUITE_START_INDEX="${SUITE_START_INDEX:-0}"
SUITE_MAX_CUBES="${SUITE_MAX_CUBES:-}"
SUITE_CUBE_FILTER="${SUITE_CUBE_FILTER:-}"
SUITE_KEEP_GOING="${SUITE_KEEP_GOING:-1}"
SUITE_DRY_RUN="${SUITE_DRY_RUN:-0}"
SUITE_DX_ROOT="${SUITE_DX_ROOT:-/data/databases/deepextremes_final}"
SUITE_CUSTOM_ROOT="${SUITE_CUSTOM_ROOT:-}"
SUITE_DX_CUBES="${SUITE_DX_CUBES:-}"

ARGS=(
    --python-bin "$PYTHON_BIN"
    --runner "$REPO_ROOT/scripts/run_terracodec_tests.py"
    --datasets "$SUITE_DATASETS"
    --dx-root "$SUITE_DX_ROOT"
    --output-root "$SUITE_OUTPUT_ROOT"
    --compression-values "$SUITE_COMPRESSION_VALUES"
    --model "$SUITE_MODEL"
    --quality "$SUITE_QUALITY"
    --device "$SUITE_DEVICE"
    --tile-size "$SUITE_TILE_SIZE"
    --time-window "$SUITE_TIME_WINDOW"
    --metrics-value-range "$SUITE_METRICS_VALUE_RANGE"
    --input-scale "$SUITE_INPUT_SCALE"
    --reflectance-scale-factor "$SUITE_REFLECTANCE_SCALE_FACTOR"
    --start-index "$SUITE_START_INDEX"
)

if [[ -n "$SUITE_CUSTOM_ROOT" ]]; then
    ARGS+=(--custom-root "$SUITE_CUSTOM_ROOT")
fi

if [[ -n "$SUITE_DX_CUBES" ]]; then
    ARGS+=(--dx-cubes "$SUITE_DX_CUBES")
fi

if [[ -n "$SUITE_MAX_CUBES" ]]; then
    ARGS+=(--max-cubes "$SUITE_MAX_CUBES")
fi

if [[ -n "$SUITE_CUBE_FILTER" ]]; then
    ARGS+=(--cube-filter "$SUITE_CUBE_FILTER")
fi

if [[ "$SUITE_KEEP_GOING" == "1" ]]; then
    ARGS+=(--keep-going)
fi

if [[ "$SUITE_DRY_RUN" == "1" ]]; then
    ARGS+=(--dry-run)
fi

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Suite datasets: $SUITE_DATASETS"
echo "Compression values: $SUITE_COMPRESSION_VALUES"
nvidia-smi || true

"$PYTHON_BIN" "$SUITE_RUNNER" "${ARGS[@]}"