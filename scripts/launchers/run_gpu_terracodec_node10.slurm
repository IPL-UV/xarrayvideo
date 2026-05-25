#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=terracodec_node10
#SBATCH --output=terracodec_node10_%j.out
#SBATCH --error=terracodec_node10_%j.err
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --ntasks=1

set -euo pipefail

REPO_ROOT="/home/oscar/xarrayvideo"
ENV_PREFIX="/home/oscar/.conda/envs/xv_terracodec"
PYTHON_BIN="$ENV_PREFIX/bin/python"
RUNNER="$REPO_ROOT/scripts/run_terracodec_tests.py"

export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
source /opt/ohpc/pub/miniforge3/bin/activate "$ENV_PREFIX"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

COMMON_MODEL="${COMMON_MODEL:-terracodec_v1_tt_s2l2a}"
COMMON_DEVICE="${COMMON_DEVICE:-auto}"
COMMON_TILE_SIZE="${COMMON_TILE_SIZE:-256}"
COMMON_TIME_WINDOW="${COMMON_TIME_WINDOW:-4}"
COMMON_COMPRESSION_VALUES="${COMMON_COMPRESSION_VALUES:-20}"
COMMON_QUALITY="${COMMON_QUALITY:-8}"

CUSTOM_INPUT_PATH="${CUSTOM_INPUT_PATH:-}"
CUSTOM_INPUT_FORMAT="${CUSTOM_INPUT_FORMAT:-pickle}"
CUSTOM_BANDS="${CUSTOM_BANDS:-B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12}"
CUSTOM_TIME_RANGE="${CUSTOM_TIME_RANGE:-:16}"
CUSTOM_SPATIAL_CROP="${CUSTOM_SPATIAL_CROP:-0:256,0:256}"
CUSTOM_BAND_FILL_MODE="${CUSTOM_BAND_FILL_MODE:-mean}"
CUSTOM_OUTPUT_STEM="${CUSTOM_OUTPUT_STEM:-results/results_terracodec_custom_node10}"

DEEPEXTREMES_INPUT_PATH="${DEEPEXTREMES_INPUT_PATH:-$(find /data/databases/deepextremes_final -maxdepth 1 -type d -name '*.zarr' | sort | head -n 1)}"
DEEPEXTREMES_BANDS="${DEEPEXTREMES_BANDS:-B02,B03,B04,B05,B06,B07,B8A}"
DEEPEXTREMES_TIME_RANGE="${DEEPEXTREMES_TIME_RANGE:-:16}"
DEEPEXTREMES_SPATIAL_CROP="${DEEPEXTREMES_SPATIAL_CROP:-}"
DEEPEXTREMES_BAND_FILL_MODE="${DEEPEXTREMES_BAND_FILL_MODE:-mean}"
DEEPEXTREMES_OUTPUT_STEM="${DEEPEXTREMES_OUTPUT_STEM:-results/results_terracodec_deepextremes_node10}"

if [[ -z "$CUSTOM_INPUT_PATH" ]]; then
    for candidate in \
        "$REPO_ROOT/../cubos_julio/cubo1_pickle" \
        "$REPO_ROOT/cubos_julio/cubo1_pickle"; do
        if [[ -f "$candidate" ]]; then
            CUSTOM_INPUT_PATH="$candidate"
            break
        fi
    done
fi

if [[ -z "$CUSTOM_INPUT_PATH" ]]; then
    echo "CUSTOM_INPUT_PATH is not set and no local custom pickle was found." >&2
    echo "Pass it with: sbatch --export=ALL,CUSTOM_INPUT_PATH=/path/to/custom_cube[,CUSTOM_INPUT_FORMAT=netcdf|zarr|pickle] scripts/launchers/run_gpu_terracodec_node10.slurm" >&2
    exit 1
fi

if [[ -z "$DEEPEXTREMES_INPUT_PATH" ]]; then
    echo "Could not resolve a DeepExtremeCubes .zarr input under /data/databases/deepextremes_final" >&2
    exit 1
fi

run_case() {
    local label="$1"
    shift
    echo
    echo "[$(date '+%F %T')] Starting ${label}"
    echo "Command: $PYTHON_BIN $RUNNER $*"
    "$PYTHON_BIN" "$RUNNER" "$@"
    echo "[$(date '+%F %T')] Finished ${label}"
}

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

run_case \
    "custom" \
    --input-path "$CUSTOM_INPUT_PATH" \
    --input-format "$CUSTOM_INPUT_FORMAT" \
    --dataset generic \
    --bands "$CUSTOM_BANDS" \
    --time-range "$CUSTOM_TIME_RANGE" \
    --spatial-crop "$CUSTOM_SPATIAL_CROP" \
    --band-fill-mode "$CUSTOM_BAND_FILL_MODE" \
    --compression-values "$COMMON_COMPRESSION_VALUES" \
    --model "$COMMON_MODEL" \
    --quality "$COMMON_QUALITY" \
    --device "$COMMON_DEVICE" \
    --tile-size "$COMMON_TILE_SIZE" \
    --time-window "$COMMON_TIME_WINDOW" \
    --output-stem "$CUSTOM_OUTPUT_STEM"

DEEPEXTREMES_ARGS=(
    --input-path "$DEEPEXTREMES_INPUT_PATH"
    --input-format zarr
    --dataset deepextremes
    --bands "$DEEPEXTREMES_BANDS"
    --time-range "$DEEPEXTREMES_TIME_RANGE"
    --band-fill-mode "$DEEPEXTREMES_BAND_FILL_MODE"
    --compression-values "$COMMON_COMPRESSION_VALUES"
    --model "$COMMON_MODEL"
    --quality "$COMMON_QUALITY"
    --device "$COMMON_DEVICE"
    --tile-size "$COMMON_TILE_SIZE"
    --time-window "$COMMON_TIME_WINDOW"
    --output-stem "$DEEPEXTREMES_OUTPUT_STEM"
)

if [[ -n "$DEEPEXTREMES_SPATIAL_CROP" ]]; then
    DEEPEXTREMES_ARGS+=(--spatial-crop "$DEEPEXTREMES_SPATIAL_CROP")
fi

run_case "deepextremes" "${DEEPEXTREMES_ARGS[@]}"

echo
echo "Results written under $REPO_ROOT/results"