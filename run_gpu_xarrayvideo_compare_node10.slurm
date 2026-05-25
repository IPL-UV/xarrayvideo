#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=xarrayvideo_cmp_n10
#SBATCH --output=xarrayvideo_cmp_node10_%j.out
#SBATCH --error=xarrayvideo_cmp_node10_%j.err
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --ntasks=1

set -euo pipefail

REPO_ROOT="/home/oscar/xarrayvideo"
ENV_PREFIX="/home/oscar/.conda/envs/xv"
PYTHON_BIN="$ENV_PREFIX/bin/python"
RUNNER="$REPO_ROOT/scripts/run_xarrayvideo_single_cube.py"

export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set +u
source /opt/ohpc/pub/miniforge3/bin/activate xv
set -u
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

COMMON_CODEC="${COMMON_CODEC:-libx265}"
COMMON_BITS="${COMMON_BITS:-12}"
COMMON_QUALITY_INDICES="${COMMON_QUALITY_INDICES:-0,1,2,3,4,5}"

CUSTOM_INPUT_PATH="${CUSTOM_INPUT_PATH:-}"
CUSTOM_INPUT_FORMAT="${CUSTOM_INPUT_FORMAT:-pickle}"
CUSTOM_BANDS="${CUSTOM_BANDS:-B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12}"
CUSTOM_TIME_RANGE="${CUSTOM_TIME_RANGE:-:16}"
CUSTOM_SPATIAL_CROP="${CUSTOM_SPATIAL_CROP:-0:256,0:256}"
CUSTOM_OUTPUT_PREFIX="${CUSTOM_OUTPUT_PREFIX:-results/results_xarrayvideo_libx265_custom}"

DEEPEXTREMES_INPUT_PATH="${DEEPEXTREMES_INPUT_PATH:-$(find /data/databases/deepextremes_final -maxdepth 1 -type d -name '*.zarr' | sort | head -n 1)}"
DEEPEXTREMES_BANDS="${DEEPEXTREMES_BANDS:-B02,B03,B04,B05,B06,B07,B8A}"
DEEPEXTREMES_TIME_RANGE="${DEEPEXTREMES_TIME_RANGE:-:16}"
DEEPEXTREMES_SPATIAL_CROP="${DEEPEXTREMES_SPATIAL_CROP:-}"
DEEPEXTREMES_OUTPUT_PREFIX="${DEEPEXTREMES_OUTPUT_PREFIX:-results/results_xarrayvideo_libx265_deepextremes}"

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
    echo "Pass it with: sbatch --export=ALL,CUSTOM_INPUT_PATH=/path/to/custom_cube[,CUSTOM_INPUT_FORMAT=netcdf|zarr|pickle] run_gpu_xarrayvideo_compare_node10.slurm" >&2
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

IFS=',' read -r -a QUALITY_LIST <<< "$COMMON_QUALITY_INDICES"

for quality_index in "${QUALITY_LIST[@]}"; do
    run_case \
        "custom q${quality_index}" \
        --input-path "$CUSTOM_INPUT_PATH" \
        --input-format "$CUSTOM_INPUT_FORMAT" \
        --dataset generic \
        --bands "$CUSTOM_BANDS" \
        --time-range "$CUSTOM_TIME_RANGE" \
        --spatial-crop "$CUSTOM_SPATIAL_CROP" \
        --codec "$COMMON_CODEC" \
        --quality-index "$quality_index" \
        --bits "$COMMON_BITS" \
        --output-stem "${CUSTOM_OUTPUT_PREFIX}_q${quality_index}"
done

for quality_index in "${QUALITY_LIST[@]}"; do
    DEEPEXTREMES_ARGS=(
        --input-path "$DEEPEXTREMES_INPUT_PATH"
        --input-format zarr
        --dataset deepextremes
        --bands "$DEEPEXTREMES_BANDS"
        --time-range "$DEEPEXTREMES_TIME_RANGE"
        --codec "$COMMON_CODEC"
        --quality-index "$quality_index"
        --bits "$COMMON_BITS"
        --output-stem "${DEEPEXTREMES_OUTPUT_PREFIX}_q${quality_index}"
    )

    if [[ -n "$DEEPEXTREMES_SPATIAL_CROP" ]]; then
        DEEPEXTREMES_ARGS+=(--spatial-crop "$DEEPEXTREMES_SPATIAL_CROP")
    fi

    run_case "deepextremes q${quality_index}" "${DEEPEXTREMES_ARGS[@]}"
done

echo
echo "Results written under $REPO_ROOT/results"