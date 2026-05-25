#!/bin/bash

#SBATCH --job-name=xv_codec_smoke
#SBATCH --output=xv_codec_smoke.log
#SBATCH --error=xv_codec_smoke_err.log
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --ntasks=1

source /opt/ohpc/pub/miniforge3/bin/activate xv
source /home/oscar/xarrayvideo/scripts/activate_codec_tools.sh
PYTHON=$(which python)

cd /home/oscar/xarrayvideo
echo "Job started: $(date)"
echo "Using vvenc/uavs3e/vtm QP ladder: 0,1,7,16,27,39"

# Keep the smoke job intentionally small: one lightweight dataset,
# one quality point, and only the external codecs we need to sanity-check.
for dataset in deepextremes; do
  echo "Fast smoke benchmark for ${dataset}"
  $PYTHON scripts/run_tests.py \
    --dataset "$dataset" \
    --rules_name smoke-vvc-avs3-vtm-fast-qplow \
    --codec_names vvenc,uavs3e,vtm \
    --sample_limit 1 \
    --quality_limit 1 \
    --skip_plot \
    --skip_latex \
    --debug > "${dataset}_smoke_vvc_avs3_vtm_fast_qplow.log" 2>&1
done

echo "Job finished: $(date)"