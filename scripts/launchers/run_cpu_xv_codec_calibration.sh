#!/bin/bash

#SBATCH --job-name=xv_codec_calib
#SBATCH --output=xv_codec_calib.log
#SBATCH --error=xv_codec_calib_err.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --ntasks=1

source /opt/ohpc/pub/miniforge3/bin/activate xv
source /home/oscar/xarrayvideo/scripts/activate_codec_tools.sh
PYTHON=$(which python)

cd /home/oscar/xarrayvideo
echo "Job started: $(date)"
echo "Using vvenc/uavs3e QP ladder: 0,1,7,16,27,39"

for dataset in custom dynamicearthnet deepextremes era5; do
  echo "Calibrating codec ladders for ${dataset}"
  $PYTHON scripts/run_tests.py \
    --dataset "$dataset" \
    --rules_name calib-vvc-avs3-qplow \
    --codec_names vvenc,uavs3e \
    --sample_limit 1 \
    --skip_plot \
    --skip_latex > "${dataset}_calib_vvc_avs3_qplow.log" 2>&1
done

echo "Job finished: $(date)"