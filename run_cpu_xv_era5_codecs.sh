#!/bin/bash

#SBATCH --job-name=xv_era5_codecs
#SBATCH --output=xv_era5_codecs.log
#SBATCH --error=xv_era5_codecs_err.log
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=260G
#SBATCH --ntasks=1

source /opt/ohpc/pub/miniforge3/bin/activate xv
source /home/oscar/xarrayvideo/scripts/activate_codec_tools.sh
PYTHON=$(which python)

cd /home/oscar/xarrayvideo
echo "Job started: $(date)"
echo "Running new codecs only: vvenc,uavs3e"
echo "Using vvenc/uavs3e QP ladder: 0,1,7,16,27,39"

$PYTHON scripts/run_tests.py --dataset era5 --rules_name era5-codecs-vvc-avs3-only-qplow --codec_names vvenc,uavs3e > era5-codecs-vvc-avs3-only-qplow.log 2>&1

echo "Job finished: $(date)"