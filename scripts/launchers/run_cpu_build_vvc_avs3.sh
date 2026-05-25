#!/bin/bash

#SBATCH --job-name=build_vvc_avs3
#SBATCH --output=build_vvc_avs3.log
#SBATCH --error=build_vvc_avs3_err.log
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --ntasks=1

set -euo pipefail

source /opt/ohpc/pub/miniforge3/bin/activate xv
source /etc/profile >/dev/null 2>&1 || true
module load gnu9/9.4.0 cmake/3.21.3 || true

cd /home/oscar/xarrayvideo
echo "Job started: $(date)"

bash scripts/install_vvc_avs3_codecs.sh "$HOME/codec_toolchains"

echo "Job finished: $(date)"