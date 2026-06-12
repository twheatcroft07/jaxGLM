#!/bin/bash
#SBATCH --job-name=jaxGLM_autoc
#SBATCH --partition=gpu_test
#SBATCH --account=bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/n/home02/twheatcroft/jaxGLM/logs/autocorr_%j.out
set -euo pipefail
mkdir -p /n/home02/twheatcroft/jaxGLM/logs
CACHE=/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/_caches
export PIP_CACHE_DIR=$CACHE/pip XDG_CACHE_HOME=$CACHE/xdg PYTHONNOUSERSITE=1
module purge || true
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate jaxGLM
cd /n/home02/twheatcroft/jaxGLM
python -u test_autocorr_calibration.py
