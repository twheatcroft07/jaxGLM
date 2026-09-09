#!/bin/bash
#SBATCH --job-name=jaxGLM_levelb
#SBATCH --partition=gpu_test
#SBATCH --account=bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/n/home02/twheatcroft/jaxGLM/logs/levelb_%j.out
set -euo pipefail
mkdir -p /n/home02/twheatcroft/jaxGLM/logs
export JAXGLM_DATA="${JAXGLM_DATA:-/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft}"   # reproduction artifacts; override to run off a copy
CACHE=/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/_caches
export PIP_CACHE_DIR=$CACHE/pip XDG_CACHE_HOME=$CACHE/xdg PYTHONNOUSERSITE=1
module purge || true
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate jaxGLM
cd /n/home02/twheatcroft/jaxGLM
python -u chantranupong/levelb_compare.py
