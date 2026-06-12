#!/bin/bash
#SBATCH --job-name=jaxGLM_verify
#SBATCH --partition=gpu_test
#SBATCH --account=bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/n/home02/twheatcroft/jaxGLM/logs/verify_%j.out
# Two checks in one GPU job (env jaxGLM):
#   1. significance calibration -- esp. the recalibrated permutation null (FPR ~ alpha?)
#   2. Chantranupong Level B step 2 -- jaxGLM reproduces lab OLS coef on their preprocessed data
set -euo pipefail
mkdir -p /n/home02/twheatcroft/jaxGLM/logs
CACHE=/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/_caches
export PIP_CACHE_DIR=$CACHE/pip XDG_CACHE_HOME=$CACHE/xdg PYTHONNOUSERSITE=1
module purge || true
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate jaxGLM
cd /n/home02/twheatcroft/jaxGLM

echo "############## 1. SIGNIFICANCE CALIBRATION ##############"
python -u test_significance_calibration.py || echo ">>> calibration test FAILED (see assert above)"

echo ""
echo "############## 2. CHANTRANUPONG LEVEL B (step 2) ##############"
python -u chantranupong/levelb_compare.py || echo ">>> level B compare FAILED"
echo "############## DONE ##############"
