#!/bin/bash
#SBATCH --job-name=jaxGLM_solver
#SBATCH --partition=gpu_test
#SBATCH --account=bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/n/home02/twheatcroft/jaxGLM/logs/solver_%j.out
set -euo pipefail
mkdir -p /n/home02/twheatcroft/jaxGLM/logs
CACHE=/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/_caches
export PIP_CACHE_DIR=$CACHE/pip XDG_CACHE_HOME=$CACHE/xdg PYTHONNOUSERSITE=1
module purge || true
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate jaxGLM
cd /n/home02/twheatcroft/jaxGLM
echo "### synthetic correctness (monotone restart must not break convergence) ###"
python -u test_synthetic.py || echo ">>> test_synthetic FAILED"
echo ""; echo "### solver robustness (no divergence on raw collinear design) ###"
python -u test_solver_robustness.py || echo ">>> robustness FAILED"
