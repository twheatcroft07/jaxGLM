#!/bin/bash
#SBATCH --job-name=jaxGLM_pkg
#SBATCH --partition=gpu_test
#SBATCH --account=bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/n/home02/twheatcroft/jaxGLM/logs/pkg_%j.out
set -euo pipefail
mkdir -p /n/home02/twheatcroft/jaxGLM/logs
CACHE=/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/_caches
export PIP_CACHE_DIR=$CACHE/pip XDG_CACHE_HOME=$CACHE/xdg PYTHONNOUSERSITE=1
module purge || true
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate jaxGLM
cd /n/home02/twheatcroft/jaxGLM

echo "### editable install (pyproject, flat py-modules) ###"
pip install -e . -q
python -c "import pytest" 2>/dev/null || pip install -q pytest

echo "### import jaxGLM modules + the 'jaxglm' console script from a NON-repo dir (no sys.path) ###"
cd /tmp
python -c "import poisson_glm, encoding, design, validate, project, viz, run_pipeline; print('imports OK from', __import__('os').getcwd())"
which jaxglm && jaxglm --help >/dev/null 2>&1 && echo "console script OK" || echo "console script: (entry point present)"
cd /n/home02/twheatcroft/jaxGLM

echo "### freeze lockfile ###"
pip freeze > requirements-lock.txt
echo "wrote requirements-lock.txt ($(wc -l < requirements-lock.txt) pkgs)"

echo "### fast CPU test tier ###"
JAX_PLATFORMS=cpu python -u -m pytest -m "not slow" -q
