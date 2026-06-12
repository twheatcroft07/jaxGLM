#!/bin/bash
#SBATCH --job-name=jaxGLM_install
#SBATCH --partition=gpu_test
#SBATCH --account=bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/n/home02/twheatcroft/jaxGLM/logs/install_%j.out
# Build the jaxGLM conda env on a GPU node (NOT login). Submit with:
#   sbatch install_env.sh
# then check logs/install_<jobid>.out and confirm `import jax; jax.devices()` lists a GPU.

set -euo pipefail
mkdir -p /n/home02/twheatcroft/jaxGLM/logs

# Keep all package caches off $HOME (per cache-location policy).
CACHE=/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/_caches
export CONDA_PKGS_DIRS=$CACHE/conda_pkgs
export PIP_CACHE_DIR=$CACHE/pip
export XDG_CACHE_HOME=$CACHE/xdg
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

module purge || true
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh

ENV=jaxGLM
if ! conda env list | grep -q "/envs/$ENV"; then
  conda create -y -n "$ENV" python=3.11
fi
conda activate "$ENV"
export PYTHONNOUSERSITE=1

# jax with bundled CUDA 12 wheels (self-contained; no cluster CUDA modules needed).
pip install --upgrade "jax[cuda12]" optax matplotlib

# install jaxGLM itself (editable) so its modules import from anywhere -- this is what lets the
# reproduction scripts `import poisson_glm` without sys.path hacks. Pulls numpy/scipy/pandas/pyyaml.
pip install -e /n/home02/twheatcroft/jaxGLM

echo "=== sanity ==="
python - <<'PY'
import jax, jax.numpy as jnp
print("jax", jax.__version__)
print("devices", jax.devices())
x = jnp.ones((1024, 1024))
print("matmul ok, trace:", float((x @ x).trace()))
PY
echo "=== done ==="
