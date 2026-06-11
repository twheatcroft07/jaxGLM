#!/bin/bash
#SBATCH --job-name=ibl_install
#SBATCH --partition=gpu_test
#SBATCH --account=bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/n/home02/twheatcroft/jaxGLM/logs/ibl_install_%j.out
# Build the `ibl` conda env (ONE-api + ibllib/brainbox + neurencoding + brainwidemap) and
# verify public-data access on a compute node. Submit: sbatch install_ibl_env.sh
# No GPU actually needed here (sklearn/CPU); gpu_test used per the big-install policy.

set -uo pipefail
mkdir -p /n/home02/twheatcroft/jaxGLM/logs

# All caches off $HOME, and the IBL ONE data cache on netscratch (it gets large).
SCR=/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft
export CONDA_PKGS_DIRS=$SCR/_caches/conda_pkgs
export PIP_CACHE_DIR=$SCR/_caches/pip
export XDG_CACHE_HOME=$SCR/_caches/xdg
export ONE_CACHE=$SCR/ibl_cache          # consumed by the probe below
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$ONE_CACHE"

module purge || true
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh

ENV=ibl
conda env list | grep -q "/envs/$ENV" || conda create -y -n "$ENV" python=3.11
conda activate "$ENV"
export PYTHONNOUSERSITE=1

echo "=== install core IBL stack ==="
pip install --upgrade ONE-api ibllib neurencoding || { echo "CORE INSTALL FAILED"; exit 1; }

echo "=== attempt brainwidemap (their encoding design-matrix code) ==="
# Non-fatal: if this fails we can build X via neurencoding's DesignMatrix directly.
pip install "git+https://github.com/int-brain-lab/paper-brain-wide-map" \
  && echo "brainwidemap OK" || echo "brainwidemap install FAILED (will fall back to neurencoding API)"

echo "=== verify imports + public ONE + a BWM session ==="
python - <<PY
import os
print("--- imports ---")
import numpy as np, pandas as pd
import one, neurencoding
print("one", one.__version__, "| neurencoding", getattr(neurencoding,"__version__","?"))
from one.api import ONE
one_ = ONE(base_url='https://openalyx.internationalbrainlab.org',
           silent=True, cache_dir=os.environ["ONE_CACHE"])
print("ONE connected; cache:", one_.cache_dir)

try:
    from brainwidemap import bwm_query
    df = bwm_query(one_)
    print("BWM insertions:", df.shape, "| cols:", list(df.columns)[:8])
    print("first pid:", df.iloc[0]["pid"], "| first eid:", df.iloc[0]["eid"])
except Exception as e:
    print("bwm_query unavailable:", repr(e))
    eids = one_.search(project='brainwide', dataset='spikes.times')
    print("fallback search, n sessions:", len(eids), "| first eid:", eids[0] if eids else None)
print("=== probe done ===")
PY
echo "=== job done ==="
