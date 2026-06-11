#!/bin/bash
#SBATCH --job-name=nwb_install
#SBATCH --partition=gpu_test
#SBATCH --account=bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/n/home02/twheatcroft/jaxGLM/logs/nwb_install_%j.out
# Build `nwb` env (dandi + pynwb + sci stack) for the Chantranupong photometry reproduction,
# and probe DANDI 001767: list assets, read one WT* session's photometry + trial events.
# No GPU needed (CPU only); gpu_test used per the big-install policy.

set -uo pipefail
mkdir -p /n/home02/twheatcroft/jaxGLM/logs
SCR=/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft
export CONDA_PKGS_DIRS=$SCR/_caches/conda_pkgs PIP_CACHE_DIR=$SCR/_caches/pip XDG_CACHE_HOME=$SCR/_caches/xdg
export DANDI_CACHE=$SCR/_caches/dandi
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$SCR/chantranupong"

module purge || true
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
ENV=nwb
conda env list | grep -q "/envs/$ENV" || conda create -y -n "$ENV" python=3.11
conda activate "$ENV"
export PYTHONNOUSERSITE=1

echo "=== install dandi + pynwb + sci stack ==="
pip install --upgrade dandi pynwb h5py numpy scipy pandas scikit-learn || { echo "INSTALL FAILED"; exit 1; }

echo "=== probe DANDI 001767 ==="
python - <<'PY'
from dandi.dandiapi import DandiAPIClient
with DandiAPIClient() as c:
    ds = c.get_dandiset("001767", "draft")
    assets = list(ds.get_assets())
    print("n assets:", len(assets))
    for a in assets[:8]:
        print(f"  {a.path}  ({a.size/1e6:.1f} MB)")
PY
echo "=== done ==="
