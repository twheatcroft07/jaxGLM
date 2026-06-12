"""Project scaffolding + config for the jaxGLM CLI (`run_pipeline.py`).

A jaxGLM project is a directory with a `config.yaml`, a `data/` folder of (already-binned) long-
format CSVs, and `results/` + `models/` for outputs -- the same shape as the Sabatini `sglm`
workflow, so the data contract is familiar:

    my_project/
      config.yaml
      data/      one or more CSVs, concatenated; each row = one time bin
      results/   kernels.png, reconstruction.png, summary.csv
      models/    fit.npz  (weights, intercepts, D2, alpha, importance, p-values)

Create one with `python run_pipeline.py --new NAME [PARENT_DIR]`, edit `config.yaml`, drop CSVs in
`data/`, then `python run_pipeline.py PATH/config.yaml`.

Data contract (per CSV row = one time bin, already on a uniform grid):
  - a session column (e.g. SessionName) -- groups bins so shift-kernels never cross session/trial
    boundaries (the boundary-aware shifting). Optional trial column refines grouping + CV folds.
  - predictor columns -- event indicators (0/1) or continuous regressors, named in config.
  - response columns -- one GLM is fit per response (a neuron / photometry channel).
"""
import os
import numpy as np

try:
    import yaml
except ImportError:                                       # pragma: no cover
    yaml = None


# Annotated template. Keys map directly onto design.build_design / encoding / poisson_glm options.
DEFAULT_CONFIG = """\
project:
  name: {name}
  path: {path}

data:
  glob: "data/*.csv"            # CSVs under the project, concatenated (must share columns)
  session_col: SessionName     # REQUIRED: groups bins for boundary-aware shifts
  trial_col: TrialNumber       # optional: refines grouping + enables trial-grouped CV folds
  time_col: null               # optional: column to sort each session by (else file order kept)
  predictors: [predictorA, predictorB]      # columns expanded into shift-kernels
  responses: [response_0]                    # columns to model (one GLM each)
  responses_prefix: null       # alternative to `responses`: auto-pick columns with this prefix

glm:
  family: gaussian             # poisson (counts) | gaussian (continuous, e.g. photometry)
  l1_ratio: 0.5                # 0=ridge .. 1=lasso ; between = elastic net
  alpha: cv                    # "cv" (CV-select per unit) | a float | a list to CV over
  shift_default: [-20, 20]     # inclusive BIN-lag window applied to every predictor
  shift_bounds: {{}}             # optional per-predictor override, e.g. {{reward: [-10, 40]}}
  standardize: true            # z-score X columns (recommended; weights are un-z-scored on save)
  max_iter: 2000
  tol: 1.0e-8

cv:
  n_folds: 5
  group_by_trial: true         # fold whole trials (avoids temporal leakage); needs trial_col

significance:
  run_wilcoxon: true           # CV-fold signed-rank full-vs-null (calibrated, conservative)
  run_permutation: false       # circular-shift null (slower; re-selects alpha per shuffle)
  n_perm: 200

outputs:
  predictor_importance: true   # held-out Delta-D2 per predictor group (ablation)
  kernels_plot: true
  reconstruction_plot: true
"""


def default_config_text(name, path):
    return DEFAULT_CONFIG.format(name=name, path=path)


def create_new_project(name, parent_dir="."):
    """Scaffold a new project directory (data/ results/ models/ + config.yaml). Returns its path.
    Refuses to overwrite an existing config.yaml."""
    path = os.path.abspath(os.path.join(parent_dir, name))
    for sub in ("", "data", "results", "models"):
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    cfg_path = os.path.join(path, "config.yaml")
    if os.path.exists(cfg_path):
        raise FileExistsError(f"{cfg_path} already exists -- not overwriting")
    with open(cfg_path, "w") as f:
        f.write(default_config_text(name, path))
    return path


def load_config(config_path):
    """Load a project config.yaml into a dict (with light validation of required keys)."""
    if yaml is None:
        raise ImportError("pyyaml is required for the CLI (pip install pyyaml)")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    d = cfg.get("data", {})
    if not d.get("session_col"):
        raise ValueError("config data.session_col is required (grouping for boundary-aware shifts)")
    if not d.get("predictors"):
        raise ValueError("config data.predictors must list at least one predictor column")
    if not d.get("responses") and not d.get("responses_prefix"):
        raise ValueError("config: set data.responses or data.responses_prefix")
    return cfg


def save_config(cfg, config_path):
    if yaml is None:
        raise ImportError("pyyaml is required for the CLI (pip install pyyaml)")
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
