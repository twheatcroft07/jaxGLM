"""End-to-end integration test for the TURNKEY path: config.yaml -> models/fit.npz.

This is the path users actually run (`jaxglm config.yaml` / run_pipeline.run), and the one where
BOTH the unstandardize_weights (P,U) broadcast crash and the frac_deviance_explained 3-tuple bug
lived -- neither caught by the unit tests, because nothing exercised run_pipeline/project. Builds a
tiny synthetic project in a tmp dir, runs the full pipeline, and asserts it produces sane outputs.
"""
import numpy as np
import pandas as pd

import run_pipeline

CONFIG = """\
project:
  name: itest
  path: {path}
data:
  glob: "data/*.csv"
  session_col: SessionName
  trial_col: TrialNumber
  time_col: null
  predictors: [evt_info, evt_noise]
  responses_prefix: "unit_"
glm:
  family: poisson
  l1_ratio: 0.5
  alpha: [0.01, 0.1]
  shift_default: [-2, 4]
  standardize: true
  max_iter: 500
  tol: 1.0e-6
cv:
  n_folds: 3
  group_by_trial: true
significance:
  run_wilcoxon: true
  run_permutation: false
outputs:
  predictor_importance: true
  kernels_plot: false
  reconstruction_plot: false
"""


def _make_project(path, n_trials=60, bins=20, U=6, seed=0):
    rng = np.random.default_rng(seed)
    (path / "data").mkdir(parents=True, exist_ok=True)
    rows = []
    for tr in range(n_trials):
        info_bin = int(rng.integers(2, bins - 5))
        noise_bin = int(rng.integers(2, bins - 5))
        info = np.zeros(bins); info[info_bin] = 1.0
        noise = np.zeros(bins); noise[noise_bin] = 1.0
        # response is elevated for 3 bins AFTER evt_info; evt_noise has no effect
        drive = np.convolve(info, [0, 1, 1, 1])[:bins]
        counts = np.zeros((bins, U))
        for u in range(U):
            rate = np.exp(-0.5 + (0.6 + 0.3 * u) * drive)
            counts[:, u] = rng.poisson(rate)
        for b in range(bins):
            row = {"SessionName": "S1", "TrialNumber": tr,
                   "evt_info": info[b], "evt_noise": noise[b]}
            for u in range(U):
                row[f"unit_{u}"] = int(counts[b, u])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path / "data" / "synth.csv", index=False)
    (path / "config.yaml").write_text(CONFIG.format(path=str(path)))


def test_run_pipeline_end_to_end(tmp_path):
    _make_project(tmp_path)
    run_pipeline.run(str(tmp_path / "config.yaml"))

    fit = tmp_path / "models" / "fit.npz"
    assert fit.exists(), "pipeline did not write models/fit.npz"
    z = np.load(fit, allow_pickle=True)

    for k in ("W", "b", "d2", "delta_d2__evt_info", "delta_d2__evt_noise"):
        assert k in z.files, f"fit.npz missing {k}"

    d2 = np.asarray(z["d2"])
    assert np.isfinite(d2).all(), "non-finite D2"
    # D2 is a fraction of deviance: must be <= 1. (The 3-tuple bug reported medians ~4141.)
    assert d2.max() <= 1.0 + 1e-6, f"D2 > 1 -- frac_deviance_explained tuple mishandled? max={d2.max()}"
    assert np.median(d2) > 0.0, "informative design should explain > 0 deviance in-sample"

    # the informative predictor must out-encode the noise predictor (held-out delta-D2)
    dd_info = np.median(np.asarray(z["delta_d2__evt_info"]))
    dd_noise = np.median(np.asarray(z["delta_d2__evt_noise"]))
    assert dd_info > dd_noise, f"evt_info ({dd_info:.3f}) should beat evt_noise ({dd_noise:.3f})"

    # summary.csv is part of the turnkey contract
    assert (tmp_path / "results" / "summary.csv").exists()
