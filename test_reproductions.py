"""Regression pins for the real-data reproductions. These read the saved comparison artifacts and
assert the headline metrics haven't drifted -- the guard that turns "we matched a published result
once" into "we still match it." They require the cluster data (on netscratch), so they SKIP
gracefully when those artifacts aren't present (e.g. in CI). Cluster workflow: re-run the relevant
`<repro>/compare_jaxglm.py` to regenerate the npz, then `pytest test_reproductions.py`.

The golden numbers + tolerances below are the known-good state (see each repro's README).
"""
import os
import numpy as np
import pytest

NS = "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft"


def _load(relpath):
    path = os.path.join(NS, relpath)
    if not os.path.exists(path):
        pytest.skip(f"reproduction artifact not present (cluster-only): {path}")
    return np.load(path, allow_pickle=True)


def test_chantranupong_gaussian_reproduction():
    # jaxGLM vs sklearn ElasticNet (the engine lab `sglm` wraps) on real photometry, identical X.
    d = _load("chantranupong/chantranupong_result.npz")
    r2_jax, r2_ref = np.asarray(d["r2_jax"]), np.asarray(d["r2_ref"])
    assert np.max(np.abs(r2_jax - r2_ref)) < 2e-3, "per-channel R^2 drifted from the sklearn reference"
    assert float(d["w_corr"]) > 0.99, "weight correlation vs sklearn dropped"


def test_kim_reinhold_reproduction():
    # jaxGLM vs sklearn AND vs Reinhold's published per-neuron holdout R^2.
    # 2 neurons are degenerate (published R^2=0, both fitters blow up) -> excluded via the sane mask.
    d = _load("kim/kim_result.npz")
    jax = np.asarray(d["r2_ho_jax"]); ref = np.asarray(d["r2_ho_ref"]); pub = np.asarray(d["pub"])
    sane = np.abs(ref) < 5
    assert sane.sum() >= 12, "too few non-degenerate neurons -- check the comparison"
    assert np.corrcoef(jax[sane], ref[sane])[0, 1] > 0.95, "solver equivalence vs sklearn regressed"
    assert np.corrcoef(jax[sane], pub[sane])[0, 1] > 0.85, "match to published per-neuron R^2 regressed"
