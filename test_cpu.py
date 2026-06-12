"""Fast CPU regression tier (runs in seconds, no GPU): input validation, boundary-aware shifting,
and core solver/encoding invariants. This is the net that catches silent breakage on every change.
Run: pytest            (this file + other fast tests)   |   pytest -m slow   for the GPU-scale ones.
"""
import warnings
import numpy as np
import jax.numpy as jnp
import pytest

import validate
import design as dz
import poisson_glm as pg
import encoding as enc


# ----------------------------------------------------------------- input validation (fail loud)
def test_check_design_rejects_nan():
    X = np.zeros((100, 3)); X[5, 1] = np.nan
    Y = np.zeros((100, 2))
    with pytest.raises(validate.DesignError, match="non-finite"):
        validate.check_design(X, Y, family="gaussian")


def test_check_design_rejects_row_mismatch():
    with pytest.raises(validate.DesignError, match="same number of rows"):
        validate.check_design(np.zeros((100, 3)), np.zeros((90, 1)), family="gaussian")


def test_check_design_rejects_1d_Y():
    with pytest.raises(validate.DesignError, match="2-D"):
        validate.check_design(np.zeros((100, 3)), np.zeros(100), family="gaussian")


def test_check_design_rejects_negative_poisson():
    Y = np.ones((100, 1)); Y[0, 0] = -1
    with pytest.raises(validate.DesignError, match="nonnegative"):
        validate.check_design(np.zeros((100, 2)), Y, family="poisson")


def test_check_design_rejects_degenerate_fold():
    fold_ids = np.zeros(100)            # single fold -> no training data for that fold
    with pytest.raises(validate.DesignError, match="no held-out or no training"):
        validate.check_design(np.random.default_rng(0).standard_normal((100, 3)),
                              np.zeros((100, 1)), family="gaussian", fold_ids=fold_ids)


def test_check_design_warns_constant_column():
    X = np.random.default_rng(0).standard_normal((100, 3)); X[:, 1] = 2.0   # constant col
    with pytest.warns(UserWarning, match="constant/all-zero"):
        rep = validate.check_design(X, np.zeros((100, 1)), family="gaussian", names=["a", "b", "c"])
    assert rep["const_cols"] == [1]


def test_check_design_passes_clean():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 4)); Y = rng.standard_normal((500, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # clean design must not even warn
        rep = validate.check_design(X, Y, family="gaussian")
    assert rep == {"T": 500, "P": 4, "U": 3, "const_cols": []}


def test_fit_health_flags_nonconverged():
    with pytest.warns(UserWarning, match="did NOT converge"):
        rep = validate.fit_health(np.zeros((5, 3)), np.zeros(3),
                                  converged=np.array([True, False, True]),
                                  n_iter=np.array([10, 100, 10]), max_iter=100)
    assert rep["n_not_converged"] == 1 and rep["n_converged"] == 2


# ----------------------------------------------------------------- boundary-aware shifting
def test_boundary_shifting_no_cross_session_leak():
    groups = dz.group_ids_from_labels(["A", "A", "A", "B", "B", "B"])
    assert list(groups) == [0, 0, 0, 1, 1, 1]
    w = np.array([0, 0, 1.0, 0, 0, 0])               # event in last bin of session A
    assert dz.shift_signal(w, 1)[3] == 1.0           # plain shift LEAKS into session B
    assert dz.shift_signal(w, 1, groups)[3] == 0.0   # group-aware shift does NOT


def test_group_ids_separate_repeated_labels():
    # same label reused in a non-adjacent block must get a distinct id
    g = dz.group_ids_from_labels(["A", "A", "B", "A"])
    assert list(g) == [0, 0, 1, 2]


# ----------------------------------------------------------------- solver / encoding invariants
def test_gaussian_recovers_known_weights():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 6)); w = rng.standard_normal(6); b = 0.5
    y = X @ w + b + 0.05 * rng.standard_normal(2000)
    W, B, n_iter, conv = pg.fit_units(jnp.asarray(X), jnp.asarray(y[:, None]),
                                      1e-4, 0.0, 1.0, 5000, 1e-9, "gaussian")
    assert np.corrcoef(np.asarray(W)[:, 0], w)[0, 1] > 0.999
    mu = np.asarray(pg.predict_rate(jnp.asarray(X), W, B, "gaussian")).ravel()
    assert 1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum() > 0.99


def test_l1_induces_sparsity():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((1500, 10))
    y = X[:, 0] * 2.0 + 0.1 * rng.standard_normal(1500)      # only 1 real predictor
    W, _, _, _ = pg.fit_units(jnp.asarray(X), jnp.asarray(y[:, None]), 0.2, 1.0, 1.0, 4000, 1e-9, "gaussian")
    assert np.mean(np.abs(np.asarray(W)[:, 0]) < 1e-8) >= 0.5   # most coeffs driven to exactly 0


def test_alpha_grid_max_zeros_all_weights():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((1500, 8)); Y = (X @ rng.standard_normal((8, 3)))[:, :]
    Xz, *_ = pg.standardize(jnp.asarray(X))
    grid = enc.alpha_grid(Xz, jnp.asarray(Y), l1_ratio=1.0, n=10)
    amax = float(np.max(grid))                                 # alpha_max = top of the KKT grid
    assert np.ptp(grid) > 0                                    # grid spans a real range
    W, _, _, _ = pg.fit_units(Xz, jnp.asarray(Y), amax, 1.0, 1.0, 3000, 1e-9, "gaussian")
    assert np.allclose(np.asarray(W), 0.0, atol=1e-6)          # alpha_max kills all weights (KKT)


def test_kernels_from_weights_roundtrip():
    e = dz.events_from_indices([10, 50, 120, 300], 500)
    X, names = dz.build_design({"evt": e, "other": np.roll(e, 3)}, range(-2, 5))
    W = np.arange(X.shape[1] * 2, dtype=float).reshape(X.shape[1], 2)
    ker = dz.kernels_from_weights(W, names)
    assert set(ker) == {"evt", "other"}
    shifts, k = ker["evt"]
    assert list(shifts) == list(range(-2, 5)) and k.shape == (7, 2)
