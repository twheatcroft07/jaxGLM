"""Regression tests for three bugs fixed in commit 60b6eb0 (2026-06). Each pins a specific failure
mode the prior unit tests did NOT exercise (they gave false-green), so the bugs can't silently
reappear.
"""
import numpy as np
import jax.numpy as jnp

import poisson_glm as pg
import encoding as enc
import design as dz


def test_unstandardize_weights_multi_unit():
    """Bug: `w_z / std` broadcast (P,U)/(P,) crashed, and the intercept term summed to a scalar
    instead of per-unit. Pin both the many-unit and the single-unit paths."""
    P, U = 5, 4
    rng = np.random.default_rng(0)
    w_z = jnp.asarray(rng.standard_normal((P, U)))
    b_z = jnp.asarray(rng.standard_normal(U))
    mean = jnp.asarray(rng.standard_normal(P))
    std = jnp.asarray(rng.random(P) + 0.5)

    w, b = pg.unstandardize_weights(w_z, b_z, mean, std)
    assert w.shape == (P, U) and b.shape == (U,)
    assert np.allclose(np.asarray(w), np.asarray(w_z) / np.asarray(std)[:, None])
    exp_b = np.asarray(b_z) - (np.asarray(w_z) * (np.asarray(mean) / np.asarray(std))[:, None]).sum(0)
    assert np.allclose(np.asarray(b), exp_b)

    # single-unit (1-D) path must still work
    w1, b1 = pg.unstandardize_weights(w_z[:, 0], float(b_z[0]), mean, std)
    assert np.asarray(w1).shape == (P,)


def test_frac_deviance_explained_returns_triple():
    """Bug: run_pipeline did `np.asarray(frac_deviance_explained(...))` expecting one array, but it
    returns (d2, dev_m, dev_0) -> medianing the whole tuple reported D2 like 4141. Pin the contract:
    3-tuple, per-unit, and d2 <= 1."""
    rng = np.random.default_rng(1)
    y = jnp.asarray(rng.poisson(2.0, (300, 3)).astype(float))
    mu = jnp.asarray(np.full((300, 3), 2.0))
    out = pg.frac_deviance_explained(y, mu, y.mean(0), "poisson")
    assert isinstance(out, tuple) and len(out) == 3
    d2, dev_m, dev_0 = out
    assert np.asarray(d2).shape == (3,)
    assert np.all(np.asarray(d2) <= 1.0 + 1e-6)


def test_one_se_never_picks_smaller_alpha():
    """Bug: plain-argmin CV pinned noisy units at the grid-floor alpha -> overfit -> held-out D2 ~
    -99. The 1-SE rule must select an alpha >= the argmin alpha (more regularization, never less)."""
    rng = np.random.default_rng(2)
    T = 1500
    events = {f"e{j}": (rng.random(T) < 0.03).astype(float) for j in range(6)}
    X, _ = dz.build_design(events, range(-3, 6))
    P = X.shape[1]
    Wr = rng.standard_normal((P, 40)) * rng.binomial(1, 0.1, (P, 40)) * 0.3   # mostly weak units
    Y = rng.poisson(np.exp(np.clip(X @ Wr - 0.5, -6, 6))).astype(float)
    Xz, *_ = pg.standardize(jnp.asarray(X))
    Yj = jnp.asarray(Y)
    fold = (np.arange(T) % 5).astype(int)
    alphas = enc.alpha_grid(Xz, Yj, l1_ratio=0.5, n=10)

    a_plain, _ = enc.cv_select_alpha(Xz, Yj, alphas, fold_ids=fold, one_se=False, max_iter=600)
    a_se, _ = enc.cv_select_alpha(Xz, Yj, alphas, fold_ids=fold, one_se=True, max_iter=600)
    assert np.all(np.asarray(a_se) >= np.asarray(a_plain) - 1e-12), "1-SE chose a smaller alpha"
    assert np.median(a_se) >= np.median(a_plain)
