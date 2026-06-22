"""Honest stress test of the significance tests on AUTOCORRELATED null data -- the case the
circular-shift permutation null is *designed* for, but which the iid-synthetic calibration
(test_significance_calibration.py) never exercised. If the permutation null is trustworthy on real
neural/photometry data, the false-positive rate must stay ~ alpha here too.

Null units are AR(1) (gaussian) or AR(1)-driven Poisson and are INDEPENDENT of the design, while
the design X is realistic: event trains expanded into shift-kernels (so its columns are themselves
autocorrelated -- exactly where a naive null would produce spurious 'significant' hits).
Run on a GPU node: conda activate jaxGLM && python test_autocorr_calibration.py
"""
import numpy as np
import jax.numpy as jnp
from scipy.stats import kstest

import design as dz
import poisson_glm as pg
import encoding as enc

N_NULL, N_SIG = 120, 20          # enough null units for an FPR estimate; keeps the refits tractable
FIXED_ALPHA = 1e-2
ALPHAS = np.array([1e-3, 1e-2, 1e-1])
MAX_ITER = 400                   # autocorrelated fits are slow; cap iters (calibration, not precision)
N_PERM = 100                     # FPR resolution ~0.01, plenty to see 0.05 vs inflation


def ar1(rng, T, n, rho):
    """Stationary AR(1): z_t = rho z_{t-1} + sqrt(1-rho^2) e_t  -> unit-variance, autocorr rho."""
    z = np.zeros((T, n))
    inn = rng.standard_normal((T, n)) * np.sqrt(1.0 - rho ** 2)
    for t in range(1, T):
        z[t] = rho * z[t - 1] + inn[t]
    return z


def make_data(family, T=3000, rho=0.95, seed=0):
    rng = np.random.default_rng(seed)
    events = {f"e{j}": (rng.random(T) < 0.03).astype(float) for j in range(4)}
    X, _ = dz.build_design(events, range(-5, 11))         # autocorrelated event-kernel design
    P = X.shape[1]
    Wr = rng.standard_normal((P, N_SIG)) * 0.3
    if family == "gaussian":
        Ynull = ar1(rng, T, N_NULL, rho)                  # autocorrelated, INDEPENDENT of X
        Ysig = X @ Wr + 0.5 * ar1(rng, T, N_SIG, rho)
    else:
        Ynull = rng.poisson(np.exp(np.clip(0.6 * ar1(rng, T, N_NULL, rho) - 0.5, -8, 8)))
        Ysig = rng.poisson(np.exp(np.clip(X @ Wr + 0.5 * ar1(rng, T, N_SIG, rho), -8, 8)))
    Y = np.concatenate([Ynull, Ysig], 1).astype(float)
    is_null = np.array([True] * N_NULL + [False] * N_SIG)
    return X, Y, is_null


def report(tag, p, is_null, a=0.05):
    pn, ps = p[is_null], p[~is_null]
    print(f"[{tag}] FPR@{a}={np.mean(pn < a):.3f} (target {a}) | power={np.mean(ps < a):.2f} | "
          f"KS-unif p={kstest(pn, 'uniform').pvalue:.3f} | frac<0.1={np.mean(pn < 0.1):.3f}")
    return float(np.mean(pn < a))


def run(family):
    X, Y, is_null = make_data(family)
    Xz, *_ = pg.standardize(jnp.asarray(X))
    Yj = jnp.asarray(Y)
    w = enc.wilcoxon_full_vs_null(Xz, Yj, ALPHAS, l1_ratio=0.5, n_folds=5, family=family, max_iter=MAX_ITER)
    report(f"{family}/wilcoxon   ", w["pvalue"], is_null)
    p = enc.permutation_null_d2(Xz, Yj, FIXED_ALPHA, l1_ratio=0.5, n_folds=5, family=family,
                                n_perm=N_PERM, max_iter=MAX_ITER)
    report(f"{family}/permutation", p["pvalue"], is_null)


import pytest


@pytest.mark.slow
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_wilcoxon_calibrated_under_autocorrelation(family):
    """The recommended test (Wilcoxon) must keep its false-positive rate near nominal even on
    rho=0.95 autocorrelated nulls -- otherwise per-unit significance on real spike trains is not
    trustworthy. (The permutation null is allowed to inflate here; that's the documented reason to
    prefer Wilcoxon for Poisson.) Slow-marked: run with `pytest -m slow`."""
    X, Y, is_null = make_data(family)
    Xz, *_ = pg.standardize(jnp.asarray(X))
    Yj = jnp.asarray(Y)
    w = enc.wilcoxon_full_vs_null(Xz, Yj, ALPHAS, l1_ratio=0.5, n_folds=5, family=family,
                                  max_iter=MAX_ITER)
    fpr = float(np.mean(np.asarray(w["pvalue"])[is_null] < 0.05))
    assert fpr <= 0.15, f"{family}: Wilcoxon FPR {fpr:.3f} inflated under autocorrelation (~0.05)"


if __name__ == "__main__":
    print("=== AUTOCORRELATED-NULL calibration (rho=0.95) ===")
    run("gaussian")
    run("poisson")
    print("\nInterpretation: FPR should stay ~0.05. Inflation here (but not in the iid test) means "
          "the permutation null does NOT fully handle real autocorrelation -> trust Wilcoxon.")
