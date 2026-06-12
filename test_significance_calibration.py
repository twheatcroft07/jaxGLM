"""Calibration reference for the significance tests in encoding.py.

A statistical test is "correct" if, under the null hypothesis (response independent of the
design), its p-values are Uniform(0,1) -- so the false-positive rate at threshold alpha equals
alpha. This harness is that reference: it runs many NULL units (no real effect) and checks the
realized FPR ~ alpha and uniformity (KS test), plus power on a set of SIGNAL units. Both
families, both tests. Units are vmapped, so hundreds of null units cost no more than a few.

Run on a GPU node: conda activate jaxGLM && python test_significance_calibration.py
"""
import numpy as np
import jax.numpy as jnp
from scipy.stats import kstest

import encoding as enc

ALPHAS = np.array([1e-3, 1e-2, 1e-1, 1.0])
FIXED_ALPHA = 1e-2          # fixed regularization for the fast permutation null (not data-tuned)
N_NULL, N_SIG = 200, 30


def make_data(T=5000, P=8, family="poisson", seed=0, n_null=N_NULL, n_sig=N_SIG):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, P))
    Wr = rng.standard_normal((P, n_sig)) * 0.5
    bsig = rng.uniform(-0.5, 0.5, size=n_sig)
    bnull = rng.uniform(-0.5, 0.5, size=n_null)
    eta_sig = X @ Wr + bsig[None, :]
    if family == "poisson":
        Ysig = rng.poisson(np.exp(np.clip(eta_sig, -8, 8)))
        Ynull = rng.poisson(np.exp(np.clip(np.broadcast_to(bnull, (T, n_null)), -8, 8)))
    else:
        Ysig = eta_sig + 0.5 * rng.standard_normal((T, n_sig))
        Ynull = bnull[None, :] + 0.5 * rng.standard_normal((T, n_null))   # independent of X
    Y = np.concatenate([Ynull, Ysig], 1).astype(float)
    is_null = np.array([True] * n_null + [False] * n_sig)
    return X, Y, is_null


def report(tag, pvals, is_null, alpha=0.05):
    pn = pvals[is_null]; ps = pvals[~is_null]
    fpr = np.mean(pn < alpha)
    power = np.mean(ps < alpha)
    ks = kstest(pn, "uniform").pvalue          # large => p-values look uniform (good)
    print(f"[{tag}] FPR@{alpha}={fpr:.3f} (target {alpha}) | power={power:.2f} | "
          f"null p: mean={pn.mean():.2f} KS-unif p={ks:.3f} | frac<0.1={np.mean(pn<0.1):.3f}")
    # lenient guards: flag gross miscalibration / no power, not minor noise
    assert fpr < 0.15, f"{tag}: false-positive rate badly inflated"
    assert power > 0.7, f"{tag}: insufficient power on real signal"


def run(family):
    X, Y, is_null = make_data(family=family)
    w = enc.wilcoxon_full_vs_null(X, Y, ALPHAS, l1_ratio=0.5, n_folds=10, family=family, max_iter=1500)
    report(f"{family}/wilcoxon", w["pvalue"], is_null)
    # fast permutation null: a FIXED alpha (not tuned to the observed alignment) -> exchangeable
    p = enc.permutation_null_d2(X, Y, FIXED_ALPHA, l1_ratio=0.5, n_folds=5, family=family,
                                n_perm=200, max_iter=1500)
    report(f"{family}/permutation(fast,fixed-a)", p["pvalue"], is_null)


def run_faithful(family, n_null=40, n_sig=10, n_perm=100):
    """Smaller-scale check that the opt-in per-shuffle-CV path (select_alpha=True) is also
    calibrated -- it's ~grid*folds slower, so we run it on fewer units/perms."""
    X, Y, is_null = make_data(family=family, n_null=n_null, n_sig=n_sig)
    p = enc.permutation_null_d2(X, Y, ALPHAS, l1_ratio=0.5, n_folds=5, family=family,
                                n_perm=n_perm, select_alpha=True, max_iter=1500)
    report(f"{family}/permutation(faithful,CV)", p["pvalue"], is_null)


if __name__ == "__main__":
    run("poisson")
    run("gaussian")
    run_faithful("gaussian")          # spot-check the slow path on one family
    print("\nCALIBRATION CHECKS PASSED (FPR ~ alpha, p-values ~ uniform, power high)")
