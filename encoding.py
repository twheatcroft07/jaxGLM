"""Encoding-model orchestration on top of poisson_glm's batched solver.

Two outputs, both per unit and both family-agnostic ('poisson' spikes / 'gaussian' continuous):

  cv_select_alpha       -- choose elastic-net strength (lambda) per unit by held-out deviance
  predictor_importance  -- held-out D^2 of the full model, and of the model with each predictor
                           GROUP removed; the drop (delta-D^2) is that group's encoding
                           contribution. This is the core scientific figure.

D^2 is computed held-out (cross-validated): for each fold we fit on the train rows and score
deviance on the held-out rows, against an intercept-only null also fit on train. So delta-D^2 is
an out-of-sample contribution, not an in-sample one. The per-unit lambda is selected first (on
the full model) and then reused for the full and ablated fits -- a controlled ablation.
"""
import numpy as np
import jax.numpy as jnp
from scipy.stats import wilcoxon

import poisson_glm as pg


def make_folds(T, n_folds=5, fold_ids=None):
    """Return a list of held-out index arrays. If fold_ids (length T) is given, fold by it
    (e.g. one fold per trial-block); otherwise split into contiguous blocks."""
    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids)
        return [np.where(fold_ids == f)[0] for f in np.unique(fold_ids)]
    edges = np.linspace(0, T, n_folds + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(n_folds)]


def _heldout_deviance(X, Y, alpha_per_unit, l1_ratio, folds, family, kw):
    """Per-unit summed held-out model and null deviance over folds. alpha_per_unit: (U,)."""
    T, U = Y.shape
    allidx = np.arange(T)
    dev = np.zeros(U)
    devnull = np.zeros(U)
    for te in folds:
        tr = np.setdiff1d(allidx, te)
        W, b, _, _ = pg.fit_units_alpha(X[tr], Y[tr], jnp.asarray(alpha_per_unit), l1_ratio,
                                        kw["L0"], kw["max_iter"], kw["tol"], family)
        mu = pg.predict_rate(X[te], W, b, family)
        dev += np.asarray(pg.deviance(Y[te], mu, family))
        mu0 = jnp.broadcast_to(Y[tr].mean(0)[None, :], Y[te].shape)   # intercept-only null
        devnull += np.asarray(pg.deviance(Y[te], mu0, family))
    return dev, devnull


def alpha_grid(X, Y, l1_ratio=0.5, n=50, ratio=1e-3):
    """Data-driven log-spaced elastic-net `alpha` grid for cross-validation.

    Returns `n` alphas from `alpha_max` (the smallest alpha that zeros ALL weights) down to
    `alpha_max * ratio`, log-spaced — the standard glmnet-style range, computed from the data
    rather than guessed. `alpha_max` is the KKT bound at w=0 for the mean-loss objective:

        alpha_max = max_j |X_jᵀ (y − mean(y))| / (T · l1_ratio)

    `X` should be standardized `(T, P)`; `Y` is `(T, U)` and the grid covers every unit (top =
    max alpha_max across units). Family-agnostic: `mean(y)` is the intercept-only prediction for
    both Poisson and Gaussian. For ridge (`l1_ratio≈0`) a small surrogate l1_ratio avoids the
    division by zero and gives a sensible scale (same grid shape). Feed the result to
    `cv_select_alpha` / `predictor_importance` / the significance tests.
    """
    X = np.asarray(X); Y = np.asarray(Y); T = Y.shape[0]
    lr = max(float(l1_ratio), 1e-2)
    r0 = Y - Y.mean(0)[None, :]                          # intercept-only residual, per unit
    amax = (np.abs(X.T @ r0) / T).max(0) / lr            # per-unit alpha_max -> (U,)
    top = max(float(amax.max()), 1e-8)
    return np.geomspace(top * ratio, top, n)


def cv_select_alpha(X, Y, alphas, l1_ratio=0.5, fold_ids=None, n_folds=5, family="poisson",
                    L0=1.0, max_iter=2000, tol=1e-8):
    """Pick alpha per unit minimizing held-out deviance. Returns (alpha_star (U,), cv_dev (A,U))."""
    X = jnp.asarray(X); Y = jnp.asarray(Y)
    T, U = Y.shape
    folds = make_folds(T, n_folds, fold_ids)
    alphas = np.asarray(alphas)
    cv_dev = np.zeros((len(alphas), U))
    allidx = np.arange(T)
    for te in folds:
        tr = np.setdiff1d(allidx, te)
        Wg, bg, _, _ = pg.fit_units_grid(X[tr], Y[tr], jnp.asarray(alphas), l1_ratio,
                                         L0, max_iter, tol, family)         # (A,P,U),(A,U)
        for a in range(len(alphas)):
            mu = pg.predict_rate(X[te], Wg[a], bg[a], family)
            cv_dev[a] += np.asarray(pg.deviance(Y[te], mu, family))
    alpha_star = alphas[np.argmin(cv_dev, axis=0)]
    return alpha_star, cv_dev


def regularization_path(X, Y, alphas, l1_ratio=0.5, fold_ids=None, n_folds=5, family="poisson",
                        L0=1.0, max_iter=2000, tol=1e-8):
    """For each alpha: full-data fit -> mean data term & mean penalty term (across units), plus
    the mean held-out CV deviance. Returns dict {alphas, data, penalty, cv_dev (each (A,)),
    alpha_cvmin} -- the 'are we in the right scale?' diagnostic. Plot with
    viz.plot_regularization_path: data term rises, penalty bumps, CV deviance dips at the sweet
    spot. Standardize X first."""
    X = jnp.asarray(X); Y = jnp.asarray(Y)
    alphas = np.asarray(alphas)
    _, cv_dev_AU = cv_select_alpha(X, Y, alphas, l1_ratio, fold_ids, n_folds, family, L0, max_iter, tol)
    cv_dev = cv_dev_AU.mean(1)                                       # mean over units
    Wg, bg, _, _ = pg.fit_units_grid(X, Y, jnp.asarray(alphas), l1_ratio, L0, max_iter, tol, family)
    recon = np.zeros(len(alphas)); penalty = np.zeros(len(alphas))
    for a in range(len(alphas)):
        ot = pg.objective_terms(X, Y, Wg[a], bg[a], float(alphas[a]), l1_ratio, family)
        recon[a] = float(np.asarray(ot["recon"]).mean())            # deviance/(2T), >= 0
        penalty[a] = float(np.asarray(ot["penalty"]).mean())
    return dict(alphas=alphas, recon=recon, penalty=penalty, cv_dev=cv_dev,
                alpha_cvmin=float(alphas[int(np.argmin(cv_dev))]))


def predictor_importance(X, Y, subsets, alphas, l1_ratio=0.5, fold_ids=None, n_folds=5,
                         family="poisson", L0=1.0, max_iter=2000, tol=1e-8):
    """Held-out D^2 of the full model and delta-D^2 for removing each predictor group.

    subsets: {group_name: array_of_column_indices_to_drop}.
    Returns dict with alpha_star (U,), d2_full (U,), delta_d2 {name: (U,)}, d2_subset {name:(U,)},
    and cv_dev (A,U).
    """
    X = jnp.asarray(X); Y = jnp.asarray(Y)
    T, U = Y.shape
    kw = dict(L0=L0, max_iter=max_iter, tol=tol)
    folds = make_folds(T, n_folds, fold_ids)

    alpha_star, cv_dev = cv_select_alpha(X, Y, alphas, l1_ratio, fold_ids, n_folds, family,
                                         L0, max_iter, tol)
    dev_full, devnull = _heldout_deviance(X, Y, alpha_star, l1_ratio, folds, family, kw)
    d2_full = 1.0 - dev_full / np.clip(devnull, 1e-10, None)

    delta_d2, d2_subset = {}, {}
    for name, cols in subsets.items():
        Xs = X.at[:, jnp.asarray(cols)].set(0.0)            # zero the group == remove it
        dev_s, _ = _heldout_deviance(Xs, Y, alpha_star, l1_ratio, folds, family, kw)
        d2s = 1.0 - dev_s / np.clip(devnull, 1e-10, None)
        d2_subset[name] = d2s
        delta_d2[name] = d2_full - d2s                       # contribution of the group
    return dict(alpha_star=alpha_star, d2_full=d2_full, delta_d2=delta_d2,
                d2_subset=d2_subset, cv_dev=cv_dev)


def _per_fold_deviance(X, Y, alpha_per_unit, l1_ratio, folds, family, kw):
    """Per-fold (not summed) held-out model and null deviance: each (n_folds, U)."""
    T, U = Y.shape
    allidx = np.arange(T)
    dev = np.zeros((len(folds), U)); devnull = np.zeros((len(folds), U))
    for i, te in enumerate(folds):
        tr = np.setdiff1d(allidx, te)
        W, b, _, _ = pg.fit_units_alpha(X[tr], Y[tr], jnp.asarray(alpha_per_unit), l1_ratio,
                                        kw["L0"], kw["max_iter"], kw["tol"], family)
        mu = pg.predict_rate(X[te], W, b, family)
        dev[i] = np.asarray(pg.deviance(Y[te], mu, family))
        mu0 = jnp.broadcast_to(Y[tr].mean(0)[None, :], Y[te].shape)
        devnull[i] = np.asarray(pg.deviance(Y[te], mu0, family))
    return dev, devnull


def wilcoxon_full_vs_null(X, Y, alphas, l1_ratio=0.5, fold_ids=None, n_folds=10, family="poisson",
                          L0=1.0, max_iter=2000, tol=1e-8):
    """Per-unit Wilcoxon signed-rank test that the full model beats the intercept-only null,
    paired across CV folds on held-out deviance (the refactoredHarveyGLM test). One-sided.
    Note: min achievable p ~ 2^-n_folds, so use enough folds (default 10). Returns dict with
    stat, pvalue, significant (U,)."""
    X = jnp.asarray(X); Y = jnp.asarray(Y); U = Y.shape[1]
    kw = dict(L0=L0, max_iter=max_iter, tol=tol)
    folds = make_folds(Y.shape[0], n_folds, fold_ids)
    alpha_star, _ = cv_select_alpha(X, Y, alphas, l1_ratio, fold_ids, n_folds, family, L0, max_iter, tol)
    dev, devnull = _per_fold_deviance(X, Y, alpha_star, l1_ratio, folds, family, kw)
    diff = devnull - dev                                     # > 0 => full model better per fold
    stat = np.full(U, np.nan); pval = np.ones(U)
    for u in range(U):
        d = diff[:, u]
        if np.allclose(d, 0):
            continue
        try:
            stat[u], pval[u] = wilcoxon(d, alternative="greater")
        except ValueError:
            pass
    return dict(stat=stat, pvalue=pval, significant=pval < 0.05, alpha_star=alpha_star)


def permutation_null_d2(X, Y, alphas, l1_ratio=0.5, fold_ids=None, n_folds=5, family="poisson",
                        n_perm=200, seed=0, L0=1.0, max_iter=2000, tol=1e-8):
    """Per-unit significance of held-out D^2 against a circular-shift null. The response is
    circularly shifted relative to the design by random offsets (preserving each signal's own
    autocorrelation while destroying the design relationship); D^2 is recomputed each time.
    p = (1 + #{null D^2 >= observed}) / (n_perm + 1). Returns dict d2_obs, pvalue, significant."""
    X = jnp.asarray(X); Y = jnp.asarray(Y); T, U = Y.shape
    kw = dict(L0=L0, max_iter=max_iter, tol=tol)
    folds = make_folds(T, n_folds, fold_ids)
    alpha_star, _ = cv_select_alpha(X, Y, alphas, l1_ratio, fold_ids, n_folds, family, L0, max_iter, tol)
    dev, devnull = _heldout_deviance(X, Y, alpha_star, l1_ratio, folds, family, kw)
    d2_obs = 1.0 - dev / np.clip(devnull, 1e-10, None)
    Ynp = np.asarray(Y)
    rng = np.random.default_rng(seed)
    null_ge = np.zeros(U)
    for _ in range(n_perm):
        sh = int(rng.integers(T // 10, T - T // 10))         # avoid near-zero shifts
        Yp = jnp.asarray(np.roll(Ynp, sh, axis=0))
        dp, d0 = _heldout_deviance(X, Yp, alpha_star, l1_ratio, folds, family, kw)
        null_ge += (1.0 - dp / np.clip(d0, 1e-10, None)) >= d2_obs
    pval = (1.0 + null_ge) / (n_perm + 1.0)
    return dict(d2_obs=d2_obs, pvalue=pval, significant=pval < 0.05, alpha_star=alpha_star)
