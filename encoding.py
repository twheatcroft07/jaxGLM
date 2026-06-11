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
