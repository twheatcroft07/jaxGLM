"""Input validation + fit-health checks: the 'fail loud, never silently wrong' layer.

A GLM that quietly returns weights for a degenerate design or an unconverged fit is worse than one
that errors -- the output looks like a scientific result. These helpers are pure numpy and run at
the Python level (before the jitted solver), so they can raise clear errors. The CLI calls them;
you can also call them directly around a library fit.
"""
import warnings
import numpy as np


class DesignError(ValueError):
    """Raised for fatal problems with (X, Y) that make a fit meaningless."""


def check_design(X, Y, family="gaussian", names=None, fold_ids=None):
    """Validate a design before fitting. RAISES DesignError on fatal problems; warns on soft ones.
    Returns a small report dict. Call once at the top level -- not inside hot refit loops.

    Fatal: wrong ndim, X/Y row mismatch, non-finite X or Y, Poisson responses with negative values,
    a fold with no train or no test data.
    Soft (warn): constant / all-zero predictor columns (not identifiable with the intercept, and
    standardization divides by ~0), non-integer Poisson responses, very few rows per parameter.
    """
    X = np.asarray(X); Y = np.asarray(Y)
    fatal = []
    if X.ndim != 2:
        fatal.append(f"X must be 2-D (T, P); got shape {X.shape}")
    if Y.ndim != 2:
        fatal.append(f"Y must be 2-D (T, U) -- one column per unit; got shape {Y.shape}. "
                     f"For a single unit pass y[:, None].")
    if X.ndim == 2 and Y.ndim == 2 and X.shape[0] != Y.shape[0]:
        fatal.append(f"X and Y must have the same number of rows; got {X.shape[0]} vs {Y.shape[0]}")
    nx = int((~np.isfinite(X)).sum()) if X.size else 0
    ny = int((~np.isfinite(Y)).sum()) if Y.size else 0
    if nx:
        fatal.append(f"X has {nx} non-finite (NaN/Inf) entries -- mask or impute before fitting")
    if ny:
        fatal.append(f"Y has {ny} non-finite (NaN/Inf) entries")
    if family == "poisson" and Y.size and np.isfinite(Y).all() and (Y < 0).any():
        fatal.append("Poisson responses must be nonnegative counts; Y has negative values")
    if fold_ids is not None and X.ndim == 2:
        fi = np.asarray(fold_ids); T = X.shape[0]
        if fi.shape[0] != T:
            fatal.append(f"fold_ids length {fi.shape[0]} != n rows {T}")
        else:
            for f in np.unique(fi):
                n_te = int((fi == f).sum())
                if n_te == 0 or n_te == T:
                    fatal.append(f"fold {f!r} has no held-out or no training data")
    if fatal:
        raise DesignError("invalid design:\n  - " + "\n  - ".join(fatal))

    # soft checks (warn, don't stop)
    report = dict(T=X.shape[0], P=X.shape[1], U=Y.shape[1])
    col_std = X.std(0)
    const = np.where(col_std == 0)[0]
    report["const_cols"] = const.tolist()
    if const.size:
        which = ", ".join(str(names[i]) if names is not None else str(i) for i in const[:8])
        warnings.warn(f"{const.size} predictor column(s) are constant/all-zero ({which}"
                      f"{'…' if const.size > 8 else ''}) -- not identifiable with the intercept "
                      f"and standardization will divide by ~0; drop them.")
    if family == "poisson" and Y.size and np.isfinite(Y).all() and not np.allclose(Y, np.round(Y)):
        warnings.warn("Poisson responses are non-integer; the deviance/D² assume counts.")
    if X.shape[0] < 5 * X.shape[1]:
        warnings.warn(f"few rows per parameter (T={X.shape[0]}, P={X.shape[1]}): the fit may be "
                      f"under-determined; rely on regularization + cross-validation.")
    return report


def fit_health(W, b, converged, n_iter, max_iter, names=None, weight_cap=1e3):
    """Summarize fit health and WARN on trouble. converged/n_iter are per-unit arrays from
    fit_units*. Returns a dict (also good for the CLI summary)."""
    conv = np.asarray(converged).astype(bool).ravel()
    Wn = np.asarray(W)
    U = conv.size
    n_conv = int(conv.sum())
    wmax = np.abs(Wn).max(0) if Wn.size else np.zeros(U)      # per-unit max |weight|
    n_extreme = int((wmax > weight_cap).sum())
    n_iter = np.asarray(n_iter).ravel()
    report = dict(n_units=U, n_converged=n_conv, n_not_converged=U - n_conv,
                  n_extreme=n_extreme, max_iter=int(max_iter),
                  median_iters=int(np.median(n_iter)) if n_iter.size else 0)
    if n_conv < U:
        warnings.warn(f"{U - n_conv}/{U} unit(s) did NOT converge within max_iter={max_iter} "
                      f"-- results may be unreliable; raise max_iter, loosen tol, or standardize X.")
    if n_extreme:
        warnings.warn(f"{n_extreme}/{U} unit(s) have extreme weights (|w|>{weight_cap:g}) -- likely "
                      f"an ill-conditioned design or too-weak regularization; standardize X.")
    return report
