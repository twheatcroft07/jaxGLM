"""Shift-kernel design-matrix construction -- a jaxGLM-native port of the Sabatini `sglm`
timeshift tooling (sglm/features/sglm_pp.py). Turns event / predictor time series into a GLM
design matrix where each predictor is expanded into a bank of time-shifted copies (its kernel).

Convention matches pandas `.shift` / sglm: a `+k` column places the predictor k samples LATER,
so the fitted coefficient on `name_+k` is the response at lag +k (k samples AFTER the event);
`-k` is k samples before. Edges are zero-filled (no wraparound).

Pure numpy (no jax/pandas) so it composes with anything; feed the resulting X straight into
poisson_glm / encoding.
"""
import re
import numpy as np


def shift_signal(v, k):
    """Shift a 1-D array by k samples with zero fill. +k moves values forward in time (later)."""
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v)
    if k == 0:
        out[:] = v
    elif k > 0:
        out[k:] = v[:-k]
    else:
        out[:k] = v[-k:]
    return out


def events_from_indices(indices, T):
    """Binary indicator vector of length T with 1.0 at each valid sample index.
    NaN / out-of-range indices are dropped (handy for trial-table event columns)."""
    v = np.zeros(T)
    idx = np.asarray(indices, dtype=float)
    idx = idx[np.isfinite(idx)]
    idx = idx[(idx >= 0) & (idx < T)].astype(int)
    v[idx] = 1.0
    return v


def build_design(predictors, shifts):
    """Expand predictors into a shift-kernel design matrix.

    predictors : dict {name: 1-D array length T} -- event indicators or continuous regressors.
    shifts     : a range / list of ints applied to every predictor, OR a dict
                 {name: list_of_shifts} for per-predictor windows.
    Returns (X (T, total_shifts), names) where names are '{predictor}_{shift}'. No intercept
    column (the solver adds the intercept). Pass X to standardize()/fit_units().
    """
    keys = list(predictors)
    cols, names = [], []
    for name in keys:
        sh = shifts[name] if isinstance(shifts, dict) else list(shifts)
        v = np.asarray(predictors[name], dtype=float)
        for k in sh:
            cols.append(shift_signal(v, k))
            names.append(f"{name}_{k}")
    return np.stack(cols, 1), names


def kernels_from_weights(W, names):
    """Parse '{pred}_{shift}' column names + fitted weights W (P, U) back into per-predictor
    kernels. Returns {pred: (shifts (S,), kernel (S, U))}, each sorted by shift. Inverse of
    build_design's column layout -- use for plotting / inspecting the fitted kernels."""
    W = np.asarray(W)
    groups = {}
    for i, nm in enumerate(names):
        m = re.match(r"^(.*)_(-?\d+)$", nm)
        if m is None:
            continue
        groups.setdefault(m.group(1), []).append((int(m.group(2)), i))
    out = {}
    for pred, lst in groups.items():
        lst.sort()
        shifts = np.array([s for s, _ in lst])
        idx = [i for _, i in lst]
        out[pred] = (shifts, W[idx, :])
    return out
