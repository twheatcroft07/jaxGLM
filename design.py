"""Design-matrix construction for jaxGLM: bin raw event/spike/signal times onto a time grid,
then expand predictors into shift-kernels. A jaxGLM-native port of the Sabatini `sglm` tooling.

Units convention (this is the only place physical time enters the pipeline):
  - raw inputs are in SECONDS: spike/event times, t_start, bin_width.
  - `bin_width` (seconds) defines the grid; everything downstream is indexed in BINS.
  - `shifts` in build_design are integer BIN lags (a +k column = predictor k bins later, so the
    coefficient on `name_+k` is the response k bins AFTER the event; -k is before). Zero-filled
    at edges (no wraparound) -- matches pandas `.shift` / sglm.
  - counts (spikes per bin) and indicator values are dimensionless.

Pure numpy (no jax/pandas) so it composes with anything; the binned X/Y feed straight into
poisson_glm / encoding.
"""
import re
import numpy as np


# ------------------------------------------------------------------- binning (seconds -> bins)
def make_grid(t_start, t_stop, bin_width):
    """Define the time grid. t_start/t_stop/bin_width in SECONDS. Returns (t_start, bin_width,
    n_bins); n_bins covers [t_start, t_stop)."""
    n_bins = int(np.floor((t_stop - t_start) / bin_width))
    return t_start, bin_width, n_bins


def bin_spikes(spike_times, spike_units, n_units, t_start, bin_width, n_bins):
    """Spike counts per bin per unit -> Y (n_bins, n_units).
    spike_times : array, SECONDS.  spike_units : int array, the unit column (0..n_units-1) for
    each spike; negative / out-of-range entries are dropped.  t_start, bin_width : SECONDS."""
    bi = np.floor((np.asarray(spike_times) - t_start) / bin_width).astype(int)
    su = np.asarray(spike_units)
    keep = (bi >= 0) & (bi < n_bins) & (su >= 0) & (su < n_units)
    Y = np.zeros((n_bins, n_units))
    np.add.at(Y, (bi[keep], su[keep]), 1.0)
    return Y


def events_from_times(event_times, t_start, bin_width, n_bins):
    """Binary indicator (n_bins,) with 1.0 in the bin containing each event time.
    event_times, t_start, bin_width : SECONDS. NaN / out-of-range events are dropped."""
    bi = np.floor((np.asarray(event_times, dtype=float) - t_start) / bin_width)
    bi = bi[np.isfinite(bi)].astype(int)
    bi = bi[(bi >= 0) & (bi < n_bins)]
    v = np.zeros(n_bins)
    v[bi] = 1.0
    return v


def resample_continuous(sample_times, values, t_start, bin_width, n_bins):
    """Average a continuous signal (e.g. photometry) onto the grid -> (n_bins,).
    sample_times, t_start, bin_width : SECONDS. Empty bins are 0."""
    bi = np.floor((np.asarray(sample_times) - t_start) / bin_width).astype(int)
    vals = np.asarray(values, dtype=float)
    keep = (bi >= 0) & (bi < n_bins)
    out = np.zeros(n_bins); cnt = np.zeros(n_bins)
    np.add.at(out, bi[keep], vals[keep]); np.add.at(cnt, bi[keep], 1.0)
    return out / np.maximum(cnt, 1.0)


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
    """Binary indicator vector of length T with 1.0 at each valid BIN index (use when events are
    already given as grid indices, e.g. NWB trial-table '*_index' columns; for event times in
    seconds use events_from_times). NaN / out-of-range indices are dropped."""
    v = np.zeros(T)
    idx = np.asarray(indices, dtype=float)
    idx = idx[np.isfinite(idx)]
    idx = idx[(idx >= 0) & (idx < T)].astype(int)
    v[idx] = 1.0
    return v


def build_design(predictors, shifts):
    """Expand predictors into a shift-kernel design matrix.

    predictors : dict {name: 1-D array length T} -- event indicators or continuous regressors,
                 already on the time grid (n_bins long).
    shifts     : integer BIN lags -- a range / list applied to every predictor, OR a dict
                 {name: list_of_shifts} for per-predictor windows. (To express a window in
                 seconds, divide by bin_width: e.g. +/-0.5 s at 20 ms -> range(-25, 26).)
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
