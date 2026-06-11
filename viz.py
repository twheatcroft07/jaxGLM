"""Visualization for fitted GLM kernels and signal reconstruction -- jaxGLM-native port of the
Sabatini `sglm` plotting (sglm/visualization). matplotlib is imported lazily so the rest of
jaxGLM has no plotting dependency.

  plot_kernels            -- per-predictor kernels vs lag (the Fig-3C-style coefficient plot)
  plot_reconstruction     -- predicted vs actual signal over a time window
  event_triggered_average -- event-aligned mean of a signal (actual or predicted)
  plot_event_reconstruction -- event-aligned actual vs predicted (reconstruct_signal analog)
"""
import numpy as np

from design import kernels_from_weights


def _lag_axis(shifts, dt):
    return shifts * dt if dt else shifts


def plot_kernels(W, names, dt=None, mean_sem=False, axes=None, figsize=None):
    """Plot each predictor's fitted kernel vs lag. W (P, U), names '{pred}_{shift}'.
    dt: seconds per sample (x-axis in seconds if given, else in shifts).
    mean_sem: if True, plot mean +/- s.e.m. across units; else one line per unit.
    Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt
    ker = kernels_from_weights(W, names)
    preds = list(ker)
    n = len(preds)
    if axes is None:
        fig, axes = plt.subplots(1, n, figsize=figsize or (3 * n, 2.8), squeeze=False)
        axes = axes[0]
    else:
        fig = axes[0].figure
    for ax, pred in zip(axes, preds):
        shifts, k = ker[pred]                       # (S,), (S, U)
        x = _lag_axis(shifts, dt)
        if mean_sem:
            m = k.mean(1); sem = k.std(1) / np.sqrt(k.shape[1])
            ax.plot(x, m, color="C0")
            ax.fill_between(x, m - sem, m + sem, alpha=0.3, color="C0")
        else:
            for u in range(k.shape[1]):
                ax.plot(x, k[:, u], alpha=0.5, lw=1)
        ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
        ax.set_title(pred); ax.set_xlabel("lag (s)" if dt else "shift")
    axes[0].set_ylabel("coefficient")
    fig.tight_layout()
    return fig


def plot_reconstruction(Y, mu, unit=0, window=None, dt=None, ax=None):
    """Predicted (mu) vs actual (Y) signal for one unit over a window. Y, mu: (T, U)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 2.6))
    a, b = window or (0, min(Y.shape[0], 2000))
    t = np.arange(a, b) * (dt or 1.0)
    ax.plot(t, np.asarray(Y)[a:b, unit], color="0.4", lw=1, label="actual")
    ax.plot(t, np.asarray(mu)[a:b, unit], color="C3", lw=1, label="predicted")
    ax.set_xlabel("time (s)" if dt else "sample"); ax.set_ylabel(f"unit {unit}")
    ax.legend(loc="upper right", fontsize=8)
    return ax.figure


def event_triggered_average(signal, event_idx, pre, post):
    """Mean of `signal` (1-D) in a window [-pre, +post] samples around each event index.
    Returns (lags (pre+post+1,), mean, sem). Events too close to the edges are skipped."""
    signal = np.asarray(signal)
    T = signal.shape[0]
    snips = []
    for e in np.asarray(event_idx, dtype=int):
        if e - pre >= 0 and e + post < T:
            snips.append(signal[e - pre:e + post + 1])
    snips = np.array(snips)
    lags = np.arange(-pre, post + 1)
    if snips.size == 0:
        return lags, np.zeros_like(lags, float), np.zeros_like(lags, float)
    return lags, snips.mean(0), snips.std(0) / np.sqrt(len(snips))


def plot_event_reconstruction(Y, mu, event_idx, unit=0, pre=20, post=40, dt=None, ax=None):
    """Event-aligned actual vs predicted average for one unit (the sglm reconstruction plot)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    lags, ya, ysem = event_triggered_average(np.asarray(Y)[:, unit], event_idx, pre, post)
    _, ma, msem = event_triggered_average(np.asarray(mu)[:, unit], event_idx, pre, post)
    x = _lag_axis(lags, dt)
    for m, s, c, lab in [(ya, ysem, "0.4", "actual"), (ma, msem, "C3", "predicted")]:
        ax.plot(x, m, color=c, label=lab)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("lag (s)" if dt else "shift"); ax.set_ylabel(f"unit {unit}")
    ax.legend(fontsize=8)
    return ax.figure
