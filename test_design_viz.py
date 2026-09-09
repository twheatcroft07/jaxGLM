"""Self-test for design.py (shift-kernel construction) and viz.py (plots).

Correctness check: synthesize a response as a KNOWN kernel convolved with an event train, build
the shift design, fit, and confirm the recovered kernel matches the true one. Also renders the
kernel / reconstruction / event-aligned plots to PNGs (on netscratch) to confirm viz runs.
Run on a GPU node: conda activate jaxGLM && python test_design_viz.py
"""
import os
import tempfile
import numpy as np
import jax.numpy as jnp

import design as dz
import viz
import poisson_glm as pg

# write figures to netscratch on the cluster, else a temp dir (so this runs in CI without netscratch)
_NS = os.path.join(os.environ.get("JAXGLM_DATA", "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft"), "jaxglm_viz")
OUT = _NS if os.path.isdir(os.path.dirname(_NS)) else os.path.join(tempfile.gettempdir(), "jaxglm_viz")
os.makedirs(OUT, exist_ok=True)


def test_shift_signal():
    v = np.array([1.0, 0, 0, 0, 0])
    assert np.allclose(dz.shift_signal(v, 2), [0, 0, 1, 0, 0]), "forward shift"
    assert np.allclose(dz.shift_signal(v, -1), [0, 0, 0, 0, 0]), "edge zero-fill"
    w = np.array([0, 0, 1.0, 0, 0])
    assert np.allclose(dz.shift_signal(w, -2), [1, 0, 0, 0, 0]), "backward shift"
    print("[1] shift_signal conventions OK")


def test_kernel_recovery():
    rng = np.random.default_rng(0)
    T, K, U = 8000, 16, 4
    ev = np.sort(rng.choice(T, size=400, replace=False))
    e = dz.events_from_indices(ev, T)
    # true kernels per unit: scaled, smooth bumps over lags 0..K
    lags = np.arange(K + 1)
    base = np.exp(-((lags - 5) ** 2) / 8.0) - 0.4 * np.exp(-((lags - 11) ** 2) / 6.0)
    ker_true = np.stack([base * s for s in (1.0, 0.6, -0.8, 0.3)], 1)      # (K+1, U)
    Y = np.zeros((T, U))
    for k in range(K + 1):
        Y += np.outer(dz.shift_signal(e, k), ker_true[k])
    Y += 0.05 * rng.standard_normal((T, U))

    X, names = dz.build_design({"evt": e}, range(0, K + 1))
    assert X.shape == (T, K + 1) and names[3] == "evt_3"
    W, b, _, _ = pg.fit_units(jnp.asarray(X), jnp.asarray(Y), 1e-4, 0.0, 1.0, 4000, 1e-10, "gaussian")
    ker = dz.kernels_from_weights(W, names)["evt"][1]                      # (K+1, U)
    corr = np.corrcoef(np.asarray(ker).ravel(), ker_true.ravel())[0, 1]
    print(f"[2] kernel recovery: corr(recovered, true) = {corr:.4f}")
    assert corr > 0.99, "shift-kernel design did not recover the true kernel"

    # viz smoke tests -> PNGs on netscratch
    mu = pg.predict_rate(jnp.asarray(X), W, b, "gaussian")
    viz.plot_kernels(np.asarray(W), names, bin_width=1 / 18.5, mean_sem=False).savefig(f"{OUT}/kernels.png", dpi=80)
    viz.plot_reconstruction(np.asarray(Y), np.asarray(mu), unit=0, window=(0, 1500)).savefig(f"{OUT}/recon.png", dpi=80)
    viz.plot_event_reconstruction(np.asarray(Y), np.asarray(mu), ev, unit=0, pre=5, post=K).savefig(f"{OUT}/event_recon.png", dpi=80)
    for f in ("kernels.png", "recon.png", "event_recon.png"):
        assert os.path.getsize(f"{OUT}/{f}") > 0
    print(f"[3] plots rendered -> {OUT}/{{kernels,recon,event_recon}}.png")


if __name__ == "__main__":
    test_shift_signal()
    test_kernel_recovery()
    print("\nDESIGN + VIZ CHECKS PASSED")
