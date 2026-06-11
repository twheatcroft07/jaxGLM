"""Self-test for the regularization diagnostics: poisson_glm.objective_terms and
encoding.regularization_path / viz.plot_regularization_path.

Validates that objective_terms really is the objective the solver minimized:
  (a) it matches an INDEPENDENT numpy recomputation of data + L1 + L2;
  (b) the fitted weights are a minimum -- perturbing them does not lower `total`.
Then renders the 'right scale' path figure to a PNG (netscratch).
Run on a GPU node: conda activate jaxGLM && python test_diagnostics.py
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)   # match numpy float64 for the exact-decomposition check

import poisson_glm as pg
import encoding as enc
import viz

OUT = "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/jaxglm_viz"
os.makedirs(OUT, exist_ok=True)


def run(family):
    rng = np.random.default_rng(0)
    T, P, U = 4000, 8, 6
    X = rng.standard_normal((T, P))
    W = rng.standard_normal((P, U)) * 0.5
    b = rng.uniform(-0.3, 0.3, size=U)
    eta = X @ W + b[None, :]
    Y = (rng.poisson(np.exp(np.clip(eta, -8, 8))) if family == "poisson"
         else eta + 0.5 * rng.standard_normal((T, U))).astype(float)
    Xz = np.asarray(pg.standardize(jnp.asarray(X))[0])
    alphas = enc.alpha_grid(Xz, Y, l1_ratio=0.5)

    a = float(alphas[len(alphas) // 2])
    Wf, bf, _, _ = pg.fit_units(jnp.asarray(Xz), jnp.asarray(Y), a, 0.5, 1.0, 4000, 1e-10, family)
    ot = {k: np.asarray(v) for k, v in pg.objective_terms(Xz, Y, Wf, bf, a, 0.5, family).items()}

    # (a) independent numpy recomputation of the SAME objective (nll data term + penalty)
    Wn = np.asarray(Wf); bn = np.asarray(bf)
    en = Xz @ Wn + bn[None, :]
    if family == "poisson":
        nll_np = np.sum(np.exp(np.clip(en, -30, 30)) - Y * en, 0) / T
    else:
        nll_np = 0.5 * np.sum((Y - en) ** 2, 0) / T
    l1_np = a * 0.5 * np.sum(np.abs(Wn), 0)
    l2_np = 0.5 * a * 0.5 * np.sum(Wn ** 2, 0)
    assert np.allclose(nll_np, ot["nll"], rtol=1e-4), "nll term mismatch"
    assert np.allclose(l1_np + l2_np, ot["penalty"], rtol=1e-4), "penalty term mismatch"
    assert np.allclose(ot["total"], ot["nll"] + ot["penalty"]), "total != nll + penalty"

    # (b) fitted weights are a minimum: perturbing W must not lower the mean objective
    Wp = jnp.asarray(Wn + 0.05 * rng.standard_normal(Wn.shape))
    tot_p = np.asarray(pg.objective_terms(Xz, Y, Wp, bf, a, 0.5, family)["total"])
    assert tot_p.mean() >= ot["total"].mean() - 1e-9, "perturbation lowered the objective (not a min)"

    # recon is a proper (>=0) reconstruction term, and the penalty fraction is well-defined in [0,1]
    assert np.all(ot["recon"] >= -1e-9), "recon (deviance/2T) should be >= 0"
    assert np.all((ot["penalty_frac"] >= 0) & (ot["penalty_frac"] <= 1)), "penalty_frac out of [0,1]"
    print(f"[{family}] objective_terms OK | recon median={np.median(ot['recon']):.3g} "
          f"penalty/(penalty+recon) median={np.median(ot['penalty_frac']):.3f}")

    # regularization path + figure
    path = enc.regularization_path(Xz, Y, alphas, l1_ratio=0.5, family=family, max_iter=1500)
    viz.plot_regularization_path(path).savefig(f"{OUT}/regpath_{family}.png", dpi=80)
    assert os.path.getsize(f"{OUT}/regpath_{family}.png") > 0
    print(f"[{family}] reg-path fig saved | CV-optimal alpha={path['alpha_cvmin']:.3g}")


if __name__ == "__main__":
    run("poisson")
    run("gaussian")
    print("\nDIAGNOSTICS CHECKS PASSED")
