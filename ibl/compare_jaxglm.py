"""Level A — step 2b (jaxGLM env): load the saved (Xz, Y) and the sklearn reference fit, fit
the SAME data with jaxGLM (ridge, l1_ratio=0, matched near-MLE alpha), and report agreement.

Pass criteria (identical X, unique convex optimum -> solvers must agree):
  - per-unit D^2: high correlation, small max |diff|
  - per-unit weights: high correlation
If these hold on real IBL spike data, jaxGLM reproduces an independent, published Poisson-GLM
fitter end-to-end.
"""
import os, sys
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, "/n/home02/twheatcroft/jaxGLM")
import poisson_glm as pg

DIR = "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/ibl_level_a"
d = np.load(os.path.join(DIR, "level_a_compare.npz"))
Xz = jnp.asarray(d["Xz"]); Y = jnp.asarray(d["Y"])
W_ref = d["W_ref"]; d2_ref = d["d2_ref"]; alpha = float(d["alpha"])
U = int(d["n_units"])
print(f"loaded: Xz {Xz.shape} | Y {Y.shape} | {U} units | ref alpha {alpha}")

# jaxGLM ridge fit on the identical standardized X
W, b, n_iter, conv = pg.fit_units(Xz, Y, alpha, 0.0, 1.0, 4000, 1e-10)
mu = pg.predict_rate(Xz, W, b)
mu_null = jnp.broadcast_to(Y.mean(0)[None, :], Y.shape)
d2_jax, _, _ = pg.frac_deviance_explained(Y, mu, mu_null)
d2_jax = np.asarray(d2_jax); W = np.asarray(W)

# ---- agreement metrics
d2_corr = np.corrcoef(d2_jax, d2_ref)[0, 1]
d2_maxdiff = np.max(np.abs(d2_jax - d2_ref))
w_corr = np.corrcoef(W.ravel(), W_ref.ravel())[0, 1]
print(f"converged units: {int(conv.sum())}/{U}")
print(f"[D^2]    corr(jax, sklearn) = {d2_corr:.4f} | max|diff| = {d2_maxdiff:.4f}")
print(f"[D^2]    median jax = {np.median(d2_jax):.4f} | median sklearn = {np.median(d2_ref):.4f}")
print(f"[weights] corr(jax, sklearn) = {w_corr:.4f}")

ok = (d2_corr > 0.99) and (d2_maxdiff < 0.02) and (w_corr > 0.99)
print("\nLEVEL A:", "PASS -- jaxGLM reproduces the sklearn/neurencoding reference" if ok
      else "REVIEW -- agreement below threshold, inspect")

np.savez(os.path.join(DIR, "level_a_result.npz"),
         d2_jax=d2_jax, d2_ref=d2_ref, W_jax=W, W_ref=W_ref,
         d2_corr=d2_corr, d2_maxdiff=d2_maxdiff, w_corr=w_corr, passed=ok)

# kernel figure: per-event encoding kernels (mean +/- s.e.m. across units)
import viz
names = d["names"]
fig = viz.plot_kernels(W, names, bin_width=0.02, mean_sem=True)
fig.savefig(os.path.join(DIR, "ibl_kernels.png"), dpi=90)
print(f"kernel figure -> {os.path.join(DIR, 'ibl_kernels.png')}")
