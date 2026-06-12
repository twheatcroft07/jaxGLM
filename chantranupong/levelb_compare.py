"""Chantranupong/Lynne LEVEL B -- step 2 (jaxGLM env): fit the lab's preprocessed Figure_6 data
with jaxGLM (gaussian, near-OLS) and confirm it reproduces their OLS engine's coefficients.
"""
import os, sys
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, "/n/home02/twheatcroft/jaxGLM")
import poisson_glm as pg
import viz

DIR = "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/chantranupong"
d = np.load(os.path.join(DIR, "levelb_compare.npz"), allow_pickle=True)
X = jnp.asarray(d["X"]); y = jnp.asarray(d["y"][:, None])
coef_ref = d["coef_ref"]; alpha = float(d["alpha"]); l1 = float(d["l1_ratio"]); names = d["names"]
print(f"loaded: X {X.shape} | y {y.shape} | alpha={alpha} l1_ratio={l1}")

W, b, _, _ = pg.fit_units(X, y, alpha, l1, 1.0, 40000, 1e-12, "gaussian")
W = np.asarray(W)[:, 0]
corr = np.corrcoef(W, coef_ref)[0, 1]
maxdiff = np.max(np.abs(W - coef_ref))
scale = np.max(np.abs(coef_ref))
print(f"[coefficients vs lab OLS] corr = {corr:.4f} | max|diff| = {maxdiff:.2e} "
      f"(coef scale {scale:.2e}, rel {maxdiff/scale:.2e})")

ok = corr > 0.999 and maxdiff / scale < 0.02
print("\nCHANTRANUPONG LEVEL B:",
      "PASS -- jaxGLM reproduces the lab's OLS coefficients on their own preprocessed data" if ok
      else "REVIEW")
fig = viz.plot_kernels(np.asarray(W)[:, None], list(names), bin_width=None, mean_sem=False)
fig.savefig(os.path.join(DIR, "levelb_kernels.png"), dpi=90)
print(f"kernel figure -> {os.path.join(DIR, 'levelb_kernels.png')}")
