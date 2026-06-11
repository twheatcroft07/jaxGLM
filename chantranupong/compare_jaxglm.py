"""Chantranupong photometry -- step 2 (jaxGLM env): load the saved (Xz, Y) and the sklearn
ElasticNet reference, fit the SAME data with jaxGLM family='gaussian' at the same (alpha,
l1_ratio), and report agreement. jaxGLM-gaussian and sklearn-ElasticNet share an identical
objective, so the weights should match closely (not just near the MLE).
"""
import os, sys
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, "/n/home02/twheatcroft/jaxGLM")
import poisson_glm as pg

DIR = "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/chantranupong"
d = np.load(os.path.join(DIR, "chantranupong_compare.npz"), allow_pickle=True)
Xz = jnp.asarray(d["Xz"]); Y = jnp.asarray(d["Y"])
W_ref = d["W_ref"]; r2_ref = d["r2_ref"]
alpha = float(d["alpha"]); l1 = float(d["l1_ratio"])
print(f"loaded: Xz {Xz.shape} | Y {Y.shape} | alpha={alpha} l1_ratio={l1}")

W, b, n_iter, conv = pg.fit_units(Xz, Y, alpha, l1, 1.0, 8000, 1e-11, "gaussian")
mu = pg.predict_rate(Xz, W, b, "gaussian")
r2_jax, _, _ = pg.frac_deviance_explained(Y, mu, Y.mean(0), "gaussian")
r2_jax = np.asarray(r2_jax); W = np.asarray(W)

w_corr = np.corrcoef(W.ravel(), W_ref.ravel())[0, 1]
w_maxdiff = np.max(np.abs(W - W_ref))
print(f"channels (greenL, greenR):")
print(f"  R^2  jax     = {np.round(r2_jax, 4)}")
print(f"  R^2  sklearn = {np.round(r2_ref, 4)}")
print(f"[weights] corr(jax, sklearn) = {w_corr:.4f} | max|diff| = {w_maxdiff:.4f}")
print(f"[R^2]     max|diff| = {np.max(np.abs(r2_jax - r2_ref)):.4f}")

ok = (w_corr > 0.99) and (w_maxdiff < 0.02) and np.max(np.abs(r2_jax - r2_ref)) < 0.01
print("\nCHANTRANUPONG GAUSSIAN LEVEL A:",
      "PASS -- jaxGLM gaussian reproduces sklearn ElasticNet on real photometry" if ok
      else "REVIEW -- agreement below threshold")

# kernel figure (design.py column names -> per-predictor kernels)
import viz
names = d["names"]
fig = viz.plot_kernels(W, names, dt=1.0 / 18.5, mean_sem=False)
fig.savefig(os.path.join(DIR, "chantranupong_kernels.png"), dpi=90)
print(f"kernel figure -> {os.path.join(DIR, 'chantranupong_kernels.png')}")
np.savez(os.path.join(DIR, "chantranupong_result.npz"),
         W_jax=W, W_ref=W_ref, r2_jax=r2_jax, r2_ref=r2_ref, w_corr=w_corr, passed=ok)
