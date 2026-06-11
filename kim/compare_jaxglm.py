"""Kim Rein reproduction -- step 2 (jaxGLM env): fit the SAME (Xz, Y) with jaxGLM
family='gaussian' at the lab's hyperparameters, and compare (a) to the sklearn ElasticNet
reference (solver isolation) and (b) the held-out R^2 to the lab's PUBLISHED scores.
"""
import os, sys
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, "/n/home02/twheatcroft/jaxGLM")
import poisson_glm as pg
import viz

DIR = "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/kim"
d = np.load(os.path.join(DIR, "kim_compare.npz"), allow_pickle=True)
Xz = jnp.asarray(d["Xz"]); Y = jnp.asarray(d["Y"])
W_ref = d["W_ref"]; r2_ho_ref = d["r2_ho_ref"]; pub = d["pub_holdout"]
alpha = float(d["alpha"]); l1 = float(d["l1_ratio"]); ntr = int(d["ntr"]); names = d["names"]
U = Y.shape[1]
print(f"loaded: Xz {Xz.shape} | Y {Y.shape} | {U} units | alpha={alpha} l1_ratio={l1}")

# fit on the SAME train split, score on the held-out tail
W, b, _, _ = pg.fit_units(Xz[:ntr], Y[:ntr], alpha, l1, 1.0, 10000, 1e-10, "gaussian")
W = np.asarray(W)
mu_ho = pg.predict_rate(Xz[ntr:], jnp.asarray(W), b, "gaussian")
r2_ho_jax, _, _ = pg.frac_deviance_explained(Y[ntr:], mu_ho, Y[ntr:].mean(0), "gaussian")
r2_ho_jax = np.asarray(r2_ho_jax)

w_corr = np.corrcoef(W.ravel(), W_ref.ravel())[0, 1]
print(f"[solver] weight corr(jax, sklearn) = {w_corr:.4f} | max|diff| = {np.max(np.abs(W - W_ref)):.4f}")
print(f"[holdout R^2] median: jax={np.median(r2_ho_jax):.3f}  sklearn={np.median(r2_ho_ref):.3f}  "
      f"published={np.median(pub):.3f}")
print(f"[holdout R^2] corr(jax, published per-unit) = {np.corrcoef(r2_ho_jax, pub)[0,1]:.3f}")

ok = (w_corr > 0.99) and (np.max(np.abs(r2_ho_jax - r2_ho_ref)) < 0.02)
print("\nKIM REPRODUCTION:",
      "PASS -- jaxGLM reproduces sklearn ElasticNet; holdout R^2 tracks the published values" if ok
      else "REVIEW")
fig = viz.plot_kernels(W[:-1], list(names[:-1]), bin_width=None, mean_sem=True)  # drop nTrial col
fig.savefig(os.path.join(DIR, "kim_kernels.png"), dpi=90)
np.savez(os.path.join(DIR, "kim_result.npz"), W_jax=W, W_ref=W_ref,
         r2_ho_jax=r2_ho_jax, r2_ho_ref=r2_ho_ref, pub=pub, w_corr=w_corr, passed=ok)
print(f"kernel figure -> {os.path.join(DIR, 'kim_kernels.png')}")
