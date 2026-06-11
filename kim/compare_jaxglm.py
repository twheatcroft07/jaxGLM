"""Kimberly Reinhold reproduction -- step 2 (jaxGLM env): fit the SAME (Xz, Y) with jaxGLM
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
Xz = jnp.asarray(d["X"]); Y = jnp.asarray(d["Y"])    # raw X (no standardization -- matches sglm)
W_ref = d["W_ref"]; r2_ho_ref = d["r2_ho_ref"]; pub = d["pub_holdout"]
alpha = float(d["alpha"]); l1 = float(d["l1_ratio"]); te = d["te"].astype(bool); names = d["names"]
tr = ~te
U = Y.shape[1]
print(f"loaded: Xz {Xz.shape} | Y {Y.shape} | {U} units | alpha={alpha} l1_ratio={l1}")

# fit on the SAME train trials, score on the held-out trials
# NOTE: the published fit used max_iter=1000 (early-stopped ElasticNet, implicitly extra-
# regularized). On this ill-conditioned raw design, fully converging overfits MORE than their
# partially-solved model, so we match their iteration budget rather than solve to high precision.
W, b, _, _ = pg.fit_units(Xz[tr], Y[tr], alpha, l1, 1.0, 1000, 1e-8, "gaussian")
W = np.asarray(W)
mu_ho = pg.predict_rate(Xz[te], jnp.asarray(W), b, "gaussian")
r2_ho_jax, _, _ = pg.frac_deviance_explained(Y[te], mu_ho, Y[te].mean(0), "gaussian")
r2_ho_jax = np.asarray(r2_ho_jax)

w_corr = np.corrcoef(W.ravel(), W_ref.ravel())[0, 1]
# two neurons (published = 0) are degenerate and blow up the holdout R^2 in both fitters; exclude
# them for the aggregate stats (they agree jax-vs-sklearn, just astronomically negative).
sane = np.abs(r2_ho_ref) < 5
print(f"[solver]   weight corr(jax, sklearn) = {w_corr:.4f} | "
      f"holdout R^2 corr(jax, sklearn) = {np.corrcoef(r2_ho_jax[sane], r2_ho_ref[sane])[0,1]:.4f}")
print(f"[vs published] median holdout R^2: jax={np.median(r2_ho_jax[sane]):.3f}  "
      f"sklearn={np.median(r2_ho_ref[sane]):.3f}  published={np.median(pub[sane]):.3f}")
pub_corr = np.corrcoef(r2_ho_jax[sane], pub[sane])[0, 1]
print(f"[vs published] per-neuron holdout R^2 corr(jax, published) = {pub_corr:.3f}")

# the meaningful test: jaxGLM's predictions match sklearn (R^2 corr) AND reproduce Reinhold's
# published per-neuron R^2. Exact weight match is NOT expected here -- the raw design is ill-
# conditioned/degenerate (collinear shifted columns, all-zero events) and the published fit is
# early-stopped (max_iter=1000), so many weight vectors give the same predictions.
r2_corr = np.corrcoef(r2_ho_jax[sane], r2_ho_ref[sane])[0, 1]
ok = (r2_corr > 0.95) and (pub_corr > 0.8)
print("\nKIM (REINHOLD) REPRODUCTION:",
      "PASS -- jaxGLM predictions match sklearn ElasticNet and reproduce the published per-neuron R^2"
      if ok else "REVIEW")
fig = viz.plot_kernels(W[:-1], list(names[:-1]), bin_width=None, mean_sem=True)  # drop nTrial col
fig.savefig(os.path.join(DIR, "kim_kernels.png"), dpi=90)
np.savez(os.path.join(DIR, "kim_result.npz"), W_jax=W, W_ref=W_ref,
         r2_ho_jax=r2_ho_jax, r2_ho_ref=r2_ho_ref, pub=pub, w_corr=w_corr, passed=ok)
print(f"kernel figure -> {os.path.join(DIR, 'kim_kernels.png')}")
