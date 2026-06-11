"""Elastic-net Poisson GLM in JAX, fit per-unit, vmappable over units / subsets / lambdas.

Goal: reproduce the *fitting target* of refactoredHarveyGLM/refactoredGLM.py (which wraps
GLM_Tensorflow_2's GLM/GLM_CV) but on the GPU via jax.vmap, so that 300 units x ~30
predictor subsets x CV folds x a lambda grid all fit in one batched pass.

Model (per unit):
    eta = X @ w + b,   mu = exp(eta),   y_t ~ Poisson(mu_t)
    minimize_{w,b}  sum_t (mu_t - y_t * eta_t)                          # Poisson NLL (const dropped)
                    + alpha * ( l1_ratio*||w||_1 + 0.5*(1-l1_ratio)*||w||_2^2 )   # elastic net

The intercept b is unpenalized. Convex problem; solved by accelerated proximal gradient
(FISTA, Beck & Teboulle 2009) with backtracking line search. The smooth part is the NLL
plus the L2 ridge term; the L1 term is handled by the soft-threshold prox.

Key structural fact exploited downstream: within a session all units share the same design
matrix X, so `X @ W` (W: P x units) is a single matmul -- ideal for the GPU. This module
keeps the per-unit solver pure and vmappable; batching lives in driver code.

Conventions:
- Standardize X columns (zero mean, unit std) before fitting so the elastic-net penalty is
  applied on a common scale. Predictions / deviance / D^2 are invariant to this; only the
  numeric value of alpha that selects a given amount of shrinkage changes. Weights can be
  mapped back to raw-column units with `unstandardize_weights`.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

# exp(30) ~ 1e13: clips runaway linear predictors to avoid inf/nan during line search
# while staying far outside the range any converged Poisson fit reaches.
ETA_CLIP = 30.0


# ----------------------------------------------------------------------------- primitives
def _mu(eta):
    return jnp.exp(jnp.clip(eta, -ETA_CLIP, ETA_CLIP))


def _smooth_f(w, b, X, y, l2):
    """Smooth part of the objective: mean Poisson NLL + ridge term (no L1).

    The NLL is averaged over timepoints (1/T), the sklearn/glmnet convention, so that
    `alpha` lives on an O(1) scale independent of session length T -- this is what makes
    the elastic-net penalty comparable across sessions and matchable to glmTF28.
    """
    eta = X @ w + b
    mu = _mu(eta)
    T = y.shape[0]
    return jnp.sum(mu - y * eta) / T + 0.5 * l2 * jnp.sum(w * w)


def _smooth_grad(w, b, X, y, l2):
    eta = X @ w + b
    mu = _mu(eta)
    r = mu - y
    T = y.shape[0]
    gw = (X.T @ r) / T + l2 * w
    gb = jnp.sum(r) / T
    return gw, gb


def _soft_threshold(z, thresh):
    return jnp.sign(z) * jnp.maximum(jnp.abs(z) - thresh, 0.0)


# ------------------------------------------------------------------------------- FISTA fit
@partial(jax.jit, static_argnums=(5,))
def fit_one(X, y, alpha, l1_ratio, L0=1.0, max_iter=500, tol=1e-7):
    """Fit one unit. Returns (w, b, n_iter, converged).

    X: (T, P) standardized design matrix (intercept NOT a column).
    y: (T,)   nonnegative counts.
    alpha, l1_ratio: elastic-net scalars (sklearn convention).
    Backtracking inflates the Lipschitz estimate L by `bt_factor` until the quadratic
    upper bound holds; L is carried (warm) across FISTA iterations.
    """
    T, P = X.shape
    l2 = alpha * (1.0 - l1_ratio)
    l1 = alpha * l1_ratio
    bt_factor = 1.5
    bt_max = 50  # backtracking halvings cap per FISTA step

    w0 = jnp.zeros(P)
    b0 = jnp.log(jnp.mean(y) + 1e-8)  # intercept-only MLE: good warm start

    def prox_step(v_w, v_b, gw, gb, L):
        # gradient step at the accelerated point, then prox (soft-threshold on w only)
        z_w = _soft_threshold(v_w - gw / L, l1 / L)
        z_b = v_b - gb / L
        return z_w, z_b

    def backtrack(v_w, v_b, L):
        fv = _smooth_f(v_w, v_b, X, y, l2)
        gw, gb = _smooth_grad(v_w, v_b, X, y, l2)

        def cond(st):
            L_, ok, _zw, _zb, i = st
            return jnp.logical_and(jnp.logical_not(ok), i < bt_max)

        def body(st):
            L_, _ok, _zw, _zb, i = st
            z_w, z_b = prox_step(v_w, v_b, gw, gb, L_)
            fz = _smooth_f(z_w, z_b, X, y, l2)
            dw = z_w - v_w
            db = z_b - v_b
            Q = fv + jnp.vdot(dw, gw) + db * gb + 0.5 * L_ * (jnp.vdot(dw, dw) + db * db)
            ok = fz <= Q + 1e-12
            return (jnp.where(ok, L_, L_ * bt_factor), ok, z_w, z_b, i + 1)

        z_w0, z_b0 = prox_step(v_w, v_b, gw, gb, L)
        L_out, _ok, z_w, z_b, _i = lax.while_loop(cond, body, (L, False, z_w0, z_b0, 0))
        return z_w, z_b, L_out

    def cond(state):
        w, b, w_prev, b_prev, t, L, it, change = state
        return jnp.logical_and(it < max_iter, change > tol)

    def body(state):
        w, b, w_prev, b_prev, t, L, it, _change = state
        mom = (t - 1.0) / ((1.0 + jnp.sqrt(1.0 + 4.0 * t * t)) / 2.0)
        v_w = w + mom * (w - w_prev)
        v_b = b + mom * (b - b_prev)
        z_w, z_b, L_new = backtrack(v_w, v_b, L)
        t_new = (1.0 + jnp.sqrt(1.0 + 4.0 * t * t)) / 2.0
        denom = jnp.maximum(jnp.sqrt(jnp.vdot(w, w)) + jnp.abs(b), 1e-12)
        change = (jnp.sqrt(jnp.vdot(z_w - w, z_w - w)) + jnp.abs(z_b - b)) / denom
        return (z_w, z_b, w, b, t_new, L_new, it + 1, change)

    init = (w0, b0, w0, b0, 1.0, L0, 0, jnp.inf)
    w, b, _wp, _bp, _t, _L, it, change = lax.while_loop(cond, body, init)
    return w, b, it, change <= tol


# ----------------------------------------------------------------- batching over units etc.
# All units in a session share X -> vmap only over y (and optionally over alpha for a grid).
fit_units = jax.jit(
    jax.vmap(fit_one, in_axes=(None, 1, None, None, None, None, None), out_axes=(1, 0, 0, 0)),
    static_argnums=(5,),
)
"""fit_units(X, Y, alpha, l1_ratio, L0, max_iter, tol):
   X: (T, P); Y: (T, U).  Returns W (P, U), b (U,), n_iter (U,), converged (U,)."""

# vmap over a lambda grid as well: alphas (A,) -> W (A, P, U), b (A, U), ...
fit_units_grid = jax.jit(
    jax.vmap(fit_units, in_axes=(None, None, 0, None, None, None, None),
             out_axes=(0, 0, 0, 0)),
    static_argnums=(5,),
)


# --------------------------------------------------------------------- deviance / D^2 / pred
def predict_rate(X, W, b):
    """mu for X:(T,P), W:(P,U), b:(U,) -> (T, U)."""
    return _mu(X @ W + b[None, :])


def poisson_deviance(y, mu, eps=1e-10):
    """Per-unit Poisson deviance. y, mu: (T, U) -> (U,)."""
    mu = jnp.clip(mu, eps, None)
    term = y * jnp.log(jnp.clip(y, eps, None) / mu) - (y - mu)
    return 2.0 * jnp.sum(term, axis=0)


def frac_deviance_explained(y, mu, mu_null, eps=1e-10):
    """D^2 = 1 - dev(model)/dev(null), per unit. y/mu:(T,U), mu_null:(T,U) or (U,)."""
    if mu_null.ndim == 1:
        mu_null = jnp.broadcast_to(mu_null[None, :], y.shape)
    dev_m = poisson_deviance(y, mu, eps)
    dev_0 = poisson_deviance(y, mu_null, eps)
    return 1.0 - dev_m / jnp.clip(dev_0, eps, None), dev_m, dev_0


# ---------------------------------------------------------------------------- standardize
def standardize(X, eps=1e-8):
    """Return (Xz, mean, std). Constant columns get std=1 (left unscaled)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = jnp.where(sd < eps, 1.0, sd)
    return (X - mu) / sd, mu, sd


def unstandardize_weights(w_z, b_z, mean, std):
    """Map weights fit on standardized X back to raw-X units."""
    w = w_z / std
    b = b_z - jnp.sum(w_z * mean / std)
    return w, b
