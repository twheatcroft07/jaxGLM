"""Elastic-net GLM in JAX, fit per-unit, vmappable over units / subsets / lambdas.

Supports two response families via a `family` argument:
    poisson  (spikes, log link)        -- the default
    gaussian (continuous, e.g. photometry; identity link / least squares)

Goal: reproduce the *fitting target* of refactoredHarveyGLM/refactoredGLM.py (which wraps
GLM_Tensorflow_2's GLM/GLM_CV) but on the GPU via jax.vmap, so that 300 units x ~30
predictor subsets x CV folds x a lambda grid all fit in one batched pass.

Model (per unit), e.g. poisson:
    eta = X @ w + b,   mu = exp(eta),   y_t ~ Poisson(mu_t)
    minimize_{w,b}  sum_t (mu_t - y_t * eta_t)                          # family loss (const dropped)
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


# ------------------------------------------------------------------------------- families
# For a canonical link the gradient of the loss wrt the weights is X^T (mu - y) for EVERY
# family, so the whole solver is family-agnostic. Only three things differ per family: the
# inverse-link mu(eta), the loss *value* (needed by the backtracking line search), and the
# intercept warm-start. `family` is always a static (trace-time) argument.
#
#   poisson  : log link,      mu = exp(eta),  loss = Σ(mu − y·eta)   -> D² = frac dev explained
#   gaussian : identity link, mu = eta,        loss = ½‖y − eta‖²     -> D² is ordinary R²
def _family_funcs(family):
    if family == "poisson":
        def mu(eta):
            return jnp.exp(jnp.clip(eta, -ETA_CLIP, ETA_CLIP))

        def loss(eta, y):
            return jnp.sum(mu(eta) - y * eta)      # Poisson NLL (constant dropped)

        def b0(y):
            return jnp.log(jnp.mean(y) + 1e-8)     # intercept-only MLE warm start
    elif family == "gaussian":
        def mu(eta):
            return eta                              # identity link, no overflow clipping

        def loss(eta, y):
            r = y - eta
            return 0.5 * jnp.sum(r * r)            # least squares

        def b0(y):
            return jnp.mean(y)
    else:
        raise ValueError(f"unknown family {family!r} (expected 'poisson' or 'gaussian')")
    return mu, loss, b0


def _soft_threshold(z, thresh):
    return jnp.sign(z) * jnp.maximum(jnp.abs(z) - thresh, 0.0)


# ------------------------------------------------------------------------------- FISTA fit
@partial(jax.jit, static_argnums=(5, 7))
def fit_one(X, y, alpha, l1_ratio, L0=1.0, max_iter=500, tol=1e-7, family="poisson"):
    """Fit one unit. Returns (w, b, n_iter, converged).

    X: (T, P) standardized design matrix (intercept NOT a column).
    y: (T,)   responses -- nonnegative counts (poisson) or continuous (gaussian).
    alpha, l1_ratio: elastic-net scalars (sklearn convention).
    family: "poisson" (spikes) or "gaussian" (continuous, e.g. photometry).
    Backtracking inflates the Lipschitz estimate L by `bt_factor` until the quadratic
    upper bound holds; L is carried (warm) across FISTA iterations.

    The smooth objective is the family loss averaged over time (1/T, sklearn/glmnet
    convention so alpha is comparable across session lengths) plus the ridge term.
    """
    T, P = X.shape
    l2 = alpha * (1.0 - l1_ratio)
    l1 = alpha * l1_ratio
    bt_factor = 1.5
    bt_max = 50  # backtracking halvings cap per FISTA step
    mu_fn, loss_fn, b0_fn = _family_funcs(family)

    def smooth_f(w, b):
        eta = X @ w + b
        return loss_fn(eta, y) / T + 0.5 * l2 * jnp.sum(w * w)

    def smooth_grad(w, b):
        eta = X @ w + b
        r = mu_fn(eta) - y                       # canonical-link gradient, family-agnostic
        return (X.T @ r) / T + l2 * w, jnp.sum(r) / T

    w0 = jnp.zeros(P)
    b0 = b0_fn(y)                                 # family-specific intercept warm start

    def prox_step(v_w, v_b, gw, gb, L):
        # gradient step at the accelerated point, then prox (soft-threshold on w only)
        z_w = _soft_threshold(v_w - gw / L, l1 / L)
        z_b = v_b - gb / L
        return z_w, z_b

    def backtrack(v_w, v_b, L):
        fv = smooth_f(v_w, v_b)
        gw, gb = smooth_grad(v_w, v_b)

        def cond(st):
            L_, ok, _zw, _zb, i = st
            return jnp.logical_and(jnp.logical_not(ok), i < bt_max)

        def body(st):
            L_, _ok, _zw, _zb, i = st
            z_w, z_b = prox_step(v_w, v_b, gw, gb, L_)
            fz = smooth_f(z_w, z_b)
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
# Thin Python wrappers give defaults (incl. family); the jitted cores take args positionally.
_fit_units = jax.jit(
    jax.vmap(fit_one, in_axes=(None, 1, None, None, None, None, None, None),
             out_axes=(1, 0, 0, 0)),
    static_argnums=(5, 7),
)
_fit_units_grid = jax.jit(
    jax.vmap(_fit_units, in_axes=(None, None, 0, None, None, None, None, None),
             out_axes=(0, 0, 0, 0)),
    static_argnums=(5, 7),
)


def fit_units(X, Y, alpha, l1_ratio, L0=1.0, max_iter=500, tol=1e-7, family="poisson"):
    """Fit every unit at once. X: (T, P); Y: (T, U).
    Returns W (P, U), b (U,), n_iter (U,), converged (U,)."""
    return _fit_units(X, Y, alpha, l1_ratio, L0, max_iter, tol, family)


def fit_units_grid(X, Y, alphas, l1_ratio, L0=1.0, max_iter=500, tol=1e-7, family="poisson"):
    """Also sweep a list of alphas. alphas: (A,) -> W (A, P, U), b (A, U), ..."""
    return _fit_units_grid(X, Y, alphas, l1_ratio, L0, max_iter, tol, family)


# Per-unit alpha: each unit fit at its OWN regularization (e.g. CV-selected lambda).
_fit_units_alpha = jax.jit(
    jax.vmap(fit_one, in_axes=(None, 1, 0, None, None, None, None, None),
             out_axes=(1, 0, 0, 0)),
    static_argnums=(5, 7),
)


def fit_units_alpha(X, Y, alphas_per_unit, l1_ratio, L0=1.0, max_iter=500, tol=1e-7,
                    family="poisson"):
    """Fit every unit with its own alpha. alphas_per_unit: (U,) -> W (P, U), b (U,), ..."""
    return _fit_units_alpha(X, Y, alphas_per_unit, l1_ratio, L0, max_iter, tol, family)


# --------------------------------------------------------------------- deviance / D^2 / pred
def predict_rate(X, W, b, family="poisson"):
    """Predicted mean mu for X:(T,P), W:(P,U), b:(U,) -> (T, U)."""
    mu_fn, _, _ = _family_funcs(family)
    return mu_fn(X @ W + b[None, :])


def deviance(y, mu, family="poisson", eps=1e-10):
    """Per-unit deviance. y, mu: (T, U) -> (U,).
    poisson: Poisson deviance; gaussian: residual sum of squares (so D^2 below is R^2)."""
    if family == "poisson":
        mu = jnp.clip(mu, eps, None)
        term = y * jnp.log(jnp.clip(y, eps, None) / mu) - (y - mu)
        return 2.0 * jnp.sum(term, axis=0)
    elif family == "gaussian":
        r = y - mu
        return jnp.sum(r * r, axis=0)
    raise ValueError(f"unknown family {family!r}")


def frac_deviance_explained(y, mu, mu_null, family="poisson", eps=1e-10):
    """D^2 = 1 - dev(model)/dev(null), per unit. y/mu:(T,U), mu_null:(T,U) or (U,).
    For gaussian this is ordinary R^2 (with mu_null = mean(y))."""
    if mu_null.ndim == 1:
        mu_null = jnp.broadcast_to(mu_null[None, :], y.shape)
    dev_m = deviance(y, mu, family, eps)
    dev_0 = deviance(y, mu_null, family, eps)
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
