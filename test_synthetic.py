"""Self-test for poisson_glm.py on synthetic Poisson data. Run on a GPU node (or CPU jax):

    conda activate jaxGLM && python test_synthetic.py

Checks:
  1. fit_units recovers known weights (high corr, high D^2) at light regularization.
  2. Ridge-only (l1_ratio=0) agrees with an independent NumPy IRLS/Newton solve -- this is
     ground truth for the smooth convex problem, so it pins down correctness of the solver.
  3. L1 actually induces sparsity as alpha grows (zeros out true-zero coefficients).
  4. Batched fit_units_grid matches looping fit_units over the lambda grid.
"""
import numpy as np
import jax
import jax.numpy as jnp

import poisson_glm as pg

jax.config.update("jax_enable_x64", True)  # tight numerical comparison vs IRLS


def make_data(T=4000, P=12, U=8, sparsity=0.4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, P))
    W = rng.standard_normal((P, U)) * 0.5
    mask = rng.random((P, U)) < sparsity
    W[mask] = 0.0                              # some truly-zero coefficients
    b = rng.uniform(-1.0, 0.0, size=U)         # baseline log-rates
    eta = X @ W + b[None, :]
    mu = np.exp(np.clip(eta, -10, 10))
    Y = rng.poisson(mu).astype(float)
    return X, Y, W, b


def irls_ridge_one(X, y, l2, iters=100):
    """Reference Newton/IRLS solve of ridge Poisson GLM for one unit (with intercept).

    Uses the same mean-NLL (1/T) normalization as poisson_glm so l2 is comparable.
    """
    Xa = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    T, P1 = Xa.shape
    beta = np.zeros(P1)
    beta[-1] = np.log(y.mean() + 1e-8)
    Rpen = np.eye(P1) * l2
    Rpen[-1, -1] = 0.0                         # intercept unpenalized
    for _ in range(iters):
        eta = Xa @ beta
        mu = np.exp(np.clip(eta, -30, 30))
        grad = (Xa.T @ (mu - y)) / T + Rpen @ beta
        H = (Xa.T @ (Xa * mu[:, None])) / T + Rpen
        step = np.linalg.solve(H, grad)
        beta = beta - step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta[:-1], beta[-1]


def main():
    X, Y, W_true, b_true = make_data()
    Xz, mean, std = pg.standardize(jnp.asarray(X))
    Xz = jnp.asarray(Xz)
    Yj = jnp.asarray(Y)

    # ---- 1. recovery at light elastic-net
    alpha, l1r = 1e-3, 0.5
    Wz, bz, nit, conv = pg.fit_units(Xz, Yj, alpha, l1r, 1.0, 1000, 1e-9)
    # map back to raw-X units for comparison with W_true
    W_raw, b_raw = jax.vmap(pg.unstandardize_weights, in_axes=(1, 0, None, None),
                            out_axes=(1, 0))(Wz, bz, mean, std)
    corr = np.corrcoef(np.asarray(W_raw).ravel(), W_true.ravel())[0, 1]
    mu = pg.predict_rate(Xz, Wz, bz)
    mu_null = jnp.broadcast_to(Yj.mean(axis=0)[None, :], Yj.shape)
    d2, _, _ = pg.frac_deviance_explained(Yj, mu, mu_null)
    print(f"[1] recovery: weight corr={corr:.3f}  median D^2={np.median(np.asarray(d2)):.3f}  "
          f"converged={int(np.sum(np.asarray(conv)))}/{Yj.shape[1]}")
    assert corr > 0.9, "weight recovery too low"
    assert np.median(np.asarray(d2)) > 0.3, "D^2 too low on well-specified data"

    # ---- 2. ridge-only vs independent IRLS (ground truth, on raw X for both)
    l2 = 5.0
    # NOTE: fit_units is vmapped -> args must be positional (vmap in_axes are positional).
    Wz2, bz2, _, _ = pg.fit_units(Xz, Yj, l2, 0.0, 1.0, 5000, 1e-12)
    W_raw2, b_raw2 = jax.vmap(pg.unstandardize_weights, in_axes=(1, 0, None, None),
                              out_axes=(1, 0))(Wz2, bz2, mean, std)
    W_raw2 = np.asarray(W_raw2)
    # IRLS solved on standardized X too (penalty scale must match), then compare mu.
    # FISTA is first-order (linear convergence) vs IRLS/Newton (quadratic), so it agrees
    # to ~1e-3 rather than machine precision -- far tighter than the D^2 tolerance that
    # the downstream glmTF28 comparison actually uses (~1e-2).
    max_mu_err = 0.0
    for u in range(Yj.shape[1]):
        w_ref, b_ref = irls_ridge_one(np.asarray(Xz), Y[:, u], l2)
        mu_ref = np.exp(np.asarray(Xz) @ w_ref + b_ref)
        mu_jax = np.asarray(pg.predict_rate(Xz, Wz2[:, u:u+1], bz2[u:u+1])).ravel()
        max_mu_err = max(max_mu_err, np.max(np.abs(mu_ref - mu_jax) / (mu_ref + 1e-6)))
    print(f"[2] ridge vs IRLS: max relative mu error={max_mu_err:.2e}")
    assert max_mu_err < 5e-3, "FISTA ridge fit disagrees with IRLS ground truth"

    # ---- 3. L1 induces sparsity
    Wz_hi, _, _, _ = pg.fit_units(Xz, Yj, 0.2, 1.0, 1.0, 3000, 1e-10)
    frac_zero = np.mean(np.abs(np.asarray(Wz_hi)) < 1e-6)
    print(f"[3] L1 sparsity: fraction zero coeffs at alpha=0.2={frac_zero:.2f}")
    assert frac_zero > 0.2, "strong L1 did not zero out coefficients"

    # ---- 4. grid fit matches looped fit
    alphas = jnp.asarray([1e-3, 1e-2, 1e-1])
    Wg, bg, _, _ = pg.fit_units_grid(Xz, Yj, alphas, 0.5, 1.0, 2000, 1e-10)
    for i, a in enumerate(np.asarray(alphas)):
        Wi, bi, _, _ = pg.fit_units(Xz, Yj, float(a), 0.5, 1.0, 2000, 1e-10)
        assert np.allclose(np.asarray(Wg[i]), np.asarray(Wi), atol=1e-5), f"grid!=loop at alpha={a}"
    print("[4] grid fit matches looped fit across lambda grid")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
