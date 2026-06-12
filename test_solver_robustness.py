"""Guard test for the monotone-FISTA restart: the solver must (a) still converge correctly on a
well-posed problem and (b) NOT diverge on a raw ill-conditioned (collinear) design even with many
iterations -- the failure that motivated the restart (plain FISTA hit R^2=-0.5 at 300k iters).
Run on a GPU node: conda activate jaxGLM && python test_solver_robustness.py
"""
import numpy as np
import jax.numpy as jnp
import poisson_glm as pg


def r2(y, mu):
    y = np.asarray(y).ravel(); mu = np.asarray(mu).ravel()
    return 1.0 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def test_wellposed_unchanged():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 12)); w = rng.standard_normal(12); b = 0.3
    y = X @ w + b + 0.1 * rng.standard_normal(4000)
    W, B, it, conv = pg.fit_units(jnp.asarray(X), jnp.asarray(y[:, None]), 1e-4, 0.5,
                                  1.0, 20000, 1e-10, "gaussian")
    mu = np.asarray(pg.predict_rate(jnp.asarray(X), W, B, "gaussian"))
    print(f"[well-posed] R^2={r2(y, mu):.4f} converged={bool(np.asarray(conv)[0])} "
          f"iters={int(np.asarray(it)[0])} weight_corr={np.corrcoef(np.asarray(W)[:,0], w)[0,1]:.4f}")
    assert r2(y, mu) > 0.98 and np.corrcoef(np.asarray(W)[:, 0], w)[0, 1] > 0.99


def test_illconditioned_no_divergence():
    # build a deliberately collinear design: duplicate/near-duplicate columns + shifted copies
    rng = np.random.default_rng(1)
    T = 8000
    base = (rng.random((T, 4)) < 0.05).astype(float)        # sparse events
    cols = [base]
    for k in range(1, 25):                                   # many near-collinear shifted copies
        cols.append(np.roll(base, k, axis=0))
    X = np.concatenate(cols, axis=1)                         # (T, 100), highly collinear
    wtrue = rng.standard_normal(X.shape[1]) * 0.3
    y = X @ wtrue + 0.5 * rng.standard_normal(T)
    Xj, yj = jnp.asarray(X), jnp.asarray(y[:, None])
    for n_iter in (40_000, 300_000):                         # the high count is what used to diverge
        W, B, it, conv = pg.fit_units(Xj, yj, 1e-3, 0.5, 1.0, n_iter, 1e-12, "gaussian")
        mu = np.asarray(pg.predict_rate(Xj, W, B, "gaussian"))
        rr = r2(y, mu)
        print(f"[ill-conditioned raw, {n_iter} iters] R^2={rr:.4f} iters={int(np.asarray(it)[0])}")
        assert rr > 0.0, f"DIVERGED at {n_iter} iters (R^2={rr:.4f})"   # plain FISTA went negative


if __name__ == "__main__":
    test_wellposed_unchanged()
    test_illconditioned_no_divergence()
    print("\nSOLVER ROBUSTNESS OK (converges on well-posed; no divergence on raw collinear design)")
