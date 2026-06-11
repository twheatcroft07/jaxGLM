"""Self-test for encoding.py: CV lambda selection + predictor-subset delta-D^2.

Builds synthetic Poisson data whose rate depends on an INFORMATIVE predictor group and is
independent of a NOISE group. Checks that removing the informative group drops held-out D^2 a
lot (large delta-D^2) while removing the noise group barely moves it. Also runs a gaussian case.
Run on a GPU node: conda activate jaxGLM && python test_encoding.py
"""
import numpy as np
import jax.numpy as jnp

import encoding as enc

ALPHAS = np.array([1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0])


def make_data(T=6000, U=8, family="poisson", seed=0):
    rng = np.random.default_rng(seed)
    Xinfo = rng.standard_normal((T, 5))      # informative group: columns 0..4
    Xnoise = rng.standard_normal((T, 5))     # noise group: columns 5..9
    X = np.concatenate([Xinfo, Xnoise], 1)
    Winfo = rng.standard_normal((5, U)) * 0.6
    b = rng.uniform(-0.5, 0.5, size=U)
    eta = Xinfo @ Winfo + b[None, :]         # rate depends ONLY on the informative group
    if family == "poisson":
        mu = np.exp(np.clip(eta, -8, 8))
        Y = rng.poisson(mu).astype(float)
    else:
        Y = eta + 0.5 * rng.standard_normal((T, U))
    return X, Y


def run(family):
    X, Y = make_data(family=family)

    # alpha_grid: the top of the data-driven grid (alpha_max) should zero (nearly) all weights
    import jax.numpy as jnp, poisson_glm as pg
    Xs = np.asarray(pg.standardize(jnp.asarray(X))[0])
    ag = enc.alpha_grid(Xs, Y, l1_ratio=0.5)
    Wtop, _, _, _ = pg.fit_units(jnp.asarray(Xs), jnp.asarray(Y), float(ag[-1]), 0.5, 1.0, 1500, 1e-8, family)
    frac_zero = np.mean(np.abs(np.asarray(Wtop)) < 1e-6)
    print(f"[{family}/alpha_grid] range=[{ag[0]:.2g}, {ag[-1]:.2g}] | weights zeroed at alpha_max={frac_zero:.2f}")
    assert frac_zero > 0.9, "alpha_max should zero (nearly) all weights"

    subsets = {"informative": np.arange(0, 5), "noise": np.arange(5, 10)}
    res = enc.predictor_importance(X, Y, subsets, ALPHAS, l1_ratio=0.5, n_folds=5,
                                   family=family, max_iter=2000, tol=1e-8)
    d2 = np.median(res["d2_full"])
    dd_info = np.median(res["delta_d2"]["informative"])
    dd_noise = np.median(res["delta_d2"]["noise"])
    a_lo, a_hi = res["alpha_star"].min(), res["alpha_star"].max()
    print(f"[{family}] median held-out D^2={d2:.3f} | "
          f"ΔD^2 informative={dd_info:.3f}  noise={dd_noise:.4f} | "
          f"alpha* range [{a_lo:.0e}, {a_hi:.0e}]")
    assert d2 > 0.2, "full model should explain real structure"
    assert dd_info > 0.1, "removing the informative group should drop D^2 a lot"
    assert dd_info > 5 * abs(dd_noise) + 0.05, "informative contribution must dominate noise"
    assert abs(dd_noise) < 0.03, "removing a noise group should barely change D^2"


if __name__ == "__main__":
    run("poisson")
    run("gaussian")
    print("\nALL ENCODING CHECKS PASSED")
