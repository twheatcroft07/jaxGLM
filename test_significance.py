"""Self-test for significance testing in encoding.py: Wilcoxon full-vs-null and the circular-
shift permutation null. Builds data where half the units genuinely depend on X and half are
pure noise, and checks both tests call the real units significant and the noise units not.
Run on a GPU node: conda activate jaxGLM && python test_significance.py
"""
import numpy as np
import jax.numpy as jnp

import encoding as enc

ALPHAS = np.array([1e-3, 1e-2, 1e-1, 1.0])


def make_data(T=6000, P=8, n_real=6, n_noise=6, family="poisson", seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, P))
    Wr = rng.standard_normal((P, n_real)) * 0.5
    b = rng.uniform(-0.5, 0.5, size=n_real + n_noise)
    eta_real = X @ Wr + b[None, :n_real]
    if family == "poisson":
        Yr = rng.poisson(np.exp(np.clip(eta_real, -8, 8)))
        Yn = rng.poisson(np.exp(np.clip(b[None, n_real:] + 0 * X[:, :n_noise], -8, 8)))
    else:
        Yr = eta_real + 0.5 * rng.standard_normal((T, n_real))
        Yn = b[None, n_real:] + 0.5 * rng.standard_normal((T, n_noise))   # independent of X
    Y = np.concatenate([Yr, Yn], 1).astype(float)
    real = np.array([True] * n_real + [False] * n_noise)
    return X, Y, real


def run(family):
    X, Y, real = make_data(family=family)
    w = enc.wilcoxon_full_vs_null(X, Y, ALPHAS, l1_ratio=0.5, n_folds=10, family=family, max_iter=1500)
    p = enc.permutation_null_d2(X, Y, ALPHAS, l1_ratio=0.5, n_folds=5, family=family,
                                n_perm=100, max_iter=1500)
    for name, res in [("wilcoxon", w), ("permutation", p)]:
        sig = res["significant"]
        tpr = sig[real].mean()        # real units flagged significant
        fpr = sig[~real].mean()       # noise units flagged significant (should be low)
        print(f"[{family}/{name}] real-unit TPR={tpr:.2f}  noise-unit FPR={fpr:.2f}  "
              f"p(real)~{np.median(res['pvalue'][real]):.3g}  p(noise)~{np.median(res['pvalue'][~real]):.3g}")
        assert tpr >= 0.8, f"{name}: should flag real units significant"
        assert fpr <= 0.2, f"{name}: should NOT flag noise units significant"


if __name__ == "__main__":
    run("poisson")
    run("gaussian")
    print("\nALL SIGNIFICANCE CHECKS PASSED")
