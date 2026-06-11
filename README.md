# jaxGLM

**GPU-accelerated elastic-net Poisson GLM for neural encoding.** Fit hundreds of single-unit
encoding models — across predictor-subset ablations, cross-validation folds, and a
regularization grid — in one batched pass on a single GPU.

> **Status: alpha / under active development.** The core solver is implemented and validated
> (see [Validation](#validation)); the data loaders, cross-validation driver, and
> significance testing are in progress (see [Roadmap](#roadmap)).

---

## Why

Single-unit encoding models are *embarrassingly batchable*: within a recording session, every
neuron is regressed onto the **same** design matrix `X` — only the response column differs. So
`X @ W` (all units at once) is a single matrix multiply. CPU tools (e.g. scikit-learn's
`PoissonRegressor`) fit one neuron at a time in a Python loop; jaxGLM expresses the whole
session as one `jax.vmap`, turning 300 units × 30 ablations × folds × λ-values into dense
GPU matmuls.

The model, per unit:

```
eta = X @ w + b,   mu = exp(eta),   y_t ~ Poisson(mu_t)

minimize_{w,b}  (1/T) Σ_t (mu_t − y_t·eta_t)                              # mean Poisson NLL
                + alpha·( l1_ratio·‖w‖₁ + ½(1−l1_ratio)·‖w‖₂² )          # elastic net
```

The intercept `b` is unpenalized. The problem is convex; it's solved by accelerated proximal
gradient (FISTA) with backtracking line search — the L1 term via soft-thresholding. The NLL is
averaged over time (sklearn/glmnet convention) so `alpha` is comparable across session lengths.

## Requirements

- A CUDA GPU (developed on the Kempner cluster; `jax[cuda12]`).
- conda/mamba. Everything installs into a self-contained env — no cluster CUDA modules.

## Installation

Builds are heavy; run them on a GPU compute node, **not** a login node.

```bash
# from a clone of this repo, on a cluster with SLURM:
sbatch install_env.sh          # creates conda env `jaxGLM` (jax[cuda12] + optax + sci stack)
```

Caches are routed off `$HOME` to scratch inside the script. After it finishes, the log should
show `devices [CudaDevice(id=0)]`.

## Quickstart

The core fitting API lives in [`poisson_glm.py`](poisson_glm.py). `fit_units` is `jax.vmap`-ed,
so **arguments must be positional** (vmap maps over positional axes).

```python
import jax.numpy as jnp
import poisson_glm as pg

# X: (T, P) design matrix for one session (NO intercept column — it's handled internally)
# Y: (T, U) spike counts; T timebins, U units that all share X
Xz, mean, std = pg.standardize(jnp.asarray(X))          # z-score columns; keep scaler

W, b, n_iter, converged = pg.fit_units(
    Xz, jnp.asarray(Y),
    0.01,    # alpha     : elastic-net strength
    0.5,     # l1_ratio  : 0.0 = ridge, 1.0 = lasso, in-between = elastic net
    1.0,     # L0        : initial Lipschitz estimate (backtracking adapts it)
    500,     # max_iter
    1e-7,    # tol        : relative-change stopping tolerance
)
# W: (P, U) weights, b: (U,) intercepts, both in standardized-X units.

# Predicted firing rates and per-unit deviance explained (D^2) vs an intercept-only null:
mu      = pg.predict_rate(Xz, W, b)                      # (T, U)
Yj      = jnp.asarray(Y)
mu_null = jnp.broadcast_to(Yj.mean(0)[None, :], Yj.shape)
d2, dev_model, dev_null = pg.frac_deviance_explained(Yj, mu, mu_null)   # d2: (U,)

# Map weights back to raw-X units if you need interpretable coefficients:
import jax
w_raw, b_raw = jax.vmap(pg.unstandardize_weights, in_axes=(1, 0, None, None),
                        out_axes=(1, 0))(W, b, mean, std)
```

Sweep a regularization grid in one batched call (adds a leading λ axis):

```python
alphas = jnp.asarray([1e-3, 1e-2, 1e-1])
Wg, bg, _, _ = pg.fit_units_grid(Xz, Yj, alphas, 0.5, 1.0, 500, 1e-7)   # Wg: (len(alphas), P, U)
```

### Run the self-test

```bash
conda activate jaxGLM
python test_synthetic.py        # on a GPU node
```

## How it scales

`fit_units` vmaps a single-unit fit over the response axis; `fit_units_grid` vmaps that over a
λ-grid. Because `X` is shared, each FISTA step is a `(T×P)·(P×U)` matmul — the natural GPU
primitive. Predictor-subset ablations are column masks on `X`, and CV folds are row subsets, so
the full *fit × subset × fold × λ* cube is expressed as nested `vmap`/`scan` rather than Python
loops. (The subset/CV drivers are on the [Roadmap](#roadmap).)

## Validation

- **Synthetic** ([`test_synthetic.py`](test_synthetic.py)): recovers known weights (corr ≈ 0.999),
  matches an independent Newton/IRLS solve in the ridge case to ~1e-10, L1 induces sparsity,
  and the batched grid fit matches a looped fit. ✅ passing.
- **Against a published reference (in progress):** "Level A" comparison vs the International
  Brain Laboratory's `neurencoding` (scikit-learn `PoissonRegressor`) on public brain-wide-map
  spike data — feed the **identical** design matrix to both and check jaxGLM reproduces their
  per-unit fits, isolating the solver from design-matrix/basis-function choices.

## Roadmap

- [x] Core FISTA elastic-net Poisson solver, vmapped over units / λ-grid
- [x] Deviance / D² helpers, standardization
- [ ] Session data loader (design matrix `X`, response counts `Y`, subset column-masks)
- [ ] Per-unit cross-validated λ selection
- [ ] Predictor-subset leave-out → ΔD² (predictor importance)
- [ ] Significance testing (intercept-null Wilcoxon; permutation/shuffle null)
- [ ] IBL Level-A comparison harness
- [ ] Gaussian/linear family (continuous signals, e.g. photometry)

## Repository layout

```
poisson_glm.py        core solver + deviance/D² helpers (the public API)
test_synthetic.py     correctness self-test on synthetic Poisson data
install_env.sh        SLURM job: build the `jaxGLM` conda env
install_ibl_env.sh    SLURM job: build the `ibl` env for the reference comparison
```

## Background

jaxGLM is a GPU rewrite of the fitting core of an internal TensorFlow Poisson-GLM pipeline
(`refactoredHarveyGLM`, itself derived from Harvey Lab's `GLM_Tensorflow_2`). It targets the
same per-unit elastic-net Poisson encoding models, but batched on the GPU and with elastic-net
(L1+L2) support that ridge-only CPU tools lack.
