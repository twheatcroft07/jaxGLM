# jaxGLM

**GPU-accelerated elastic-net Poisson GLM for neural encoding.** Fits hundreds of single-unit
encoding models — across predictor-subset ablations, cross-validation folds, and a
regularization grid — in one batched pass on a single GPU.

> **Status: alpha / under active development.** The core solver is implemented and validated
> (see [Validation](#validation)); the data loaders, cross-validation driver, and significance
> testing are in progress (see [Roadmap](#roadmap)).

---

## Installation

Builds are heavy, so they run as SLURM jobs on a GPU node, not on a login node:

```bash
sbatch install_env.sh          # creates conda env `jaxGLM` (jax[cuda12] + optax + sci stack)
```

The script routes package caches off `$HOME`. When it finishes, its log shows
`devices [CudaDevice(id=0)]`, confirming JAX sees the GPU.

## How it works

This section explains what the pipeline does and how the pieces fit together. (Agents run the
code; this is here so a person can understand the model and the data flow.)

### What it fits

For each recorded unit (neuron), jaxGLM fits a **Poisson generalized linear model** that predicts
the unit's spike count in each time bin from a set of behavioral/task predictors:

```
eta = X · w + b           (linear predictor)
mu  = exp(eta)            (expected spike count; log link)
y   ~ Poisson(mu)         (counts are Poisson around that rate)
```

Each unit gets its own weight vector `w` and intercept `b`. The weights are fit by minimizing
the Poisson negative log-likelihood plus an **elastic-net** penalty (a mix of L1 and L2) that
shrinks weights and, via the L1 part, can zero out uninformative predictors:

```
minimize  (1/T) Σ_t (mu_t − y_t·eta_t)  +  alpha·( l1_ratio·‖w‖₁ + ½(1−l1_ratio)·‖w‖₂² )
```

The intercept is never penalized. The problem is convex, so there is a single best solution;
jaxGLM finds it with accelerated proximal gradient (FISTA). The likelihood is averaged over time
bins, which keeps the penalty strength `alpha` on a consistent scale regardless of how long a
session is.

### The two inputs: `X` and `Y`

Everything is organized around two matrices for a recording session:

- **`X`** — the **design matrix**, shape `(T, P)`: `T` time bins by `P` predictors. Predictors
  are things like task events (stimulus, choice, reward), often expanded into time-shifted or
  basis-function kernels. `X` does **not** include an intercept column — that's handled
  internally.
- **`Y`** — the **response matrix**, shape `(T, U)`: the same `T` time bins by `U` units. Each
  column is one neuron's binned spike counts.

The key fact that makes this fast: **every unit in a session shares the same `X`** — only its
column of `Y` differs. So fitting all `U` units at once is one matrix multiply (`X · W`) rather
than a Python loop over neurons. jaxGLM is built around this: it `vmap`s a single-unit fit across
all units (and across a regularization grid, and across predictor subsets and CV folds), turning
the whole workload into dense GPU matmuls. This is the main thing it does that CPU tools
(scikit-learn's `PoissonRegressor`, one neuron at a time) cannot.

### What you get out

Fitting returns, for every unit at once: the weight matrix `W` (`P × U`), the intercepts `b`,
and convergence info. From those you can compute each unit's **predicted firing rate** over time
and its **fraction of deviance explained (D²)** — how much better the full model predicts the
unit's spikes than an intercept-only baseline. D² is the headline per-unit quality measure, the
GLM analogue of variance explained.

The intended scientific workflow (subset ablations) drops groups of predictors from `X`, refits,
and measures how much D² falls — the drop attributable to each predictor group is that variable's
**encoding contribution**. Significance is assessed by comparing against a null (intercept-only,
and later a shuffle/permutation null). These drivers are on the [Roadmap](#roadmap); the solver
they build on is done.

### The knobs

| Parameter   | Meaning |
|-------------|---------|
| `alpha`     | Overall regularization strength (larger ⇒ more shrinkage). |
| `l1_ratio`  | Balance of penalties: `0.0` = pure ridge (L2), `1.0` = pure lasso (L1), in between = elastic net. |
| `max_iter`  | Iteration cap for the FISTA solver. |
| `tol`       | Stopping tolerance on the relative change between iterations. |

The public entry points live in [`poisson_glm.py`](poisson_glm.py): `fit_units` (fit every unit
for one `alpha`), `fit_units_grid` (also sweep a list of `alpha`s), plus `predict_rate`,
`frac_deviance_explained`, and `standardize` helpers. `fit_units` is `jax.vmap`-ed, so its
arguments are positional. [`test_synthetic.py`](test_synthetic.py) is a worked end-to-end example.

## Validation

- **Synthetic** ([`test_synthetic.py`](test_synthetic.py)): recovers known weights (corr ≈ 0.999),
  matches an independent Newton/IRLS solve in the ridge case to ~1e-10, L1 induces sparsity, and
  the batched grid fit matches a looped fit. ✅ passing.
- **Against a published reference (in progress):** a "Level A" comparison vs the International
  Brain Laboratory's `neurencoding` (scikit-learn `PoissonRegressor`) on public brain-wide-map
  spike data. Both fit the **identical** design matrix, so the comparison isolates the solver
  from design-matrix/basis-function choices — if jaxGLM doesn't reproduce their per-unit fits,
  it's a bug in jaxGLM.

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
ibl/                  Level-A comparison scaffolding (IBL data extraction)
```
