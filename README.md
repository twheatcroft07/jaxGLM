# jaxGLM

**GPU-accelerated elastic-net GLM for neural encoding.** A full pipeline — from event-kernel
design matrix to cross-validated, significance-tested encoding kernels — that fits **all units in
a session at once** on a single GPU. Supports **Poisson** (spike counts) and **Gaussian**
(continuous signals, e.g. fiber photometry) responses.

> **Status: research code, but the pipeline is complete and validated.** The solver and the
> encoding/significance drivers are implemented and tested on synthetic data and reproduced
> against published references on real public datasets (see [Validation](#validation)).

---

## Pipeline at a glance

| module | what it does |
|---|---|
| [`design.py`](design.py) | Build a **shift-kernel design matrix** `X` from event/regressor time series (`build_design`); parse fitted weights back into per-predictor kernels (`kernels_from_weights`). |
| [`poisson_glm.py`](poisson_glm.py) | The **GPU solver**: batched elastic-net GLM fit over all units (FISTA), `family="poisson"`/`"gaussian"`, plus deviance / D² / R² helpers. |
| [`encoding.py`](encoding.py) | The **encoding model**: per-unit cross-validated λ (`cv_select_alpha`), held-out predictor-subset **ΔD²** (`predictor_importance`), and **significance** (`wilcoxon_full_vs_null`, `permutation_null_d2`). |
| [`viz.py`](viz.py) | **Plots**: per-predictor kernels vs lag, predicted-vs-actual reconstruction, event-aligned averages. |

A typical end-to-end run:

```python
import jax.numpy as jnp
import design as dz, poisson_glm as pg, encoding as enc, viz

# 1. design matrix: expand events into shift-kernels (here a ±0.5 s window at 20 ms bins)
X, names = dz.build_design({"stim": stim_onset, "reward": reward_onset}, shifts=range(-10, 30))

# 2. fit every unit at once  (family="gaussian" for photometry)
Xz, mean, std = pg.standardize(jnp.asarray(X))
W, b, _, _ = pg.fit_units(Xz, jnp.asarray(Y), 0.01, 0.5, family="poisson")

# 3. predictor importance (held-out ΔD²) and significance
imp = enc.predictor_importance(X, Y, subsets={"stim": [...], "reward": [...]}, alphas=[1e-3, 1e-2, 1e-1])
sig = enc.permutation_null_d2(X, Y, alphas=[1e-3, 1e-2, 1e-1])     # per-unit p-values

# 4. plot the fitted kernels
viz.plot_kernels(W, names, dt=0.02, mean_sem=True).savefig("kernels.png")
```

## Installation

Builds are heavy, so they run as SLURM jobs on a GPU node, not on a login node:

```bash
sbatch install_env.sh          # creates conda env `jaxGLM` (jax[cuda12] + optax + sci stack)
```

The script routes package caches off `$HOME`. When it finishes, its log shows
`devices [CudaDevice(id=0)]`, confirming JAX sees the GPU. (`matplotlib` is needed only for
`viz.py`.)

## How it works

(Agents run the code; this section is so a person can understand the model and data flow.)

### What it fits

jaxGLM fits a **generalized linear model** for each recorded unit. Two response families, chosen
with `family`: **`"poisson"`** (default, spike counts) and **`"gaussian"`** (continuous signals).
The Poisson case predicts a unit's spike count per time bin from behavioral/task predictors:

```
eta = X · w + b      (linear predictor)
mu  = exp(eta)       (expected count; log link.  gaussian: mu = eta, identity link)
y   ~ Poisson(mu)
```

Each unit gets its own `w`, `b`, fit by minimizing the negative log-likelihood plus an
**elastic-net** penalty (L1 + L2):

```
minimize  (1/T) Σ_t loss(eta_t, y_t)  +  alpha·( l1_ratio·‖w‖₁ + ½(1−l1_ratio)·‖w‖₂² )
```

The intercept is unpenalized. The problem is convex; jaxGLM solves it with accelerated proximal
gradient (FISTA). The loss is averaged over time bins, keeping `alpha` on a consistent scale
across sessions. For gaussian the loss is least squares and **D² is ordinary R²**.

### The two inputs: `X` and `Y`

- **`X`** — design matrix `(T, P)`: `T` time bins × `P` predictors (events expanded into shift
  kernels by `design.build_design`). No intercept column — handled internally.
- **`Y`** — responses `(T, U)`: same `T` bins × `U` units (one column per neuron / channel).

The key fact that makes this fast: **every unit shares the same `X`** — only its column of `Y`
differs. So fitting all `U` units is one matrix multiply (`X · W`). jaxGLM `vmap`s a single-unit
fit across units (and across the λ-grid, predictor subsets, and CV folds), turning the workload
into dense GPU matmuls — what CPU tools (sklearn one neuron at a time) cannot do.

### What you get out

- **Per-unit weights `W`, predicted rates, and D²/R²** (fraction of deviance explained vs an
  intercept-only baseline) — the headline quality measure.
- **Predictor importance**: `encoding.predictor_importance` drops groups of predictors, refits,
  and reports the held-out **ΔD²** — each variable's encoding contribution.
- **Significance**: `wilcoxon_full_vs_null` (CV-fold signed-rank, full vs intercept null) and
  `permutation_null_d2` (circular-shift null) give per-unit p-values.
- **Kernels & figures**: `viz.plot_kernels` etc. turn the fitted weights into kernel plots.

### Inputs and the choices that matter

Beyond `X` and `Y`, a handful of choices drive the results. What you provide, and how to pick it:

**Bin size / sampling (`dt`).** `X` and `Y` must be on the *same* time grid; the bin width sets
temporal resolution. Finer bins → sharper kernels but more bins (more compute) and sparser counts
(Poisson). Photometry: use the native rate (e.g. 18.5 Hz). Spikes: 10–50 ms bins are typical.
`dt` is also what `viz.plot_kernels(..., dt=…)` uses to label lags in seconds. Binning `Y` onto
this grid is on you.

**Predictors and the kernel window (`shifts`).** `design.build_design(predictors, shifts)`
expands each event into time-shifted copies. `shifts` is the lag window — e.g. `range(-10, 30)`
at 20 ms = −0.2 to +0.6 s around the event. Cover the response you expect (negative lags for
anticipatory, positive for evoked). Pass a `{name: range}` dict for **per-predictor windows**
(e.g. a long reward kernel, short lick kernel). Predictors can be event indicators (0/1, via
`events_from_indices`) or continuous regressors.

**Standardize `X`.** `pg.standardize(X)` z-scores columns so `alpha` acts evenly across
predictors; recover interpretable weights with `unstandardize_weights`. Recommended (the
reproductions do this).

**Regularization (`alpha`, `l1_ratio`) — select it, don't guess.** `l1_ratio` sets the penalty
type (`0`=ridge, `1`=lasso, between=elastic net); `alpha` sets the strength. Don't hand-pick
`alpha` — cross-validate it: `encoding.cv_select_alpha(X, Y, alphas, …)` returns a **per-unit**
`alpha`, or `fit_units_grid` sweeps the grid. Provide `alphas` spanning a few orders of magnitude
(e.g. `[1e-4, 1e-3, 1e-2, 1e-1, 1]`).

**CV folds — avoid temporal leakage.** `n_folds` / `fold_ids` control the splits. Bins within a
trial are correlated, so random per-bin folds leak signal across train/test and inflate D². Pass
`fold_ids` grouping **whole trials** (one id per trial or block) so held-out data is genuinely
independent — this matters for honest D² *and* significance. Contiguous-block folds (the default)
are a reasonable fallback.

**Predictor subsets (`subsets`).** For ΔD² you define the groups to ablate as
`{name: column_indices}`. The `names` returned by `build_design` make it easy to grab all shifts
of a predictor (e.g. every `reward_*` column).

**Permutations (`n_perm`).** `permutation_null_d2` resolves p to `1/(n_perm+1)`; use ≥200 for
p≈0.005, more for stricter thresholds. Each permutation refits, so it's the main cost driver.

**Response (`Y`) must match the family.** Poisson: nonnegative integer counts per bin. Gaussian:
any continuous value (e.g. z-scored dF/F).

| param | where | quick guidance |
|---|---|---|
| `family` | `fit_units`, `encoding` | `"poisson"` counts · `"gaussian"` continuous |
| `shifts` | `build_design` | lag window per predictor; cover expected response |
| `alpha` | `fit_units` / `alphas` grid | **CV-select**, don't hardcode |
| `l1_ratio` | `fit_units` | `0` ridge · `1` lasso · between elastic net |
| `fold_ids` | `encoding.*` | group **whole trials** to avoid leakage |
| `n_perm` | `permutation_null_d2` | ≥200; trades runtime for p-resolution |
| `max_iter`/`tol` | `fit_units` | FISTA iteration cap / stopping tolerance |

## Validation

- **Synthetic** ([`test_synthetic.py`](test_synthetic.py)): recovers known weights (corr ≈ 0.999),
  matches an independent Newton/IRLS solve (Poisson ridge) and the OLS closed form (Gaussian) to
  ~1e-10, L1 induces sparsity, batched grid fit matches a looped fit.
- **Encoding & significance** ([`test_encoding.py`](test_encoding.py),
  [`test_significance.py`](test_significance.py)): ΔD² isolates an informative predictor group
  from a noise group; the significance tests flag real-signal units and not noise units.
- **Published references on real public data — both families.** Each fits an *identical* design
  matrix with jaxGLM and with the standard scikit-learn fitter the relevant lab tool wraps,
  isolating the solver:
  - **Poisson / spikes** — IBL brain-wide-map Neuropixels vs `PoissonRegressor` (≈ IBL
    `neurencoding`): per-unit **D² corr 1.0000**, **weight corr 0.9988** ([`ibl/`](ibl)).
  - **Gaussian / photometry** — Chantranupong 2023 (DANDI 001767) vs `ElasticNet` (≈ Sabatini
    `sglm`): **R² agree to 7e-4**, **weight corr 0.9963** ([`chantranupong/`](chantranupong)).

## Roadmap

- [x] Core FISTA elastic-net solver (Poisson + Gaussian), vmapped over units / λ-grid
- [x] Shift-kernel design-matrix construction (`design.py`)
- [x] Per-unit cross-validated λ selection + held-out subset ΔD² (`encoding.py`)
- [x] Significance testing — Wilcoxon full-vs-null + circular-shift permutation null
- [x] Visualization — kernels, reconstruction, event-aligned averages (`viz.py`)
- [x] Real-data reproductions: IBL (Poisson) and Chantranupong (Gaussian)
- [ ] Logistic / multinomial family (choice / RL behavioral models)
- [ ] Match a published kernel figure exactly (Chantranupong "Level B": replicate `lynne_pp` preprocessing)
- [ ] Trial-aware CV-fold helper; config-driven batch runner

## Repository layout

```
design.py             shift-kernel design-matrix construction
poisson_glm.py        GPU elastic-net GLM solver (Poisson/Gaussian) + deviance/D²
encoding.py           CV λ, held-out subset ΔD², significance tests
viz.py                kernel / reconstruction / event-aligned plots
test_*.py             self-tests (synthetic, encoding, significance, design+viz)
install_env.sh        SLURM job: build the `jaxGLM` conda env
install_ibl_env.sh    SLURM job: build the `ibl` env (ONE-api/ibllib/neurencoding)
install_nwb_env.sh    SLURM job: build the `nwb` env (dandi/pynwb/sklearn)
ibl/                  IBL Poisson reproduction (README + scripts + kernel figure)
chantranupong/        Chantranupong Gaussian reproduction (README + scripts + kernel figure)
```
