# Related work: why this is fast, and what's prior art

**Short version:** jaxGLM sits in the (apparently empty) intersection of two well-populated areas
— *penalized neural-encoding GLMs* and *GPU-batched permutation inference*. Each ingredient has
prior art; the **combination** — a penalized **Poisson+Gaussian** encoding GLM (FISTA elastic-net)
whose **entire refit-heavy outer loop** (units × λ-grid × CV folds × subset ablations × permutation
shuffles) is `vmap`/`jit`-batched into one GPU kernel with **fixed-`X` reuse** — we could not find
packaged anywhere as of June 2026. This is a *novel combination*, not a novel primitive: we should
cite the cousins below rather than claim we invented GPU permutation testing.

This note exists so a reader (or reviewer) asking "isn't this just X?" gets a precise answer.

## The actual claim, stated carefully

The fast thing is **not** "a GPU GLM" — those are common. It is that the work which dominates an
encoding analysis is *fitting the same design structure thousands of times* for statistics, and
jaxGLM turns every one of those repetitions into a batch axis of a single compiled kernel:

- `vmap` over **units × λ × CV folds × predictor-subset ablations × permutation shuffles** at once.
- **Fixed-`X` reuse**: across permutation shuffles only `y` is circularly shifted, so the heavy
  linear algebra is computed once; for the Gaussian/ridge case a shuffle is a matmul against a
  precomputed `(XᵀX+λI)⁻¹Xᵀ`.
- `jit`: compiled once, run thousands of times. No Python loop over the outer axes.

Measured consequence: a permutation-null test that takes **>25 min** with a single-fit library
(loop of `.fit()` calls) runs in **seconds** here.

## Closest matches

| What | Link | How close | What it's missing vs. jaxGLM |
|---|---|---|---|
| **NeMoS** (Flatiron) | [github](https://github.com/flatironinstitute/nemos) | Same domain, same JAX. `PopulationGLM` batches over **neurons**. | **CV/λ are done by an external scikit-learn `GridSearchCV` — one `.fit()` per fold×λ** (confirmed from their own [CV how-to](https://nemos.readthedocs.io/en/latest/how_to_guide/plot_06_sklearn_pipeline_cv_demo.html)). The outer loop is a Python loop, not a kernel. This is precisely the gap in [engine-decision.md](engine-decision.md). |
| **GPU permutation inference in neuroimaging** — FSL `randomise` (Winkler et al. 2014/2016), `CompletePT`/Arrayfire | [randomise theory](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/Randomise(2f)Theory.html), [Winkler 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5035139/), [CompletePT](https://github.com/felipegb94/CompletePT) | **Strongest prior art for the permutation idea** — they *do* batch thousands of shuffles on GPU against a fixed design matrix. | Different model class: **mass-univariate, unpenalized OLS/F-tests per voxel.** No elastic-net, no Poisson/log-link, no FISTA, no λ-grid. Validates "batch the null on GPU"; not the penalized-GLM solver. |
| **glmnet / celer / pyglmnet / yaglm / adelie** | [pyglmnet](https://github.com/glm-tools/pyglmnet), [yaglm](https://arxiv.org/pdf/2110.05567), [adelie](https://arxiv.org/html/2405.08631v1) | Mature penalized GLMs; pathwise over λ (one axis batched *algorithmically* via warm starts). | **CPU coordinate descent.** No GPU, no `vmap` over folds/permutations/units. A λ-path ≠ batching the whole outer loop. |
| **memming/pyglm** | [github](https://github.com/memming/pyglm) | GPU, neural spike GLM. | Fully-Bayesian MCMC of a single model — a different problem. |
| General **"vmap the CV folds"** JAX pattern | e.g. [Bayesian CV by parallel MCMC](https://arxiv.org/pdf/2310.07002) | The pattern exists in the wild. | Nobody packages it for a penalized neural-encoding GLM with a permutation null + fixed-`X` reuse. |

## Per-area findings

- **NeMoS** — the right model family and the right backend (JAX), and it vectorizes over neurons via
  `PopulationGLM`. But hyperparameter/CV selection is delegated to scikit-learn `GridSearchCV`,
  which calls `.fit()` once per (fold, λ) — an external Python loop, no fixed-`X` reuse. For a
  refit-heavy workload this is the bottleneck (see [engine-decision.md](engine-decision.md)).
- **Neuroimaging GPU permutation** (FSL `randomise`, Winkler et al., CompletePT/Arrayfire) — the
  closest conceptual cousin to our permutation-null design and the right thing to cite when someone
  asks "isn't this FSL randomise?". They batch many sign-flips/shuffles on GPU with a fixed design,
  but operate on **unpenalized mass-univariate Gaussian** models (one GLM per voxel), not a
  penalized Poisson/Gaussian encoding GLM.
- **glmnet family** (glmnet, celer, pyglmnet, yaglm, adelie) — state-of-the-art penalized GLM
  solvers, but **CPU coordinate descent**. They batch the λ-*path* algorithmically (warm starts);
  they do not batch CV folds × permutations × units onto a GPU.
- **Neuro encoding toolboxes** (pyglmnet, neuroGLM/Pillow, IBL `neurencoding`, spykes) — correct
  model class, but single-fit / CPU; the statistics loop is external Python.
- **Broader ML** — `jax.vmap` over CV folds / bootstrap is a known JAX idiom (and `jaxopt`/`lineax`
  give vmap-able solvers), but it is not packaged as a neural-encoding GLM with a calibrated
  permutation null. The pattern is available; the assembled tool is what's missing.

## Gap assessment

The specific combination — **penalized (elastic-net) Poisson *and* Gaussian encoding GLM, FISTA,
full outer-loop `vmap` including the permutation null with fixed-`X` reuse, on GPU** — appears
genuinely unoccupied as a packaged tool. The tools with the right *model* (NeMoS, pyglmnet) loop the
outer axes; the tools that batch the outer *loop* on GPU (neuroimaging permutation) have the wrong
model (unpenalized, Gaussian-only, mass-univariate). jaxGLM lives in the empty intersection.

That is a real, citable gap — **not** a claim that GPU permutation testing or GPU GLMs are new. When
writing this up, cite Winkler et al. for the GPU-permutation lineage and position jaxGLM as bringing
that batching to penalized encoding GLMs.

## Citations (read June 2026)

- NeMoS: https://github.com/flatironinstitute/nemos · CV how-to: https://nemos.readthedocs.io/en/latest/how_to_guide/plot_06_sklearn_pipeline_cv_demo.html
- FSL randomise theory: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/Randomise(2f)Theory.html
- Winkler et al., *Faster permutation inference in brain imaging* (2016): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5035139/
- CompletePT (GPU permutation testing, Arrayfire): https://github.com/felipegb94/CompletePT
- pyglmnet: https://github.com/glm-tools/pyglmnet
- yaglm: https://arxiv.org/pdf/2110.05567
- adelie (block-CD pathwise group-lasso/elastic-net): https://arxiv.org/html/2405.08631v1
- memming/pyglm (GPU Bayesian spike GLM): https://github.com/memming/pyglm
- Bayesian cross-validation by parallel MCMC (vmap-over-folds example): https://arxiv.org/pdf/2310.07002
