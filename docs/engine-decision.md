# Engine decision: keep jaxGLM's batched solver (evaluated NeMoS, rejected for this workload)

**Status:** decided — 2026-06-12. jaxGLM stays its own batched solver. NeMoS is *not* adopted as
the engine. (This is an architecture-decision record so we don't re-litigate it.)

## Context

jaxGLM implements its own GPU elastic-net GLM (FISTA, `vmap`-batched). We seriously evaluated
replacing that solver with **[NeMoS](https://github.com/flatironinstitute/nemos)** — the Flatiron
Institute's mature, well-tested JAX GLM library — on the reasonable premise of "don't maintain a
bespoke solver when a maintained one exists." We went as far as scaffolding a `saba-glm` repo that
wrapped NeMoS and porting `encoding`/`significance`/`diagnostics` onto it.

We then dropped that and came back to jaxGLM. Here's why.

## The decisive fact: our workload is refit-heavy

The encoding-analysis pipeline is dominated by **fitting the same model structure thousands of
times**: cross-validation folds, predictor-subset ablation across many units, and especially the
**permutation null** (hundreds of shuffles × folds). jaxGLM is built precisely for this:

- **`vmap` over every axis at once** — units × λ-grid × CV folds × subsets × shuffles all become
  batch dimensions, compiled into *one* GPU kernel. No Python loop.
- **Fixed-`X` reuse** — across permutation shuffles the design `X` is fixed (only `y` shifts), so
  the heavy linear algebra is computed once and reused. For Gaussian ridge a shuffle is a matmul
  against a precomputed `(XᵀX+λI)⁻¹Xᵀ`.
- **`jit`** — compiled once, run thousands of times.

NeMoS exposes a stateful, scikit-learn-style **single-fit** API (`GLM`/`PopulationGLM` batch over
*units* but not over the outer loops). Driving CV/ablation/permutation means calling `.fit()` in a
Python loop — hundreds to thousands of **separate LBFGS solves**, no fixed-`X` reuse. The gap is
**architectural, not incidental**: it cannot be closed without reaching into NeMoS internals.

This is confirmed by NeMoS's own documentation, not just our measurement: their
[cross-validation how-to](https://nemos.readthedocs.io/en/latest/how_to_guide/plot_06_sklearn_pipeline_cv_demo.html)
delegates CV/λ selection to scikit-learn `GridSearchCV`, which calls `.fit()` once per (fold, λ)
combination — an external Python loop by design. (See the broader survey in
[related-work.md](related-work.md): no existing tool batches the whole outer loop of a penalized
encoding GLM onto the GPU.)

**Measured:** a permutation-null *test* (n_perm=100, both families) ran **>25 minutes** on NeMoS
before we killed it; the equivalent batched version on jaxGLM runs in **seconds**.

## Why NeMoS's advantages don't justify the speed loss

| NeMoS advantage | Assessment for our use |
|---|---|
| Extra families (Bernoulli, Gamma, NegBinomial) | **Cheap to add to jaxGLM.** Gaussian was ~40 lines; the canonical-link gradient `Xᵀ(μ−y)` is family-independent, so a new family is just link + loss + deviance + intercept warm-start. |
| Basis functions (raised-cosine, splines, convolution) | The one genuinely useful, non-trivial piece — **but it's a fit-once, upstream step** (build `X`, then fit). If we want it, call NeMoS's basis module *just* to construct `X` and still fit with jaxGLM. No speed cost. |
| GroupLasso, identifiability handling | Niche; addable (group prox / drop-redundant-columns) if needed. |
| Maturity / tests / GLM-HMM | Real value *if shipping a general tool or doing state-space models*. Not our case — jaxGLM is already validated against three lab references (see below). |

## Validation status that makes "keep jaxGLM" safe

jaxGLM's solver is not unverified bespoke code — it reproduces independent references on real data:
- **Poisson / IBL spikes** vs sklearn `PoissonRegressor` (≈ `neurencoding`): D² corr 1.0000.
- **Gaussian / Chantranupong photometry** vs sklearn `ElasticNet` (≈ lab `sglm`): R² Δ 7e-4.
- **Gaussian / Reinhold neural** vs the lab's published per-neuron scores: R² corr 0.915.
- Synthetic: matches Newton/IRLS and OLS closed form to ~1e-10.

## Decision

- **Keep jaxGLM's batched solver as the engine.** Its `vmap` batching is exactly what NeMoS can't
  do for a refit-heavy workload — that's the whole value, not redundant overhead.
- **Use NeMoS only where it fits without the speed cost:** as an upstream **basis-function** source
  (fit-once design construction), or later if the lab needs **GLM-HMM / single-fit Bernoulli** at a
  scale where batching isn't the bottleneck.
- **Add families to jaxGLM directly** (Bernoulli/logistic next, if behavioral models come up).
- `saba-glm` (the NeMoS-wrapping prototype) is dropped; this record + `validate_nemos.py` history
  capture what was learned.

## One-line takeaway

The speed we need comes from **batching the outer loops**, which is jaxGLM's core design and which
NeMoS's single-fit API structurally cannot match. NeMoS is a good library for a different shape of
problem.
