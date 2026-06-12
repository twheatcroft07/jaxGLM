# Sparse design matrices: assessed, deliberately not adopted

**Question:** the lab `sglm` workflow holds the shift-kernel design `X` as a `scipy.sparse`
(`csr_array`) to save memory on big designs. Should jaxGLM do the same?

**Verdict: no — sparse `X` is the wrong fit for jaxGLM's design.** It's *possible*
(`jax.experimental.sparse` has BCOO), but it doesn't help here, for four concrete reasons. Where
memory is actually a concern, chunking beats sparsity.

## Why sparse helps in sglm but not here

sglm fits **sklearn** models on **CPU**, often on the **raw** (un-standardized) design. There, a
shift-expanded event design is genuinely sparse (each event column is mostly zeros; its shifts too),
sklearn's coordinate descent consumes `scipy.sparse` directly, and CPU RAM is the binding
constraint. Sparsity is a clean win.

jaxGLM is a different machine, and three of its core choices each independently defeat sparsity:

1. **Standardization densifies `X`.** The recommended path z-scores every column
   (`poisson_glm.standardize`) so `alpha` acts evenly across predictors. Z-scoring a binary/event
   column turns every `0` into `-mean/std` — **the matrix becomes fully dense before it ever
   reaches the solver.** Sparsity is destroyed by the very preprocessing we recommend. (sglm avoids
   this by fitting raw `X`.)

2. **The speed comes from dense batched matmuls.** jaxGLM's whole value is `vmap`/`jit` batching the
   fit over units × λ × folds × shuffles into dense GPU GEMMs (`X @ W`, `Xᵀ(μ−y)`). GPU sparse
   matmul (BCOO) is only faster at very high sparsity *and* large size, and it does **not** compose
   cleanly with `vmap`/`jit` across all the axes we batch — adopting it would fight the architecture
   that makes jaxGLM fast in the first place.

3. **Typical encoding designs are small dense.** `X` is `(T bins, P=predictors×shifts)` in float32.
   Realistic cases sit comfortably in GPU memory dense:
   - Chantranupong-scale: T≈21k, P≈290 → **~24 MB**.
   - A big photometry session: T≈100k, P≈1000 → **~400 MB**.
   - It takes T≈500k × P≈2000 → **~4 GB**, or T≈1M × P≈5000 → **~20 GB**, before dense `X` alone
     stresses a 40–80 GB GPU. Those are extreme (very fine bins × very many predictor-shifts).

4. **JAX sparse is experimental.** BCOO + the batching transforms + a custom FISTA prox is a lot of
   fragile surface area for a narrow payoff.

## What to do instead when memory *is* the problem

The regime where dense `X` won't fit is narrow (huge T **and** huge P **and** willing to skip
standardization). If you land there, these help more than sparsity and keep the dense-GEMM speed:

- **Stay float32** (already the default) — half the memory of float64.
- **Chunk the batch axes**: fit units (or λ-grid points, or folds) in groups rather than one giant
  `vmap`, so peak memory is `X` plus a slice of the batch, not the full cross-product. This is the
  natural scaling knob and is on the roadmap if a real workload needs it.
- **Thin the design**: coarser bins or fewer shifts per predictor cut `P`/`T` directly and usually
  cost little (kernels are smooth).
- Only if all that fails and `X` itself (not the batch) is the wall, revisit a sparse *un-standardized*
  path as a special mode — but that's a different code path with its own correctness caveats, not a
  drop-in.

**Bottom line:** keep `X` dense. Sparsity buys little here because standardization densifies it and
dense GPU GEMMs are the point; chunking is the right lever for the rare oversized design.
