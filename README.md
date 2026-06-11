# jaxGLM

GPU elastic-net Poisson GLM for SC neural encoding, in JAX. Rewrite of the fitting core of
`~/refactoredHarveyGLM/refactoredGLM.py` (which wraps the TensorFlow `GLM_Tensorflow_2`),
designed so that **300 units × ~30 predictor subsets × CV folds × a λ grid** fit in one
batched `vmap` pass on a single GPU.

## Why a rewrite
The current pipeline fits per-unit Poisson GLMs through a TF/Keras gradient loop. The work is
embarrassingly batchable because **all units in a session share the design matrix `X`**
(only the response column differs), so `X @ W` is one matmul. JAX `vmap` expresses that
directly; TensorFlow makes it awkward. Target scale per session: T≈28,000 timebins
(7 s trials × 200 trials @ 50 ms), P design columns, 300 units.

## Status
- [x] `poisson_glm.py` — core FISTA proximal-gradient solver, vmappable over units/λ; deviance/D².
- [x] `test_synthetic.py` — correctness self-test (weight recovery; ridge case vs independent IRLS).
- [ ] env built on GPU node (`install_env.sh`)
- [ ] data loader for `intermediate_data.pkl` (df_predictors_shift → X, responses → Y, subsets)
- [ ] per-unit CV λ selection + intercept-null Wilcoxon (reproduce refactoredGLM stats)
- [ ] subset leave-out ΔD² (predictor importance)
- [ ] comparison harness vs glmTF28 on 1–2 sessions (D², λ*, significance calls)
- [ ] (later) permutation null as an upgrade over the intercept-only null

## Setup (do NOT run on login)
```bash
sbatch install_env.sh            # builds conda env `jaxGLM` on gpu_test
# then, on a GPU node:
conda activate jaxGLM && python test_synthetic.py
```

## Reproduction target
First milestone is *exact reproduction* of refactoredGLM's outputs (intercept-only null +
CV-fold Wilcoxon signed-rank, per-unit CV λ). Validation = per-unit, per-subset D² agreeing
within tolerance, matching λ* (±1 grid step), and matching significance decisions. Only once
that matches do we add the permutation null.
