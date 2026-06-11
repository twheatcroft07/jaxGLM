# From R² to D²: deviance explained, made accessible

This note derives **D² (fraction of deviance explained)** — the headline per-unit quality
measure in jaxGLM — and shows that ordinary **R²** is just the Gaussian special case of it.
The goal is intuition first, then the algebra. By the end, the two helpers in
[`poisson_glm.py`](../poisson_glm.py) (`deviance` and `frac_deviance_explained`) should read as
one idea applied to two response families.

The punchline up front:

```
            dev(your model)
D²  =  1 −  ───────────────              dev = a measure of "how badly predictions miss"
            dev(null model)

Gaussian:  dev = Σ (y − μ)²      →   D²  =  1 − SS_res / SS_tot   =   R²
Poisson :  dev = 2 Σ [ y·log(y/μ) − (y − μ) ]
```

Same formula. Only the definition of "miss" changes with the noise model.

---

## 1. The thing you already know: R²

For ordinary least squares, R² is the **fraction of variance explained**:

```
        SS_res     Σ (yₜ − μₜ)²          μₜ = your model's prediction
R² = 1 ─────── = 1 ─────────────         ȳ  = the mean of y
        SS_tot     Σ (yₜ − ȳ)²
```

Read it as a comparison of two models:

- **Numerator `SS_res`** — squared error your *fitted model* leaves behind.
- **Denominator `SS_tot`** — squared error of the dumbest honest baseline: predict the
  **mean** `ȳ` for every time bin, ignoring all predictors. This is the **null model**.

So R² is not really "variance explained" in the abstract — it is **error of my model,
relative to error of the intercept-only model**:

```
R²  =  1 − (error with predictors) / (error without them)
```

R² = 1 → perfect fit (no residual error). R² = 0 → your predictors did no better than the
mean. R² < 0 → you did *worse* than the mean (possible on held-out data; see §6).

This "1 − model/null" shape is the whole idea. Everything below is about choosing the right
notion of *error* when the response isn't Gaussian.

---

## 2. Why squared error is the wrong ruler for spike counts

Squared error `(y − μ)²` is the natural error measure when noise is **Gaussian with constant
variance**: minimizing `Σ(y − μ)²` *is* maximizing the Gaussian likelihood. But spike counts
aren't Gaussian:

- They're non-negative integers, not real numbers.
- Their variance grows with their mean (a unit firing at rate 20 is noisier in absolute terms
  than one at rate 0.2). Constant-variance squared error doesn't know this — it penalizes a
  miss of 5 spikes the same whether the cell fires at 1 Hz or 100 Hz.

We need an error ruler derived from the **Poisson** likelihood instead. That ruler is the
**deviance**.

---

## 3. Deviance: a likelihood-based ruler for "how badly did I miss?"

Deviance generalizes "squared error" to any GLM family. The definition is a **likelihood
ratio against a perfect model**.

Let `ℓ(μ; y)` be the log-likelihood of the data `y` under predictions `μ`. Two reference points:

- **Saturated model** — predicts `μₜ = yₜ` exactly for every bin. This is the best any model
  could *possibly* do; it has the highest achievable log-likelihood, `ℓ(y; y)`.
- **Your model** — predicts some `μₜ`, with log-likelihood `ℓ(μ; y)`.

The deviance is twice the gap between them:

```
D(y, μ)  =  2 · [ ℓ(y; y) − ℓ(μ; y) ]
            └────────┬────────┘
            log-likelihood lost by predicting μ instead of the perfect y
```

Three things to notice, because they're what make it a sensible ruler:

1. **It's ≥ 0.** The saturated model maximizes the likelihood, so the bracket can't be negative.
2. **It's 0 only for a perfect fit** (`μ = y` everywhere).
3. **Bigger = worse**, just like squared error. It is, literally, "negative log-likelihood,
   shifted so that a perfect prediction scores 0."

That's the entire concept. Now we just plug each family's likelihood into the definition.

---

## 4. Plug in Gaussian → recover squared error

Gaussian log-likelihood for one bin, variance `σ²`:

```
ℓ(μ; yₜ)  =  − (yₜ − μₜ)² / (2σ²)  −  ½ log(2πσ²)
```

The saturated model has `μₜ = yₜ`, so its squared-error term vanishes and only the constant
`−½log(2πσ²)` survives. Subtract and double:

```
D(y, μ)  =  2 Σₜ [ ℓ(yₜ; yₜ) − ℓ(μₜ; yₜ) ]
         =  2 Σₜ [  − 0  +  (yₜ − μₜ)²/(2σ²)  ]        (the log-2πσ² constants cancel)

         =  (1/σ²) · Σₜ (yₜ − μₜ)²   =   SS_res / σ²
```

**Gaussian deviance is just the residual sum of squares** (scaled by the noise variance). This
is exactly what `deviance(..., family="gaussian")` returns —
[`poisson_glm.py:217`](../poisson_glm.py#L217) computes `Σ(y − μ)²`, dropping the `σ²` because
it cancels in the ratio of §5.

---

## 5. Plug in Poisson → the spike-count deviance

Poisson log-likelihood for one bin (rate `μₜ`, observed count `yₜ`):

```
ℓ(μ; yₜ)  =  yₜ · log μₜ  −  μₜ  −  log(yₜ!)
```

The `log(yₜ!)` term doesn't depend on `μ`, so it cancels in the saturated-minus-model
difference. Saturated model: `μₜ = yₜ`, giving `ℓ(yₜ; yₜ) = yₜ log yₜ − yₜ − log(yₜ!)`.
Subtract and double:

```
D(y, μ)  =  2 Σₜ [ ℓ(yₜ; yₜ) − ℓ(μₜ; yₜ) ]

         =  2 Σₜ [ (yₜ log yₜ − yₜ) − (yₜ log μₜ − μₜ) ]

         =  2 Σₜ [ yₜ · log(yₜ / μₜ)  −  (yₜ − μₜ) ]
```

That last line is **exactly** the Poisson deviance in
[`poisson_glm.py:214`](../poisson_glm.py#L214). Reading the two terms:

- `yₜ · log(yₜ/μₜ)` — the core penalty: predicting the wrong *rate* costs in proportion to
  the count, on a log scale. Crucially, the penalty for a fixed multiplicative miss (say,
  predicting 2× too low) is the same at high and low rates — the right behavior for counts.
- `−(yₜ − μₜ)` — a correction that keeps the deviance ≥ 0 and makes it 0 at `μ = y`.

(At `yₜ = 0` the `y·log y` term is taken as 0 by the usual limit `y log y → 0`; the code's
`eps` clips guard the logs numerically — [`poisson_glm.py:213-214`](../poisson_glm.py#L213-L214).)

For small residuals this Poisson deviance reduces to a *variance-weighted* squared error,
`≈ Σ (yₜ − μₜ)² / μₜ` (second-order Taylor expansion). That's the Poisson analogue of `SS_res`,
but with each bin's error divided by its own mean — automatically down-weighting the
high-rate bins that are *supposed* to be noisier. This is precisely the fix §2 asked for.

---

## 6. D²: wrap deviance in the "1 − model/null" shell

Now reuse the R² shape, with deviance as the error ruler:

```
       dev(model)        D(y, μ_model)
D² = 1 ────────── = 1 − ───────────────
       dev(null)         D(y, μ_null)
```

This is `frac_deviance_explained` — [`poisson_glm.py:222-229`](../poisson_glm.py#L222-L229).

- **`μ_null`** is the intercept-only model: the best constant prediction. For Poisson that
  constant is the **mean rate** `ȳ` (the MLE of a single Poisson rate); for Gaussian it is
  also the mean `ȳ`. So in both families the null model predicts `ȳ` everywhere — the same
  baseline R² uses.
- **Gaussian**: numerator → `SS_res`, denominator → `SS_tot`, and `D² = 1 − SS_res/SS_tot = R²`
  **exactly**. The `σ²` scaling cancels in the ratio, which is why the code never needs it.
- **Poisson**: same formula, Poisson deviances. This is the "GLM analogue of variance
  explained" the README advertises — a genuine generalization, not an analogy.

### How to read D² values

| D²        | meaning                                                              |
|-----------|---------------------------------------------------------------------|
| `1`       | perfect fit — model reproduces every count / value                  |
| `0`       | predictors add nothing beyond the mean rate                         |
| `(0, 1)`  | fraction of the null model's deviance your predictors remove        |
| `< 0`     | model predicts **worse** than the mean — expect this on held-out folds for units with no real tuning |

The negative case isn't a bug. On *training* data D² ∈ [0, 1] because the fitted model can't do
worse than its own null. But jaxGLM reports **held-out / cross-validated** D² (and ΔD² for
predictor importance), and out-of-sample a model can overfit and underperform the mean — so
negative D² is both possible and informative: it flags units the predictors don't actually
explain.

---

## 7. One-paragraph summary

R² answers "how much of the squared error around the mean did my predictors remove?" That
question only makes sense when error *means* squared error — i.e. when noise is Gaussian.
Deviance is the general answer to "how badly did I miss?", derived as a likelihood-ratio
against a perfect model; it equals `SS_res` for Gaussian and `2 Σ[y·log(y/μ) − (y − μ)]` for
Poisson. Wrapping it in `1 − model/null` gives **D²**, which *is* R² for Gaussian and is the
correct, variance-aware version of it for spike counts. Two families, one formula — exactly how
the code is written.

---

### References

- McCullagh & Nelder (1989), *Generalized Linear Models*, §2.3 (deviance) — the canonical source.
- Implementation: [`poisson_glm.py`](../poisson_glm.py), functions `deviance` and
  `frac_deviance_explained`.
