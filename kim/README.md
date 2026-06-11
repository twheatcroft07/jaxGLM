# Reinhold reproduction — Gaussian, real neural data + published scores

Shows jaxGLM (`family="gaussian"`) reproduces Kimberly Reinhold's GLM on her own example data
(from Harvard Dataverse DVN/QPQEC9, "Striatum supports fast learning but not memory recall"),
fit with the lab's `sglm`/`k-glm` engine (sklearn ElasticNet, `model_type='Normal'`). Uniquely,
this dataset ships the **published per-neuron fitted scores**, so we check against ground truth.

## Data (public)
`example datasets to run code/glm/forglm_trainingSet_wreach/` from QPQEC9:
- `behEvents.mat` — 10 task events x 13064 bins (cue, opto, distract, success, drop, miss,
  cXsuc, cXdro, cXmis, reach) + `nTrial`.
- `neuron_data_matrix.mat` — 16 cue-aligned units x 13064 bins (continuous -> Gaussian).
- `output/neuronN_glm_metadata.csv` — the lab's published hyperparameters + holdout R^2.

## Design / fit
Each event -> shift-kernel [-20, 50] (via `design.build_design`) + `nTrial`; 711 features.
Fit at the lab's hyperparameters: `alpha=0.01, l1_ratio=0.1, max_iter=1000`. Held out whole
interleaved trials. **X is left RAW (not standardized)** -- sglm fits raw events, so alpha is on
that scale; z-scoring makes alpha ~100x too weak and overfits to a negative holdout R^2.

## Result
| metric | value |
|---|---|
| per-neuron holdout **R^2 corr (jaxGLM vs published)** | **0.915** |
| median holdout R^2 (jaxGLM / published) | -0.000 / 0.000 |
| holdout R^2 corr (jaxGLM vs sklearn) | 0.976 |

jaxGLM reproduces the published per-neuron encoding. Note: exact weight-for-weight match is *not*
expected here -- the raw design is ill-conditioned (collinear shifted columns, all-zero events)
and the published fit is early-stopped (`max_iter=1000`), so many weight vectors give the same
predictions (weight corr ~0.84 while prediction/R^2 corr ~0.98).
