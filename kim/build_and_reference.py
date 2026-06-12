"""Kimberly Reinhold reproduction -- step 1 (nwb env): build the GLM design from the example dataset
(behEvents + neuron_data_matrix) shipped with QPQEC9, fit the reference with sklearn ElasticNet
at the lab's own hyperparameters (alpha=0.01, l1_ratio=0.1 -- model_type='Normal' = Gaussian),
and load the lab's PUBLISHED per-neuron scores for comparison. Saves for the jaxGLM stage.

Design (verified from the published feature_names): 10 event kernels (cue, opto, distract,
success, drop, miss, cXsuc, cXdro, cXmis, reach) each shifted [-20, 50], + nTrial covariate.
Responses: neuron_data_matrix (16 cue-aligned units, continuous -> Gaussian).
"""
import os, glob, sys
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.linear_model import ElasticNet

import design as dz

G = "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/kim/example datasets to run code/glm/forglm_trainingSet_wreach"
EVENTS = ["cue", "opto", "distract", "success", "drop", "miss", "cXsuc", "cXdro", "cXmis", "reach"]
SHIFTS = range(-20, 51)                         # [-20, 50] inclusive, 71 lags (matches feature_names)
ALPHA, L1 = 0.01, 0.1                            # the lab's saved hyperparameters

beh = np.asarray(sio.loadmat(os.path.join(G, "behEvents.mat"))["behEvents"]).astype(float)   # (11, T)
Y = np.asarray(sio.loadmat(os.path.join(G, "neuron_data_matrix.mat"))["neuron_data_matrix"]).T  # (T, 16)
T, U = Y.shape
print(f"behEvents {beh.shape} | Y {Y.shape}")

# X: 10 events x shifts, + nTrial covariate (last behEvents row)
predictors = {name: beh[i] for i, name in enumerate(EVENTS)}
X, names = dz.build_design(predictors, SHIFTS)
X = np.concatenate([X, beh[10][:, None]], axis=1); names = names + ["nTrial"]
# NOTE: do NOT standardize -- the lab's sglm fits RAW binary events, so alpha=0.01 is on that
# scale. Standardizing makes alpha far too weak (overfits to negative holdout R^2).
print(f"X {X.shape} ({len(names)} features)  [raw, matches sglm]")

# published per-neuron holdout scores (lab's GLM output)
pub = []
for f in sorted(glob.glob(os.path.join(G, "output", "*_glm_metadata.csv")),
                key=lambda p: int(os.path.basename(p).split("_")[0].replace("neuron", ""))):
    pub.append(float(pd.read_csv(f)["score_holdout"].iloc[0]))
pub = np.array(pub[:U])

# hold out whole trials, interleaved (so the nTrial ramp interpolates -- a contiguous tail would
# extrapolate nTrial outside the training range and blow up).
nTrial = beh[10].astype(int)
ho_trials = np.unique(nTrial)[::5]               # every 5th trial held out
te = np.isin(nTrial, ho_trials)
print(f"holdout: {ho_trials.size} trials / {np.unique(nTrial).size} | {te.sum()}/{T} bins")

# reference fit: sklearn ElasticNet per neuron at the lab's hyperparams
W_ref = np.zeros((X.shape[1], U)); r2_tr = np.zeros(U); r2_ho = np.zeros(U)
for u in range(U):
    m = ElasticNet(alpha=ALPHA, l1_ratio=L1, fit_intercept=True, max_iter=1000, tol=1e-4)  # the lab's setting
    m.fit(X[~te], Y[~te, u])
    W_ref[:, u] = m.coef_
    r2_tr[u] = m.score(X[~te], Y[~te, u]); r2_ho[u] = m.score(X[te], Y[te, u])
print(f"sklearn ElasticNet: median holdout R^2 = {np.median(r2_ho):.3f} | "
      f"published median = {np.median(pub):.3f}")

out = "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/kim/kim_compare.npz"
np.savez(out, X=X, Y=Y, W_ref=W_ref, r2_tr_ref=r2_tr, r2_ho_ref=r2_ho, pub_holdout=pub,
         alpha=ALPHA, l1_ratio=L1, te=te, names=np.array(names))
print(f"saved -> {out}")
