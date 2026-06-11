"""Level A — step 2a (ibl env): build a design matrix X + spike-count matrix Y for the BWM
session, and fit the reference Poisson GLM with scikit-learn's PoissonRegressor (which is
exactly the engine IBL's neurencoding.PoissonGLM wraps). Saves everything for the jaxGLM stage.

Design: per-trial windows aligned to stimulus onset, 20 ms bins. Predictors are FIR kernels
(banks of time-lagged indicators) for three task events plus signed stimulus contrast:
    stimOn FIR | firstMovement FIR | feedback FIR | signed_contrast
The exact basis doesn't matter for Level A: the SAME X is handed to both fitters, so the
comparison isolates the solver. We fit near the MLE (tiny ridge) so a unique convex optimum
exists and both solvers must converge to it regardless of regularization conventions.
"""
import os
import numpy as np
from sklearn.linear_model import PoissonRegressor

DIR = "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/ibl_level_a"
d = np.load(os.path.join(DIR, "session_raw.npz"), allow_pickle=True)

st = np.asarray(d["spike_times"]); sc = np.asarray(d["spike_clusters"]).astype(int)
label = np.asarray(d["cluster_label"])
stimOn = np.asarray(d["trials_stimOn_times"])
move = np.asarray(d["trials_firstMovement_times"])
feedback = np.asarray(d["trials_feedback_times"])
cL = np.nan_to_num(np.asarray(d["trials_contrastLeft"]))
cR = np.nan_to_num(np.asarray(d["trials_contrastRight"]))
signed = cR - cL

# ---- good units -> contiguous columns
good = np.where(label >= 1.0)[0]
U = good.size
lut = -np.ones(label.size, dtype=int); lut[good] = np.arange(U)
print(f"good units: {U}")

# ---- binning / window
DT = 0.02
PRE, POST = 0.5, 2.0
NB = int(round((PRE + POST) / DT))          # bins per trial
S0 = int(round(PRE / DT))                    # bin index of stimulus onset
L_STIM, L_MOVE, L_FB = 20, 25, 25            # FIR lag counts per event
NCOL = L_STIM + L_MOVE + L_FB + 1
valid = ~np.isnan(stimOn)
tidx = np.where(valid)[0]
print(f"trials used: {tidx.size} | bins/trial: {NB} | predictors: {NCOL}")

X_blocks, Y_blocks = [], []
for ti in tidx:
    t0 = stimOn[ti] - PRE
    edges0 = t0
    m = (st >= edges0) & (st < edges0 + NB * DT)
    bi = np.clip(((st[m] - edges0) / DT).astype(int), 0, NB - 1)
    col = lut[sc[m]]
    keep = col >= 0
    Ytr = np.zeros((NB, U))
    np.add.at(Ytr, (bi[keep], col[keep]), 1.0)

    Xtr = np.zeros((NB, NCOL))
    for k in range(L_STIM):                          # stimulus onset FIR
        if S0 + k < NB: Xtr[S0 + k, k] = 1.0
    if not np.isnan(move[ti]):                        # first-movement FIR
        mb = int(round((move[ti] - stimOn[ti] + PRE) / DT))
        for k in range(L_MOVE):
            b = mb + k
            if 0 <= b < NB: Xtr[b, L_STIM + k] = 1.0
    if not np.isnan(feedback[ti]):                    # feedback FIR
        fb = int(round((feedback[ti] - stimOn[ti] + PRE) / DT))
        for k in range(L_FB):
            b = fb + k
            if 0 <= b < NB: Xtr[b, L_STIM + L_MOVE + k] = 1.0
    Xtr[S0:, NCOL - 1] = signed[ti]                   # signed contrast, on from stim onset

    X_blocks.append(Xtr); Y_blocks.append(Ytr)

X = np.concatenate(X_blocks, 0)
Y = np.concatenate(Y_blocks, 0)
print(f"X: {X.shape} | Y: {Y.shape} | mean count/bin: {Y.mean():.3f}")

# ---- standardize X (constant cols -> std 1); both fitters use this same Xz
mean = X.mean(0); std = X.std(0); std[std < 1e-8] = 1.0
Xz = (X - mean) / std

# ---- reference fit: sklearn PoissonRegressor per unit, tiny ridge (near MLE)
ALPHA = 1e-4
W_ref = np.zeros((NCOL, U)); b_ref = np.zeros(U); d2_ref = np.zeros(U)
for u in range(U):
    m = PoissonRegressor(alpha=ALPHA, fit_intercept=True, max_iter=500, tol=1e-9)
    m.fit(Xz, Y[:, u])
    W_ref[:, u] = m.coef_; b_ref[u] = m.intercept_
    d2_ref[u] = m.score(Xz, Y[:, u])     # sklearn .score = fraction Poisson deviance explained
print(f"sklearn reference: median D^2 = {np.median(d2_ref):.4f} | "
      f"frac units D^2>0.01: {np.mean(d2_ref > 0.01):.2f}")

out = os.path.join(DIR, "level_a_compare.npz")
np.savez(out, Xz=Xz, Y=Y, W_ref=W_ref, b_ref=b_ref, d2_ref=d2_ref,
         alpha=ALPHA, mean=mean, std=std, ncol=NCOL, n_units=U)
print(f"saved -> {out}")
