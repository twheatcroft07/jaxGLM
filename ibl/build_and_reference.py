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
import os, sys
import numpy as np
from sklearn.linear_model import PoissonRegressor

sys.path.insert(0, "/n/home02/twheatcroft/jaxGLM")
import design as dz                                        # shift-kernel construction

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
BIN_S = 0.02
PRE, POST = 0.5, 2.0
NB = int(round((PRE + POST) / BIN_S))          # bins per trial
S0 = int(round(PRE / BIN_S))                    # bin index of stimulus onset
L_STIM, L_MOVE, L_FB = 20, 25, 25            # FIR lag counts per event
NCOL = L_STIM + L_MOVE + L_FB + 1
valid = ~np.isnan(stimOn)
tidx = np.where(valid)[0]
print(f"trials used: {tidx.size} | bins/trial: {NB} | predictors: {NCOL}")

# FIR kernels via design.build_design, built per trial window. Because each window is isolated,
# build_design's zero-fill clips forward shifts at the window edge -- identical to the previous
# hand-coded `if b < NB` construction, but now with proper '{event}_{lag}' column names.
SHIFTS = {"stim": range(0, L_STIM), "move": range(0, L_MOVE),
          "feedback": range(0, L_FB), "contrast": [0]}
X_blocks, Y_blocks, names = [], [], None
for ti in tidx:
    t0 = stimOn[ti] - PRE                                       # window start (s)
    m = (st >= t0) & (st < t0 + NB * BIN_S)                     # spikes in this window
    Ytr = dz.bin_spikes(st[m], lut[sc[m]], U, t0, BIN_S, NB)    # counts (NB, U)
    # event indicators from event TIMES (events_from_times drops NaN / out-of-window)
    stim = dz.events_from_times([stimOn[ti]], t0, BIN_S, NB)
    mv = dz.events_from_times([move[ti]], t0, BIN_S, NB)
    fbv = dz.events_from_times([feedback[ti]], t0, BIN_S, NB)
    contrast = np.zeros(NB); contrast[S0:] = signed[ti]
    Xtr, names = dz.build_design({"stim": stim, "move": mv, "feedback": fbv, "contrast": contrast}, SHIFTS)
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
         alpha=ALPHA, mean=mean, std=std, ncol=NCOL, n_units=U, names=np.array(names))
print(f"saved -> {out}")
