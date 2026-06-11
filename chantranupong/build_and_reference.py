"""Chantranupong gaussian Level-A, step 1 (nwb env): build the photometry GLM design from a
WT63 NWB session and fit the reference with sklearn ElasticNet (the engine sabatinilab-glm's
`sglm` wraps for model_type='Normal'). Saves (Xz, Y, reference coefs) for the jaxGLM stage.

Design follows the paper's sglm spec (verified from coefficients-for-bernardo-WT61-63-64.csv +
lynne_pp.py): 7 behavioral-event predictors, each expanded into a bank of time-shift columns
(the kernel), predicting the z-scored green photometry signals.

  predictors: cpn cpx (center poke in/exit) | spnr spnnr spxr spxnr (side poke in/exit x reward)
              | sl (side licks, left+right pooled)
  responses : detrended (z-scored) green photometry, both channels (ACh3.0 & dLight)

NOTE: this is the *solver-isolation* check -- the SAME X is fed to sklearn and jaxGLM, so the
event-construction details need not match lynne_pp exactly; that matters only for the separate
published-CSV reproduction. jaxGLM-gaussian and sklearn-ElasticNet share an identical objective,
so at the same (alpha, l1_ratio) they should converge to the same weights.
"""
import os, sys
import numpy as np
from pynwb import NWBHDF5IO
from sklearn.linear_model import ElasticNet

sys.path.insert(0, "/n/home02/twheatcroft/jaxGLM")
import design as dz                                        # shift-kernel construction

DIR = "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft/chantranupong"
NWB = os.path.join(DIR, "WT63_20211112.nwb")
SHIFTS = range(-20, 21)                                   # +/-20 bins @ 18.5 Hz ~ +/-1.1 s
PRED = ["cpn", "cpx", "spnr", "spnnr", "spxr", "spxnr", "sl"]
ALPHA, L1 = 0.01, 0.5

io = NWBHDF5IO(NWB, "r"); nwb = io.read()
gL = np.nan_to_num(np.asarray(nwb.processing["ophys"]["detrended_photometry_greenL"].data[:]))
gR = np.nan_to_num(np.asarray(nwb.processing["ophys"]["detrended_photometry_greenR"].data[:]))
Y = np.stack([gL, gR], 1)                                 # (T, 2) two green channels
T = Y.shape[0]

tr = nwb.trials
g = lambda n: np.asarray(tr[n][:])
ci, co = g("photometry_center_in_index"), g("photometry_center_out_index")
si, so = g("photometry_side_in_index"), g("photometry_side_out_index")
rew = g("was_rewarded").astype(float)
valid = g("is_photometry_trial").astype(bool) & g("has_all_photometry_data").astype(bool)
print(f"T={T} | trials={len(rew)} valid_photometry={valid.sum()} | "
      f"side_in_index range [{np.nanmin(si):.0f}, {np.nanmax(si):.0f}]")


indicator = lambda idx: dz.events_from_indices(idx, T)     # design.py builds the indicators

vr, vn = valid & (rew > 0), valid & (rew == 0)
base = {
    "cpn": indicator(ci[valid]), "cpx": indicator(co[valid]),
    "spnr": indicator(si[vr]), "spnnr": indicator(si[vn]),
    "spxr": indicator(so[vr]), "spxnr": indicator(so[vn]),
}
eo = nwb.processing["behavior"]["event_onsets"]
licks = (np.asarray(eo["left_lick_event_onsets"].data[:]) +
         np.asarray(eo["right_lick_event_onsets"].data[:]))
base["sl"] = (licks > 0).astype(float)
print("events per predictor:", {p: int(base[p].sum()) for p in PRED})

X, names = dz.build_design({p: base[p] for p in PRED}, SHIFTS)   # zero-fill shift kernels
mean = X.mean(0); std = X.std(0); std[std < 1e-8] = 1.0
Xz = (X - mean) / std
print(f"X: {X.shape} | Y: {Y.shape}")

W_ref = np.zeros((X.shape[1], 2)); b_ref = np.zeros(2); r2_ref = np.zeros(2)
for u in range(2):
    m = ElasticNet(alpha=ALPHA, l1_ratio=L1, fit_intercept=True, max_iter=20000, tol=1e-10)
    m.fit(Xz, Y[:, u])
    W_ref[:, u] = m.coef_; b_ref[u] = m.intercept_; r2_ref[u] = m.score(Xz, Y[:, u])
print(f"sklearn ElasticNet R^2: greenL={r2_ref[0]:.4f}  greenR={r2_ref[1]:.4f}")

out = os.path.join(DIR, "chantranupong_compare.npz")
np.savez(out, Xz=Xz, Y=Y, W_ref=W_ref, b_ref=b_ref, r2_ref=r2_ref,
         alpha=ALPHA, l1_ratio=L1, names=np.array(names))
print("saved ->", out)
io.close()
