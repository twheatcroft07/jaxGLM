"""Chantranupong/Lynne LEVEL B -- step 1 (nwb env): exact in-repo reproduction. Use the lab's
own preprocessed dataframe (sabatinilab-glm Figure_6 combo_df.csv, the post-lynne_pp output) and
their exact model spec (X_cols, y='gGLUr', shifts -20..+20, OLS alpha=0), build the shift-kernel
design with design.build_design, and fit the reference with sklearn (alpha=0 OLS == what sglm
uses for model_type='Normal'). No NWB adapter, no reconstruction -- their preprocessing, their
config. tiny ridge (1e-8) is used so the rank-deficient shifted-event design has a unique solution
(== the minimum-norm OLS their LinearRegression returns).
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet   # mean-normalized objective == jaxGLM's (Ridge is NOT)

import design as dz

REPO = "/n/home02/twheatcroft/code/sabatinilab-glm/sglm/outputs_clean/Figure_6/g1/-20_+20"
NS = os.environ.get("JAXGLM_DATA", "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft")
OUT = os.path.join(NS, "chantranupong", "levelb_compare.npz")
XCOLS = ["photometryCenterInIndex", "photometrySideInIndex", "photometrySideInIndexr",
         "photometrySideOutIndex", "sl", "spnnrOff"]               # their base_simple spec
YCOL = "gGLUr"
# their published fit is alpha=0 (pure OLS), but the shifted-event design is collinear/rank-
# deficient, so the alpha=0 solution is NON-unique. A small elastic-net makes it unique so jaxGLM
# and sklearn must agree on the lab's own preprocessed data. ElasticNet (NOT Ridge) is used
# because its objective is mean-normalized like jaxGLM's, so alpha is on the same scale.
ALPHA, L1 = 1e-3, 0.5

df = pd.read_csv(os.path.join(REPO, "iXyp_0", "combo_df.csv"))
# 'photometrySideInIndex' (all side-ins) is derived = rewarded + unrewarded, not stored
if "photometrySideInIndex" not in df.columns:
    df["photometrySideInIndex"] = df["photometrySideInIndexr"] + df["photometrySideInIndexnr"]
print(f"combo_df: {df.shape} | has cols: {[c for c in XCOLS+[YCOL] if c in df.columns]}")
if "wi_trial_keep" in df.columns:                                  # their within-trial (non-ITI) mask
    df = df[df["wi_trial_keep"].astype(float) > 0]

base = {c: np.nan_to_num(df[c].to_numpy(dtype=float)) for c in XCOLS}
X, names = dz.build_design(base, range(-20, 21))                    # their shift window
y = df[YCOL].to_numpy(dtype=float)
keep = np.isfinite(y)
X, y = X[keep], y[keep]
print(f"X {X.shape} | y {y.shape} ({len(names)} features)")

m = ElasticNet(alpha=ALPHA, l1_ratio=L1, fit_intercept=True, max_iter=20000, tol=1e-10).fit(X, y)
print(f"sklearn ElasticNet R^2 on their data = {m.score(X, y):.4f}")
np.savez(OUT, X=X, y=y, coef_ref=m.coef_, intercept_ref=m.intercept_,
         alpha=ALPHA, l1_ratio=L1, names=np.array(names))
print(f"saved -> {OUT}")
