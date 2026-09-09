"""Level A — step 1: download one BWM session and introspect the IBL encoding design-matrix
API so we can build X/Y identically to neurencoding. Saves raw spikes+trials to netscratch
for reuse, and prints the structure of brainwidemap.encoding / neurencoding.design_matrix.

Run via SLURM (see ibl/run_extract.sh); needs the `ibl` conda env + network.
"""
import os
import numpy as np

PID = "56f2a378-78d2-4132-b3c8-8c1ba82be598"
EID = "6713a4a7-faed-4df2-acab-ee4e63326f8d"
NS = os.environ.get("JAXGLM_DATA", "/n/netscratch/kempner_bsabatini_lab/Lab/twheatcroft")
OUT = os.path.join(NS, "ibl_level_a")
os.makedirs(OUT, exist_ok=True)

from one.api import ONE
one = ONE(base_url="https://openalyx.internationalbrainlab.org", silent=True,
          cache_dir=os.environ.get("ONE_CACHE",
                                   os.path.join(NS, "ibl_cache")))

# ----------------------------------------------------------------- introspect the encoding API
def show(title, obj, hide_private=True):
    names = [n for n in dir(obj) if not (hide_private and n.startswith("_"))]
    print(f"--- {title} ---")
    print(names)

print("================ ENCODING API INTROSPECTION ================")
try:
    import neurencoding
    show("neurencoding", neurencoding)
    import neurencoding.design_matrix as ndm
    show("neurencoding.design_matrix", ndm)
    import neurencoding.linear as nlin
    show("neurencoding.linear", nlin)
    import neurencoding.poisson as npois
    show("neurencoding.poisson", npois)
except Exception as e:
    print("neurencoding introspection error:", repr(e))

try:
    import pkgutil, brainwidemap.encoding as benc
    show("brainwidemap.encoding", benc)
    print("--- brainwidemap.encoding submodules ---")
    print([m.name for m in pkgutil.iter_modules(benc.__path__)])
    # the BWM design-matrix construction usually lives in design.py / params.py
    for sub in ("design", "params", "glm_predict", "utils"):
        try:
            mod = __import__(f"brainwidemap.encoding.{sub}", fromlist=["x"])
            show(f"brainwidemap.encoding.{sub}", mod)
        except Exception as e:
            print(f"  ({sub} not importable: {e})")
except Exception as e:
    print("brainwidemap.encoding introspection error:", repr(e))

# ----------------------------------------------------------------------- download session data
print("\n================ SESSION DATA ================")
from brainbox.io.one import SpikeSortingLoader
ssl = SpikeSortingLoader(pid=PID, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)
st, sc = np.asarray(spikes["times"]), np.asarray(spikes["clusters"])
label = np.asarray(clusters.get("label", []))
good = np.where(label >= 1.0)[0] if label.size else np.array([])
print(f"spikes: {st.shape[0]:,} | clusters: {label.size} | good(label>=1): {good.size}")
print(f"recording span: {st.min():.1f}-{st.max():.1f} s")

trials = one.load_object(EID, "trials")
tkeys = [k for k in trials.keys()]
print("trial keys:", tkeys)
n_tr = np.asarray(trials["stimOn_times"]).shape[0]
print(f"n trials: {n_tr}")

# save raw arrays for the next step (build X/Y); keep it lean
np.savez(os.path.join(OUT, "session_raw.npz"),
         pid=PID, eid=EID, spike_times=st, spike_clusters=sc,
         cluster_label=label,
         **{f"trials_{k}": np.asarray(trials[k]) for k in tkeys
            if np.asarray(trials[k]).dtype != object})
print(f"\nsaved -> {os.path.join(OUT, 'session_raw.npz')}")
print("================ done ================")
