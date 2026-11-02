"""Temporal occlusion of the frozen CBraMod + logistic regression model, chb01 test seizure.

Run from anywhere:  python C:\\rtds-group\\scripts\\explain_cbramod_chb01.py [--stride 2]
Needs: chb01_25.edf and chb01_26.edf in EDF_DIR, the weights in third_party/cbramod, the artifacts folder.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

REPO = Path(r"C:\rtds-group")
sys.path.insert(0, str(REPO))
from rtds import cbramod_feat as cf, protocol as pr          # noqa: E402
from rtds.chb import parse_summary                           # noqa: E402
from rtds.config import Protocol                             # noqa: E402

EDF_DIR = Path(r"C:\seizure-xai\data\raw\chb01")
ART = Path(r"C:\rtds-artifacts\unzipped")
OUT = Path(r"C:\seizure-xai\results")
ONSET = 91350            # test seizure onset on the chb01 patient timeline (s), from find_test_seizure.py
N_SEC, TOP_K, N_RAND, NOISE = 10, 3, 5, 0.05

ap = argparse.ArgumentParser()
ap.add_argument("--stride", type=int, default=1, help="explain one window out of N (faster)")
a = ap.parse_args()
rng = np.random.default_rng(0)
P = Protocol()
lo, hi = ONSET - P.sph - P.sop, ONSET - P.sph      # pre-ictal window: 35 to 5 min before onset
CONTEXT = 160                                      # seconds before `lo`, for the causal smoothing

# --- classifier, threshold, recordings
bundle = joblib.load(ART / "models" / "cbramod" / "chb01.joblib")
print("joblib keys:", list(bundle.keys()))
clf, mu = bundle["clf"], bundle["mu"]
sd = bundle.get("sd", bundle.get("std"))
if sd is None:
    raise SystemExit(f"No 'sd' entry in the joblib file, keys are {list(bundle.keys())}")
thr = json.load(open(ART / "results_cbramod.json"))["chb01"]["threshold"]
recs = {r.name: r for r in parse_summary(EDF_DIR / "chb01-summary.txt")}


def score(emb):
    return clf.decision_function((emb - mu) / sd)


# --- windows of the two recordings that cover the pre-ictal period
model = cf.load_cbramod("cpu")
T, W = [], []
for name in ["chb01_25.edf", "chb01_26.edf"]:
    w = cf.read_edf_windows(EDF_DIR / name)
    if w is None:
        raise SystemExit(f"{name}: missing channels")
    T.append(recs[name].start + cf.WIN_S * (np.arange(len(w)) + 1))
    W.append(w)
T, W = np.concatenate(T), np.concatenate(W)
keep = (T >= lo - CONTEXT) & (T <= hi)
T, W = T[keep], W[keep]
t0 = time.time()
E = cf.embed(model, W, "cpu")
print(f"Embedded {len(W)} windows in {time.time() - t0:.0f} s")

# --- check 1: do we reproduce the cached embeddings of the result files?
cache = np.load(ART / "cache" / "chb01_cbramod.npz")
tc, ec = cache["t"], cache["e"]
idx = np.minimum(np.searchsorted(tc, T), len(tc) - 1)
ok = np.isclose(tc[idx], T)
print("Windows matched with the cache:", int(ok.sum()), "of", len(T))
print("Max |embedding - cached embedding|:", float(np.abs(ec[idx[ok]] - E[ok]).max()))

# --- check 2: does the model miss this seizure, as in the result file (hits = 0)?
s = score(E)
s_smooth = pr.smooth(s, 16)
inwin = (T >= lo) & (T <= hi)
print(f"Pre-ictal windows: {int(inwin.sum())} | max smoothed score: {s_smooth[inwin].max():.3f} "
      f"| threshold: {thr:.3f} | alarm raised: {bool((s_smooth[inwin] >= thr).any())}")

# --- temporal occlusion, faithfulness and stability, window by window
ex = np.flatnonzero(inwin)[::a.stride]
drops = np.zeros((len(ex), N_SEC))
d_top, d_rand, rho = [], [], []


def occlusions(w):
    X = np.repeat(w[None], N_SEC, 0).copy()
    for j in range(N_SEC):
        X[j, :, j, :] = 0.0                  # remove second j (zero = channel mean after band-pass)
    return X


t0 = time.time()
for n, i in enumerate(ex):
    w, base = W[i], s[i]
    noisy = (w + rng.normal(0, NOISE * w.std(), w.shape)).astype(np.float32)
    # random deletions use the same K, top-K deletion is added after we know the ranking
    X_occ = occlusions(w)
    X_rand = np.repeat(w[None], N_RAND, 0).copy()
    for r in range(N_RAND):
        for sec in rng.choice(N_SEC, TOP_K, replace=False):
            X_rand[r, :, sec, :] = 0.0
    X_noise = np.concatenate([noisy[None], occlusions(noisy)])
    sc = score(cf.embed(model, np.concatenate([X_occ, X_rand, X_noise])))
    drops[n] = base - sc[:N_SEC]
    d_rand.append(base - sc[N_SEC:N_SEC + N_RAND].mean())
    base_n, sc_n = sc[N_SEC + N_RAND], sc[N_SEC + N_RAND + 1:]
    rho.append(spearmanr(drops[n], base_n - sc_n).correlation)
    X_top = w[None].copy()
    for sec in np.argsort(drops[n])[-TOP_K:]:
        X_top[0, :, sec, :] = 0.0
    d_top.append(base - score(cf.embed(model, X_top))[0])
    if n == 4:
        print(f"~{(time.time() - t0) / 5 * len(ex) / 60:.1f} min expected for {len(ex)} windows")

d_top, d_rand, rho = np.array(d_top), np.array(d_rand), np.array(rho, float)
print("\n=== Results (decision-function units) ===")
print("Mean importance per second:", np.round(drops.mean(0), 3))
print(f"Faithfulness: removing the top-{TOP_K} seconds lowers the score by {d_top.mean():.3f}, "
      f"random {TOP_K} seconds by {d_rand.mean():.3f}; top > random in {(d_top > d_rand).mean():.0%} of windows")
print("  Wilcoxon signed-rank (top > random), p =", f"{wilcoxon(d_top, d_rand, alternative='greater').pvalue:.2e}")
print(f"Stability (Spearman, noise {NOISE:.0%}): median {np.nanmedian(rho):.2f} (IQR "
      f"{np.nanpercentile(rho, 25):.2f} to {np.nanpercentile(rho, 75):.2f})")

# --- figure
OUT.mkdir(exist_ok=True)
minutes = (ONSET - T[ex]) / 60
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.3, 1]})
vmax = np.abs(drops).max()
im = ax1.imshow(drops, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax1.set_xlabel("Second inside the 10-s window")
ax1.set_ylabel("Minutes before seizure onset (window end)")
yt = np.linspace(0, len(ex) - 1, 6).astype(int)
ax1.set_yticks(yt)
ax1.set_yticklabels([f"{minutes[k]:.0f}" for k in yt])
fig.colorbar(im, ax=ax1, label="Score drop when the second is removed")
m, se = drops.mean(0), drops.std(0, ddof=1) / np.sqrt(len(ex))
ax2.bar(np.arange(N_SEC), m, yerr=se, color="#0B3D91")
ax2.axhline(0, color="grey", lw=0.8)
ax2.set_xlabel("Second inside the 10-s window")
ax2.set_ylabel("Mean importance (+/- s.e.m.)")
fig.suptitle("chb01, test seizure: CBraMod + logistic regression, temporal occlusion")
fig.tight_layout()
fig.savefig(OUT / "cbramod_chb01_occlusion.png", dpi=200)
np.savez(OUT / "cbramod_chb01_occlusion.npz", drops=drops, minutes=minutes, d_top=d_top, d_rand=d_rand, rho=rho)
print("Saved", OUT / "cbramod_chb01_occlusion.png")
