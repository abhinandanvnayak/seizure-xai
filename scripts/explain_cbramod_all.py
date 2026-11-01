"""Temporal occlusion of frozen CBraMod + logistic regression on EVERY test seizure of EVERY patient.

Run:   python C:\\rtds-group\\scripts\\explain_cbramod_all.py --patients chb01          (check: same numbers as before)
       python C:\\rtds-group\\scripts\\explain_cbramod_all.py                          (all 16 patients)
Resumable: finished seizures are stored in results\\occlusion\\ and skipped when you run it again.
Downloads the needed EDF files from PhysioNet with aws (then deletes them, use --keep to keep them).
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu

REPO = Path(r"C:\rtds-group")
sys.path.insert(0, str(REPO))
from rtds import cbramod_feat as cf, protocol as pr          # noqa: E402
from rtds.chb import parse_summary                           # noqa: E402
from rtds.config import Protocol                             # noqa: E402

ART = Path(r"C:\rtds-artifacts\unzipped")
RAW = Path(r"C:\seizure-xai\data\raw")
OUT = Path(r"C:\seizure-xai\results")
S3 = "s3://physionet-open/chbmit/1.0.0"
N_SEC, TOP_K, N_RAND, NOISE = 10, 3, 5, 0.05

ap = argparse.ArgumentParser()
ap.add_argument("--patients", nargs="*", default=None)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--keep", action="store_true", help="keep downloaded EDF files")
a = ap.parse_args()
P = Protocol()
(OUT / "occlusion").mkdir(parents=True, exist_ok=True)
res_all = json.load(open(ART / "results_cbramod.json"))
patients = a.patients or sorted(res_all)


def aws(src, dst):
    r = subprocess.run(f'aws s3 cp --no-sign-request "{src}" "{dst}" --only-show-errors', shell=True)
    return r.returncode == 0


def occlusions(w):
    X = np.repeat(w[None], N_SEC, 0).copy()
    for j in range(N_SEC):
        X[j, :, j, :] = 0.0
    return X


model = None
rng = np.random.default_rng(0)
for pat in patients:
    pdir = RAW / pat
    pdir.mkdir(parents=True, exist_ok=True)
    summ = pdir / f"{pat}-summary.txt"
    if not summ.exists() and not aws(f"{S3}/{pat}/{pat}-summary.txt", summ):
        print(pat, "summary not available, skipped"); continue
    recs = parse_summary(summ)
    sz = sorted(s for r in recs for s in r.seizures)
    split = pr.chrono_split(sz, P)
    if split is None:
        print(pat, ": <2 target seizures, skipped"); continue
    test_targets = split[2]
    bundle = joblib.load(ART / "models" / "cbramod" / f"{pat}.joblib")
    clf, mu, sd = bundle["clf"], bundle["mu"], bundle["sd"]
    k_smooth = int(bundle.get("smooth_steps", 16))
    thr = res_all[pat]["threshold"]
    cache = np.load(ART / "cache" / f"{pat}_cbramod.npz")
    tc, ec = cache["t"], cache["e"]
    score = lambda emb: clf.decision_function((emb - mu) / sd)
    for on, off in test_targets:
        tag = f"{pat}_{on}"
        f_out = OUT / "occlusion" / f"{tag}.npz"
        if f_out.exists():
            print(tag, "already done"); continue
        lo, hi = on - P.sph - P.sop, on - P.sph
        need = [r for r in recs if r.end >= lo - 160 and r.start <= hi]
        got, T, W = [], [], []
        ok = True
        for r in need:
            f = pdir / r.name
            if not f.exists():
                if not aws(f"{S3}/{pat}/{r.name}", f):
                    ok = False; break
                got.append(f)
            w = cf.read_edf_windows(f)
            if w is None:
                ok = False; break
            T.append(r.start + cf.WIN_S * (np.arange(len(w)) + 1)); W.append(w)
        if not ok or not T:
            print(tag, ": files unavailable or channels missing, skipped")
            if not a.keep:
                for f in got: f.unlink(missing_ok=True)
            continue
        T, W = np.concatenate(T), np.concatenate(W)
        keep = (T >= lo - 160) & (T <= hi)
        T, W = T[keep], W[keep]
        inwin = (T >= lo) & (T <= hi)
        if inwin.sum() < 0.5 * (hi - lo) / cf.WIN_S:
            print(tag, ": less than half of the pre-ictal window is covered, skipped")
            if not a.keep:
                for f in got: f.unlink(missing_ok=True)
            continue
        if model is None:
            model = cf.load_cbramod("cpu")
        E = cf.embed(model, W, "cpu")
        idx = np.minimum(np.searchsorted(tc, T), len(tc) - 1)
        m = np.isclose(tc[idx], T)
        diff = float(np.abs(ec[idx[m]] - E[m]).max()) if m.any() else float("nan")
        s = score(E)
        ss = pr.smooth(s, k_smooth)
        maxs = float(ss[inwin].max())
        hit = bool((ss[inwin] >= thr).any())
        ex = np.flatnonzero(inwin)[::a.stride]
        drops = np.zeros((len(ex), N_SEC)); d_top, d_rand, rho = [], [], []
        t0 = time.time()
        for n, i in enumerate(ex):
            w, base = W[i], s[i]
            noisy = (w + rng.normal(0, NOISE * w.std(), w.shape)).astype(np.float32)
            X_rand = np.repeat(w[None], N_RAND, 0).copy()
            for r_ in range(N_RAND):
                for sec in rng.choice(N_SEC, TOP_K, replace=False):
                    X_rand[r_, :, sec, :] = 0.0
            Xall = np.concatenate([occlusions(w), X_rand, noisy[None], occlusions(noisy)])
            sc = score(cf.embed(model, Xall))
            drops[n] = base - sc[:N_SEC]
            d_rand.append(base - sc[N_SEC:N_SEC + N_RAND].mean())
            rho.append(spearmanr(drops[n], sc[N_SEC + N_RAND] - sc[N_SEC + N_RAND + 1:]).correlation)
            Xt = w[None].copy()
            for sec in np.argsort(drops[n])[-TOP_K:]:
                Xt[0, :, sec, :] = 0.0
            d_top.append(base - score(cf.embed(model, Xt))[0])
        np.savez(f_out, patient=pat, onset=on, hit=hit, maxs=maxs, thr=thr, cache_diff=diff,
                 drops=drops, minutes=(on - T[ex]) / 60, d_top=np.array(d_top), d_rand=np.array(d_rand),
                 rho=np.array(rho, float))
        print(f"{tag}: windows {len(ex)} | hit={hit} | max score {maxs:.2f} vs thr {thr:.2f} | "
              f"cache diff {diff:.1e} | {time.time() - t0:.0f} s", flush=True)
        if not a.keep:
            for f in got: f.unlink(missing_ok=True)

# ------------------------------------------------------------------ aggregation
files = sorted((OUT / "occlusion").glob("*.npz"))
if not files:
    raise SystemExit("No seizure processed.")
rows = []
for f in files:
    d = np.load(f)
    rows.append(dict(tag=f.stem, hit=bool(d["hit"]), diff=float(d["cache_diff"]), prof=d["drops"].mean(0),
                     top=float(d["d_top"].mean()), rnd=float(d["d_rand"].mean()),
                     frac=float((d["d_top"] > d["d_rand"]).mean()), rho=float(np.nanmedian(d["rho"])),
                     mean_imp=float(d["drops"].mean()), minutes=d["minutes"], wimp=d["drops"].mean(1)))
n = len(rows)
hits = np.array([r["hit"] for r in rows])
top, rnd = np.array([r["top"] for r in rows]), np.array([r["rnd"] for r in rows])
print(f"\n=== {n} test seizures ({hits.sum()} detected, {(~hits).sum()} missed) ===")
print("Largest |embedding - cache| over seizures:", max(r["diff"] for r in rows))
print(f"Faithfulness per seizure: top-{TOP_K} drop {top.mean():.2f} vs random {rnd.mean():.2f}; "
      f"top > random in {(top > rnd).sum()}/{n} seizures")
if n >= 6:
    print("  Wilcoxon over seizures (top > random): p =", f"{wilcoxon(top, rnd, alternative='greater').pvalue:.3g}")
print("Stability (median Spearman per seizure): median", np.round(np.median([r["rho"] for r in rows]), 2),
      "min", np.round(min(r["rho"] for r in rows), 2))
mi = np.array([r["mean_imp"] for r in rows])
if hits.any() and (~hits).any():
    print(f"Mean importance: detected {mi[hits].mean():.2f} vs missed {mi[~hits].mean():.2f}; "
          f"Mann-Whitney p = {mannwhitneyu(mi[hits], mi[~hits]).pvalue:.3g}")
    gap = top - rnd
    print(f"Faithfulness gap (top - random): detected {gap[hits].mean():.2f} vs missed {gap[~hits].mean():.2f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for i in range(n):
    ax[0].plot([0, 1], [rnd[i], top[i]], color="#B4472B" if hits[i] else "#0B3D91", alpha=.6, marker="o")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["3 random\nseconds", "top-3\nseconds"])
ax[0].set_ylabel("Score drop when removed"); ax[0].set_title("Faithfulness (one line per seizure)")
P_ = np.array([r["prof"] for r in rows])
for flag, col, lab in [(True, "#B4472B", "detected"), (False, "#0B3D91", "missed")]:
    if (hits == flag).any():
        ax[1].plot(range(N_SEC), P_[hits == flag].mean(0), color=col, marker="o", label=f"{lab} (n={int((hits == flag).sum())})")
ax[1].set_xlabel("Second inside the 10-s window"); ax[1].set_ylabel("Mean importance"); ax[1].legend()
ax[1].set_title("Importance inside the window")
bins = np.arange(5, 40, 5)
for flag, col, lab in [(True, "#B4472B", "detected"), (False, "#0B3D91", "missed")]:
    sel = [r for r, h in zip(rows, hits) if h == flag]
    if sel:
        mn = np.concatenate([r["minutes"] for r in sel]); wi = np.concatenate([r["wimp"] for r in sel])
        y = [wi[(mn >= b) & (mn < b + 5)].mean() if ((mn >= b) & (mn < b + 5)).any() else np.nan for b in bins[:-1]]
        ax[2].plot(bins[:-1] + 2.5, y, color=col, marker="o", label=lab)
ax[2].invert_xaxis(); ax[2].set_xlabel("Minutes before onset"); ax[2].set_ylabel("Mean importance")
ax[2].set_title("Importance as the seizure gets closer"); ax[2].legend()
fig.suptitle(f"CBraMod + logistic regression, temporal occlusion on {n} test seizures")
fig.tight_layout(); fig.savefig(OUT / "cbramod_all_occlusion.png", dpi=200)
print("Saved", OUT / "cbramod_all_occlusion.png")
