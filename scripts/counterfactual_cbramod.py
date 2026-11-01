"""Counterfactual explanation for MISSED seizures of the frozen CBraMod + logistic regression model.

Question: what is the smallest change of 1 to 3 seconds of the EEG window that would push the model score
above its alarm threshold?  The seconds are chosen by the temporal occlusion (most important), compared with
the least important seconds and with random seconds (control).

Run:  python C:\\rtds-group\\scripts\\counterfactual_cbramod.py            (all missed seizures)
      python C:\\rtds-group\\scripts\\counterfactual_cbramod.py --k 3 --patients chb01      (quick test)
Needs results\\occlusion\\*.npz (from explain_cbramod_all.py).  Output: results\\counterfactual_cbramod.csv / .png
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np

ART = Path(r"C:\rtds-artifacts\unzipped"); RAW = Path(r"C:\seizure-xai\data\raw")
OUT = Path(r"C:\seizure-xai\results"); S3 = "s3://physionet-open/chbmit/1.0.0"
REPO = Path(r"C:\rtds-group")
PRE = 1200            # the window ends 20 min before onset (same window as the SHAP/LIME/saliency comparison)


# ---------------------------------------------------------------- numerics (numpy only, testable without torch)
def pgd(scorer, x, mask, eps, steps):
    """Maximise the score with a perturbation restricted to the masked seconds and an L2 budget of
    eps * ||selected part of the window||.  x: (B,16,10,200), mask: (B,1,10,1), eps: (B,)."""
    sel_norm = np.sqrt(((x * mask) ** 2).sum(axis=(1, 2, 3)))
    R = eps * sel_norm
    delta = np.zeros_like(x)
    best = np.full(len(x), -np.inf)
    for it in range(steps + 1):
        s, g = scorer.score_grad(x + delta)
        best = np.maximum(best, s)
        if it == steps:
            break
        g = g * mask
        gn = np.sqrt((g ** 2).sum(axis=(1, 2, 3))) + 1e-12
        delta = delta + ((R / 3.0) * (0.85 ** it) / gn)[:, None, None, None] * g
        dn = np.sqrt((delta ** 2).sum(axis=(1, 2, 3))) + 1e-12
        delta = delta * np.minimum(1.0, R / dn)[:, None, None, None]
    return best


def min_eps(scorer, x, mask, thr, eps_max, steps, iters):
    """Smallest relative budget (bisection) for which the score reaches thr; NaN if eps_max is not enough."""
    B = len(x)
    feasible = pgd(scorer, x, mask, np.full(B, eps_max), steps) >= thr
    lo, hi = np.full(B, eps_max * 1e-4), np.full(B, eps_max)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        ok = pgd(scorer, x, mask, mid, steps) >= thr
        hi, lo = np.where(ok, mid, hi), np.where(ok, lo, mid)
    return np.where(feasible, hi, np.nan)


def seconds_mask(sel, n=10):
    m = np.zeros(n, np.float32); m[list(sel)] = 1.0
    return m


def run_window(scorer, w, thr, ks, n_draws, eps_max, steps, iters, rng):
    """All arms for one window. Returns a list of dict rows."""
    base = float(scorer.score(w[None])[0])
    M = np.ones((10, 10), np.float32) - np.eye(10, dtype=np.float32)
    drops = base - scorer.score((w[None] * M[:, None, :, None]).astype(np.float32))
    rows = []
    for k in ks:
        sels = [("occlusion-top", 0, np.argsort(drops)[-k:]), ("occlusion-bottom", 0, np.argsort(drops)[:k])]
        sels += [("random", d, rng.choice(10, k, replace=False)) for d in range(n_draws)]
        mask = np.stack([seconds_mask(s)[None, :, None] for _, _, s in sels]).astype(np.float32)   # (B,1,10,1)
        x = np.repeat(w[None], len(sels), 0).astype(np.float32)
        e = min_eps(scorer, x, mask, thr, eps_max, steps, iters)
        sel_norm = np.sqrt(((x * mask) ** 2).sum(axis=(1, 2, 3))); win_norm = np.sqrt((w ** 2).sum())
        for (arm, d, s), ei, sn in zip(sels, e, sel_norm):
            rows.append(dict(k=k, arm=arm, draw=d, seconds=",".join(map(str, sorted(s))), eps_sel=ei,
                             eps_window=ei * sn / win_norm, base=base, thr=thr))
    return rows


def summarise(df, eps_max):
    """Per seizure and arm: validity (share of draws reaching the threshold) and median eps (NaN counted as eps_max)."""
    d = df.copy()
    d["valid"] = d.eps_sel.notna()
    d["eps_c"] = d.eps_sel.fillna(eps_max)
    g = d.groupby(["patient", "onset", "k", "arm"]).agg(valid=("valid", "mean"), eps=("eps_c", "median")).reset_index()
    return g


# ---------------------------------------------------------------- torch wrapper
class TorchScorer:
    def __init__(self, model, clf, mu, sd):
        import torch
        self.t = torch; self.model = model
        for p in model.parameters(): p.requires_grad_(False)
        self.coef = torch.tensor(clf.coef_[0], dtype=torch.float32)
        self.b = torch.tensor(clf.intercept_[0], dtype=torch.float32)
        self.mu = torch.tensor(mu, dtype=torch.float32); self.sd = torch.tensor(sd, dtype=torch.float32)

    def _s(self, xt):
        f = self.model.encoder(self.model.patch_embedding(xt)).mean(dim=(1, 2))
        return ((f - self.mu) / self.sd) @ self.coef + self.b

    def score(self, x):
        with self.t.no_grad():
            out = [self._s(self.t.from_numpy(np.ascontiguousarray(x[i:i + 16], dtype=np.float32))).numpy()
                   for i in range(0, len(x), 16)]
        return np.concatenate(out)

    def score_grad(self, x):
        xt = self.t.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).clone().requires_grad_(True)
        s = self._s(xt); s.sum().backward()
        return s.detach().numpy(), xt.grad.numpy()


# ---------------------------------------------------------------- main
def main():
    import joblib, json
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    sys.path.insert(0, str(REPO))
    from rtds import cbramod_feat as cf
    from rtds.chb import parse_summary

    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--k", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--eps-max", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    res_all = json.load(open(ART / "results_cbramod.json"))
    missed = []
    for f in sorted((OUT / "occlusion").glob("*.npz")):
        d = np.load(f)
        if not bool(d["hit"]):
            missed.append((str(d["patient"]), float(d["onset"])))
    if a.patients: missed = [m for m in missed if m[0] in a.patients]
    print(len(missed), "missed seizures:", missed, flush=True)
    model = cf.load_cbramod("cpu")

    def aws(src, dst):
        return subprocess.run(f'aws s3 cp --no-sign-request "{src}" "{dst}" --only-show-errors', shell=True).returncode == 0

    rows, skipped = [], []
    for pat, onset in missed:
        bundle = joblib.load(ART / "models" / "cbramod" / f"{pat}.joblib")
        scorer = TorchScorer(model, bundle["clf"], bundle["mu"], bundle["sd"])
        thr = res_all[pat]["threshold"]
        pdir = RAW / pat; pdir.mkdir(parents=True, exist_ok=True)
        summ = pdir / f"{pat}-summary.txt"
        if not summ.exists(): aws(f"{S3}/{pat}/{pat}-summary.txt", summ)
        target = onset - PRE
        rec = next(r for r in parse_summary(summ) if r.start <= target - 10 and r.end >= target)
        fpath = pdir / rec.name; downloaded = False
        if not fpath.exists(): downloaded = aws(f"{S3}/{pat}/{rec.name}", fpath)
        wins = cf.read_edf_windows(fpath)
        end = int(round(target - rec.start))
        w = wins.transpose(1, 0, 2, 3).reshape(16, -1)[:, end * 200 - 2000: end * 200].reshape(16, 10, 200).astype(np.float32)
        if downloaded and not a.keep: fpath.unlink(missing_ok=True)
        base = float(scorer.score(w[None])[0])
        if base >= thr:
            print(f"{pat} {int(onset)}: score {base:.2f} already >= threshold {thr:.2f}, skipped", flush=True)
            skipped.append((pat, onset)); continue
        r = run_window(scorer, w, thr, a.k, a.draws, a.eps_max, a.steps, a.iters, rng)
        for x in r: x.update(patient=pat, onset=onset)
        rows += r
        top3 = [x for x in r if x["arm"] == "occlusion-top" and x["k"] == max(a.k)][0]
        print(f"{pat} {int(onset)}: score {base:.2f} vs thr {thr:.2f} | occlusion-top k={top3['k']}: "
              f"eps = {top3['eps_sel']:.3f}", flush=True)
    if not rows: raise SystemExit("Nothing computed.")
    df = pd.DataFrame(rows); df.to_csv(OUT / "counterfactual_cbramod.csv", index=False)
    g = summarise(df, a.eps_max); n = g[["patient", "onset"]].drop_duplicates().shape[0]
    print(f"\n=== {n} missed seizures analysed ({len(skipped)} skipped: already above threshold) ===")
    for k in a.k:
        print(f"\nk = {k} second(s)")
        for arm in ["occlusion-top", "occlusion-bottom", "random"]:
            s = g[(g.k == k) & (g.arm == arm)]
            v = s[s.valid > 0]
            print(f"  {arm:17s} validity {s.valid.mean():.0%} | median eps (all, censored at {a.eps_max}) {s.eps.median():.2f}"
                  f" | median eps (valid only) {v.eps.median() if len(v) else float('nan'):.2f}")
        t = g[(g.k == k) & (g.arm == "occlusion-top")].set_index(["patient", "onset"]).eps
        for other in ["random", "occlusion-bottom"]:
            o = g[(g.k == k) & (g.arm == other)].set_index(["patient", "onset"]).eps.reindex(t.index)
            if n >= 6 and (t - o).abs().sum() > 0:
                print(f"  Wilcoxon eps(occlusion-top) < eps({other}): p = {wilcoxon(t, o, alternative='less').pvalue:.3g}")

    # ---- figure
    NAVY, GREY, ORANGE = "#0B3D91", "#9AA3B5", "#D9822B"
    plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    arms = [("occlusion-top", "most important\n(occlusion)", NAVY), ("random", "random", GREY), ("occlusion-bottom", "least important\n(occlusion)", ORANGE)]
    ks = sorted(g.k.unique()); wd = 0.26
    for i, (arm, lab, c) in enumerate(arms):
        v = [g[(g.k == k) & (g.arm == arm)].valid.mean() * 100 for k in ks]
        ax[0].bar(np.arange(len(ks)) + (i - 1) * wd, v, wd, color=c, label=lab.replace("\n", " "))
    ax[0].set_xticks(range(len(ks))); ax[0].set_xticklabels([f"{k} s" for k in ks])
    ax[0].set_xlabel("Number of modified seconds"); ax[0].set_ylabel("Missed seizures turned into alarms (%)")
    ax[0].set_ylim(0, 105); ax[0].set_title("Validity"); ax[0].legend(frameon=False, fontsize=13, loc="upper left")
    kk = max(ks)
    for i, (arm, lab, c) in enumerate(arms):
        e = g[(g.k == kk) & (g.arm == arm)].eps.values * 100
        ax[1].boxplot(e, positions=[i], widths=0.5, patch_artist=True, boxprops=dict(facecolor="white", edgecolor=c, lw=2),
                      medianprops=dict(color=c, lw=2.5), whiskerprops=dict(color=c), capprops=dict(color=c), showfliers=False)
        ax[1].scatter(np.full(len(e), i) + rng.uniform(-.12, .12, len(e)), e, color=c, s=40, zorder=3)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([l for _, l, _ in arms], fontsize=13)
    ax[1].set_ylabel("Change needed (% of the signal)"); ax[1].set_title(f"Proximity ({kk} seconds modified)")
    ax[1].axhline(a.eps_max * 100, color="k", lw=1, ls=(0, (4, 3))); ax[1].text(2.4, a.eps_max * 100 + 1, "not reached", ha="right", fontsize=12)
    fig.suptitle(f"Counterfactuals for {n} missed seizures (CBraMod + logistic regression)", fontweight="bold", color=NAVY)
    fig.tight_layout(); fig.savefig(OUT / "counterfactual_cbramod.png", dpi=200)
    print("Saved", OUT / "counterfactual_cbramod.png")


if __name__ == "__main__":
    main()
