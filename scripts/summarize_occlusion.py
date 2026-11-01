"""Summarise results\\occlusion\\*.npz (temporal occlusion on all test seizures). Read-only, fast."""
import sys, glob
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
OUT = sys.argv[1] if len(sys.argv) > 1 else r"C:\seizure-xai\results"
rows = []
for f in sorted(glob.glob(f"{OUT}/occlusion/*.npz")):
    d = np.load(f, allow_pickle=True)
    m = lambda k: float(np.nanmean(np.asarray(d[k], float)))
    dr=np.asarray(d['drops'],float)
    rows.append(dict(imp=float(np.nanmean(dr)), edge=float(np.nanmean(dr[:,[0,9]])), mid=float(np.nanmean(dr[:,1:9])), pat=str(d["patient"]), onset=float(d["onset"]), hit=bool(d["hit"]),
                     top=m("d_top"), rand=m("d_rand"), rho=m("rho"), cache=float(np.nanmax(np.abs(d["cache_diff"]))),
                     nwin=int(np.asarray(d["drops"]).shape[0])))
n = len(rows); hit = np.array([r["hit"] for r in rows])
top = np.array([r["top"] for r in rows]); rnd = np.array([r["rand"] for r in rows]); rho = np.array([r["rho"] for r in rows])
print(f"{n} seizures: {hit.sum()} detected, {(~hit).sum()} missed | windows per seizure: min {min(r['nwin'] for r in rows)}, median {int(np.median([r['nwin'] for r in rows]))}")
print(f"max |cache difference| = {max(r['cache'] for r in rows):.2g}")
print(f"\nFAITHFULNESS (score drop, mean per seizure)")
print(f"  top-3 removed : median {np.median(top):.3f} (IQR {np.percentile(top,25):.3f} to {np.percentile(top,75):.3f})")
print(f"  random-3      : median {np.median(rnd):.3f} (IQR {np.percentile(rnd,25):.3f} to {np.percentile(rnd,75):.3f})")
print(f"  top > random in {(top>rnd).sum()}/{n} seizures | Wilcoxon one-sided p = {wilcoxon(top, rnd, alternative='greater').pvalue:.4f}")
print(f"\nSTABILITY (Spearman rho with 5% noise): median {np.median(rho):.2f} (IQR {np.percentile(rho,25):.2f} to {np.percentile(rho,75):.2f}), min {rho.min():.2f}")
imp=np.array([r['imp'] for r in rows]); edge=np.array([r['edge'] for r in rows]); mid=np.array([r['mid'] for r in rows])
print(f"\nMEAN IMPORTANCE per seizure (all windows, all seconds): median {np.median(imp):.2f}")
print(f"  edge seconds (0 and 9) median {np.median(edge):.2f} vs middle seconds (1-8) median {np.median(mid):.2f} | edge < middle in {(edge<mid).sum()}/{n} seizures | Wilcoxon p = {wilcoxon(edge, mid).pvalue:.4f}")
print(f"  detected median {np.median(imp[hit]):.2f} | missed median {np.median(imp[~hit]):.2f} | Mann-Whitney p = {mannwhitneyu(imp[hit], imp[~hit]).pvalue:.3f}")
print("\nDETECTED vs MISSED (Mann-Whitney, two-sided)")
for name, v in [("top-3 drop", top), ("top - random", top - rnd), ("stability rho", rho)]:
    a, b = v[hit], v[~hit]
    print(f"  {name:14s} detected median {np.median(a):.3f} | missed median {np.median(b):.3f} | p = {mannwhitneyu(a, b).pvalue:.3f}")
