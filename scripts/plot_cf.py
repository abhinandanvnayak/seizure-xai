import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
OUT = r"C:\seizure-xai\results"
import sys
if len(sys.argv) > 1: OUT = sys.argv[1]
d = pd.read_csv(f"{OUT}/counterfactual_cbramod.csv")
d["valid"] = d.eps_sel.notna(); d["pct"] = d.eps_window.fillna(1.0) * 100
g = d.groupby(["patient", "onset", "k", "arm"]).agg(valid=("valid", "mean"), pct=("pct", "median")).reset_index()
n = g[["patient", "onset"]].drop_duplicates().shape[0]
NAVY, GREY, ORANGE = "#0B3D91", "#9AA3B5", "#D9822B"
arms = [("occlusion-top", "Most important (occlusion)", NAVY), ("random", "Random", GREY), ("occlusion-bottom", "Least important (occlusion)", ORANGE)]
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.15]})
ks = sorted(g.k.unique()); wd = 0.26
for i, (arm, lab, c) in enumerate(arms):
    ax[0].bar(np.arange(len(ks)) + (i - 1) * wd, [g[(g.k == k) & (g.arm == arm)].valid.mean() * 100 for k in ks], wd, color=c, label=lab)
ax[0].set_xticks(range(len(ks))); ax[0].set_xticklabels([f"{k} s" for k in ks])
ax[0].set_xlabel("Number of modified seconds"); ax[0].set_ylabel("Missed seizures turned into alarms (%)")
ax[0].set_ylim(0, 100); ax[0].set_title("Validity")
w = 0.22; rng = np.random.default_rng(0)
for j, k in enumerate(ks):
    for i, (arm, lab, c) in enumerate(arms):
        e = g[(g.k == k) & (g.arm == arm)].pct.values; x = j + (i - 1) * w
        ax[1].boxplot(e, positions=[x], widths=w * 0.8, patch_artist=True, boxprops=dict(facecolor="white", edgecolor=c, lw=2),
                      medianprops=dict(color=c, lw=2.5), whiskerprops=dict(color=c), capprops=dict(color=c), showfliers=False)
        ax[1].scatter(x + rng.uniform(-.05, .05, len(e)), e, color=c, s=22, zorder=3)
ax[1].set_xticks(range(len(ks))); ax[1].set_xticklabels([f"{k} s" for k in ks]); ax[1].set_xlim(-.5, len(ks) - .5)
ax[1].set_xlabel("Number of modified seconds"); ax[1].set_ylabel("Change needed (% of the whole window)")
ax[1].set_title("Proximity (smaller = closer)")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=14)
fig.suptitle(f"Counterfactuals for {n} missed seizures (CBraMod + logistic regression)", fontweight="bold", color=NAVY)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{OUT}/counterfactual_cbramod_v2.png", dpi=200)
for k in ks:
    t = g[(g.k == k) & (g.arm == "occlusion-top")].set_index(["patient", "onset"]).pct
    for o in ["random", "occlusion-bottom"]:
        v = g[(g.k == k) & (g.arm == o)].set_index(["patient", "onset"]).pct.reindex(t.index)
        print(f"k={k} {o}: median top {t.median():.2f} vs {v.median():.2f} | Wilcoxon (two-sided) p = {wilcoxon(t, v).pvalue:.3f}")
