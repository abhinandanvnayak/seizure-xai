"""Poster figure from the finished per-seizure files (nothing is recomputed).
Run: python C:\\rtds-group\\scripts\\make_poster_figure.py
Writes results\\poster_2x2.png (3 panels + free quarter for text) and the 3 panels alone."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\seizure-xai\results")
RED, BLUE = "#B4472B", "#0B3D91"
plt.rcParams.update({"font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
                     "axes.spines.top": False, "axes.spines.right": False})

rows = []
for f in sorted((OUT / "occlusion").glob("*.npz")):
    d = np.load(f)
    rows.append(dict(hit=bool(d["hit"]), top=float(d["d_top"].mean()), rnd=float(d["d_rand"].mean()),
                     prof=d["drops"].mean(0), minutes=d["minutes"], wimp=d["drops"].mean(1)))
n = len(rows)
hits = np.array([r["hit"] for r in rows])
top, rnd = np.array([r["top"] for r in rows]), np.array([r["rnd"] for r in rows])
better = int((top > rnd).sum())
prof = np.array([r["prof"] for r in rows])
LAB = {True: f"detected (n={int(hits.sum())})", False: f"missed (n={int((~hits).sum())})"}


def p_faith(ax):
    for i in range(n):
        ax.plot([0, 1], [rnd[i], top[i]], color=RED if hits[i] else BLUE, alpha=.6, marker="o", ms=6, lw=2)
    ax.set_xlim(-.25, 1.25)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["3 random\nseconds", "top-3 ranked\nseconds"])
    ax.set_ylabel("Score drop when removed")
    ax.set_title(f"Faithfulness\ntop-3 > random in {better}/{n} seizures", fontsize=19)


def p_window(ax):
    for flag, col in [(True, RED), (False, BLUE)]:
        if (hits == flag).any():
            ax.plot(range(10), prof[hits == flag].mean(0), color=col, marker="o", ms=8, lw=3, label=LAB[flag])
    ax.set_xlabel("Second inside the 10-s window"); ax.set_ylabel("Mean importance")
    ax.set_xticks(range(0, 10, 2)); ax.set_title("Importance inside the window"); ax.legend(frameon=False)


def p_time(ax):
    bins = np.arange(5, 40, 5)
    for flag, col in [(True, RED), (False, BLUE)]:
        sel = [r for r, h in zip(rows, hits) if h == flag]
        if sel:
            mn = np.concatenate([r["minutes"] for r in sel]); wi = np.concatenate([r["wimp"] for r in sel])
            y = [wi[(mn >= b) & (mn < b + 5)].mean() if ((mn >= b) & (mn < b + 5)).any() else np.nan for b in bins[:-1]]
            ax.plot(bins[:-1] + 2.5, y, color=col, marker="o", ms=8, lw=3, label=LAB[flag])
    ax.invert_xaxis(); ax.set_xlabel("Minutes before seizure onset"); ax.set_ylabel("Mean importance")
    ax.set_title("Importance vs time to onset"); ax.legend(frameon=False)


panels = [("faithfulness", p_faith), ("window", p_window), ("time", p_time)]
fig, axs = plt.subplots(2, 2, figsize=(14, 11))
for ax, (_, fn) in zip(axs.flat, panels):
    fn(ax)
axs[1, 1].axis("off")           # free quarter for text
fig.tight_layout(h_pad=3, w_pad=3)
fig.savefig(OUT / "poster_2x2.png", dpi=200)
for name, fn in panels:
    f1, a1 = plt.subplots(figsize=(7, 5.5)); fn(a1); f1.tight_layout()
    f1.savefig(OUT / f"poster_panel_{name}.png", dpi=200); plt.close(f1)
print("Saved poster_2x2.png and 3 single panels in", OUT)
