"""Nested-validation tuning of the zero-shot score post-processing (smoothing span, alarm rule).

No weights are trained. For held-out patient P the smoothing span and the alarm rule are chosen only by running
the full protocol on the other evaluable patients (never P), then applied once to P.

usage: python scripts/tune_zeroshot.py --model zs      (reads cache/<patient>_zs.npz)
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtds import frozen as fz

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="zs")           # suffix of the score cache; zs = TimesFM
ap.add_argument("--cache", default="cache")
ap.add_argument("--data", default="dataset")
a = ap.parse_args()

SMOOTH, RULES = [96, 160, 320, 640], ["static", "rolling"]
BASELINE = (160, "static")
GRID = list(itertools.product(SMOOTH, RULES))
names = [f"chb{i:02d}" for i in range(1, 25)
         if (Path(a.cache) / f"chb{i:02d}_{a.model}.npz").exists() and (Path(a.cache) / f"chb{i:02d}_x.npz").exists()]
data = {n: fz.Patient(n, a.model, a.cache, a.data) for n in names}
ev = [n for n in names if data[n].evaluable]
print(f"{a.model}: {len(ev)} evaluable patients", flush=True)

res = {}
for P in ev:
    inner = {g: [fz.evaluate(data[Q], data[Q].raw, *g) for Q in ev if Q != P] for g in GRID}
    util = {g: fz.excess(r) for g, r in inner.items()}
    best = max(GRID, key=lambda g: (round(util[g], 6), g == BASELINE))
    res[P] = {}
    for label, g in (("tuned", best), ("baseline", BASELINE)):
        res[P][label] = fz.evaluate(data[P], data[P].raw, *g)
        res[P][label]["config"] = dict(smooth_s=g[0], rule=g[1])
    res[P]["inner_utility"] = {f"{s}|{r}": u for (s, r), u in util.items()}
    print(P, "chose", best, flush=True)

for label in ("baseline", "tuned"):
    rs = [res[p][label] for p in ev]
    print(fz.pooled_line(f"{a.model} {label}", rs))
    print(fz.pooled_line(f"{a.model} {label} (>=1h interictal)", [r for r in rs if r["interictal_hours"] >= 1]))
Path(f"results_{a.model}_tuned.json").write_text(json.dumps(res, indent=1, default=float))
