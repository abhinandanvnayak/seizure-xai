"""Nested-validation tuning for frozen-embedding seizure prediction (leakage-free).

For each held-out patient P, the classifier strength C, the causal smoothing span and the alarm rule
('static' threshold vs 'rolling' percentile) are chosen ONLY by running the full protocol on the other
evaluable patients Q, with classifiers trained on everyone except {P, Q}. P's data (train or test period)
never influences the choice. The chosen configuration is then evaluated once on P.

usage: python scripts/tune_frozen.py --model cbramod
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtds import frozen as fz

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="cbramod")
ap.add_argument("--cache", default="cache")
ap.add_argument("--data", default="dataset")
ap.add_argument("--neg_per_pos", type=int, default=20)
ap.add_argument("--jobs", type=int, default=8)
a = ap.parse_args()

CS, SMOOTH, RULES = [0.001, 0.01, 0.1], [60, 160, 320], ["static", "rolling"]
BASELINE = (0.01, 160, "static")           # the configuration used in the original run
GRID = list(itertools.product(CS, SMOOTH, RULES))

names = [f"chb{i:02d}" for i in range(1, 25)
         if (Path(a.cache) / f"chb{i:02d}_{a.model}.npz").exists() and (Path(a.cache) / f"chb{i:02d}_x.npz").exists()]
data = {n: fz.Patient(n, a.model, a.cache, a.data) for n in names}
ev = [n for n in names if data[n].evaluable]
pool = [n for n in names if data[n].pos.any()]
mats = {n: tuple(m.astype(np.float32) if m.ndim == 2 else m for m in fz.training_matrix(data[n], a.neg_per_pos))
        for n in pool}
print(f"{a.model}: {len(names)} patients loaded, {len(ev)} evaluable, {len(pool)} in the training pool", flush=True)


def fit_one(excl, C):
    return (frozenset(excl), C), fz.fit(mats, [n for n in pool if n not in excl], C)


jobs = [((p, q), C) for p, q in itertools.combinations(ev, 2) for C in CS] + [((p,), C) for p in ev for C in CS]
print(f"fitting {len(jobs)} classifiers ...", flush=True)
models = dict(Parallel(n_jobs=a.jobs)(delayed(fit_one)(e, C) for e, C in jobs))


def process(P):
    inner = {g: [] for g in GRID}
    for Q in ev:
        if Q == P:
            continue
        for C in CS:
            raw = fz.raw_scores(models[(frozenset((P, Q)), C)], data[Q])
            for sm, rule in itertools.product(SMOOTH, RULES):
                inner[(C, sm, rule)].append(fz.evaluate(data[Q], raw, sm, rule))
    util = {g: fz.excess(r) for g, r in inner.items()}
    best = max(GRID, key=lambda g: (round(util[g], 6), g == BASELINE))      # ties -> baseline config
    out = {}
    for label, g in (("tuned", best), ("baseline", BASELINE)):
        raw = fz.raw_scores(models[(frozenset((P,)), g[0])], data[P])
        out[label] = fz.evaluate(data[P], raw, g[1], g[2])
        out[label]["config"] = dict(C=g[0], smooth_s=g[1], rule=g[2])
    out["inner_utility"] = {f"{c}|{s}|{r}": u for (c, s, r), u in util.items()}
    return P, out


res = dict(Parallel(n_jobs=a.jobs)(delayed(process)(P) for P in ev))
for label in ("baseline", "tuned"):
    rs = [res[p][label] for p in ev]
    print(fz.pooled_line(f"{a.model} {label}", rs))
    print(fz.pooled_line(f"{a.model} {label} (>=1h interictal)", [r for r in rs if r["interictal_hours"] >= 1]))
print("chosen configs:", {p: tuple(res[p]["tuned"]["config"].values()) for p in ev})
Path(f"results_{a.model}_tuned.json").write_text(json.dumps(res, indent=1, default=float))
