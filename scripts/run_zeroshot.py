"""Zero-shot TimesFM seizure prediction on CHB-MIT, per patient, chronological split.

usage: python scripts/run_zeroshot.py chb01 chb02 ...  [--cache cache]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtds import chb, protocol as pr, zeroshot
from rtds.config import Protocol, ZeroShot

ap = argparse.ArgumentParser()
ap.add_argument("patients", nargs="+")
ap.add_argument("--data", default="dataset")
ap.add_argument("--cache", default="cache")
ap.add_argument("--out", default="results_zeroshot.json")
a = ap.parse_args()
for _p in list(a.patients):
    if not chb.has_times(Path(a.data) / _p / f"{_p}-summary.txt"):
        print(f"{_p}: summary has no recording times, skipped")
        a.patients.remove(_p)

P, Z = Protocol(), ZeroShot()
cache = Path(a.cache)
cache.mkdir(exist_ok=True)
model, results = None, {}
for pat in a.patients:
    f = cache / f"{pat}_zs.npz"
    fx = cache / f"{pat}_x.npz"
    if fx.exists():
        d = np.load(fx)
        x, seizures = d["x"], [tuple(v) for v in d["sz"]]
    else:
        x, seizures = chb.build_patient_series(Path(a.data) / pat, Z.bin_s)
        np.savez(fx, x=x, sz=np.array(seizures).reshape(-1, 2))
    split = pr.chrono_split(seizures, P)
    if split is None:
        print(f"{pat}: <2 target seizures, skipped")
        continue
    if f.exists():
        d = np.load(f)
        times, scores = d["times"], d["scores"]
    else:
        model = model or zeroshot.load_timesfm(Z)
        times, scores = zeroshot.score_series(x, Z, model)
        np.savez(f, times=times, scores=scores)
    valid = ~np.isnan(scores)
    times, scores = times[valid], pr.smooth(scores[valid], P.smooth_steps)
    train_end, _, test_targets = split
    test_targets = pr.covered_targets(test_targets, x, P, Z.bin_s)
    if not test_targets:
        print(f"{pat}: no test seizure with data coverage, skipped")
        continue
    tr, te = times < train_end, times >= train_end
    train_inter = tr & pr.interictal_mask(times, seizures, P)
    thr = pr.calibrate(times, scores, train_inter, P.target_fah, P.sop)
    alarms = pr.raise_alarms(times[te], scores[te], thr, P.sop)
    res = pr.evaluate(alarms, times[te], seizures, test_targets, P)
    rate = len(alarms) / max(1e-9, (times[te][-1] - times[te][0]) / 3600)
    res.update(pr.poisson_baseline(rate, res["n_seizures"], res["hits"], P), threshold=float(thr))
    results[pat] = res
    print(pat, json.dumps(res, default=float))

def pooled(rs, label):
    if not rs:
        return
    h = sum(r["hits"] for r in rs); n = sum(r["n_seizures"] for r in rs)
    fa = sum(r["false_alarms"] for r in rs); hrs = sum(r["interictal_hours"] for r in rs)
    print(f"POOLED {label}: patients {len(rs)}  sens {h}/{n}  FAH {fa / max(hrs, 1e-9):.3f} over {hrs:.1f} h")


pooled(list(results.values()), "all")
pooled([r for r in results.values() if r["interictal_hours"] >= 1], "interictal>=1h")
Path(a.out).write_text(json.dumps(results, indent=1, default=float))
