"""Frozen-embedding seizure prediction: leave-one-patient-out classifier + chronological test period.

Per held-out patient P:
  * classifier is fit ONLY on the other patients' windows (patient-disjoint);
  * P's embeddings are z-scored with statistics from P's own train period only;
  * P's alarm threshold is calibrated on P's train-period interictal scores only (no labels);
  * P is evaluated on its later test period with causal smoothing, SPH/SOP event metrics, Poisson baseline.

usage: python scripts/run_frozen.py --model cbramod chb01 chb02 ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtds import chb, protocol as pr
from rtds.config import Protocol

ap = argparse.ArgumentParser()
ap.add_argument("patients", nargs="+")
ap.add_argument("--model", default="cbramod")
ap.add_argument("--data", default="dataset")
ap.add_argument("--cache", default="cache")
ap.add_argument("--device", default="cpu")
ap.add_argument("--smooth_s", type=int, default=160, help="causal smoothing span in seconds")
ap.add_argument("--neg_per_pos", type=int, default=20)
ap.add_argument("--out", default=None)
a = ap.parse_args()
for _p in list(a.patients):
    if not chb.has_times(Path(a.data) / _p / f"{_p}-summary.txt"):
        print(f"{_p}: summary has no recording times, skipped")
        a.patients.remove(_p)
P = Protocol()
cache = Path(a.cache)
cache.mkdir(exist_ok=True)


def embeddings(pat):
    f = cache / f"{pat}_{a.model}.npz"
    if f.exists():
        d = np.load(f)
        return d["t"], d["e"]
    if a.model == "cbramod":
        from rtds import cbramod_feat as cf
        t, e = cf.patient_embeddings(Path(a.data) / pat, embeddings.model, a.device)
    else:
        raise SystemExit(f"unknown model {a.model}")
    np.savez(f, t=t, e=e)
    return t, e


if a.model == "cbramod":
    from rtds import cbramod_feat as cf
    embeddings.model = cf.load_cbramod(a.device)

data = {}
for pat in a.patients:
    t, e = embeddings(pat)
    sz = sorted(s for r in chb.parse_summary(Path(a.data) / pat / f"{pat}-summary.txt") for s in r.seizures)
    fx = cache / f"{pat}_x.npz"        # raw feature series written by run_zeroshot.py
    if fx.exists():
        xs = np.load(fx)["x"]
    else:
        xs, _ = chb.build_patient_series(Path(a.data) / pat, 1)
        np.savez(fx, x=xs, sz=np.array(sz).reshape(-1, 2))
    data[pat] = dict(x=xs, t=t, e=e, sz=sz, split=pr.chrono_split(sz, P), targets=pr.target_seizures(sz, P))


def labels(t, d):
    pos = np.zeros(len(t), bool)
    for on, _ in d["targets"]:
        pos |= (t >= on - P.sph - P.sop) & (t <= on - P.sph)
    neg = pr.interictal_mask(t, d["sz"], P)
    return pos, neg & ~pos


rng = np.random.default_rng(0)
results = {}
for test in a.patients:
    d = data[test]
    if d["split"] is None:
        print(f"{test}: <2 target seizures, skipped")
        continue
    train_end, _, test_targets = d["split"]
    test_targets = pr.covered_targets(test_targets, d["x"], P, 1)
    if not test_targets:
        print(f"{test}: no test seizure with data coverage, skipped")
        continue
    # --- patient-disjoint training set from the other patients
    X, y = [], []
    for pat, o in data.items():
        if pat == test:
            continue
        pos, neg = labels(o["t"], o)
        ni = rng.choice(np.flatnonzero(neg), size=min(neg.sum(), max(1, pos.sum() * a.neg_per_pos)), replace=False)
        idx = np.concatenate([np.flatnonzero(pos), ni])
        mu, sd = o["e"][neg].mean(0), o["e"][neg].std(0) + 1e-6      # other patients' own normalisation
        X.append((o["e"][idx] - mu) / sd)
        y.append(np.r_[np.ones(pos.sum()), np.zeros(len(ni))])
    X, y = np.concatenate(X), np.concatenate(y)
    clf = LogisticRegression(C=0.01, class_weight="balanced", max_iter=500).fit(X, y)
    # --- held-out patient: normalise with its train period only, score everything causally
    t, e = d["t"], d["e"]
    tr = t < train_end
    tr_inter = tr & pr.interictal_mask(t, d["sz"], P)
    mu, sd = e[tr_inter].mean(0), e[tr_inter].std(0) + 1e-6
    s = pr.smooth(clf.decision_function((e - mu) / sd), max(1, a.smooth_s // 10))
    thr = pr.calibrate(t, s, tr_inter, P.target_fah, P.sop)
    te = t >= train_end
    alarms = pr.raise_alarms(t[te], s[te], thr, P.sop)
    res = pr.evaluate(alarms, t[te], d["sz"], test_targets, P)
    rate = len(alarms) / max(1e-9, (t[te][-1] - t[te][0]) / 3600)
    res.update(pr.poisson_baseline(rate, res["n_seizures"], res["hits"], P), threshold=float(thr))
    results[test] = res
    print(test, json.dumps(res, default=float), flush=True)

def pooled(rs, label):
    if not rs:
        return
    h = sum(r["hits"] for r in rs); n = sum(r["n_seizures"] for r in rs)
    fa = sum(r["false_alarms"] for r in rs); hrs = sum(r["interictal_hours"] for r in rs)
    print(f"POOLED {a.model} {label}: patients {len(rs)}  sens {h}/{n}  FAH {fa / max(hrs, 1e-9):.3f} over {hrs:.1f} h")


pooled(list(results.values()), "all")
pooled([r for r in results.values() if r["interictal_hours"] >= 1], "interictal>=1h")
Path(a.out or f"results_{a.model}.json").write_text(json.dumps(results, indent=1, default=float))
