"""Shared machinery for frozen-embedding seizure prediction (any embedding model).

Everything here is leakage-free by construction:
  * a classifier for held-out patient P is fit only on other patients;
  * P's embeddings are z-scored with P's own train-period interictal statistics only;
  * alarm thresholds are calibrated on P's train-period interictal scores only (no seizure labels);
  * smoothing and rolling statistics are causal.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import chb, protocol as pr
from .config import Protocol

P = Protocol()


class Patient:
    def __init__(self, name, model, cache="cache", data="dataset"):
        d = np.load(Path(cache) / f"{name}_{model}.npz")
        self.name = name
        if "scores" in d:           # zero-shot score file (rtds.zeroshot): one raw anomaly score per step
            ok = ~np.isnan(d["scores"])
            self.t, self.raw = d["times"][ok], d["scores"][ok]
            self.e = self.raw[:, None]
        else:
            self.t, self.e = d["t"], d["e"]
        self.step = float(np.median(np.diff(self.t)))
        self.x = np.load(Path(cache) / f"{name}_x.npz")["x"]
        self.sz = sorted(s for r in chb.parse_summary(Path(data) / name / f"{name}-summary.txt") for s in r.seizures)
        self.targets = pr.target_seizures(self.sz, P)
        self.split = pr.chrono_split(self.sz, P)
        self.test_targets = []
        if self.split is not None:
            self.train_end = self.split[0]
            self.test_targets = pr.covered_targets(self.split[2], self.x, P, 1)
        self.evaluable = bool(self.test_targets)
        self.pos = np.zeros(len(self.t), bool)
        for on, _ in self.targets:
            self.pos |= (self.t >= on - P.sph - P.sop) & (self.t <= on - P.sph)
        self.neg = pr.interictal_mask(self.t, self.sz, P) & ~self.pos
        self._tr_inter = None

    def train_inter(self):
        """Train-period interictal mask (only meaningful for evaluable patients)."""
        return (self.t < self.train_end) & pr.interictal_mask(self.t, self.sz, P)

    def own_norm(self):
        m = self.train_inter()
        return self.e[m].mean(0), self.e[m].std(0) + 1e-6


def training_matrix(p: Patient, neg_per_pos, seed=0):
    """Positives + a fixed random negative subsample of one patient, z-scored by its own negatives."""
    rng = np.random.default_rng(seed + int(p.name[3:]))
    neg = np.flatnonzero(p.neg)
    ni = rng.choice(neg, size=min(len(neg), max(1, int(p.pos.sum()) * neg_per_pos)), replace=False)
    idx = np.concatenate([np.flatnonzero(p.pos), ni])
    mu, sd = p.e[p.neg].mean(0), p.e[p.neg].std(0) + 1e-6
    return (p.e[idx] - mu) / sd, np.r_[np.ones(int(p.pos.sum())), np.zeros(len(ni))]


def fit(mats, names, C):
    X = np.concatenate([mats[n][0] for n in names])
    y = np.concatenate([mats[n][1] for n in names])
    return LogisticRegression(C=C, class_weight="balanced", max_iter=500).fit(X, y)


def raw_scores(clf, p: Patient):
    mu, sd = p.own_norm()
    return clf.decision_function((p.e - mu) / sd)


def rolling_percentile(s, w):
    """Causal percentile of s[t] within the previous w samples (excluding itself); needs >= w//4 history."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(s)
    win = sliding_window_view(np.concatenate([np.full(w, np.nan), s]), w)[:n]   # win[i] = s[i-w:i]
    out = np.zeros(n)
    hist = np.minimum(np.arange(n), w)
    for a in range(0, n, 2048):
        blk = win[a:a + 2048]
        out[a:a + len(blk)] = (blk <= s[a:a + len(blk), None]).sum(1) / np.maximum(hist[a:a + len(blk)], 1)
    out[hist < max(8, w // 4)] = 0.0
    return out


def evaluate(p: Patient, raw, smooth_s, rule, rolling_h=6.0):
    """rule: 'static' (train-calibrated score threshold) or 'rolling' (train-calibrated percentile of the
    trailing `rolling_h` hours, which tracks slow drift). Returns the standard result dict."""
    k = max(1, int(round(smooth_s / p.step)))
    s = pr.smooth(raw, k)
    tr = p.train_inter()
    te = p.t >= p.train_end
    if rule == "static":
        thr = pr.calibrate(p.t, s, tr, P.target_fah, P.sop)
        alarms = pr.raise_alarms(p.t[te], s[te], thr, P.sop)
    else:
        pct = rolling_percentile(s, int(rolling_h * 3600 / p.step))
        thr = pr.calibrate(p.t, pct, tr, P.target_fah, P.sop)
        alarms = pr.raise_alarms(p.t[te], pct[te], thr, P.sop)
    res = pr.evaluate(alarms, p.t[te], p.sz, p.test_targets, P)
    rate = len(alarms) / max(1e-9, (p.t[te][-1] - p.t[te][0]) / 3600)
    res.update(pr.poisson_baseline(rate, res["n_seizures"], res["hits"], P), threshold=float(thr),
               seizures=pr.seizure_outcomes(alarms, p.test_targets, P), alarms=[float(a) for a in alarms],
               train_end=float(p.train_end))
    return res


def excess(results):
    """Pooled (hits - hits expected from a matched-rate random predictor) per seizure."""
    n = sum(r["n_seizures"] for r in results)
    return (sum(r["hits"] - r["n_seizures"] * r["random_sensitivity"] for r in results) / n) if n else 0.0


def pooled_line(label, rs):
    h = sum(r["hits"] for r in rs); n = sum(r["n_seizures"] for r in rs)
    fa = sum(r["false_alarms"] for r in rs); hrs = sum(r["interictal_hours"] for r in rs)
    return f"{label}: patients {len(rs)}  sens {h}/{n}  FAH {fa / max(hrs, 1e-9):.3f} over {hrs:.1f} h  excess/seizure {excess(rs):+.3f}"
