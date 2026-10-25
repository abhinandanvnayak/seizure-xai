"""Leakage-free prediction protocol: chronological split, alarms, event-level metrics, Poisson baseline."""
import numpy as np
from scipy.stats import binom

from .config import Protocol


def target_seizures(seizures, p: Protocol):
    """First seizure of each cluster; later ones are not prediction targets."""
    out, last_end = [], -np.inf
    for on, off in seizures:
        if on - last_end >= p.cluster_gap:
            out.append((on, off))
        last_end = off
    return out


def chrono_split(seizures, p: Protocol):
    """Return (train_end_time, train_targets, test_targets). Train period = everything before
    the test period starts (after the last train seizure's postictal buffer)."""
    tg = target_seizures(seizures, p)
    n_train = max(1, int(np.ceil(len(tg) * p.train_frac)))
    if len(tg) - n_train < 1:
        return None
    train_end = tg[n_train][0] - p.sph - p.sop   # nothing from the first test preictal leaks into train
    return train_end, tg[:n_train], tg[n_train:]


def interictal_mask(times, seizures, p: Protocol):
    """True where time is >postictal after every seizure end and not within SPH+SOP before an onset."""
    m = np.ones(len(times), bool)
    for on, off in seizures:
        m &= ~((times >= on - p.sph - p.sop) & (times <= off + p.postictal))
    return m


def smooth(scores, k):
    """Causal moving average: the value at step t uses only steps <= t (no future leakage).
    NaNs are forward-filled; leading NaNs are dropped to the first valid value."""
    s = np.asarray(scores, float).copy()
    valid = ~np.isnan(s)
    if not valid.any():
        return s
    idx = np.where(valid, np.arange(len(s)), 0)
    np.maximum.accumulate(idx, out=idx)
    s = s[idx]
    s[: np.argmax(valid)] = s[np.argmax(valid)]
    if k <= 1:
        return s
    c = np.cumsum(np.insert(s, 0, 0.0))
    n = np.minimum(np.arange(1, len(s) + 1), k)
    return (c[1:] - c[np.arange(len(s)) + 1 - n]) / n


def calibrate(times, scores, train_mask, target_fah, refractory):
    """Pick a threshold such that train interictal data yields ~target_fah alarms/hour."""
    t, s = times[train_mask], scores[train_mask]
    hours = (t[-1] - t[0]) / 3600 if len(t) > 1 else 0
    if hours <= 0:
        return np.inf
    for q in np.linspace(0.999, 0.5, 500):
        thr = np.quantile(s, q)
        if len(raise_alarms(t, s, thr, refractory)) / hours >= target_fah:
            return thr
    return np.quantile(s, 0.5)


def raise_alarms(times, scores, thr, refractory):
    out, last = [], -np.inf
    for t, s in zip(times, scores):
        if s >= thr and t - last >= refractory:
            out.append(t)
            last = t
    return np.array(out)


def evaluate(alarms, test_times, test_seizures_all, test_targets, p: Protocol):
    """Event-level sensitivity (alarm in [onset-SPH-SOP, onset-SPH]) and false alarms/hour
    over interictal test time. Alarms inside any SPH window are ignored (neither hit nor FA)."""
    hits = 0
    for on, _ in test_targets:
        hits += bool(((alarms >= on - p.sph - p.sop) & (alarms <= on - p.sph)).any())
    inter = interictal_mask(test_times, test_seizures_all, p)
    in_sph = np.zeros(len(alarms), bool)
    for on, _ in test_seizures_all:
        in_sph |= (alarms > on - p.sph) & (alarms <= on)
    step = np.median(np.diff(test_times)) if len(test_times) > 1 else 0
    inter_hours = inter.sum() * step / 3600
    fa = [a for a, sp in zip(alarms, in_sph)
          if not sp and np.interp(a, test_times, inter.astype(float)) > 0.5]
    return dict(n_seizures=len(test_targets), hits=hits, sensitivity=hits / max(1, len(test_targets)),
                false_alarms=len(fa), interictal_hours=inter_hours,
                fah=len(fa) / inter_hours if inter_hours > 0 else np.nan)


def poisson_baseline(rate_per_hour, n_seizures, hits, p: Protocol):
    """Random predictor issuing alarms as a Poisson process at the same rate.
    P(an SOP window contains >=1 alarm) = 1-exp(-rate*SOP); p-value = P(Binomial >= hits)."""
    q = 1 - np.exp(-rate_per_hour * p.sop / 3600)
    return dict(random_sensitivity=q, p_value=float(binom.sf(hits - 1, n_seizures, q)) if n_seizures else np.nan)


def covered_targets(targets, x, p: Protocol, bin_s=1, min_frac=0.5):
    """Keep target seizures whose pre-ictal window [onset-SPH-SOP, onset-SPH] has >= min_frac of its raw
    feature bins present (x is the NaN-gapped per-bin series). Method-independent, so every model is
    evaluated on the same seizures; seizures in unreadable/missing recordings are not counted as misses."""
    keep = []
    for on, off in targets:
        a, b = int((on - p.sph - p.sop) // bin_s), int((on - p.sph) // bin_s)
        seg = x[max(a, 0):max(b, 0)]
        if len(seg) and np.isfinite(seg).sum() * bin_s >= min_frac * p.sop:
            keep.append((on, off))
    return keep


def seizure_outcomes(alarms, test_targets, p: Protocol):
    """Per test seizure: onset/end, whether an alarm fell in [onset-SPH-SOP, onset-SPH], and the first such alarm time."""
    out = []
    for on, off in test_targets:
        a = alarms[(alarms >= on - p.sph - p.sop) & (alarms <= on - p.sph)]
        out.append(dict(onset=float(on), end=float(off), detected=bool(len(a)),
                        first_alarm=float(a[0]) if len(a) else None))
    return out
