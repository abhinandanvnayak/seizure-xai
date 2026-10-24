"""CHB-MIT loading: summary parsing, seizure timeline, per-second line-length features."""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import CHANNELS


@dataclass
class Rec:
    name: str
    start: float          # seconds since the patient's first recording started
    end: float
    seizures: list        # [(onset, end)] in patient timeline seconds


def _hms(s):
    h, m, sec = map(int, s.split(":"))
    return h * 3600 + m * 60 + sec


def has_times(path):
    """chb24's summary has no file start/end times, so its recordings cannot be placed on a timeline."""
    return "File Start Time" in Path(path).read_text()


def parse_summary(path):
    """Return recordings on a continuous patient timeline (handles midnight rollover)."""
    text = Path(path).read_text()
    recs, day, prev, t0 = [], 0, None, None
    for block in re.split(r"\n\s*\n(?=File Name:)", text):
        m = re.search(r"File Name:\s*(\S+)", block)
        if not m:
            continue
        s = _hms(re.search(r"File Start Time:\s*(\S+)", block).group(1))
        e = _hms(re.search(r"File End Time:\s*(\S+)", block).group(1))
        if prev is not None and s < prev:
            day += 1
        prev = s
        start = s + day * 86400
        if e < s:
            day += 1
        end = e + day * 86400
        prev = e
        t0 = start if t0 is None else t0
        sz = []
        for a, b in zip(re.findall(r"Seizure(?: \d+)? Start Time:\s*(\d+)", block),
                        re.findall(r"Seizure(?: \d+)? End Time:\s*(\d+)", block)):
            sz.append((start - t0 + int(a), start - t0 + int(b)))
        recs.append(Rec(m.group(1), start - t0, end - t0, sz))
    return recs


def line_length_series(edf_path, fs_bin=1, bandpass=(0.5, 40.0)):
    """Mean over channels of log line length per bin for one EDF. Returns (series, fs)."""
    import pyedflib
    with pyedflib.EdfReader(str(edf_path)) as f:
        labels = f.getSignalLabels()
        idx, seen = [], set()
        for i, l in enumerate(labels):
            l = l.strip().upper()
            if l in CHANNELS and l not in seen:
                idx.append(i)
                seen.add(l)
        if len(idx) < 12:        # e.g. chb12_27-29 use referential labels; treat as a gap
            return None
        fs = int(round(f.getSampleFrequency(idx[0])))
        sig = np.stack([f.readSignal(i) for i in idx])
    sos = butter(4, bandpass, btype="band", fs=fs, output="sos")
    sig = sosfiltfilt(sos, sig, axis=1)
    n = sig.shape[1] // (fs * fs_bin)
    d = np.abs(np.diff(sig, axis=1, prepend=sig[:, :1]))[:, : n * fs * fs_bin]
    ll = d.reshape(len(idx), n, fs * fs_bin).sum(-1)
    return np.log1p(ll).mean(0).astype(np.float32)


def build_patient_series(patient_dir, bin_s=1):
    """Patient-timeline feature series (NaN in gaps / missing files) and seizure list."""
    patient_dir = Path(patient_dir)
    recs = parse_summary(patient_dir / f"{patient_dir.name}-summary.txt")
    total = int(max(r.end for r in recs) // bin_s) + 1
    x = np.full(total, np.nan, np.float32)
    for r in recs:
        p = patient_dir / r.name
        if not p.exists():
            continue
        s = line_length_series(p, bin_s)
        if s is None:
            continue
        a = int(r.start // bin_s)
        x[a:a + len(s)] = s[: total - a]
    seizures = sorted(sz for r in recs for sz in r.seizures)
    return x, seizures
