"""Frozen CBraMod embeddings for CHB-MIT: 10 s windows, 16 bipolar channels, 200 Hz."""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

from .chb import parse_summary
from .config import CHANNELS

_TP = Path(__file__).resolve().parents[1] / "third_party" / "cbramod"
sys.path.insert(0, str(_TP))
from models.cbramod import CBraMod  # noqa: E402

CH16 = CHANNELS[:16]          # same 16 bipolar channels CBraMod's own CHB-MIT recipe uses
FS, WIN_S = 200, 10


def load_cbramod(device="cpu"):
    m = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    m.load_state_dict(torch.load(_TP / "pretrained_weights.pth", map_location="cpu"))
    return m.to(device).eval()


@torch.no_grad()
def embed(model, x, device="cpu", batch=32):
    """x: (n, 16, 10, 200) float32 (EEG in µV/100). Returns (n, 200): mean-pooled encoder output."""
    out = []
    for i in range(0, len(x), batch):
        b = torch.from_numpy(x[i:i + batch]).to(device)
        h = model.encoder(model.patch_embedding(b))
        out.append(h.mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def read_edf_windows(edf_path):
    """Return (n_windows, 16, 10, 200) float32 from one EDF, or None if channels are missing."""
    import pyedflib
    with pyedflib.EdfReader(str(edf_path)) as f:
        labels = [l.strip().upper() for l in f.getSignalLabels()]
        idx = []
        for c in CH16:
            if c not in labels:
                return None
            idx.append(labels.index(c))     # first occurrence (some files repeat T8-P8)
        fs = int(round(f.getSampleFrequency(idx[0])))
        sig = np.stack([f.readSignal(i) for i in idx])
    sos = butter(4, (0.5, 75.0), btype="band", fs=fs, output="sos")
    sig = resample_poly(sosfiltfilt(sos, sig, axis=1), FS, fs, axis=1)
    n = sig.shape[1] // (FS * WIN_S)
    sig = sig[:, : n * FS * WIN_S] / 100.0
    return sig.reshape(16, n, WIN_S, FS).transpose(1, 0, 2, 3).astype(np.float32)


def patient_embeddings(patient_dir, model, device="cpu"):
    """(end_times_s, embeddings) on the patient timeline for every available window."""
    patient_dir = Path(patient_dir)
    recs = parse_summary(patient_dir / f"{patient_dir.name}-summary.txt")
    T, E = [], []
    for r in recs:
        p = patient_dir / r.name
        if not p.exists():
            continue
        w = read_edf_windows(p)
        if w is None or len(w) == 0:
            continue
        E.append(embed(model, w, device))
        T.append(r.start + WIN_S * (np.arange(len(w)) + 1))
    return np.concatenate(T), np.concatenate(E)
