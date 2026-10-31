"""Extract frozen EEG embeddings for several models in ONE pass over the EDF files.

usage: python scripts/extract_embeddings.py --models biot labram chb01 chb03 ...
Writes cache/<patient>_<model>.npz with t (window end time on the patient timeline) and e (embeddings).
Windows are the same 10 s windows used for CBraMod, so every model is scored on identical time points.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtds import cbramod_feat as cf, chb
from rtds.embedders import EMBEDDERS

ap = argparse.ArgumentParser()
ap.add_argument("patients", nargs="+")
ap.add_argument("--models", nargs="+", required=True, choices=sorted(EMBEDDERS))
ap.add_argument("--data", default="dataset")
ap.add_argument("--cache", default="cache")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

cache = Path(a.cache)
fns = {m: EMBEDDERS[m](a.device) for m in a.models}
for pat in a.patients:
    summary = Path(a.data) / pat / f"{pat}-summary.txt"
    if not chb.has_times(summary):
        print(f"{pat}: summary has no recording times, skipped", flush=True)
        continue
    todo = [m for m in a.models if not (cache / f"{pat}_{m}.npz").exists()]
    if not todo:
        continue
    T, E = [], {m: [] for m in todo}
    for r in chb.parse_summary(summary):
        p = Path(a.data) / pat / r.name
        w = cf.read_edf_windows(p) if p.exists() else None
        if w is None or len(w) == 0:
            continue
        T.append(r.start + cf.WIN_S * (np.arange(len(w)) + 1))
        for m in todo:
            E[m].append(fns[m](w))
    t = np.concatenate(T)
    for m in todo:
        np.savez(cache / f"{pat}_{m}.npz", t=t, e=np.concatenate(E[m]))
    print(f"{pat}: {len(t)} windows -> {todo}", flush=True)
