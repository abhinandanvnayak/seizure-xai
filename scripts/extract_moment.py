"""Frozen MOMENT-1-small embeddings of the per-second line-length series, on the TimesFM step grid.
usage: python scripts/extract_moment.py chb01 chb03 ...   -> cache/<patient>_moment.npz (t, e)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtds.seriesmodels import moment_embeddings

for pat in sys.argv[1:]:
    fx, out = Path("cache") / f"{pat}_x.npz", Path("cache") / f"{pat}_moment.npz"
    if out.exists() or not fx.exists():
        continue
    t, e = moment_embeddings(np.load(fx)["x"])
    np.savez(out, t=t, e=e)
    print(f"{pat}: {len(t)} windows", flush=True)
