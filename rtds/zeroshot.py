"""Zero-shot TimesFM anomaly score: how badly does the forecast miss the next horizon?"""
import numpy as np

from .config import ZeroShot


def load_timesfm(cfg: ZeroShot, device=None):
    """device: "cpu" (default; MPS gave no speedup for this model) | "mps". TimesFM only auto-selects cuda/cpu."""
    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    device = device or "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        model.model.device = torch.device(device)
        model.model.to(device)
    model.compile(timesfm.ForecastConfig(
        max_context=cfg.context, max_horizon=cfg.horizon, normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True,
        infer_is_positive=False, fix_quantile_crossing=True))
    return model


def score_series(x, cfg: ZeroShot = ZeroShot(), model=None):
    """Return (times, scores): score at time t uses x[t-context:t] to forecast x[t:t+horizon].

    Score = MAE of the median forecast, scaled by the context's median absolute deviation.
    Steps whose context or horizon touch NaN are skipped (NaN score).
    """
    model = model or load_timesfm(cfg)
    starts = np.arange(cfg.context, len(x) - cfg.horizon, cfg.stride)
    ok = np.array([not np.isnan(x[s - cfg.context:s + cfg.horizon]).any() for s in starts])
    scores = np.full(len(starts), np.nan, np.float32)
    todo = np.flatnonzero(ok)
    for b in range(0, len(todo), cfg.batch):
        ii = todo[b:b + cfg.batch]
        ctx = [x[starts[i] - cfg.context:starts[i]] for i in ii]
        point, _ = model.forecast(horizon=cfg.horizon, inputs=ctx)
        for k, i in enumerate(ii):
            s = starts[i]
            mad = np.median(np.abs(ctx[k] - np.median(ctx[k]))) + 1e-3
            scores[i] = np.abs(point[k] - x[s:s + cfg.horizon]).mean() / mad
    return starts.astype(np.float64), scores
