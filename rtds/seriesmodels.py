"""General-purpose time-series foundation models applied to the per-second line-length series x.

  * chronos_forecaster(): zero-shot forecaster for rtds.zeroshot.score_series (same scoring as TimesFM)
  * moment_embeddings(): frozen MOMENT embeddings of the trailing 512 s of x, on the TimesFM step grid
"""
import numpy as np
import torch


def chronos_forecaster(name="amazon/chronos-bolt-small"):
    """Return fn(list_of_context_arrays, horizon) -> (n, horizon) median forecasts."""
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)

    def fn(contexts, horizon):
        q, _ = pipe.predict_quantiles(inputs=[torch.tensor(c, dtype=torch.float32) for c in contexts],
                                      prediction_length=horizon, quantile_levels=[0.5])
        return q[..., 0].numpy()
    return fn


def moment_embeddings(x, stride=32, length=512, batch=64, name="AutonLab/MOMENT-1-small"):
    """Embeddings for windows x[s-length:s], s = length, length+stride, ... (NaN windows skipped).
    Returns (times s, embeddings)."""
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(name, model_kwargs={"task_name": "embedding"})
    model.init()
    model.eval()
    starts = np.arange(length, len(x), stride)
    ok = np.array([not np.isnan(x[s - length:s]).any() for s in starts])
    starts = starts[ok]
    out = []
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            b = np.stack([x[s - length:s] for s in starts[i:i + batch]])[:, None, :].astype(np.float32)
            out.append(model(x_enc=torch.from_numpy(b)).embeddings.numpy())
    return starts.astype(np.float64), np.concatenate(out)
