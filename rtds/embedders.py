"""Frozen EEG embedders sharing one input format: windows (n, 16, 10, 200) float32, 16 bipolar channels in
rtds.config.CHANNELS[:16] order, 200 Hz, 10 s, band-passed, values = microvolts / 100.
Checkpoints are loaded with torch.load(weights_only=True) so no pickled code is executed."""
import sys
from pathlib import Path

import numpy as np
import torch

_TP = Path(__file__).resolve().parents[1] / "third_party"

# BIOT's 16-channel order -> indices into CHANNELS[:16]
_BIOT_ORDER = [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]
# LaBraM: electrode name used for each of our 16 bipolar channels (first electrode of the pair).
# NOTE: LaBraM expects potentials at electrodes; bipolar pairs are an approximation (FP1/FP2 repeat).
_LABRAM_ELECTRODES = ["FP1", "F7", "T7", "P7", "FP1", "F3", "C3", "P3",
                      "FP2", "F4", "C4", "P4", "FP2", "F8", "T8", "P8"]
_STD_1020 = (['FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
              'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
              'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
              'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
              'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
              'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10'])


def _cbramod(device):
    from . import cbramod_feat as cf
    m = cf.load_cbramod(device)
    return lambda w: cf.embed(m, w, device)


def _biot(device):
    sys.path.insert(0, str(_TP / "biot"))
    from biot import BIOTEncoder
    enc = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=16)
    enc.load_state_dict(torch.load(_TP / "biot" / "EEG-PREST-16-channels.ckpt", map_location="cpu",
                                   weights_only=True))
    enc = enc.to(device).eval()

    @torch.no_grad()
    def fn(w, batch=64):
        x = w.reshape(len(w), 16, -1)[:, _BIOT_ORDER]
        x = x / (np.quantile(np.abs(x), 0.95, axis=-1, keepdims=True) + 1e-8)   # BIOT's own per-channel scaling
        out = [enc(torch.from_numpy(x[i:i + batch].astype(np.float32)).to(device)).cpu().numpy()
               for i in range(0, len(x), batch)]
        return np.concatenate(out)
    return fn


def _labram(device):
    sys.path.insert(0, str(_TP / "labram"))
    from modeling_finetune import labram_base_patch200_200
    # Structure of the pretraining checkpoint: no q/v bias, layer-scale (gamma) present, final LayerNorm `norm`.
    model = labram_base_patch200_200(num_classes=0, qkv_bias=False, use_mean_pooling=False, init_values=0.1,
                                     use_rel_pos_bias=False, use_abs_pos_emb=True)
    # The checkpoint also stores training metadata (an argparse.Namespace and numpy scalars); allowlist exactly
    # those inert types instead of disabling weights_only.
    import argparse
    allow = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, argparse.Namespace,
             np.dtypes.Float32DType, np.dtypes.Float64DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allow):
        ck = torch.load(_TP / "labram" / "labram-base.pth", map_location="cpu", weights_only=True)
    sd = ck.get("model", ck.get("state_dict", ck))
    sd = {k[len("student."):]: v for k, v in sd.items()
          if k.startswith("student.") and not k.startswith("student.lm_head")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith(("head", "fc_norm"))]
    if bad:
        raise RuntimeError(f"LaBraM checkpoint did not match the model, missing: {bad[:5]}")
    print(f"LaBraM loaded: {len(sd) - len(unexpected)} tensors matched, ignored {sorted(unexpected)}", flush=True)
    model = model.to(device).eval()
    chans = [0] + [_STD_1020.index(c) + 1 for c in _LABRAM_ELECTRODES]

    @torch.no_grad()
    def fn(w, batch=32):
        # mean over all patch tokens (the final LayerNorm is applied inside forward_features)
        out = [model.forward_features(torch.from_numpy(w[i:i + batch]).to(device), input_chans=chans,
                                      return_patch_tokens=True).mean(1).cpu().numpy()
               for i in range(0, len(w), batch)]
        return np.concatenate(out)
    return fn


EMBEDDERS = {"cbramod": _cbramod, "biot": _biot, "labram": _labram}
