import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import SmallCNN

FS, WINDOW_S, SEG_S = 256, 10, 1          # sampling rate, window length, segment length
ONSET = 1467                               # seizure start in chb01_04 (seconds)

d = np.load("data/processed/windows_test.npz")
X, y, source, t0 = d["X"], d["y"], d["source"], d["t0"]

model = SmallCNN()
model.load_state_dict(torch.load("results/small_cnn.pt"))
model.eval()


def prob(x):
    with torch.no_grad():
        return torch.sigmoid(model(x))


# Pre-seizure windows of chb01_04, sorted in time
sel = np.where((source == "chb01_04") & (y == 1))[0]
sel = sel[np.argsort(t0[sel])]
x = torch.tensor(X[sel])
p0 = prob(x)
print("Mean P(seizure soon) on these windows:", round(p0.mean().item(), 3))

seg = SEG_S * FS
n_seg = WINDOW_S // SEG_S
drops = np.zeros((len(sel), n_seg))
for j in range(n_seg):
    xm = x.clone()
    xm[:, :, j * seg:(j + 1) * seg] = 0.0   # occlude one segment (= channel mean)
    drops[:, j] = (p0 - prob(xm)).numpy()    # drop in P(seizure soon)

print("Mean drop per 1-second segment:", np.round(drops.mean(axis=0), 3))

time_to_onset = ONSET - (t0[sel] + WINDOW_S)
fig, ax = plt.subplots(figsize=(7, 5))
vmax = np.abs(drops).max()
im = ax.imshow(drops, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xlabel("Second inside the 10-s window")
ax.set_ylabel("Seconds before seizure onset (window end)")
ax.set_yticks(range(0, len(sel), 5))
ax.set_yticklabels(time_to_onset[::5])
fig.colorbar(im, label="Drop in P(seizure soon) when the segment is removed")
fig.tight_layout()
fig.savefig("results/temporal_occlusion.png", dpi=200)
print("Saved results/temporal_occlusion.png")