from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    """Prediction protocol. All times in seconds."""
    sph: float = 5 * 60            # seizure prediction horizon (intervention time)
    sop: float = 30 * 60           # seizure occurrence period (alarm validity window)
    postictal: float = 2 * 3600    # excluded after each seizure from interictal time
    cluster_gap: float = 4 * 3600  # seizures closer than this to the previous one are not prediction targets
    train_frac: float = 0.5        # fraction of lead seizures used for the chronological train period
    target_fah: float = 0.5        # false alarms/hour used to calibrate the threshold on train data
    smooth_steps: int = 5          # moving-average over score steps before thresholding


@dataclass(frozen=True)
class ZeroShot:
    bin_s: int = 1        # feature bin (line length per second)
    context: int = 256    # context bins fed to TimesFM (multiple of 32)
    horizon: int = 32     # forecast bins scored per step
    stride: int = 32      # seconds between scoring steps
    batch: int = 64


CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ"]
