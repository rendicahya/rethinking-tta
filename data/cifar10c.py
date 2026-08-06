"""CIFAR-10-C dataset: reads the per-corruption .npy files produced by
scripts/download_data.sh (Hendrycks & Dietterich, 2019)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

SEVERITIES = (1, 2, 3, 4, 5)
_N_PER_SEVERITY = 10_000


class CIFAR10C(Dataset):
    """A single (corruption, severity) split of CIFAR-10-C.

    In each `{corruption}.npy` file, the first 10,000 images are severity 1 and
    the last 10,000 are severity 5; `labels.npy` (10,000 labels) applies to every
    severity block.
    """

    def __init__(self, root: str | Path, corruption: str, severity: int, transform=None):
        if corruption not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption '{corruption}'. Available: {CORRUPTIONS}")
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")

        root = Path(root)
        images = np.load(root / f"{corruption}.npy")
        labels = np.load(root / "labels.npy")

        start = (severity - 1) * _N_PER_SEVERITY
        end = severity * _N_PER_SEVERITY
        self.images = images[start:end]
        self.labels = labels[start:end]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        label = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label
