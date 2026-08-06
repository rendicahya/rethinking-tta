"""Classic augmentation-averaging TTA methods (no gradient-based adaptation)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseTTA


class BaselineTTA(BaseTTA):
    """No augmentation: a single plain forward pass. The cost/accuracy reference point."""

    name = "baseline"

    @torch.no_grad()
    def predict(self, batch):
        images, _ = batch
        return self.model(images)


class FlipTTA(BaseTTA):
    """Averages softmax predictions over the original image and its horizontal flip."""

    name = "flip"

    @torch.no_grad()
    def predict(self, batch):
        images, _ = batch
        views = [images, torch.flip(images, dims=[3])]
        probs = [F.softmax(self.model(v), dim=1) for v in views]
        return torch.stack(probs).mean(0)

    @property
    def n_forward_passes(self) -> int:
        return 2


class MultiCropTTA(BaseTTA):
    """Averages softmax predictions over `n_views` random-crop (+ optional flip) views."""

    name = "multicrop"

    def __init__(self, model, n_views: int = 10, crop_size: int = 28, **kwargs):
        super().__init__(model)
        self.n_views = n_views
        self.crop_size = crop_size

    @torch.no_grad()
    def predict(self, batch):
        images, _ = batch
        _, _, h, w = images.shape
        probs = []
        for _ in range(self.n_views):
            top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
            left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
            crop = images[:, :, top : top + self.crop_size, left : left + self.crop_size]
            crop = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)
            if torch.rand(1).item() < 0.5:
                crop = torch.flip(crop, dims=[3])
            probs.append(F.softmax(self.model(crop), dim=1))
        return torch.stack(probs).mean(0)

    @property
    def n_forward_passes(self) -> int:
        return self.n_views
