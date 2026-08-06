"""Test-time augmentation / adaptation (TTA) methods.

Every method wraps a trained model and, given a batch of (possibly corrupted)
images, returns predictions (logits or averaged probabilities). `n_forward_passes`
reports how many backbone forward passes are needed per sample, used downstream
to estimate compute cost.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTTA(ABC):
    """Base class every TTA method implements."""

    name: str = "base"

    def __init__(self, model: nn.Module, **kwargs):
        self.model = model

    def prepare(self) -> None:
        """Called once before evaluating a dataloader (set model mode, etc.)."""
        self.model.eval()

    @abstractmethod
    def predict(self, batch):
        """Return logits/probabilities of shape (B, num_classes) for a batch."""
        raise NotImplementedError

    @property
    def n_forward_passes(self) -> int:
        """Backbone forward passes per input sample (cost accounting)."""
        return 1


# --- classic augmentation-averaging methods (no gradient-based adaptation) --------


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


# --- TENT: test-time entropy minimization (Wang et al., 2021), simplified ---------


def configure_bn_for_tent(model: nn.Module) -> nn.Module:
    """Freeze everything except BatchNorm affine params; disable running stats."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        else:
            for p in module.parameters(recurse=False):
                p.requires_grad_(False)
    return model


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(1).mean()


class TentTTA(BaseTTA):
    """Adapts BatchNorm affine params for `steps` gradient steps per batch, then predicts."""

    name = "tent"

    def __init__(self, model, lr: float = 1e-3, steps: int = 1, **kwargs):
        super().__init__(model)
        configure_bn_for_tent(self.model)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        self.steps = steps

    def prepare(self) -> None:
        # TENT keeps the model in train() mode so BatchNorm uses batch statistics.
        self.model.train()

    def predict(self, batch):
        images, _ = batch
        # Explicitly re-enable autograd: under a Lightning test loop the outer
        # context is torch.no_grad() (with Trainer(inference_mode=False)), which
        # torch.enable_grad() can locally override.
        with torch.enable_grad():
            for _ in range(self.steps):
                logits = self.model(images)
                loss = entropy_loss(logits)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        with torch.no_grad():
            logits = self.model(images)
        return logits

    @property
    def n_forward_passes(self) -> int:
        return self.steps + 1


# --- registry -----------------------------------------------------------------

TTA_METHODS = {
    "baseline": BaselineTTA,
    "flip": FlipTTA,
    "multicrop": MultiCropTTA,
    "tent": TentTTA,
}


def build_tta(cfg, model):
    """Build a TTA method from a config `tta = dict(type=..., ...)` section."""
    cls = TTA_METHODS.get(cfg.type)
    if cls is None:
        raise ValueError(f"Unknown TTA method '{cfg.type}'. Available: {list(TTA_METHODS)}")

    kwargs = cfg.to_dict()
    kwargs.pop("type")
    return cls(model, **kwargs)
