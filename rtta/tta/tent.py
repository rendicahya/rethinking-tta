"""TENT: Test-time entropy minimization (Wang et al., 2021), simplified.

Adapts only the affine parameters of BatchNorm layers by minimizing prediction
entropy on each incoming test batch, using batch statistics instead of the
running statistics collected during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseTTA


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
