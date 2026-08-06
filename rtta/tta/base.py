"""Common interface for test-time augmentation / adaptation (TTA) methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseTTA(ABC):
    """Base class every TTA method implements.

    Subclasses wrap a trained model and, given a batch of (possibly corrupted)
    images, return predictions (logits or averaged probabilities). `n_forward_passes`
    reports how many backbone forward passes are needed per sample, which is used
    downstream to estimate compute cost.
    """

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
