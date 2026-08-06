"""Maps a config's `tta.type` string to a TTA class, and builds it."""

from __future__ import annotations

from .classic import BaselineTTA, FlipTTA, MultiCropTTA
from .tent import TentTTA

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
