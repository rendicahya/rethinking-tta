"""
Python-file-based config system.

A config file (e.g. config/resnet-50-cifar-10-c-baseline-n1-default.py) is a plain
Python module that defines top-level dict variables such as `model`, `data`, `tta`,
and `output`. This module loads that file and wraps it in a `Config` object that
supports both attribute access (cfg.model.type) and dict-style access/.get().
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path
from typing import Any


class Config(types.SimpleNamespace):
    """Recursive, dict-like namespace built from a config file's top-level variables."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return {
            k: (v.to_dict() if isinstance(v, Config) else v)
            for k, v in vars(self).items()
        }


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(**{k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    """Load a config/*.py file and return a Config namespace."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cfg_dict = {
        k: v
        for k, v in vars(module).items()
        if not k.startswith("_") and not isinstance(v, types.ModuleType)
    }
    cfg = _wrap(cfg_dict)
    cfg.config_path = str(path)
    cfg.config_name = path.stem
    return cfg
