"""Compute-cost measurement utilities: latency, FLOPs, GPU memory, energy.

FLOPs counting needs the optional `fvcore` dependency, and energy sampling needs
the optional `pynvml` dependency (both installable via `uv sync --extra metrics`).
Both degrade gracefully to `None` if unavailable, so evaluation still runs.
"""

from __future__ import annotations

import contextlib
import time

import torch


class LatencyTimer:
    """CUDA-synchronized timer accumulating total elapsed wall-clock time."""

    def __init__(self):
        self.total_seconds = 0.0
        self.n_batches = 0

    @contextlib.contextmanager
    def measure(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total_seconds += time.perf_counter() - start
        self.n_batches += 1

    @property
    def avg_seconds_per_batch(self) -> float:
        return self.total_seconds / self.n_batches if self.n_batches else 0.0


def count_flops(model: torch.nn.Module, input_size=(1, 3, 32, 32)):
    """Estimate FLOPs for a single forward pass. Returns None if fvcore is not installed."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("[count_flops] fvcore not installed, skipping FLOPs count "
              "(install with `uv sync --extra metrics`).")
        return None

    device = next(model.parameters()).device
    dummy = torch.randn(*input_size, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        analysis = FlopCountAnalysis(model, dummy)
        analysis.unsupported_ops_warnings(False)
        flops = analysis.total()
    model.train(was_training)
    return flops


def reset_gpu_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**2)


class GPUEnergyMeter:
    """Samples instantaneous GPU power draw via pynvml (NVML). Energy = avg power x time."""

    def __init__(self, device_index: int = 0):
        self._handle = None
        self._pynvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._handle is not None

    def sample_power_watts(self):
        """Instantaneous power draw in watts, or None if pynvml/NVML is unavailable."""
        if not self.available:
            return None
        return self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
