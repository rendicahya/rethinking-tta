"""LightningModule wrapping a backbone model + a BaseTTA method for evaluation.

Intended to be used with `Trainer.test(dataloaders=[...])` where each dataloader
is one (corruption, severity) split (see test.py). Accuracy and compute-cost
metrics (latency, GPU memory, energy) are accumulated per dataloader and can be
read back afterwards via `.results()`.

Note: the Trainer must be created with `inference_mode=False` so that TTA
methods needing gradients at test time (e.g. TENT) can locally re-enable
autograd with `torch.enable_grad()` inside `predict()` -- `torch.inference_mode()`
taints tensors in a way that can't be undone locally, but `torch.no_grad()` can.
"""

from __future__ import annotations

import lightning as L

from .metrics.cost import GPUEnergyMeter, LatencyTimer, peak_gpu_memory_mb, reset_gpu_memory_stats
from .tta.base import BaseTTA


class TTAModule(L.LightningModule):
    def __init__(self, model, tta: BaseTTA, loader_names: list[str] | None = None):
        super().__init__()
        self.model = model
        self.tta = tta
        self.loader_names = loader_names or []
        self.energy_meter = GPUEnergyMeter()

        self._timers: dict[int, LatencyTimer] = {}
        self._correct: dict[int, int] = {}
        self._total: dict[int, int] = {}
        self._power_samples: dict[int, list[float]] = {}
        self._peak_mem_mb: dict[int, float | None] = {}
        self._current_loader_idx: int | None = None

    def on_test_epoch_start(self) -> None:
        self.tta.prepare()
        self._timers.clear()
        self._correct.clear()
        self._total.clear()
        self._power_samples.clear()
        self._peak_mem_mb.clear()
        self._current_loader_idx = None

    def on_test_batch_start(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if dataloader_idx != self._current_loader_idx:
            self._current_loader_idx = dataloader_idx
            reset_gpu_memory_stats()

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        images, labels = batch
        timer = self._timers.setdefault(dataloader_idx, LatencyTimer())

        with timer.measure():
            logits = self.tta.predict((images, labels))

        power = self.energy_meter.sample_power_watts()
        if power is not None:
            self._power_samples.setdefault(dataloader_idx, []).append(power)

        preds = logits.argmax(dim=1)
        self._correct[dataloader_idx] = self._correct.get(dataloader_idx, 0) + (preds == labels).sum().item()
        self._total[dataloader_idx] = self._total.get(dataloader_idx, 0) + labels.size(0)
        self._peak_mem_mb[dataloader_idx] = peak_gpu_memory_mb()

    def results(self) -> list[dict]:
        """Per-dataloader accuracy + cost metrics, read after `Trainer.test()` finishes."""
        rows = []
        for idx in sorted(self._timers):
            timer = self._timers[idx]
            correct = self._correct.get(idx, 0)
            total = self._total.get(idx, 0)
            power_samples = self._power_samples.get(idx, [])
            avg_power = sum(power_samples) / len(power_samples) if power_samples else None
            energy_j = avg_power * timer.total_seconds if avg_power is not None else None
            name = self.loader_names[idx] if idx < len(self.loader_names) else str(idx)

            rows.append(
                {
                    "name": name,
                    "accuracy": correct / total if total else 0.0,
                    "n_samples": total,
                    "avg_latency_ms_per_batch": timer.avg_seconds_per_batch * 1000,
                    "avg_power_w": avg_power,
                    "energy_j": energy_j,
                    "peak_gpu_memory_mb": self._peak_mem_mb.get(idx),
                }
            )
        return rows
