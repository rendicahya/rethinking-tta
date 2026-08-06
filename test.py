"""
Evaluate a trained model + TTA method on clean CIFAR-10 and corrupted CIFAR-10-C
using PyTorch Lightning, reporting accuracy alongside compute-cost metrics
(latency, FLOPs, GPU memory, energy) for cost-effectiveness analysis.

Usage:
    uv run test.py config/resnet-50-cifar-10-c-baseline-n1-default.py
    uv run test.py config/resnet-50-cifar-10-c-flip-n2-default.py --accelerator cpu
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import lightning as L

from rtta.config import load_config
from rtta.data.datamodule import CIFAR10DataModule
from rtta.lightning_module import TTAModule
from rtta.metrics.cost import count_flops
from rtta.models.resnet import build_model
from rtta.tta.registry import build_tta

RESULT_FIELDS = [
    "corruption",
    "severity",
    "accuracy",
    "n_samples",
    "avg_latency_ms_per_batch",
    "avg_power_w",
    "energy_j",
    "peak_gpu_memory_mb",
]


def build_test_dataloaders(dm: CIFAR10DataModule, cfg):
    """Returns (dataloaders, loader_meta) where loader_meta[i] = (corruption, severity)."""
    dataloaders = [dm.clean_test_dataloader()]
    loader_meta = [("clean", 0)]

    corruptions = cfg.data.get("corruptions", "all")
    corruptions = dm.corruptions() if corruptions == "all" else corruptions
    severities = cfg.data.get("severities", [1, 2, 3, 4, 5])

    for corruption in corruptions:
        for severity in severities:
            dataloaders.append(dm.corrupted_dataloader(corruption, severity))
            loader_meta.append((corruption, severity))

    return dataloaders, loader_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="Path to a config/*.py file")
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator: auto|gpu|cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[config] {cfg.config_name}")

    model = build_model(cfg.model)
    tta = build_tta(cfg.tta, model)
    flops = count_flops(model, input_size=(1, 3, 32, 32))

    dm = CIFAR10DataModule(
        root=cfg.data.get("root", "data"),
        batch_size=cfg.data.get("batch_size", 128),
        num_workers=cfg.data.get("num_workers", 4),
    )
    dm.setup("validate")

    dataloaders, loader_meta = build_test_dataloaders(dm, cfg)
    loader_names = [f"{c}_sev{s}" for c, s in loader_meta]
    lit_module = TTAModule(model, tta, loader_names=loader_names)

    # inference_mode=False lets TTA methods (e.g. TENT) run a local backward pass
    # inside test_step via torch.enable_grad(), which torch.inference_mode() would block.
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        inference_mode=False,
    )
    trainer.test(lit_module, dataloaders=dataloaders)

    results = lit_module.results()
    for row, (corruption, severity) in zip(results, loader_meta):
        row["corruption"] = corruption
        row["severity"] = severity
        tag = "clean" if corruption == "clean" else f"{corruption} sev{severity}"
        print(f"[{tag:<24}] acc={row['accuracy']:.4f}  latency={row['avg_latency_ms_per_batch']:.1f}ms/batch")

    output_cfg = cfg.get("output", None)
    out_dir = Path(output_cfg.get("dir", "results")) if output_cfg else Path("results")
    tag = output_cfg.get("tag", cfg.config_name) if output_cfg else cfg.config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in RESULT_FIELDS})

    print(f"\nSaved per-corruption/severity results to {out_path}")
    print(f"Backbone FLOPs (single forward pass): {flops if flops is not None else 'N/A'}")
    print(f"TTA forward passes per sample: {tta.n_forward_passes}")
    if flops is not None:
        print(f"Estimated FLOPs per sample with TTA: {flops * tta.n_forward_passes:,.0f}")


if __name__ == "__main__":
    main()
