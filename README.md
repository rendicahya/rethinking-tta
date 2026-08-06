# Rethinking Test-Time Augmentation (TTA)

Cost-effectiveness analysis of Test-Time Augmentation (TTA) for image
classification under distribution shift, evaluated on CIFAR-10-C and
Tiny-ImageNet-C.

## Requirements

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An NVIDIA GPU (RTX 3060 12GB or equivalent recommended) with CUDA installed
- Bash (native on Linux/macOS; use WSL or Git Bash on Windows)
- [aria2](https://aria2.github.io/) (optional, for faster dataset downloads —
  the download script auto-installs it if missing, see below)

## Setup

1. Clone the repo and enter it:

   ```bash
   git clone https://github.com/rendicahya/rethinking-tta.git
   cd rethinking-tta
   ```

2. Install dependencies with uv:

   ```bash
   uv sync
   ```

   This creates a virtual environment (`.venv`) and installs all dependencies
   from `pyproject.toml` (PyTorch, torchvision, PyTorch Lightning).

## Download the dataset

Datasets used:

- **CIFAR-10** — clean data, used to train the CIFAR base model. Downloaded to `data/cifar10/`.
- **CIFAR-10-C** — CIFAR-10 with 15 corruption types x 5 severity levels
  (Hendrycks & Dietterich, 2019), used to evaluate TTA under distribution shift.
  Downloaded from Zenodo (~2.9 GB) to `data/cifar10-c/`.
- **Tiny-ImageNet** — clean data (64x64, 200 classes), used to train the Tiny-ImageNet
  base model. Downloaded to `data/tiny-imagenet/` (~248 MB).
- **Tiny-ImageNet-C** — Tiny-ImageNet with the same 15 corruption types x 5 severity
  levels, used to evaluate TTA under distribution shift at a second scale.
  Downloaded from Zenodo (~2.8 GB) to `data/tiny-imagenet-c/`. This stands in for
  full-size ImageNet-C (~62 GB), which is impractical on a 12 GB GPU / 16 GB RAM setup.

Run:

```bash
bash data/download.sh
```

Available options:

```bash
bash data/download.sh --skip-cifar10
bash data/download.sh --skip-cifar10-c
bash data/download.sh --skip-tiny-imagenet
bash data/download.sh --skip-tiny-imagenet-c
bash data/download.sh --force               # re-download even if files already exist
bash data/download.sh --no-aria2-install     # don't try to auto-install aria2c
bash data/download.sh --no-unzip-install     # don't try to auto-install unzip
bash data/download.sh --cleanup              # delete the archive after extraction
```

The script uses `aria2c` (multi-connection, much faster) if available, and
`unzip` (for the Tiny-ImageNet zip archive), and tries to install whichever is
missing automatically via `apt`/`brew`/`dnf`/`pacman` — pass
`--no-aria2-install`/`--no-unzip-install` to skip that (falls back to `curl`
and Python's `zipfile` module respectively). It also verifies each file's MD5
checksum, supports resuming if the connection drops mid-download, and
automatically skips a dataset that's already been downloaded/extracted (pass
`--force` to redo it anyway).

## Project structure

```
config/                    # experiment configs + loader
  config_loader.py            # loader for config/*.py files
  README.md                   # config naming scheme
  resnet-50-*.py               # experiment configs
data/                       # dataset code, download script, and downloaded datasets
  download.sh                  # dataset download script: bash data/download.sh
  cifar10c.py                  # CIFAR-10-C dataset
  datamodule.py                # Lightning DataModule (CIFAR-10 + CIFAR-10-C)
  cifar10/, cifar10-c/, ...    # downloaded datasets (gitignored)
model.py                   # ResNet backbone
tta.py                     # TTA methods (baseline, flip, multicrop, TENT) + registry
metrics.py                 # cost metrics: latency, FLOPs, GPU memory, energy
lightning_module.py        # LightningModule wrapping model + TTA method
test.py                    # evaluation entrypoint: uv run test.py config/xxx.py
```

Only the downloaded dataset folders/archives inside `data/` are gitignored — code
files (`data/*.py`, `data/*.sh`) stay tracked.

## Next steps

Base model training, evaluation usage docs, and experiment results will be
added next.
