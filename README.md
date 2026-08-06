# Rethinking Test-Time Augmentation (TTA)

Cost-effectiveness analysis of Test-Time Augmentation (TTA) for image
classification under distribution shift, evaluated on CIFAR-10 (clean) and
CIFAR-10-C (corrupted).

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

- **CIFAR-10** — clean data, used to train the base model. Downloaded to `data/cifar10/`.
- **CIFAR-10-C** — CIFAR-10 with 15 corruption types x 5 severity levels
  (Hendrycks & Dietterich, 2019), used to evaluate TTA under distribution shift.
  Downloaded from Zenodo (~2.9 GB) to `data/cifar10-c/`.

Run:

```bash
bash scripts/download_data.sh
```

Available options:

```bash
bash scripts/download_data.sh --skip-cifar10        # download CIFAR-10-C only
bash scripts/download_data.sh --skip-cifar10-c      # download CIFAR-10 only
bash scripts/download_data.sh --force               # re-download even if files already exist
bash scripts/download_data.sh --no-aria2-install     # don't try to auto-install aria2c
```

The script uses `aria2c` (multi-connection, much faster) if available, and
tries to install it automatically via `apt`/`brew`/`dnf`/`pacman` if it isn't
found — pass `--no-aria2-install` to skip that and fall back straight to
`curl`. It also verifies each file's MD5 checksum and supports resuming if
the connection drops mid-download.

## Project structure

```
config/           # experiment configs (see config/README.md for the naming scheme)
rtta/             # core code (models, data, TTA methods, metrics)
scripts/          # supporting scripts (dataset download, etc.)
data/             # downloaded datasets (gitignored)
test.py           # evaluation entrypoint: uv run test.py config/xxx.py
```

## Next steps

Base model training, evaluation usage docs, and experiment results will be
added next.
