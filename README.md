# Rethinking Test-Time Augmentation (TTA)

Cost-effectiveness analysis of Test-Time Augmentation (TTA) for image
classification under distribution shift, evaluated on CIFAR-10-C and
Tiny-ImageNet-C.

## Requirements

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An NVIDIA GPU (RTX 3060 12GB or equivalent recommended) with CUDA installed
- Bash (Linux/macOS) or PowerShell 5.1+ (Windows) to run the download
  scripts — every `download.sh` (Linux/macOS) has a matching `download.ps1`
  (Windows) with the same options (kebab-case flags become PascalCase, e.g.
  `--skip-cifar10` -> `-SkipCifar10`)
- [aria2](https://aria2.github.io/) (optional, for faster dataset downloads —
  the `.sh` script auto-installs it if missing; on Windows install it
  yourself first, e.g. `winget install aria2.aria2`, see below)

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

Run (Linux/macOS: `.sh`, Windows: `.ps1`):

```bash
bash data/download.sh
```

```powershell
.\data\download.ps1
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

```powershell
.\data\download.ps1 -SkipCifar10
.\data\download.ps1 -SkipCifar10C
.\data\download.ps1 -SkipTinyImagenet
.\data\download.ps1 -SkipTinyImagenetC
.\data\download.ps1 -Force                   # re-download even if files already exist
.\data\download.ps1 -Cleanup                 # delete the archive after extraction
```

The `.sh` script uses `aria2c` (multi-connection, much faster) if available,
and `unzip` (for the Tiny-ImageNet zip archive), and tries to install
whichever is missing automatically via `apt`/`brew`/`dnf`/`pacman` — pass
`--no-aria2-install`/`--no-unzip-install` to skip that (falls back to `curl`
and Python's `zipfile` module respectively). The `.ps1` script uses `aria2c`
too if it's already on `PATH`, but doesn't auto-install it (no single
standard Windows package manager); it falls back to `curl.exe` (ships with
Windows 10/11) and extracts with the built-in `tar` (for `.tar`/`.tar.gz`)
and `Expand-Archive` (for the Tiny-ImageNet `.zip`) — no `unzip` needed. Both
scripts verify each file's MD5 checksum, support resuming if the connection
drops mid-download (`.sh`/curl and aria2c only), and automatically skip a
dataset that's already been downloaded/extracted (pass `--force`/`-Force` to
redo it anyway).

## Models

Two backbones are supported, matching the original research plan:

| Model | Params | Pretrained checkpoint | Input normalization |
|---|---|---|---|
| ResNet-50 (adapted stem for 32x32) | ~23.5M | [edadaltocg/resnet50_cifar10](https://huggingface.co/edadaltocg/resnet50_cifar10) (94.65% acc, MIT) | CIFAR-10 mean/std |
| WideResNet-28-10 | ~36.5M | [RobustBench](https://github.com/RobustBench/robustbench) "Standard" CIFAR-10 model (~94.8% acc, MIT) | none (raw [0,1] pixels) |

Both fit comfortably on a 12 GB GPU at CIFAR resolution (32x32) with plenty of
headroom for batch sizes of 256+ and TTA methods that run multiple forward
passes per sample (e.g. `multicrop` with `n_views=10`) — VRAM is not a
constraint for either model at this scale.

## Download a pretrained checkpoint

To run an evaluation without training a model from scratch, download both
pretrained checkpoints (Linux/macOS: `.sh`, Windows: `.ps1`):

```bash
bash checkpoints/download.sh
```

```powershell
.\checkpoints\download.ps1
```

This saves `checkpoints/resnet-50-cifar-10.pt` and
`checkpoints/wideresnet-28-10-cifar-10.pt`, which the example configs in
`config/` already point to. Options:

```bash
bash checkpoints/download.sh --force                  # re-download even if files exist
bash checkpoints/download.sh --skip-resnet50
bash checkpoints/download.sh --skip-wideresnet2810
```

```powershell
.\checkpoints\download.ps1 -Force                     # re-download even if files exist
.\checkpoints\download.ps1 -SkipResnet50
.\checkpoints\download.ps1 -SkipWideresnet2810
```

The `.ps1` script downloads the WideResNet-28-10 checkpoint from Google
Drive via `uvx gdown` too, falling back to a `curl.exe` cookie-jar trick if
`uv`/`gdown` aren't available.

The ResNet-50 checkpoint's architecture (conv1: 3x3 stride 1, no maxpool, fc:
10 classes) matches `model.py`'s `build_resnet50`, and its expected input
normalization matches `data/datamodule.py`'s `CIFAR10_MEAN`/`CIFAR10_STD`
defaults. The WideResNet-28-10 checkpoint's architecture matches `model.py`'s
`WideResNet` (attribute names kept identical to RobustBench's implementation
for state_dict compatibility) and expects raw `[0, 1]` pixel inputs — see the
`wideresnet2810` configs, which set `data.mean=(0,0,0), data.std=(1,1,1)`.

The WideResNet-28-10 checkpoint is hosted on Google Drive (no direct HTTPS
mirror), downloaded via `gdown` (through `uvx`, falling back to a plain
`curl` cookie trick) and sanity-checked (file size, not-HTML) since Google
Drive doesn't publish a checksum for it. If the download script reports an
error, you can also fetch it manually from
`https://drive.google.com/uc?id=1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC` and save it
as `checkpoints/wideresnet-28-10-cifar-10.pt`.

If you swap in a different checkpoint, double-check its expected input
normalization and set `data.mean`/`data.std` in the config accordingly.

## Run an evaluation

```bash
uv run test.py config/resnet-50-cifar-10-c-baseline-n1-default.py
uv run test.py config/wideresnet-28-10-cifar-10-c-baseline-n1-default.py
```

This runs the configured TTA method against clean CIFAR-10 and every
CIFAR-10-C corruption/severity, and writes per-split accuracy and cost
metrics (latency, FLOPs, GPU memory, energy) to `results/`. See
`config/README.md` for the config naming scheme and other example configs
(`flip`, `multicrop`, `tent`) across both backbones.

## Project structure

```
config/                    # experiment configs + loader
  config_loader.py            # loader for config/*.py files
  README.md                   # config naming scheme
  resnet-50-*.py               # ResNet-50 experiment configs
  wideresnet-28-10-*.py        # WideResNet-28-10 experiment configs
data/                       # dataset code, download scripts, and downloaded datasets
  download.sh                  # dataset download script, Linux/macOS: bash data/download.sh
  download.ps1                 # dataset download script, Windows: .\data\download.ps1
  cifar10c.py                  # CIFAR-10-C dataset
  datamodule.py                # Lightning DataModule (CIFAR-10 + CIFAR-10-C)
  cifar10/, cifar10-c/, ...    # downloaded datasets (gitignored)
checkpoints/                # pretrained/trained model checkpoints
  download.sh                  # checkpoint download script, Linux/macOS: bash checkpoints/download.sh
  download.ps1                 # checkpoint download script, Windows: .\checkpoints\download.ps1
  resnet-50-cifar-10.pt        # downloaded checkpoint (gitignored)
  wideresnet-28-10-cifar-10.pt # downloaded checkpoint (gitignored)
model.py                   # ResNet-50 + WideResNet-28-10 backbones
tta.py                     # TTA methods (baseline, flip, multicrop, TENT) + registry
metrics.py                 # cost metrics: latency, FLOPs, GPU memory, energy
lightning_module.py        # LightningModule wrapping model + TTA method
test.py                    # evaluation entrypoint: uv run test.py config/xxx.py
```

Only the downloaded dataset/checkpoint files are gitignored — download scripts
(`data/*.sh`, `data/*.ps1`, `checkpoints/*.sh`, `checkpoints/*.ps1`) and code
files (`data/*.py`) stay tracked.

## Next steps

Base model training, evaluation usage docs, and experiment results will be
added next.
