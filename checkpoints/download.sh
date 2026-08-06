#!/usr/bin/env bash
#
# Download pretrained CIFAR-10 checkpoints.
#
# 1. ResNet-50, trained by Eduardo Dadalto:
#    https://huggingface.co/edadaltocg/resnet50_cifar10 (MIT license, self-reported
#    test accuracy 0.9465). Its architecture (conv1: 3x3 stride 1, no maxpool, fc: 10
#    classes) matches model.py's build_resnet50, so it loads directly via
#    model.load_state_dict() -- see model.py's build_model().
#    NOTE: data/datamodule.py's default CIFAR10_STD was set to match the
#    normalization this checkpoint was trained with.
#
# 2. WideResNet-28-10, RobustBench's "Standard" CIFAR-10 model:
#    https://github.com/RobustBench/robustbench (MIT license, ~94.8% clean test
#    accuracy). Hosted on Google Drive (RobustBench doesn't mirror it elsewhere).
#    Its architecture matches model.py's WideResNet, and it expects RAW [0,1]
#    pixel inputs (no mean/std normalization) -- the wideresnet2810 configs in
#    config/ set data.mean=(0,0,0), data.std=(1,1,1) accordingly.
#
# Usage:
#   ./checkpoints/download.sh
#   ./checkpoints/download.sh --force
#   ./checkpoints/download.sh --skip-resnet50
#   ./checkpoints/download.sh --skip-wideresnet2810

set -euo pipefail

CHECKPOINTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESNET50_CIFAR10_URL="https://huggingface.co/edadaltocg/resnet50_cifar10/resolve/main/pytorch_model.bin?download=true"
RESNET50_CIFAR10_SHA256="600e3f79fa41bc7c91d751a65b29b1ed2733345f8b6908381caa79a32ff28a6e"
RESNET50_CIFAR10_DEST="$CHECKPOINTS_DIR/resnet-50-cifar-10.pt"

# RobustBench "Standard" WRN-28-10 CIFAR-10 checkpoint, Google Drive file ID.
# No published checksum exists for this file (unlike the HuggingFace/Zenodo
# downloads above), so it's only sanity-checked (size + not-HTML) below.
WRN2810_CIFAR10_GDRIVE_ID="1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC"
WRN2810_CIFAR10_DEST="$CHECKPOINTS_DIR/wideresnet-28-10-cifar-10.pt"
WRN2810_CIFAR10_MIN_BYTES=100000000 # ~100 MB; the real file is ~140 MB

FORCE=false
SKIP_RESNET50=false
SKIP_WRN2810=false
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=true ;;
    --skip-resnet50) SKIP_RESNET50=true ;;
    --skip-wideresnet2810) SKIP_WRN2810=true ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

check_sha256() {
  # check_sha256 <file> <expected_sha256>
  local file="$1"
  local expected="$2"
  local actual

  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$file" | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    actual=$(shasum -a 256 "$file" | awk '{print $1}')
  else
    echo "[check_sha256] No sha256sum/shasum tool found, skipping checksum verification." >&2
    return 0
  fi

  [ "$actual" = "$expected" ]
}

download_resnet50() {
  if [ -f "$RESNET50_CIFAR10_DEST" ] && [ "$FORCE" = false ]; then
    echo "[resnet-50-cifar-10] Already present at $RESNET50_CIFAR10_DEST, skipping."
    return 0
  fi

  mkdir -p "$CHECKPOINTS_DIR"
  echo "[resnet-50-cifar-10] Downloading (~94 MB) ..."

  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x16 -s16 -k1M -c -o "$(basename "$RESNET50_CIFAR10_DEST")" -d "$CHECKPOINTS_DIR" "$RESNET50_CIFAR10_URL"
  else
    curl -L -C - --retry 5 --retry-delay 5 -o "$RESNET50_CIFAR10_DEST" "$RESNET50_CIFAR10_URL"
  fi

  echo "[resnet-50-cifar-10] Verifying SHA-256 checksum ..."
  if ! check_sha256 "$RESNET50_CIFAR10_DEST" "$RESNET50_CIFAR10_SHA256"; then
    echo "[resnet-50-cifar-10] ERROR: checksum mismatch. Delete $RESNET50_CIFAR10_DEST and re-run." >&2
    exit 1
  fi

  echo "[resnet-50-cifar-10] Checksum OK. Saved to $RESNET50_CIFAR10_DEST"
}

# download_gdrive <file_id> <dest>
#
# Prefers `uvx gdown` (this project already uses uv, and gdown handles Google
# Drive's confirmation-token/virus-scan-warning redirect for large files
# properly). Falls back to a curl cookie-jar trick if uvx/gdown isn't usable.
# Google Drive downloads are inherently less reliable than a direct HTTPS host
# (no published checksum either), so the caller must sanity-check the result.
download_gdrive() {
  local file_id="$1"
  local dest="$2"

  if command -v uvx >/dev/null 2>&1; then
    echo "[download_gdrive] Trying 'uvx gdown' ..."
    if uvx gdown "$file_id" -O "$dest"; then
      return 0
    fi
    echo "[download_gdrive] uvx gdown failed, falling back to curl ..." >&2
  else
    echo "[download_gdrive] uvx not found, falling back to curl ..." >&2
  fi

  local cookie_jar
  cookie_jar="$(mktemp)"
  local confirm_page="$dest.confirm.html"
  local confirm_code

  curl -sc "$cookie_jar" "https://drive.google.com/uc?export=download&id=${file_id}" -o "$confirm_page"
  confirm_code=$(grep -o 'confirm=[a-zA-Z0-9_-]*' "$confirm_page" | head -1 | cut -d= -f2)
  rm -f "$confirm_page"

  if [ -n "${confirm_code:-}" ]; then
    curl -Lb "$cookie_jar" \
      "https://drive.google.com/uc?export=download&confirm=${confirm_code}&id=${file_id}" -o "$dest"
  else
    # Small files (no virus-scan warning) download directly without a confirm token.
    curl -Lb "$cookie_jar" "https://drive.google.com/uc?export=download&id=${file_id}" -o "$dest"
  fi

  rm -f "$cookie_jar"
}

download_wideresnet2810() {
  if [ -f "$WRN2810_CIFAR10_DEST" ] && [ "$FORCE" = false ]; then
    echo "[wideresnet-28-10-cifar-10] Already present at $WRN2810_CIFAR10_DEST, skipping."
    return 0
  fi

  mkdir -p "$CHECKPOINTS_DIR"
  echo "[wideresnet-28-10-cifar-10] Downloading from Google Drive (~140 MB) ..."
  download_gdrive "$WRN2810_CIFAR10_GDRIVE_ID" "$WRN2810_CIFAR10_DEST"

  echo "[wideresnet-28-10-cifar-10] Sanity-checking the download ..."
  if [ ! -s "$WRN2810_CIFAR10_DEST" ]; then
    echo "[wideresnet-28-10-cifar-10] ERROR: download is empty. Delete $WRN2810_CIFAR10_DEST and re-run." >&2
    exit 1
  fi

  local size
  size=$(wc -c < "$WRN2810_CIFAR10_DEST")
  if [ "$size" -lt "$WRN2810_CIFAR10_MIN_BYTES" ]; then
    if head -c 200 "$WRN2810_CIFAR10_DEST" | grep -qi '<html'; then
      echo "[wideresnet-28-10-cifar-10] ERROR: got an HTML page instead of the checkpoint" \
           "(Google Drive quota/confirmation issue). Delete $WRN2810_CIFAR10_DEST and retry," \
           "or download it manually from" \
           "https://drive.google.com/uc?id=$WRN2810_CIFAR10_GDRIVE_ID" >&2
    else
      echo "[wideresnet-28-10-cifar-10] ERROR: file is only $size bytes" \
           "(expected >= $WRN2810_CIFAR10_MIN_BYTES). Delete $WRN2810_CIFAR10_DEST and re-run." >&2
    fi
    exit 1
  fi

  echo "[wideresnet-28-10-cifar-10] Looks OK ($size bytes, no published checksum to verify against)."
  echo "[wideresnet-28-10-cifar-10] Saved to $WRN2810_CIFAR10_DEST"
}

[ "$SKIP_RESNET50" = true ] || download_resnet50
[ "$SKIP_WRN2810" = true ] || download_wideresnet2810
