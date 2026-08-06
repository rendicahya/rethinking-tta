#!/usr/bin/env bash
#
# Download datasets for the cost-effectiveness study of TTA under distribution shift.
# Faster alternative to download_data.py: uses aria2c (multi-connection) when available,
# falling back to curl with resume support.
#
# Datasets:
# - CIFAR-10   : clean (in-distribution) data -> data/cifar10/
# - CIFAR-10-C : 15 corruption types x 5 severities (Hendrycks & Dietterich, 2019) -> data/cifar10-c/
#
# Usage:
#   ./scripts/download_data.sh              # download both
#   ./scripts/download_data.sh --skip-cifar10
#   ./scripts/download_data.sh --skip-cifar10-c
#   ./scripts/download_data.sh --force              # re-download even if files already exist
#   ./scripts/download_data.sh --no-aria2-install    # don't try to auto-install aria2c

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

CIFAR10_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5="c58f30108f718f92721af3b95e74349a"

CIFAR10_C_URL="https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
CIFAR10_C_MD5="56bf5dcef84df0e2308c6dcbcbbd8499"

SKIP_CIFAR10=false
SKIP_CIFAR10_C=false
FORCE=false
NO_ARIA2_INSTALL=false

for arg in "$@"; do
  case "$arg" in
    --skip-cifar10) SKIP_CIFAR10=true ;;
    --skip-cifar10-c) SKIP_CIFAR10_C=true ;;
    --force) FORCE=true ;;
    --no-aria2-install) NO_ARIA2_INSTALL=true ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p "$DATA_DIR"

# --- helpers -----------------------------------------------------------

ensure_aria2c() {
  if command -v aria2c >/dev/null 2>&1; then
    return
  fi

  if [ "$NO_ARIA2_INSTALL" = true ]; then
    echo "[setup] aria2c not found, skipping auto-install (--no-aria2-install). Falling back to curl."
    return
  fi

  echo "[setup] aria2c not found, attempting to install it for faster multi-connection downloads ..."

  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y aria2
  elif command -v brew >/dev/null 2>&1; then
    brew install aria2
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y aria2
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -Sy --noconfirm aria2
  else
    echo "[setup] No supported package manager found (apt/brew/dnf/pacman)." >&2
    echo "[setup] Install aria2c manually: https://aria2.github.io/" >&2
  fi

  if command -v aria2c >/dev/null 2>&1; then
    echo "[setup] aria2c installed successfully."
  else
    echo "[setup] Could not install aria2c, falling back to curl (slower)." >&2
  fi
}

download() {
  # download <url> <dest_file>
  local url="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"

  if command -v aria2c >/dev/null 2>&1; then
    echo "[download] Using aria2c (multi-connection) for $(basename "$dest")"
    aria2c -x16 -s16 -k1M -c -o "$(basename "$dest")" -d "$(dirname "$dest")" "$url"
  else
    echo "[download] aria2c not found, using curl (single connection) for $(basename "$dest")"
    echo "[download] Tip: install aria2c for much faster parallel downloads."
    curl -L -C - --retry 5 --retry-delay 5 -o "$dest" "$url"
  fi
}

check_md5() {
  # check_md5 <file> <expected_md5>
  local file="$1"
  local expected="$2"
  local actual

  if command -v md5sum >/dev/null 2>&1; then
    actual=$(md5sum "$file" | awk '{print $1}')
  elif command -v md5 >/dev/null 2>&1; then
    actual=$(md5 -q "$file")
  else
    echo "[check_md5] No md5sum/md5 tool found, skipping checksum verification." >&2
    return 0
  fi

  [ "$actual" = "$expected" ]
}

# --- CIFAR-10 ------------------------------------------------------------

download_cifar10() {
  local out_dir="$DATA_DIR/cifar10"
  local archive="$DATA_DIR/cifar-10-python.tar.gz"

  if [ -d "$out_dir/cifar-10-batches-py" ] && [ "$FORCE" = false ]; then
    echo "[CIFAR-10] Already present at $out_dir, skipping."
    return
  fi

  if [ ! -f "$archive" ] || [ "$FORCE" = true ]; then
    echo "[CIFAR-10] Downloading (~170 MB) ..."
    download "$CIFAR10_URL" "$archive"
  fi

  echo "[CIFAR-10] Verifying MD5 checksum ..."
  if ! check_md5 "$archive" "$CIFAR10_MD5"; then
    echo "[CIFAR-10] ERROR: MD5 checksum mismatch. Delete $archive and re-run." >&2
    exit 1
  fi
  echo "[CIFAR-10] Checksum OK."

  mkdir -p "$out_dir"
  echo "[CIFAR-10] Extracting to $out_dir ..."
  tar -xzf "$archive" -C "$out_dir"
  echo "[CIFAR-10] Done."
}

# --- CIFAR-10-C ------------------------------------------------------------

download_cifar10_c() {
  local out_dir="$DATA_DIR/cifar10-c"
  local archive="$DATA_DIR/CIFAR-10-C.tar"

  if [ -f "$out_dir/labels.npy" ] && [ "$FORCE" = false ]; then
    echo "[CIFAR-10-C] Already present at $out_dir, skipping."
    return
  fi

  if [ ! -f "$archive" ] || [ "$FORCE" = true ]; then
    echo "[CIFAR-10-C] Downloading (~2.9 GB) ..."
    download "$CIFAR10_C_URL" "$archive"
  fi

  echo "[CIFAR-10-C] Verifying MD5 checksum ..."
  if ! check_md5 "$archive" "$CIFAR10_C_MD5"; then
    echo "[CIFAR-10-C] ERROR: MD5 checksum mismatch. Delete $archive and re-run." >&2
    exit 1
  fi
  echo "[CIFAR-10-C] Checksum OK."

  echo "[CIFAR-10-C] Extracting to $out_dir ..."
  mkdir -p "$out_dir"
  tar -xf "$archive" -C "$DATA_DIR"

  # The archive extracts into a "CIFAR-10-C/" folder; move contents to out_dir.
  if [ -d "$DATA_DIR/CIFAR-10-C" ] && [ "$DATA_DIR/CIFAR-10-C" != "$out_dir" ]; then
    mv "$DATA_DIR/CIFAR-10-C"/* "$out_dir/"
    rmdir "$DATA_DIR/CIFAR-10-C"
  fi

  echo "[CIFAR-10-C] Done. Archive kept at $archive (delete manually to save space)."
}

# --- main ------------------------------------------------------------------

ensure_aria2c

[ "$SKIP_CIFAR10" = false ] && download_cifar10
[ "$SKIP_CIFAR10_C" = false ] && download_cifar10_c

echo "All downloads finished."
