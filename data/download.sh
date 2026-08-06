#!/usr/bin/env bash
#
# Download datasets for the cost-effectiveness study of TTA under distribution shift.
# Uses aria2c (multi-connection) when available, falling back to curl with resume support.
#
# Datasets:
# - CIFAR-10         : clean (in-distribution) data -> data/cifar10/
# - CIFAR-10-C       : 15 corruption types x 5 severities (Hendrycks & Dietterich, 2019) -> data/cifar10-c/
# - Tiny-ImageNet    : clean (in-distribution) data, 64x64, 200 classes -> data/tiny-imagenet/
# - Tiny-ImageNet-C  : 15 corruption types x 5 severities, 64x64 -> data/tiny-imagenet-c/
#                      (smaller stand-in for full ImageNet-C, ~2.8GB vs ~62GB)
#
# Usage:
#   ./data/download.sh              # download everything
#   ./data/download.sh --skip-cifar10
#   ./data/download.sh --skip-cifar10-c
#   ./data/download.sh --skip-tiny-imagenet
#   ./data/download.sh --skip-tiny-imagenet-c
#   ./data/download.sh --force              # re-download even if files already exist
#   ./data/download.sh --no-aria2-install    # don't try to auto-install aria2c
#   ./data/download.sh --no-unzip-install    # don't try to auto-install unzip
#   ./data/download.sh --cleanup             # delete the downloaded archive after extraction
#
# Already-downloaded/extracted datasets are skipped automatically (use --force to redo).

set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CIFAR10_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5="c58f30108f718f92721af3b95e74349a"

CIFAR10_C_URL="https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
CIFAR10_C_MD5="56bf5dcef84df0e2308c6dcbcbbd8499"

TINY_IMAGENET_URL="https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1"
TINY_IMAGENET_MD5="90528d7ca1a48142e341f4ef8d21d0de"

TINY_IMAGENET_C_URL="https://zenodo.org/records/2469796/files/TinyImageNet-C.tar?download=1"
TINY_IMAGENET_C_MD5="3d9c6e89c2609aeb4198f84c8edd1ff0"

SKIP_CIFAR10=false
SKIP_CIFAR10_C=false
SKIP_TINY_IMAGENET=false
SKIP_TINY_IMAGENET_C=false
FORCE=false
NO_ARIA2_INSTALL=false
NO_UNZIP_INSTALL=false
CLEANUP=false

for arg in "$@"; do
  case "$arg" in
    --skip-cifar10) SKIP_CIFAR10=true ;;
    --skip-cifar10-c) SKIP_CIFAR10_C=true ;;
    --skip-tiny-imagenet) SKIP_TINY_IMAGENET=true ;;
    --skip-tiny-imagenet-c) SKIP_TINY_IMAGENET_C=true ;;
    --force) FORCE=true ;;
    --no-aria2-install) NO_ARIA2_INSTALL=true ;;
    --no-unzip-install) NO_UNZIP_INSTALL=true ;;
    --cleanup) CLEANUP=true ;;
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

ensure_unzip() {
  if command -v unzip >/dev/null 2>&1; then
    return
  fi

  if [ "$NO_UNZIP_INSTALL" = true ]; then
    echo "[setup] unzip not found, skipping auto-install (--no-unzip-install). Falling back to Python's zipfile module."
    return
  fi

  echo "[setup] unzip not found, attempting to install it ..."

  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y unzip
  elif command -v brew >/dev/null 2>&1; then
    brew install unzip
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y unzip
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -Sy --noconfirm unzip
  else
    echo "[setup] No supported package manager found (apt/brew/dnf/pacman)." >&2
    echo "[setup] Install unzip manually, or rely on the Python zipfile fallback." >&2
  fi

  if command -v unzip >/dev/null 2>&1; then
    echo "[setup] unzip installed successfully."
  else
    echo "[setup] Could not install unzip, falling back to Python's zipfile module." >&2
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

cleanup_archive() {
  # cleanup_archive <archive_file> <label>
  local archive="$1"
  local label="$2"

  if [ "$CLEANUP" = true ]; then
    rm -f "$archive"
    echo "[$label] Removed archive $archive (--cleanup)."
  else
    echo "[$label] Done. Archive kept at $archive (pass --cleanup to remove it after extraction)."
  fi
}

extract_zip() {
  # extract_zip <zip_file> <dest_dir>
  local zip_file="$1"
  local dest_dir="$2"
  mkdir -p "$dest_dir"

  if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "$zip_file" -d "$dest_dir"
  else
    echo "[extract_zip] unzip not found, falling back to Python's zipfile module."
    python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip_file" "$dest_dir"
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

  cleanup_archive "$archive" "CIFAR-10"
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

  cleanup_archive "$archive" "CIFAR-10-C"
}

# --- Tiny-ImageNet ---------------------------------------------------------

download_tiny_imagenet() {
  local out_dir="$DATA_DIR/tiny-imagenet"
  local archive="$DATA_DIR/tiny-imagenet-200.zip"

  if [ -d "$out_dir/tiny-imagenet-200/train" ] && [ "$FORCE" = false ]; then
    echo "[Tiny-ImageNet] Already present at $out_dir, skipping."
    return
  fi

  if [ ! -f "$archive" ] || [ "$FORCE" = true ]; then
    echo "[Tiny-ImageNet] Downloading (~248 MB) ..."
    download "$TINY_IMAGENET_URL" "$archive"
  fi

  echo "[Tiny-ImageNet] Verifying MD5 checksum ..."
  if ! check_md5 "$archive" "$TINY_IMAGENET_MD5"; then
    echo "[Tiny-ImageNet] ERROR: MD5 checksum mismatch. Delete $archive and re-run." >&2
    exit 1
  fi
  echo "[Tiny-ImageNet] Checksum OK."

  echo "[Tiny-ImageNet] Extracting to $out_dir ..."
  extract_zip "$archive" "$out_dir"

  cleanup_archive "$archive" "Tiny-ImageNet"
}

# --- Tiny-ImageNet-C ---------------------------------------------------------

download_tiny_imagenet_c() {
  local out_dir="$DATA_DIR/tiny-imagenet-c"
  local archive="$DATA_DIR/TinyImageNet-C.tar"

  if [ -d "$out_dir" ] && [ "$(ls -A "$out_dir" 2>/dev/null)" ] && [ "$FORCE" = false ]; then
    echo "[Tiny-ImageNet-C] Already present at $out_dir, skipping."
    return
  fi

  if [ ! -f "$archive" ] || [ "$FORCE" = true ]; then
    echo "[Tiny-ImageNet-C] Downloading (~2.8 GB) ..."
    download "$TINY_IMAGENET_C_URL" "$archive"
  fi

  echo "[Tiny-ImageNet-C] Verifying MD5 checksum ..."
  if ! check_md5 "$archive" "$TINY_IMAGENET_C_MD5"; then
    echo "[Tiny-ImageNet-C] ERROR: MD5 checksum mismatch. Delete $archive and re-run." >&2
    exit 1
  fi
  echo "[Tiny-ImageNet-C] Checksum OK."

  echo "[Tiny-ImageNet-C] Extracting to $out_dir ..."
  mkdir -p "$out_dir"
  # Extracted directly into out_dir (unlike CIFAR-10-C's archive, this one's exact
  # top-level folder name inside the tar hasn't been confirmed) -- if the tar contains
  # its own wrapper folder, contents will just end up one level deeper, e.g.
  # out_dir/Tiny-ImageNet-C/<corruption>/<severity>/... instead of out_dir/<corruption>/...
  # Check `ls "$out_dir"` after the first run and adjust cifar10c.py/datamodule.py if needed.
  tar -xf "$archive" -C "$out_dir"

  cleanup_archive "$archive" "Tiny-ImageNet-C"
}

# --- main ------------------------------------------------------------------

ensure_aria2c
[ "$SKIP_TINY_IMAGENET" = false ] && ensure_unzip

[ "$SKIP_CIFAR10" = false ] && download_cifar10
[ "$SKIP_CIFAR10_C" = false ] && download_cifar10_c
[ "$SKIP_TINY_IMAGENET" = false ] && download_tiny_imagenet
[ "$SKIP_TINY_IMAGENET_C" = false ] && download_tiny_imagenet_c

echo "All downloads finished."
