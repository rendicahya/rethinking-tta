<#
.SYNOPSIS
    Download datasets for the cost-effectiveness study of TTA under distribution
    shift (Windows PowerShell version of download.sh).

.DESCRIPTION
    Uses aria2c (multi-connection) when available, falling back to curl.exe with
    resume support, falling back to Invoke-WebRequest.

    Datasets:
    - CIFAR-10         : clean (in-distribution) data -> data/cifar10/
    - CIFAR-10-C       : 15 corruption types x 5 severities (Hendrycks & Dietterich, 2019) -> data/cifar10-c/
    - Tiny-ImageNet    : clean (in-distribution) data, 64x64, 200 classes -> data/tiny-imagenet/
    - Tiny-ImageNet-C  : 15 corruption types x 5 severities, 64x64 -> data/tiny-imagenet-c/
                         (smaller stand-in for full ImageNet-C, ~2.8GB vs ~62GB)

    Already-downloaded/extracted datasets are skipped automatically (use -Force to redo).

    Unlike download.sh, this script does not attempt to install aria2c (there's no
    single standard Windows package manager to install it with) -- if aria2c isn't
    found, it just falls back to curl.exe / Invoke-WebRequest. Install aria2c
    yourself (e.g. via winget/scoop/choco) for faster parallel downloads:
        winget install aria2.aria2

.PARAMETER Force
    Re-download even if a dataset already exists.

.PARAMETER Cleanup
    Delete downloaded archives (including leftovers from earlier runs) once
    their dataset is extracted.

.PARAMETER SkipCifar10
.PARAMETER SkipCifar10C
.PARAMETER SkipTinyImagenet
.PARAMETER SkipTinyImagenetC
    Skip the corresponding dataset.

.EXAMPLE
    .\data\download.ps1
.EXAMPLE
    .\data\download.ps1 -SkipTinyImagenet -SkipTinyImagenetC
.EXAMPLE
    .\data\download.ps1 -Force -Cleanup
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$Cleanup,
    [switch]$SkipCifar10,
    [switch]$SkipCifar10C,
    [switch]$SkipTinyImagenet,
    [switch]$SkipTinyImagenetC
)

$ErrorActionPreference = "Stop"

$DataDir = $PSScriptRoot

$Cifar10Url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
$Cifar10Md5 = "c58f30108f718f92721af3b95e74349a"

$Cifar10CUrl = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
$Cifar10CMd5 = "56bf5dcef84df0e2308c6dcbcbbd8499"

$TinyImagenetUrl = "https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1"
$TinyImagenetMd5 = "90528d7ca1a48142e341f4ef8d21d0de"

$TinyImagenetCUrl = "https://zenodo.org/records/2469796/files/TinyImageNet-C.tar?download=1"
$TinyImagenetCMd5 = "3d9c6e89c2609aeb4198f84c8edd1ff0"

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

# --- helpers -----------------------------------------------------------

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-Md5 {
    param([string]$Path)
    return (Get-FileHash -Path $Path -Algorithm MD5).Hash.ToLower()
}

function Invoke-Download {
    # Invoke-Download <url> <dest_file>
    param([string]$Url, [string]$Dest)

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dest) | Out-Null
    $destDir = Split-Path -Parent $Dest
    $destName = Split-Path -Leaf $Dest

    if (Test-CommandExists "aria2c") {
        Write-Host "[download] Using aria2c (multi-connection) for $destName"
        & aria2c -x16 -s16 -k1M -c -o $destName -d $destDir $Url
        if ($LASTEXITCODE -ne 0) { throw "aria2c failed with exit code $LASTEXITCODE" }
    }
    elseif (Test-CommandExists "curl.exe") {
        Write-Host "[download] aria2c not found, using curl.exe (single connection) for $destName"
        Write-Host "[download] Tip: install aria2c for much faster parallel downloads (winget install aria2.aria2)."
        & curl.exe -L -C - --retry 5 --retry-delay 5 -o $Dest $Url
        if ($LASTEXITCODE -ne 0) { throw "curl.exe failed with exit code $LASTEXITCODE" }
    }
    else {
        Write-Host "[download] aria2c/curl.exe not found, using Invoke-WebRequest (no resume support) for $destName"
        Invoke-WebRequest -Uri $Url -OutFile $Dest
    }
}

function Remove-ArchiveIfRequested {
    # Remove-ArchiveIfRequested <archive_file> <label>
    param([string]$Archive, [string]$Label)

    if ($Cleanup) {
        Remove-Item -Force -ErrorAction SilentlyContinue $Archive
        Write-Host "[$Label] Removed archive $Archive (-Cleanup)."
    }
    else {
        Write-Host "[$Label] Done. Archive kept at $Archive (pass -Cleanup to remove it after extraction)."
    }
}

function Expand-TarArchive {
    # Expand-TarArchive <tar_file> <dest_dir> [-Gzip]
    param([string]$TarFile, [string]$DestDir, [switch]$Gzip)

    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    if (-not (Test-CommandExists "tar")) {
        throw "'tar' not found. Windows 10 1803+ / Windows 11 ship it built in; " +
              "update Windows or install bsdtar/7-Zip and extract $TarFile to $DestDir manually."
    }
    if ($Gzip) {
        & tar -xzf $TarFile -C $DestDir
    }
    else {
        & tar -xf $TarFile -C $DestDir
    }
    if ($LASTEXITCODE -ne 0) { throw "tar extraction failed with exit code $LASTEXITCODE" }
}

function Expand-ZipArchive {
    # Expand-ZipArchive <zip_file> <dest_dir>
    param([string]$ZipFile, [string]$DestDir)

    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    Expand-Archive -Path $ZipFile -DestinationPath $DestDir -Force
}

function Confirm-Md5 {
    # Confirm-Md5 <file> <expected_md5> <label> -- exits the script on mismatch.
    param([string]$File, [string]$Expected, [string]$Label)

    Write-Host "[$Label] Verifying MD5 checksum ..."
    $actual = Get-Md5 -Path $File
    if ($actual -ne $Expected) {
        Write-Error "[$Label] ERROR: MD5 checksum mismatch (got $actual, expected $Expected). Delete $File and re-run."
        exit 1
    }
    Write-Host "[$Label] Checksum OK."
}

# --- CIFAR-10 ------------------------------------------------------------

function Get-Cifar10 {
    $outDir = Join-Path $DataDir "cifar10"
    $archive = Join-Path $DataDir "cifar-10-python.tar.gz"

    if ((Test-Path (Join-Path $outDir "cifar-10-batches-py")) -and (-not $Force)) {
        Write-Host "[CIFAR-10] Already present at $outDir, skipping."
        if (Test-Path $archive) { Remove-ArchiveIfRequested -Archive $archive -Label "CIFAR-10" }
        return
    }

    if ((-not (Test-Path $archive)) -or $Force) {
        Write-Host "[CIFAR-10] Downloading (~170 MB) ..."
        Invoke-Download -Url $Cifar10Url -Dest $archive
    }

    Confirm-Md5 -File $archive -Expected $Cifar10Md5 -Label "CIFAR-10"

    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host "[CIFAR-10] Extracting to $outDir ..."
    Expand-TarArchive -TarFile $archive -DestDir $outDir -Gzip

    Remove-ArchiveIfRequested -Archive $archive -Label "CIFAR-10"
}

# --- CIFAR-10-C ------------------------------------------------------------

function Get-Cifar10C {
    $outDir = Join-Path $DataDir "cifar10-c"
    $archive = Join-Path $DataDir "CIFAR-10-C.tar"

    if ((Test-Path (Join-Path $outDir "labels.npy")) -and (-not $Force)) {
        Write-Host "[CIFAR-10-C] Already present at $outDir, skipping."
        if (Test-Path $archive) { Remove-ArchiveIfRequested -Archive $archive -Label "CIFAR-10-C" }
        return
    }

    if ((-not (Test-Path $archive)) -or $Force) {
        Write-Host "[CIFAR-10-C] Downloading (~2.9 GB) ..."
        Invoke-Download -Url $Cifar10CUrl -Dest $archive
    }

    Confirm-Md5 -File $archive -Expected $Cifar10CMd5 -Label "CIFAR-10-C"

    Write-Host "[CIFAR-10-C] Extracting to $outDir ..."
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Expand-TarArchive -TarFile $archive -DestDir $DataDir

    # The archive extracts into a "CIFAR-10-C/" folder; move contents to outDir.
    $extractedDir = Join-Path $DataDir "CIFAR-10-C"
    if ((Test-Path $extractedDir) -and ($extractedDir -ne $outDir)) {
        Get-ChildItem -Path $extractedDir | Move-Item -Destination $outDir -Force
        Remove-Item -Force -Recurse $extractedDir
    }

    Remove-ArchiveIfRequested -Archive $archive -Label "CIFAR-10-C"
}

# --- Tiny-ImageNet ---------------------------------------------------------

function Get-TinyImagenet {
    $outDir = Join-Path $DataDir "tiny-imagenet"
    $archive = Join-Path $DataDir "tiny-imagenet-200.zip"

    if ((Test-Path (Join-Path $outDir "tiny-imagenet-200\train")) -and (-not $Force)) {
        Write-Host "[Tiny-ImageNet] Already present at $outDir, skipping."
        if (Test-Path $archive) { Remove-ArchiveIfRequested -Archive $archive -Label "Tiny-ImageNet" }
        return
    }

    if ((-not (Test-Path $archive)) -or $Force) {
        Write-Host "[Tiny-ImageNet] Downloading (~248 MB) ..."
        Invoke-Download -Url $TinyImagenetUrl -Dest $archive
    }

    Confirm-Md5 -File $archive -Expected $TinyImagenetMd5 -Label "Tiny-ImageNet"

    Write-Host "[Tiny-ImageNet] Extracting to $outDir ..."
    Expand-ZipArchive -ZipFile $archive -DestDir $outDir

    Remove-ArchiveIfRequested -Archive $archive -Label "Tiny-ImageNet"
}

# --- Tiny-ImageNet-C ---------------------------------------------------------

function Get-TinyImagenetC {
    $outDir = Join-Path $DataDir "tiny-imagenet-c"
    $archive = Join-Path $DataDir "TinyImageNet-C.tar"

    if ((Test-Path $outDir) -and (Get-ChildItem -Path $outDir -ErrorAction SilentlyContinue) -and (-not $Force)) {
        Write-Host "[Tiny-ImageNet-C] Already present at $outDir, skipping."
        if (Test-Path $archive) { Remove-ArchiveIfRequested -Archive $archive -Label "Tiny-ImageNet-C" }
        return
    }

    if ((-not (Test-Path $archive)) -or $Force) {
        Write-Host "[Tiny-ImageNet-C] Downloading (~2.8 GB) ..."
        Invoke-Download -Url $TinyImagenetCUrl -Dest $archive
    }

    Confirm-Md5 -File $archive -Expected $TinyImagenetCMd5 -Label "Tiny-ImageNet-C"

    Write-Host "[Tiny-ImageNet-C] Extracting to $outDir ..."
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    # Extracted directly into outDir (unlike CIFAR-10-C's archive, this one's exact
    # top-level folder name inside the tar hasn't been confirmed) -- if the tar contains
    # its own wrapper folder, contents will just end up one level deeper, e.g.
    # outDir/Tiny-ImageNet-C/<corruption>/<severity>/... instead of outDir/<corruption>/...
    # Check the folder contents after the first run and adjust cifar10c.py/datamodule.py if needed.
    Expand-TarArchive -TarFile $archive -DestDir $outDir

    Remove-ArchiveIfRequested -Archive $archive -Label "Tiny-ImageNet-C"
}

# --- main ------------------------------------------------------------------

if (-not $SkipCifar10) { Get-Cifar10 }
if (-not $SkipCifar10C) { Get-Cifar10C }
if (-not $SkipTinyImagenet) { Get-TinyImagenet }
if (-not $SkipTinyImagenetC) { Get-TinyImagenetC }

Write-Host "All downloads finished."
