<#
.SYNOPSIS
    Download pretrained CIFAR-10 checkpoints (Windows PowerShell version of download.sh).

.DESCRIPTION
    1. ResNet-50, trained by Eduardo Dadalto:
       https://huggingface.co/edadaltocg/resnet50_cifar10 (MIT license, self-reported
       test accuracy 0.9465). Its architecture (conv1: 3x3 stride 1, no maxpool, fc: 10
       classes) matches model.py's build_resnet50, so it loads directly via
       model.load_state_dict() -- see model.py's build_model().
       NOTE: data/datamodule.py's default CIFAR10_STD was set to match the
       normalization this checkpoint was trained with.

    2. WideResNet-28-10, RobustBench's "Standard" CIFAR-10 model:
       https://github.com/RobustBench/robustbench (MIT license, ~94.8% clean test
       accuracy). Hosted on Google Drive (RobustBench doesn't mirror it elsewhere).
       Its architecture matches model.py's WideResNet, and it expects RAW [0,1]
       pixel inputs (no mean/std normalization) -- the wideresnet2810 configs in
       config/ set data.mean=(0,0,0), data.std=(1,1,1) accordingly.

.PARAMETER Force
    Re-download even if the checkpoint file already exists.

.PARAMETER SkipResnet50
    Don't download the ResNet-50 checkpoint.

.PARAMETER SkipWideresnet2810
    Don't download the WideResNet-28-10 checkpoint.

.EXAMPLE
    .\checkpoints\download.ps1
.EXAMPLE
    .\checkpoints\download.ps1 -Force
.EXAMPLE
    .\checkpoints\download.ps1 -SkipWideresnet2810
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$SkipResnet50,
    [switch]$SkipWideresnet2810
)

$ErrorActionPreference = "Stop"

$CheckpointsDir = $PSScriptRoot

$Resnet50Url    = "https://huggingface.co/edadaltocg/resnet50_cifar10/resolve/main/pytorch_model.bin?download=true"
$Resnet50Sha256 = "600e3f79fa41bc7c91d751a65b29b1ed2733345f8b6908381caa79a32ff28a6e"
$Resnet50Dest   = Join-Path $CheckpointsDir "resnet-50-cifar-10.pt"

# RobustBench "Standard" WRN-28-10 CIFAR-10 checkpoint, Google Drive file ID.
# No published checksum exists for this file (unlike the HuggingFace download
# above), so it's only sanity-checked (size + not-HTML) below.
$Wrn2810GDriveId = "1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC"
$Wrn2810Dest      = Join-Path $CheckpointsDir "wideresnet-28-10-cifar-10.pt"
$Wrn2810MinBytes  = 100000000  # ~100 MB; the real file is ~140 MB

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLower()
}

function Invoke-Download {
    # Simple resumable-ish download: prefers aria2c, falls back to curl.exe,
    # falls back to Invoke-WebRequest.
    param(
        [string]$Url,
        [string]$Dest
    )

    $destDir = Split-Path -Parent $Dest
    $destName = Split-Path -Leaf $Dest

    if (Test-CommandExists "aria2c") {
        & aria2c -x16 -s16 -k1M -c -o $destName -d $destDir $Url
        if ($LASTEXITCODE -ne 0) { throw "aria2c failed with exit code $LASTEXITCODE" }
    }
    elseif (Test-CommandExists "curl.exe") {
        & curl.exe -L -C - --retry 5 --retry-delay 5 -o $Dest $Url
        if ($LASTEXITCODE -ne 0) { throw "curl failed with exit code $LASTEXITCODE" }
    }
    else {
        Invoke-WebRequest -Uri $Url -OutFile $Dest
    }
}

function Get-Resnet50Checkpoint {
    if ((Test-Path $Resnet50Dest) -and (-not $Force)) {
        Write-Host "[resnet-50-cifar-10] Already present at $Resnet50Dest, skipping."
        return
    }

    New-Item -ItemType Directory -Force -Path $CheckpointsDir | Out-Null
    Write-Host "[resnet-50-cifar-10] Downloading (~94 MB) ..."
    Invoke-Download -Url $Resnet50Url -Dest $Resnet50Dest

    Write-Host "[resnet-50-cifar-10] Verifying SHA-256 checksum ..."
    $actual = Get-Sha256 -Path $Resnet50Dest
    if ($actual -ne $Resnet50Sha256) {
        Write-Error "[resnet-50-cifar-10] ERROR: checksum mismatch (got $actual). Delete $Resnet50Dest and re-run."
        exit 1
    }

    Write-Host "[resnet-50-cifar-10] Checksum OK. Saved to $Resnet50Dest"
}

function Get-GoogleDriveFile {
    # Prefers `uvx gdown` (this project already uses uv, and gdown handles Google
    # Drive's confirmation-token/virus-scan-warning redirect for large files
    # properly). Falls back to a curl.exe cookie-jar trick if uvx/gdown isn't
    # usable. Google Drive downloads are inherently less reliable than a direct
    # HTTPS host (no published checksum either), so the caller must sanity-check
    # the result.
    param(
        [string]$FileId,
        [string]$Dest
    )

    if (Test-CommandExists "uvx") {
        Write-Host "[Get-GoogleDriveFile] Trying 'uvx gdown' ..."
        & uvx gdown $FileId -O $Dest
        if ($LASTEXITCODE -eq 0) { return }
        Write-Warning "[Get-GoogleDriveFile] uvx gdown failed, falling back to curl.exe ..."
    }
    else {
        Write-Warning "[Get-GoogleDriveFile] uvx not found, falling back to curl.exe ..."
    }

    if (-not (Test-CommandExists "curl.exe")) {
        throw "Neither 'uvx gdown' nor 'curl.exe' is available. Install uv (https://docs.astral.sh/uv/) " +
              "or curl, or download the file manually from https://drive.google.com/uc?id=$FileId"
    }

    $cookieJar = New-TemporaryFile
    $confirmPage = "$Dest.confirm.html"

    try {
        & curl.exe -sc $cookieJar "https://drive.google.com/uc?export=download&id=$FileId" -o $confirmPage
        $confirmCode = $null
        if (Test-Path $confirmPage) {
            $match = Select-String -Path $confirmPage -Pattern 'confirm=([a-zA-Z0-9_-]+)' | Select-Object -First 1
            if ($match) { $confirmCode = $match.Matches[0].Groups[1].Value }
            Remove-Item $confirmPage -ErrorAction SilentlyContinue
        }

        if ($confirmCode) {
            & curl.exe -Lb $cookieJar "https://drive.google.com/uc?export=download&confirm=$confirmCode&id=$FileId" -o $Dest
        }
        else {
            # Small files (no virus-scan warning) download directly without a confirm token.
            & curl.exe -Lb $cookieJar "https://drive.google.com/uc?export=download&id=$FileId" -o $Dest
        }
        if ($LASTEXITCODE -ne 0) { throw "curl.exe failed with exit code $LASTEXITCODE" }
    }
    finally {
        Remove-Item $cookieJar -ErrorAction SilentlyContinue
    }
}

function Get-Wideresnet2810Checkpoint {
    if ((Test-Path $Wrn2810Dest) -and (-not $Force)) {
        Write-Host "[wideresnet-28-10-cifar-10] Already present at $Wrn2810Dest, skipping."
        return
    }

    New-Item -ItemType Directory -Force -Path $CheckpointsDir | Out-Null
    Write-Host "[wideresnet-28-10-cifar-10] Downloading from Google Drive (~140 MB) ..."
    Get-GoogleDriveFile -FileId $Wrn2810GDriveId -Dest $Wrn2810Dest

    Write-Host "[wideresnet-28-10-cifar-10] Sanity-checking the download ..."
    if (-not (Test-Path $Wrn2810Dest)) {
        Write-Error "[wideresnet-28-10-cifar-10] ERROR: download failed, no file at $Wrn2810Dest."
        exit 1
    }

    $size = (Get-Item $Wrn2810Dest).Length
    if ($size -eq 0) {
        Write-Error "[wideresnet-28-10-cifar-10] ERROR: download is empty. Delete $Wrn2810Dest and re-run."
        exit 1
    }

    if ($size -lt $Wrn2810MinBytes) {
        $head = Get-Content -Path $Wrn2810Dest -TotalCount 5 -ErrorAction SilentlyContinue
        if ($head -join "" -match '(?i)<html') {
            Write-Error ("[wideresnet-28-10-cifar-10] ERROR: got an HTML page instead of the checkpoint " +
                "(Google Drive quota/confirmation issue). Delete $Wrn2810Dest and retry, or download it " +
                "manually from https://drive.google.com/uc?id=$Wrn2810GDriveId")
        }
        else {
            Write-Error ("[wideresnet-28-10-cifar-10] ERROR: file is only $size bytes " +
                "(expected >= $Wrn2810MinBytes). Delete $Wrn2810Dest and re-run.")
        }
        exit 1
    }

    Write-Host "[wideresnet-28-10-cifar-10] Looks OK ($size bytes, no published checksum to verify against)."
    Write-Host "[wideresnet-28-10-cifar-10] Saved to $Wrn2810Dest"
}

if (-not $SkipResnet50) { Get-Resnet50Checkpoint }
if (-not $SkipWideresnet2810) { Get-Wideresnet2810Checkpoint }
