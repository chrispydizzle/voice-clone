[CmdletBinding()]
param(
    [switch]$Dev,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"

function Install-SystemPackage {
    param(
        [string]$Command,
        [string]$PackageId
    )

    if (Get-Command $Command -ErrorAction SilentlyContinue) {
        return
    }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "$Command is required. Install $PackageId with WinGet, then rerun setup."
    }

    & winget install --id $PackageId --exact --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $PackageId with WinGet."
    }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher is required. Install 64-bit Python 3.12 from python.org."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 is required. Install it and rerun setup."
    }
}

$venvPython = Resolve-Path ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip

Install-SystemPackage -Command "ffmpeg" -PackageId "Gyan.FFmpeg"
Install-SystemPackage -Command "sox" -PackageId "ChrisBagwell.SoX"

if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "Installing CUDA-enabled PyTorch..."
    & $venvPython -m pip install "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu126
} else {
    Write-Host "Installing CPU-only PyTorch..."
    & $venvPython -m pip install "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installation failed."
}

$requirements = if ($Dev) { "requirements-dev.txt" } else { "requirements.txt" }
& $venvPython -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Python dependency installation failed."
}

Write-Host ""
if ($Dev) {
    Write-Host "Development setup complete. Verify with: .\check.ps1"
} else {
    Write-Host "Setup complete. Start the app with: .\run.ps1"
}
