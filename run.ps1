$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

function Add-WinGetToolToPath {
    param(
        [string]$Command,
        [string]$PackagePattern
    )

    if (Get-Command $Command -ErrorAction SilentlyContinue) {
        return
    }

    $packageRoot = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
    $tool = Get-ChildItem $packageRoot -Directory -Filter $PackagePattern -ErrorAction SilentlyContinue |
        ForEach-Object {
            Get-ChildItem $_.FullName -File -Filter "$Command.exe" -Recurse -ErrorAction SilentlyContinue
        } |
        Select-Object -First 1
    if ($tool) {
        $env:PATH = "$($tool.DirectoryName);$env:PATH"
    }
}

Add-WinGetToolToPath -Command "ffmpeg" -PackagePattern "Gyan.FFmpeg_*"
Add-WinGetToolToPath -Command "sox" -PackagePattern "ChrisBagwell.SoX_*"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command sox -ErrorAction SilentlyContinue)) {
    throw "FFmpeg and SoX are required. Run .\setup.ps1 first."
}

& ".venv\Scripts\python.exe" app.py
