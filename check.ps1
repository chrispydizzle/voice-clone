$ErrorActionPreference = "Stop"

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run .\setup.ps1 -Dev first."
}

& $python -m pip check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -m compileall -q app.py voice_clone tests
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -m pytest -q
exit $LASTEXITCODE

