param([switch]$InstallOnly,[switch]$TrainOnly)
$ErrorActionPreference="Stop"
Set-Location $PSScriptRoot
Write-Host "=== Multi-Organ Clinical Risk Intelligence ===" -ForegroundColor Cyan
if(-not (Get-Command python -ErrorAction SilentlyContinue)){throw "Python not found. Install Python 3.11 and enable Add Python to PATH."}
if(-not (Test-Path ".venv\Scripts\python.exe")){python -m venv .venv}
$Py=Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip
if($LASTEXITCODE -ne 0){throw "pip upgrade failed"}
& $Py -m pip install -r requirements.txt
if($LASTEXITCODE -ne 0){throw "Dependency installation failed"}
if($InstallOnly){Write-Host "Setup complete. Run .\launch.ps1" -ForegroundColor Green;Read-Host "Press Enter to close";exit 0}
if($TrainOnly){& $Py train_model.py;if($LASTEXITCODE -ne 0){throw "Training failed"};Read-Host "Press Enter to close";exit 0}
Write-Host "Starting Streamlit. Keep this window open." -ForegroundColor Green
& $Py -m streamlit run app.py
