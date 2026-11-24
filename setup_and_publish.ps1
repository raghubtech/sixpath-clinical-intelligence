param([string]$RemoteUrl="",[switch]$SkipRun,[switch]$OpenStreamlitCloud)
$ErrorActionPreference="Stop"
Set-Location $PSScriptRoot
Write-Host "=== Setup, train and publish to GitHub ===" -ForegroundColor Cyan
if(-not (Get-Command python -ErrorAction SilentlyContinue)){throw "Install Python 3.11 and enable Add Python to PATH."}
if(-not (Get-Command git -ErrorAction SilentlyContinue)){throw "Install Git for Windows: https://git-scm.com/download/win"}
if(-not (Test-Path ".venv\Scripts\python.exe")){python -m venv .venv}
$Py=Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip
if($LASTEXITCODE -ne 0){throw "pip upgrade failed"}
& $Py -m pip install -r requirements.txt
if($LASTEXITCODE -ne 0){throw "Dependency installation failed"}
Write-Host "Training/evaluating six outputs..." -ForegroundColor Yellow
& $Py train_model.py
if($LASTEXITCODE -ne 0){throw "Model training failed"}
if(-not (Test-Path ".git")){git init}
git branch -M main
git add .
if(git status --porcelain){git commit -m "Add multi-organ clinical risk intelligence project";if($LASTEXITCODE -ne 0){throw "Git commit failed. Configure git user.name and user.email."}}else{Write-Host "No new changes to commit."}
if(-not $RemoteUrl){$RemoteUrl=Read-Host "Paste the URL of your EMPTY GitHub repository"}
if(-not $RemoteUrl){throw "GitHub repository URL is required"}
$remotes=git remote
if($remotes -contains "origin"){git remote set-url origin $RemoteUrl}else{git remote add origin $RemoteUrl}
Write-Host "Pushing main to GitHub..." -ForegroundColor Yellow
git push -u origin main
if($LASTEXITCODE -ne 0){throw "Push failed. Verify repo URL and sign in with Git Credential Manager; then retry git push -u origin main."}
Write-Host "GitHub push completed." -ForegroundColor Green
if($OpenStreamlitCloud){Start-Process "https://share.streamlit.io/";Write-Host "On Streamlit Cloud choose your repository, branch main, app file app.py, then click Deploy." -ForegroundColor Yellow}
if(-not $SkipRun){Write-Host "Launching local app. Keep this PowerShell window open." -ForegroundColor Green;& $Py -m streamlit run app.py}
