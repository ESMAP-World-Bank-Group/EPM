# sync.ps1: fetches code and data (started by Sync.bat on Windows).
#   git pull   (code and pointers)
#   dvc pull   (data from the store)
$ErrorActionPreference = "Stop"
$TOOLS = $PSScriptRoot
$REPO  = Split-Path $TOOLS -Parent

$envFile = Join-Path $TOOLS ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
        $k, $v = $_ -split '=', 2
        Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
    }
}
Set-Location $REPO
Write-Host "[1/2] git pull ..." -ForegroundColor Cyan
git pull
Write-Host "[2/2] dvc pull ..." -ForegroundColor Cyan
dvc pull
Write-Host "OK - code and data up to date." -ForegroundColor Green
