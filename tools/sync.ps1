# sync.ps1 — recupere code + donnees (lance par Sync.bat sur Windows).
#   git pull   (code + pointeurs)
#   dvc pull   (donnees depuis le store)
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
Write-Host "OK - code + donnees a jour." -ForegroundColor Green
