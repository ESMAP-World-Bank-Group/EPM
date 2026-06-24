# setup_model.ps1 — onboard a model's data to the private store (one-time, per model).
#
#   Usage:   powershell -File tools/setup_model.ps1 data_<model>
#   Example: powershell -File tools/setup_model.ps1 data_sapp
#
# Moves a data folder OUT of git into DVC (the store). Files stay on your disk.
# This script only makes LOCAL changes — review them, then publish with Publish.bat.

param([Parameter(Mandatory = $true)][string]$DataFolder)

$ErrorActionPreference = "Stop"
$TOOLS = $PSScriptRoot
$REPO  = Split-Path $TOOLS -Parent
Set-Location $REPO

$path = "epm/input/$DataFolder"
if (-not (Test-Path $path)) { Write-Error "Not found: $path"; exit 1 }

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Host ""
Write-Host "Repo   : $REPO"            -ForegroundColor DarkGray
Write-Host "Branch : $branch"          -ForegroundColor DarkGray
Write-Host "Folder : $path"            -ForegroundColor DarkGray
Write-Host ""
Write-Host "This moves '$path' OUT of git into DVC/the store." -ForegroundColor Yellow
Write-Host "Files STAY on your disk; on your next push the data leaves the public repo." -ForegroundColor Yellow
if ((Read-Host "Continue? (y/N)") -ne "y") { Write-Host "Aborted."; exit 0 }

# 1. DVC init (once per repo)
if (-not (Test-Path ".dvc")) {
    Write-Host "[1/4] dvc init"
    dvc init | Out-Null
    Write-Host "      /!\ remote not configured. Run once, then re-run this script:" -ForegroundColor DarkYellow
    Write-Host "          dvc remote add -d store s3://<bucket>/dvcstore" -ForegroundColor DarkYellow
    Write-Host "          dvc remote modify store endpointurl <endpoint>" -ForegroundColor DarkYellow
} else {
    Write-Host "[1/4] dvc already initialised"
}

# 2. .gitignore: drop the whitelist of the folder, keep only its .dvc pointer tracked
Write-Host "[2/4] update .gitignore"
$gi  = ".gitignore"
$esc = [regex]::Escape("epm/input/$DataFolder")
$lines = Get-Content $gi | Where-Object { $_ -notmatch "^\!$esc(/|$)" }   # remove !epm/input/<folder>/  and  /**
$ptr = "!epm/input/$DataFolder.dvc"
if ($lines -notcontains $ptr) {
    # make sure the data folder is ignored at all (covers the 'epm/input/*' pattern repos)
    if (-not (Select-String -Path $gi -SimpleMatch "epm/input/*" -Quiet)) { $lines += "epm/input/$DataFolder/" }
    $lines += $ptr
}
Set-Content $gi $lines

# 3+4. stop tracking in git (files stay on disk), then hand the folder to DVC
Write-Host "[3/4] git rm --cached (untrack, files stay on disk)"
git rm -r --cached $path | Out-Null
Write-Host "[4/4] dvc add (create the pointer)"
dvc add $path

Write-Host ""
Write-Host "DONE (local changes only)." -ForegroundColor Green
Write-Host "Next:" -ForegroundColor Green
Write-Host "  1. review        : git status"
Write-Host "  2. publish       : double-click Publish.bat"
Write-Host "  3. EPM View      : add '$branch' to R2_BRANCHES in"
Write-Host "                     epm-data-explorer/src/utils/epmFetch.js"
