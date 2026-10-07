# publish.ps1 - publishes EVERYTHING in one go (launched by Publish.bat):
#   re-hash (dvc add) -> commit+push the pointers -> dvc push (data, for the server)
#   -> readable copies of inputs + output_view (for EPM View)
# Auto-detects the repo and the branch. Reads the keys from tools/.env (gitignored).

$ErrorActionPreference = "Stop"
$TOOLS = $PSScriptRoot
$REPO  = Split-Path $TOOLS -Parent

# --- 1. load the keys from tools/.env ---
$envFile = Join-Path $TOOLS ".env"
if (-not (Test-Path $envFile)) {
    Write-Error "Missing $envFile  ->  copy tools/.env.example to tools/.env and fill in your keys."
    exit 1
}
Get-Content $envFile | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
}

# --- 2. auto-detect repo + branch ---
Set-Location $REPO
$env:EPM_REPO   = $REPO
$env:EPM_BRANCH = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Host ""
Write-Host "Repo : $REPO" -ForegroundColor DarkGray
Write-Host "Branch : $($env:EPM_BRANCH)" -ForegroundColor DarkGray

# --- 3. for each DVC-tracked data folder: re-hash + readable copy ---
$pointers = Get-ChildItem "epm\input" -Filter "data_*.dvc" -ErrorAction SilentlyContinue
if (-not $pointers) { Write-Host "No data_*.dvc folder found (model not migrated to DVC yet?)" -ForegroundColor Yellow }
foreach ($p in $pointers) {
    $folder = $p.BaseName    # e.g. data_blacksea
    Write-Host ""
    Write-Host "[data] $folder : dvc add + readable upload ..." -ForegroundColor Cyan
    dvc add "epm/input/$folder"
    $env:EPM_DATA_FOLDER = $folder
    python "$TOOLS\upload_to_r2.py"
}

# --- 4. commit + push the pointers (if changed) ---
Write-Host ""
Write-Host "[git] commit + push the pointers (if changed) ..." -ForegroundColor Cyan
git add epm/input/*.dvc
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "update data ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))"
    git push
} else {
    Write-Host "   (pointers unchanged -> nothing to commit)" -ForegroundColor DarkGray
}

# --- 5. data -> store (DVC, for the server) ---
Write-Host ""
Write-Host "[dvc] pushing the data to the store (for the server) ..." -ForegroundColor Cyan
dvc push

# --- 6. readable results -> store (for EPM View) ---
Write-Host ""
Write-Host "[results] upload epm/output_view -> store (for EPM View) ..." -ForegroundColor Cyan
python "$TOOLS\upload_output_view_to_r2.py"

Write-Host ""
Write-Host "OK - published: GitHub (pointers) + store (DVC + readable for EPM View)." -ForegroundColor Green
