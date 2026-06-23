# publish.ps1 — publie TOUT en un geste (lance par Publish.bat) :
#   re-hash (dvc add) -> commit+push des pointeurs -> dvc push (donnees, pour le serveur)
#   -> copies lisibles inputs + output_view (pour EPM View)
# Auto-detecte le repo et la branche. Lit les cles dans tools/.env (gitignore).

$ErrorActionPreference = "Stop"
$TOOLS = $PSScriptRoot
$REPO  = Split-Path $TOOLS -Parent

# --- 1. charger les cles depuis tools/.env ---
$envFile = Join-Path $TOOLS ".env"
if (-not (Test-Path $envFile)) {
    Write-Error "Manque $envFile  ->  copie tools/.env.example en tools/.env et mets tes cles."
    exit 1
}
Get-Content $envFile | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
}

# --- 2. auto-detect repo + branche ---
Set-Location $REPO
$env:EPM_REPO   = $REPO
$env:EPM_BRANCH = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Host ""
Write-Host "Repo : $REPO" -ForegroundColor DarkGray
Write-Host "Branche : $($env:EPM_BRANCH)" -ForegroundColor DarkGray

# --- 3. pour chaque dossier de donnees suivi par DVC : re-hash + copie lisible ---
$pointers = Get-ChildItem "epm\input" -Filter "data_*.dvc" -ErrorAction SilentlyContinue
if (-not $pointers) { Write-Host "Aucun dossier data_*.dvc (modele pas encore migre vers DVC ?)" -ForegroundColor Yellow }
foreach ($p in $pointers) {
    $folder = $p.BaseName    # ex: data_blacksea
    Write-Host ""
    Write-Host "[data] $folder : dvc add + upload lisible ..." -ForegroundColor Cyan
    dvc add "epm/input/$folder"
    $env:EPM_DATA_FOLDER = $folder
    python "$TOOLS\upload_to_r2.py"
}

# --- 4. commit + push des pointeurs (si change) ---
Write-Host ""
Write-Host "[git] commit + push des pointeurs (si change) ..." -ForegroundColor Cyan
git add epm/input/*.dvc
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "update data ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))"
    git push
} else {
    Write-Host "   (pointeurs inchanges -> rien a committer)" -ForegroundColor DarkGray
}

# --- 5. donnees -> store (DVC, pour le serveur) ---
Write-Host ""
Write-Host "[dvc] push des donnees vers le store (pour le serveur) ..." -ForegroundColor Cyan
dvc push

# --- 6. resultats lisibles -> store (pour EPM View) ---
Write-Host ""
Write-Host "[results] upload epm/output_view -> store (pour EPM View) ..." -ForegroundColor Cyan
python "$TOOLS\upload_output_view_to_r2.py"

Write-Host ""
Write-Host "OK - publie : GitHub (pointeurs) + store (DVC + lisible pour EPM View)." -ForegroundColor Green
