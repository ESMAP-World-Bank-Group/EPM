# publish.ps1 — publie TOUT en un geste (lance par Publish.bat) :
#   re-hash (dvc add) -> commit+push des pointeurs -> dvc push (donnees, pour le serveur)
#   -> copies lisibles inputs + output_view (pour EPM View)
# Auto-detecte le repo et la branche. Lit les cles dans tools/.env (gitignore).
#
# Sans argument (double-clic sur Publish.bat) : comportement inchange, tout est publie.
# Les uploads ne renvoient que ce qui manque au store ou y est plus vieux (voir r2_sync).
#
# Options, pour la ligne de commande :
#   -Check          compare local et distant, n'envoie rien, ne commit rien
#   -Force          renvoie tout, meme ce qui est deja a jour
#   -SkipData       ne touche pas aux inputs (ni dvc add, ni dvc push)
#   -SkipResults    ne touche pas a epm/output_view
#   -Only <motif>   restreint les resultats a un glob (ex: "simulations_run_*/npv_external.csv")

param(
    [switch]$Check,
    [switch]$Force,
    [switch]$SkipData,
    [switch]$SkipResults,
    [string]$Only
)

$ErrorActionPreference = "Stop"
$TOOLS = $PSScriptRoot
$REPO  = Split-Path $TOOLS -Parent

# --- 0. options passees aux uploaders ---
$syncArgs = @()
if ($Check) { $syncArgs += "--check" }
if ($Force) { $syncArgs += "--force" }

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
if ($Check) { Write-Host "Mode : CHECK (rien ne sera envoye ni commite)" -ForegroundColor Yellow }

# --- 3. pour chaque dossier de donnees suivi par DVC : re-hash + copie lisible ---
if ($SkipData) {
    Write-Host ""
    Write-Host "[data] -SkipData -> inputs ignores" -ForegroundColor DarkGray
} else {
    $pointers = Get-ChildItem "epm\input" -Filter "data_*.dvc" -ErrorAction SilentlyContinue
    if (-not $pointers) { Write-Host "Aucun dossier data_*.dvc (modele pas encore migre vers DVC ?)" -ForegroundColor Yellow }
    foreach ($p in $pointers) {
        $folder = $p.BaseName    # ex: data_blacksea
        Write-Host ""
        Write-Host "[data] $folder : dvc add + upload lisible ..." -ForegroundColor Cyan
        if (-not $Check) { dvc add "epm/input/$folder" }
        $env:EPM_DATA_FOLDER = $folder
        python "$TOOLS\upload_to_r2.py" @syncArgs
        if ($LASTEXITCODE -ne 0) { Write-Error "upload_to_r2.py a signale des echecs (voir ci-dessus)." ; exit 1 }
    }

    # --- 4. commit + push des pointeurs (si change) ---
    if (-not $Check) {
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
        dvc push -r r2 --jobs 8
    }
}

# --- 6. resultats lisibles -> store (pour EPM View) ---
if ($SkipResults) {
    Write-Host ""
    Write-Host "[results] -SkipResults -> epm/output_view ignore" -ForegroundColor DarkGray
} else {
    Write-Host ""
    Write-Host "[results] upload epm/output_view -> store (pour EPM View) ..." -ForegroundColor Cyan
    $resArgs = $syncArgs
    if ($Only) { $resArgs = $resArgs + @("--only", $Only) }
    python "$TOOLS\upload_output_view_to_r2.py" @resArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "upload_output_view_to_r2.py a signale des echecs (voir ci-dessus)." ; exit 1 }
}

Write-Host ""
if ($Check) {
    Write-Host "OK - check termine, rien n'a ete envoye." -ForegroundColor Green
} else {
    Write-Host "OK - publie : GitHub (pointeurs) + store (DVC + lisible pour EPM View)." -ForegroundColor Green
}
