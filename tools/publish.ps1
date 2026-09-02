# publish.ps1: publishes EVERYTHING in one gesture (started by Publish.bat):
#   re-hash (dvc add) -> commit and push the pointers -> dvc push (data, for the server)
#   -> readable copies of inputs and output_view (for EPM View)
# Auto-detects the repo and the branch. Reads the keys from tools/.env (gitignored).
#
# With no argument (double-click on Publish.bat): same behaviour as always, all is published.
# The uploads only send what the store is missing or holds older (see r2_sync).
#
# Options, for the command line:
#   -Check          compare local and remote, send nothing, commit nothing
#   -Force          send everything again, even what is already up to date
#   -SkipData       leave the inputs alone (no dvc add, no dvc push)
#   -SkipResults    leave epm/output_view alone
#   -Only <pattern> restrict the results to a glob (e.g. "simulations_run_*/npv_external.csv")

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

# --- 0. options passed on to the uploaders ---
$syncArgs = @()
if ($Check) { $syncArgs += "--check" }
if ($Force) { $syncArgs += "--force" }

# --- 1. load the keys from tools/.env ---
$envFile = Join-Path $TOOLS ".env"
if (-not (Test-Path $envFile)) {
    Write-Error "Missing $envFile  ->  copy tools/.env.example to tools/.env and put your keys in it."
    exit 1
}
Get-Content $envFile | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
}

# --- 2. auto-detect repo and branch ---
Set-Location $REPO
$env:EPM_REPO   = $REPO
$env:EPM_BRANCH = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Host ""
Write-Host "Repo   : $REPO" -ForegroundColor DarkGray
Write-Host "Branch : $($env:EPM_BRANCH)" -ForegroundColor DarkGray
if ($Check) { Write-Host "Mode : CHECK (nothing will be sent or committed)" -ForegroundColor Yellow }

# --- 3. for each data folder tracked by DVC: re-hash + readable copy ---
if ($SkipData) {
    Write-Host ""
    Write-Host "[data] -SkipData -> inputs left alone" -ForegroundColor DarkGray
} else {
    $pointers = Get-ChildItem "epm\input" -Filter "data_*.dvc" -ErrorAction SilentlyContinue
    if (-not $pointers) { Write-Host "No data_*.dvc folder (model not migrated to DVC yet?)" -ForegroundColor Yellow }
    foreach ($p in $pointers) {
        $folder = $p.BaseName    # e.g. data_blacksea
        Write-Host ""
        Write-Host "[data] $folder : dvc add + readable upload ..." -ForegroundColor Cyan
        if (-not $Check) { dvc add "epm/input/$folder" }
        $env:EPM_DATA_FOLDER = $folder
        python "$TOOLS\upload_to_r2.py" @syncArgs
        if ($LASTEXITCODE -ne 0) { Write-Error "upload_to_r2.py reported failures (see above)." ; exit 1 }
    }

    # --- 4. commit and push the pointers (when they changed) ---
    if (-not $Check) {
        Write-Host ""
        Write-Host "[git] commit + push the pointers (when they changed) ..." -ForegroundColor Cyan
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
        Write-Host "[dvc] push the data to the store (for the server) ..." -ForegroundColor Cyan
        dvc push -r r2 --jobs 8
    }
}

# --- 6. readable results -> store (for EPM View) ---
if ($SkipResults) {
    Write-Host ""
    Write-Host "[results] -SkipResults -> epm/output_view left alone" -ForegroundColor DarkGray
} else {
    Write-Host ""
    Write-Host "[results] upload epm/output_view -> store (for EPM View) ..." -ForegroundColor Cyan
    $resArgs = $syncArgs
    if ($Only) { $resArgs = $resArgs + @("--only", $Only) }
    python "$TOOLS\upload_output_view_to_r2.py" @resArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "upload_output_view_to_r2.py reported failures (see above)." ; exit 1 }
}

Write-Host ""
if ($Check) {
    Write-Host "OK - check done, nothing was sent." -ForegroundColor Green
} else {
    Write-Host "OK - published: GitHub (pointers) + store (DVC + readable for EPM View)." -ForegroundColor Green
}
