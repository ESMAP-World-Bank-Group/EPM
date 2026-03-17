# EPM Installer Script
# Clones the EPM repo, sets up conda environment, and creates a desktop launcher.

$REPO_URL = "https://github.com/ESMAP-World-Bank-Group/EPM.git"
$REPO_BRANCH = "main"
$ENV_NAME = "esmap_env"
$PYTHON_VERSION = "3.10"
$MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$MINICONDA_INSTALLER = "$env:TEMP\Miniconda3-installer.exe"

# ── Helpers ──────────────────────────────────────────────────────────────────

function Write-Step($msg) {
    Write-Host ""
    Write-Host ">>> $msg" -ForegroundColor Cyan
}

function Write-Success($msg) {
    Write-Host "    OK: $msg" -ForegroundColor Green
}

function Write-Fail($msg) {
    Write-Host "    ERROR: $msg" -ForegroundColor Red
}

function Pause-Exit($code) {
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit $code
}

# ── Banner ────────────────────────────────────────────────────────────────────

Clear-Host
Write-Host "=============================================" -ForegroundColor Yellow
Write-Host "   EPM - Electricity Planning Model"          -ForegroundColor Yellow
Write-Host "   Installer"                                  -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow

# ── Step 1: Choose install folder ─────────────────────────────────────────────

Write-Step "Choose installation folder"

Add-Type -AssemblyName System.Windows.Forms
$browser = New-Object System.Windows.Forms.FolderBrowserDialog
$browser.Description = "Select the folder where EPM will be installed"
$browser.ShowNewFolderButton = $true
$browser.RootFolder = [System.Environment+SpecialFolder]::UserProfile

$result = $browser.ShowDialog()
if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
    Write-Fail "No folder selected. Installation cancelled."
    Pause-Exit 1
}

$INSTALL_DIR = Join-Path $browser.SelectedPath "EPM"
Write-Success "Install location: $INSTALL_DIR"

# ── Step 2: Check / install Git ───────────────────────────────────────────────

Write-Step "Checking for Git"

$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    Write-Success "Git found: $($git.Source)"
} else {
    Write-Host "    Git not found. Installing via winget..." -ForegroundColor Yellow
    winget install --id Git.Git -e --source winget --silent
    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) {
        Write-Fail "Git installation failed. Please install Git manually from https://git-scm.com and re-run."
        Pause-Exit 1
    }
    Write-Success "Git installed."
}

# ── Step 3: Clone repository ──────────────────────────────────────────────────

Write-Step "Cloning EPM repository"

if (Test-Path $INSTALL_DIR) {
    Write-Host "    Folder already exists. Pulling latest changes..." -ForegroundColor Yellow
    git -C $INSTALL_DIR pull origin $REPO_BRANCH
} else {
    git clone --branch $REPO_BRANCH $REPO_URL $INSTALL_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Clone failed. Check your internet connection."
        Pause-Exit 1
    }
}
Write-Success "Repository ready at $INSTALL_DIR"

# ── Step 4: Check / install Conda ─────────────────────────────────────────────

Write-Step "Checking for Conda"

# Try to find conda (Anaconda or Miniconda)
$condaCmd = $null
$condaCandidates = @(
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\anaconda3\Scripts\conda.exe",
    "C:\ProgramData\miniconda3\Scripts\conda.exe"
)
foreach ($c in $condaCandidates) {
    if (Test-Path $c) { $condaCmd = $c; break }
}
if (-not $condaCmd) {
    $condaCmd = (Get-Command conda -ErrorAction SilentlyContinue)?.Source
}

if ($condaCmd) {
    Write-Success "Conda found: $condaCmd"
} else {
    Write-Host "    Conda not found. Downloading Miniconda..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $MINICONDA_URL -OutFile $MINICONDA_INSTALLER -UseBasicParsing
    Write-Host "    Installing Miniconda (this may take a few minutes)..." -ForegroundColor Yellow
    Start-Process -FilePath $MINICONDA_INSTALLER `
        -ArgumentList "/S /D=$env:USERPROFILE\miniconda3" `
        -Wait -NoNewWindow
    $condaCmd = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
    if (-not (Test-Path $condaCmd)) {
        Write-Fail "Miniconda installation failed."
        Pause-Exit 1
    }
    Write-Success "Miniconda installed."
}

# Derive conda base dir and activation script
$condaBase = Split-Path (Split-Path $condaCmd)
$condaActivate = Join-Path $condaBase "Scripts\activate.ps1"
if (-not (Test-Path $condaActivate)) {
    $condaActivate = Join-Path $condaBase "shell\condabin\conda-hook.ps1"
}

# ── Step 5: Create conda environment ──────────────────────────────────────────

Write-Step "Setting up Python environment ($ENV_NAME)"

$reqFile = Join-Path $INSTALL_DIR "requirements.txt"

# Check if env already exists
$envExists = (& $condaCmd env list) -match "^\s*$ENV_NAME\s"
if ($envExists) {
    Write-Host "    Environment already exists. Updating packages..." -ForegroundColor Yellow
    & $condaCmd run -n $ENV_NAME pip install -r $reqFile --quiet
} else {
    & $condaCmd create -n $ENV_NAME python=$PYTHON_VERSION -y --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Failed to create conda environment."
        Pause-Exit 1
    }
    & $condaCmd run -n $ENV_NAME pip install -r $reqFile --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Failed to install Python packages."
        Pause-Exit 1
    }
}
Write-Success "Environment ready."

# ── Step 6: Create desktop launcher ───────────────────────────────────────────

Write-Step "Creating desktop launcher"

$desktop = [System.Environment]::GetFolderPath("Desktop")
$launcherPath = Join-Path $desktop "Launch EPM Dashboard.bat"

$batContent = @"
@echo off
title EPM Dashboard
echo =============================================
echo   EPM - Electricity Planning Model
echo   Starting Dashboard...
echo =============================================
echo.

SET CONDA_BASE=$condaBase
SET ENV_NAME=$ENV_NAME
SET INSTALL_DIR=$INSTALL_DIR

CALL "%CONDA_BASE%\Scripts\activate.bat" %ENV_NAME%
cd /d "%INSTALL_DIR%"
start "" "http://localhost:8050"
python dashboard/app.py

pause
"@

Set-Content -Path $launcherPath -Value $batContent -Encoding UTF8
Write-Success "Launcher created on Desktop: Launch EPM Dashboard.bat"

# ── Step 7: GAMS check ────────────────────────────────────────────────────────

Write-Step "Checking for GAMS"

$gams = Get-Command gams -ErrorAction SilentlyContinue
if ($gams) {
    Write-Success "GAMS found: $($gams.Source)"
} else {
    Write-Host ""
    Write-Host "  !! GAMS not detected on this machine." -ForegroundColor Yellow
    Write-Host "     EPM requires GAMS with a valid license to run optimizations." -ForegroundColor Yellow
    Write-Host "     Download GAMS from: https://www.gams.com/download/" -ForegroundColor Yellow
    Write-Host "     Contact your ESMAP/World Bank contact for a license file." -ForegroundColor Yellow
}

# ── Done ──────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "   Installation complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "   EPM installed at : $INSTALL_DIR" -ForegroundColor White
Write-Host "   To start the dashboard, double-click:" -ForegroundColor White
Write-Host "   'Launch EPM Dashboard' on your Desktop" -ForegroundColor White
Write-Host ""

Pause-Exit 0
