# EPM Installer Script

$REPO_URL    = "https://github.com/ESMAP-World-Bank-Group/EPM.git"
$REPO_BRANCH = "main"
$ENV_NAME    = "epm_env"
$PYTHON_VER  = "3.10"
$MINICONDA_URL       = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$MINICONDA_INSTALLER = "$env:TEMP\Miniconda3-installer.exe"

function Write-Step { Write-Host ""; Write-Host ">>> $args" -ForegroundColor Cyan }
function Write-Ok   { Write-Host "    OK: $args" -ForegroundColor Green }
function Write-Warn { Write-Host "    !! $args" -ForegroundColor Yellow }
function Write-Err  { Write-Host "    ERROR: $args" -ForegroundColor Red }
function Stop-Install { Write-Host "Press Enter to exit..."; $null = Read-Host; exit 1 }

Clear-Host
Write-Host "=============================================" -ForegroundColor Yellow
Write-Host "   EPM - Electricity Planning Model"          -ForegroundColor Yellow
Write-Host "   Installer"                                  -ForegroundColor Yellow
Write-Host "=============================================" -ForegroundColor Yellow

# --- Step 1: Install folder ---

Write-Step "Choose installation folder"
Write-Host "    Press Enter for default: $env:USERPROFILE\EPM" -ForegroundColor Gray
$userInput = Read-Host "    Folder"
if ($userInput -eq "") {
    $INSTALL_DIR = "$env:USERPROFILE\EPM"
} else {
    $INSTALL_DIR = $userInput
}
Write-Ok "Install location: $INSTALL_DIR"

# --- Step 2: Git ---

Write-Step "Checking for Git"
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    Write-Ok "Git found: $($git.Source)"
} else {
    Write-Warn "Git not found. Installing via winget..."
    winget install --id Git.Git -e --source winget --silent
    $env:PATH = "$env:PATH;C:\Program Files\Git\cmd"
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) {
        Write-Err "Git install failed. Install from https://git-scm.com then re-run."
        Stop-Install
    }
    Write-Ok "Git installed."
}

# --- Step 3: Clone ---

Write-Step "Cloning EPM repository"
$isGitRepo = Test-Path (Join-Path $INSTALL_DIR ".git")
if ($isGitRepo) {
    Write-Warn "Folder exists - pulling latest changes..."
    & git -C "$INSTALL_DIR" pull --quiet origin $REPO_BRANCH
} else {
    if (Test-Path $INSTALL_DIR) {
        Write-Warn "Folder exists but is not a git repo - cloning into it..."
    }
    & git clone --quiet --branch $REPO_BRANCH $REPO_URL "$INSTALL_DIR"
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Clone failed. Check your internet connection."
        Stop-Install
    }
}
Write-Ok "Repository ready at $INSTALL_DIR"

# --- Step 4: Conda ---

Write-Step "Checking for Conda"
$condaCmd = $null
$candidates = @(
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\anaconda3\Scripts\conda.exe",
    "C:\ProgramData\miniconda3\Scripts\conda.exe"
)
foreach ($c in $candidates) {
    if (Test-Path $c) {
        $condaCmd = $c
        break
    }
}
if (-not $condaCmd) {
    $found = Get-Command conda -ErrorAction SilentlyContinue
    if ($found) {
        $condaCmd = $found.Source
    }
}

if ($condaCmd) {
    Write-Ok "Conda found: $condaCmd"
} else {
    Write-Warn "Conda not found. Downloading Miniconda..."
    Invoke-WebRequest -Uri $MINICONDA_URL -OutFile $MINICONDA_INSTALLER -UseBasicParsing
    Start-Process -FilePath $MINICONDA_INSTALLER -ArgumentList "/S /D=$env:USERPROFILE\miniconda3" -Wait -NoNewWindow
    $condaCmd = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
    if (-not (Test-Path $condaCmd)) {
        Write-Err "Miniconda install failed."
        Stop-Install
    }
    Write-Ok "Miniconda installed."
}

$condaBase = Split-Path (Split-Path $condaCmd)

# --- Step 5: Python environment ---

Write-Step "Setting up Python environment ($ENV_NAME)"

$reqFile = "$INSTALL_DIR\requirements.txt"
if (-not (Test-Path $reqFile)) {
    Write-Err "requirements.txt not found - repository may not have cloned correctly."
    Stop-Install
}

$envList   = & "$condaCmd" env list 2>&1
$envExists = $envList | Select-String -SimpleMatch $ENV_NAME

if ($envExists) {
    Write-Warn "Environment exists. Updating packages..."
    & "$condaCmd" run -n $ENV_NAME pip install -r "$reqFile"
    if ($LASTEXITCODE -ne 0) {
        Write-Err "pip install failed."
        Stop-Install
    }
} else {
    Write-Warn "Creating environment (may take a few minutes)..."
    & "$condaCmd" create -n $ENV_NAME "python=$PYTHON_VER" -y
    if ($LASTEXITCODE -ne 0) {
        Write-Err "conda create failed."
        Stop-Install
    }
    & "$condaCmd" run -n $ENV_NAME pip install -r "$reqFile"
    if ($LASTEXITCODE -ne 0) {
        Write-Err "pip install failed."
        Stop-Install
    }
}
Write-Ok "Environment ready."

# --- Step 6: Desktop launcher ---

Write-Step "Creating desktop launcher"

$desktop = [System.Environment]::GetFolderPath("Desktop")
if (-not (Test-Path $desktop)) {
    $dlist = @(
        "$env:USERPROFILE\OneDrive\Desktop",
        "$env:USERPROFILE\OneDrive - World Bank Group\Desktop",
        "$env:USERPROFILE\Desktop"
    )
    foreach ($d in $dlist) {
        if (Test-Path $d) {
            $desktop = $d
            break
        }
    }
}
Write-Host "    Desktop: $desktop" -ForegroundColor Gray

$activateBat  = "$condaBase\Scripts\activate.bat"
$launcherPath = "$desktop\Launch EPM Dashboard.bat"

$line1  = "@echo off"
$line2  = "title EPM Dashboard"
$line3  = "echo ============================================="
$line4  = "echo   EPM - Electricity Planning Model"
$line5  = "echo   Starting Dashboard..."
$line6  = "echo ============================================="
$line7  = "echo."
$line8  = "CALL `"$activateBat`" $ENV_NAME"
$line9  = "cd /d `"$INSTALL_DIR`""
$line10 = "start `"`" `"http://localhost:8080`""
$line11 = "python dashboard/app.py"
$line12 = "pause"

$batContent = $line1, $line2, $line3, $line4, $line5, $line6, $line7, $line8, $line9, $line10, $line11, $line12

Set-Content -Path $launcherPath -Value $batContent -Encoding UTF8

if (Test-Path $launcherPath) {
    Write-Ok "Launcher created: $launcherPath"
} else {
    Write-Err "Could not create launcher on Desktop."
    Write-Warn "No problem - a launcher has been saved in the install folder instead:"
    $fallback = "$INSTALL_DIR\launch_dashboard.bat"
    Set-Content -Path $fallback -Value $batContent -Encoding UTF8
    Write-Ok "Fallback launcher: $fallback"
}

# --- Step 7: GAMS ---

Write-Step "Checking for GAMS"
$gams = Get-Command gams -ErrorAction SilentlyContinue
if ($gams) {
    Write-Ok "GAMS found: $($gams.Source)"
} else {
    Write-Warn "GAMS not detected. EPM requires GAMS with a valid license."
    Write-Warn "Download: https://www.gams.com/download/"
}

# --- Done ---

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "   Installation complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "   EPM installed at : $INSTALL_DIR"
Write-Host "   Launch the dashboard by double-clicking:"
Write-Host "   'Launch EPM Dashboard' on your Desktop"
Write-Host ""
Write-Host "Press Enter to exit..."
$null = Read-Host
exit 0
