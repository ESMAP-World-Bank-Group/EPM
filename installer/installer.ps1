# EPM Installer Script

$REPO_URL    = "https://github.com/ESMAP-World-Bank-Group/EPM.git"
$REPO_BRANCH = "main"
$ENV_NAME    = "epm_env"
$PYTHON_VER  = "3.10"
$MINICONDA_URL       = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$MINICONDA_INSTALLER = "$env:TEMP\Miniconda3-installer.exe"

# A written trace is the difference between "it failed" and a diagnosis: without
# one, reporting a problem means sending a screenshot. It goes to TEMP because
# the install folder is not known yet at this point, and may not exist.
#
# Start-Transcript is a cmdlet, so it survives ConstrainedLanguage. If the host
# refuses to transcribe, the installer carries on without a log rather than
# failing over it.
$LOG_FILE = "$env:TEMP\epm_install_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Start-Transcript -Path $LOG_FILE -ErrorAction SilentlyContinue | Out-Null

function Write-Step { Write-Host ""; Write-Host ">>> $args" -ForegroundColor Cyan }
function Write-Ok   { Write-Host "    OK: $args" -ForegroundColor Green }
function Write-Warn { Write-Host "    !! $args" -ForegroundColor Yellow }
function Write-Err  { Write-Host "    ERROR: $args" -ForegroundColor Red }

function Stop-Install {
    Write-Host ""
    Write-Host "    A full log of this run was saved to:" -ForegroundColor Yellow
    Write-Host "    $LOG_FILE"                            -ForegroundColor Yellow
    Write-Host "    Please attach it when reporting the problem." -ForegroundColor Yellow
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    Write-Host "Press Enter to exit..."
    $null = Read-Host
    exit 1
}

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
    # Keep a copy of the output so the failure below can be diagnosed instead of
    # just reported. Tee-Object still shows it live.
    $createLog = "$env:TEMP\epm_conda_create.log"
    & "$condaCmd" create -n $ENV_NAME "python=$PYTHON_VER" -y --override-channels -c conda-forge 2>&1 |
        Tee-Object -FilePath $createLog
    if ($LASTEXITCODE -ne 0) {
        Write-Err "conda create failed."
        # This is what blocked users before conda-forge was pinned. It should no
        # longer happen - if it does, something is putting Anaconda's channels
        # back in. Say so, rather than leaving a bare "failed".
        if (Select-String -Path $createLog -Pattern "Terms of Service" -Quiet -ErrorAction SilentlyContinue) {
            Write-Warn "conda is refusing to download until Anaconda's Terms of Service are accepted."
            Write-Warn "This installer deliberately avoids Anaconda's channels, so this normally"
            Write-Warn "means a .condarc on this machine forces them back in. Check with:"
            Write-Warn "    conda config --show channels"
        }
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

# Locating the Desktop is surprisingly hard on a managed laptop:
#   - GetFolderPath is a .NET method call, which PowerShell refuses in
#     ConstrainedLanguage mode - the default under WDAC/AppLocker.
#   - "$env:USERPROFILE\Desktop" often does not exist at all: OneDrive Known
#     Folder Move redirects it into "OneDrive - <tenant>", and the tenant name
#     varies by organization ("OneDrive - WBG", "OneDrive - Contoso", ...).
#   - on a localized Windows the folder may not even be named "Desktop".
#
# The registry knows the answer in every one of those cases, and reading it
# needs only cmdlets - so it survives ConstrainedLanguage. Everything below it
# is a fallback for the rare machine where the key is missing.
$desktop = $null

$shellFolders = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders' -ErrorAction SilentlyContinue
if ($shellFolders -and $shellFolders.Desktop) {
    # -replace is an operator, not a method: ExpandEnvironmentVariables would be
    # blocked here. This key is normally already expanded.
    $fromReg = $shellFolders.Desktop -replace '%USERPROFILE%', $env:USERPROFILE
    if (Test-Path $fromReg) { $desktop = $fromReg }
}

if (-not $desktop) {
    try { $fromDotNet = [System.Environment]::GetFolderPath("Desktop") } catch { $fromDotNet = $null }
    if ($fromDotNet -and (Test-Path $fromDotNet)) { $desktop = $fromDotNet }
}

if (-not $desktop) {
    # Any "OneDrive*" folder, whatever this organization happens to call it.
    $oneDrives = Get-ChildItem -Path $env:USERPROFILE -Directory -Filter "OneDrive*" -ErrorAction SilentlyContinue
    foreach ($od in $oneDrives) {
        $candidate = Join-Path $od.FullName "Desktop"
        if (Test-Path $candidate) {
            $desktop = $candidate
            break
        }
    }
}

if (-not $desktop -and (Test-Path "$env:USERPROFILE\Desktop")) {
    $desktop = "$env:USERPROFILE\Desktop"
}

if ($desktop) {
    Write-Host "    Desktop: $desktop" -ForegroundColor Gray
} else {
    Write-Warn "Could not locate the Desktop folder."
}

$activateBat  = "$condaBase\Scripts\activate.bat"

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

$launcherCreated = $false
if ($desktop) {
    $launcherPath = Join-Path $desktop "Launch EPM Dashboard.bat"
    Set-Content -Path $launcherPath -Value $batContent -Encoding UTF8 -ErrorAction SilentlyContinue
    if (Test-Path $launcherPath) {
        Write-Ok "Launcher created: $launcherPath"
        $launcherCreated = $true
    }
}

if (-not $launcherCreated) {
    Write-Warn "Could not create the launcher on the Desktop."
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

    # `gams --version` does not exist: it exits 6 trying to open "--version" as
    # an input file. `gams audit` prints the version and exits 0. curdir keeps
    # any stray listing file out of the install folder.
    $auditOut = & gams audit lo=3 curdir="$env:TEMP" 2>&1 | Out-String
    if ($auditOut -match '(\d+)\.(\d+)\.(\d+)') {
        $gamsVer = $matches[0]
        $maj = [int]$matches[1]
        $min = [int]$matches[2]
        Write-Ok "GAMS version: $gamsVer"
        if ($maj -ge 54) {
            Write-Warn "GAMS $maj.x is NOT yet supported - input treatment will fail at run time."
            Write-Warn "Please install GAMS 53.x or earlier: https://www.gams.com/download/"
        } elseif ($maj -lt 48 -or ($maj -eq 48 -and $min -lt 2)) {
            Write-Warn "GAMS $gamsVer is older than the required 48.2.0."
            Write-Warn "Please upgrade: https://www.gams.com/download/"
        }
    } else {
        Write-Warn "Could not determine the GAMS version. EPM requires 48.2.0 to 53.x."
    }
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
Write-Host "   Installation log : $LOG_FILE"
Write-Host ""
Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
Write-Host "Press Enter to exit..."
$null = Read-Host
exit 0
