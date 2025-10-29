@echo off
setlocal ENABLEDELAYEDEXPANSION
title EPM One-Click Setup Script

:: ---------- Configuration ----------
set REPO_URL=https://github.com/ESMAP-World-Bank-Group/EPM.git
set REPO_DIR=%~dp0EPM
set CONDA_ENV=epm_env
set GAMS_MAIN=%REPO_DIR%\main.gms
set PYTHON_SCRIPT=%REPO_DIR%\epm\epm.py
set REQ_FILE=%REPO_DIR%\requirements.txt
set LOG_FILE=%~dp0setup_log.txt

echo ----------------------------------------
echo EPM One-Click Installer and Tester for Windows
echo ----------------------------------------
echo Logging output to: %LOG_FILE%
echo.

:: ---------- Step 1: Check Git ----------
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Git not found. Please install Git first.
    powershell -Command "Start-Process 'https://git-scm.com/download/win' -UseNewEnvironment"
    pause
    exit /b
)
echo [+] Git found.

:: ---------- Step 2: Check Conda ----------
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Conda not found. Installing Miniconda...
    powershell -Command "Start-Process 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -UseNewEnvironment"
    echo Please complete Miniconda installation, then rerun this script.
    pause
    exit /b
)
echo [+] Conda found.

:: ---------- Step 3: Check GAMS in PATH ----------
where gams >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] GAMS not found in PATH. Please install GAMS >= 48.2.0 and enable "Add to PATH" during installation.
    pause
    exit /b
)
echo [+] GAMS found in PATH.

:: ---------- Step 4: Clone or update repo ----------
if not exist "%REPO_DIR%" (
    echo [*] Cloning EPM repository...
    git clone %REPO_URL% "%REPO_DIR%" >> "%LOG_FILE%" 2>&1
) else (
    echo [+] Repository already exists. Pulling latest changes...
    cd "%REPO_DIR%"
    git pull >> "%LOG_FILE%" 2>&1
)
echo [+] Repository ready at %REPO_DIR%.

:: ---------- Step 5: Test GAMS ----------
echo [*] Testing GAMS installation with MODELTYPE=RMIP ...
cd "%REPO_DIR%"
gams "%GAMS_MAIN%" lo=2 --MODELTYPE=RMIP >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [!] GAMS test failed — please check installation, license, or RMIP solver. See %LOG_FILE% for details.
    pause
    exit /b
) else (
    echo [+] GAMS executed successfully with MODELTYPE=RMIP.
)

:: ---------- Step 6: Recreate Conda environment ----------
echo [!] Removing old environment '%CONDA_ENV%' (if any)...
conda env remove -y -n %CONDA_ENV% >> "%LOG_FILE%" 2>&1

echo [*] Creating new environment '%CONDA_ENV%' ...
conda create -y -n %CONDA_ENV% python=3.10 >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [!] Failed to create environment. See %LOG_FILE% for details.
    pause
    exit /b
)
echo [+] Environment created successfully.

:: ---------- Step 7: Install dependencies ----------
call conda activate %CONDA_ENV%
echo [*] Installing dependencies from requirements.txt ...
pip install --upgrade pip >> "%LOG_FILE%" 2>&1
pip install -r "%REQ_FILE%" >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [!] Dependency installation failed — see %LOG_FILE%.
    pause
    exit /b
)
echo [+] All Python dependencies installed successfully.

:: ---------- Step 8: Run EPM Python test ----------
echo [*] Running EPM Python test ...
cd "%REPO_DIR%"
python "%PYTHON_SCRIPT%" --solver RMIP --simple >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [!] Python EPM test failed — please check logs or Python/GAMS integration.
    pause
    exit /b
) else (
    echo [+] Python EPM test completed successfully.
)

echo ----------------------------------------
echo ✅ All tests passed — EPM environment ready to use.
echo ----------------------------------------
echo See log for details: %LOG_FILE%
pause
endlocal
