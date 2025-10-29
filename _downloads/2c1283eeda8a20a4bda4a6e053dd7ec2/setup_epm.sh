#!/usr/bin/env bash
set -e  # stop on first error
set -o pipefail

# ---------- Configuration ----------
REPO_URL="https://github.com/ESMAP-World-Bank-Group/EPM.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/EPM"
CONDA_ENV="epm_env"
REQ_FILE="$REPO_DIR/requirements.txt"
LOG_FILE="$SCRIPT_DIR/setup_log.txt"
PYTHON_SCRIPT="$REPO_DIR/epm/epm.py"
GAMS_MAIN="$REPO_DIR/main.gms"

echo "----------------------------------------"
echo "EPM One-Click Installer and Tester for macOS/Linux"
echo "----------------------------------------"
echo "Logging output to: $LOG_FILE"
echo

# ---------- Step 1: Check Git ----------
if ! command -v git &> /dev/null; then
    echo "[!] Git not found. Installing via Homebrew (requires sudo)..."
    if command -v brew &> /dev/null; then
        brew install git
    else
        echo "Homebrew not found. Please install Git manually: https://git-scm.com/downloads"
        exit 1
    fi
else
    echo "[+] Git found."
fi

# ---------- Step 2: Check Conda ----------
if ! command -v conda &> /dev/null; then
    echo "[!] Conda not found. Installing Miniconda..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda"
    export PATH="$HOME/miniconda/bin:$PATH"
    echo "[+] Miniconda installed."
else
    echo "[+] Conda found."
fi

# ---------- Step 3: Check GAMS ----------
if ! command -v gams &> /dev/null; then
    echo "[!] GAMS not found in PATH."
    echo "Please install GAMS >= 48.2.0 and add it to PATH (see GAMS docs)."
    exit 1
else
    echo "[+] GAMS found in PATH."
fi

# ---------- Step 4: Clone or update repo ----------
if [ ! -d "$REPO_DIR" ]; then
    echo "[*] Cloning EPM repository..."
    git clone "$REPO_URL" "$REPO_DIR" >> "$LOG_FILE" 2>&1
else
    echo "[+] Repository already exists. Pulling latest changes..."
    (cd "$REPO_DIR" && git pull >> "$LOG_FILE" 2>&1)
fi
echo "[+] Repository ready at $REPO_DIR."

# ---------- Step 5: Test GAMS ----------
echo "[*] Testing GAMS installation with MODELTYPE=RMIP ..."
(cd "$REPO_DIR" && gams "$GAMS_MAIN" lo=2 --MODELTYPE=RMIP >> "$LOG_FILE" 2>&1)
if [ $? -ne 0 ]; then
    echo "[!] GAMS test failed — check installation, license, or RMIP solver."
    echo "See log for details: $LOG_FILE"
    exit 1
else
    echo "[+] GAMS executed successfully with MODELTYPE=RMIP."
fi

# ---------- Step 6: Recreate Conda environment ----------
echo "[!] Removing old environment '$CONDA_ENV' (if any)..."
conda env remove -y -n "$CONDA_ENV" >> "$LOG_FILE" 2>&1 || true

echo "[*] Creating new environment '$CONDA_ENV'..."
conda create -y -n "$CONDA_ENV" python=3.10 >> "$LOG_FILE" 2>&1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ---------- Step 7: Install dependencies ----------
echo "[*] Installing dependencies from $REQ_FILE ..."
pip install --upgrade pip >> "$LOG_FILE" 2>&1
pip install -r "$REQ_FILE" >> "$LOG_FILE" 2>&1
echo "[+] All Python dependencies installed successfully."

# ---------- Step 8: Run EPM Python test ----------
echo "[*] Running EPM Python test ..."
(cd "$REPO_DIR" && python "$PYTHON_SCRIPT" --solver RMIP --simple >> "$LOG_FILE" 2>&1)
if [ $? -ne 0 ]; then
    echo "[!] Python EPM test failed — see $LOG_FILE for details."
    exit 1
else
    echo "[+] Python EPM test completed successfully."
fi

echo "----------------------------------------"
echo "✅ All tests passed — EPM environment ready."
echo "----------------------------------------"
echo "See detailed logs in: $LOG_FILE"