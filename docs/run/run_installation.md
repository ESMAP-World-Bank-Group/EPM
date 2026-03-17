# Installation

## Prerequisites

EPM's optimization engine is **GAMS**, which must be installed on your machine before proceeding.

!!! info "Install GAMS"
    [Download GAMS](https://www.gams.com/download/), version **48.2.0 or higher** required.
    Contact your institution or the World Bank team for a license.

    Add GAMS to your system PATH so it can be called from the terminal:

    - **Windows:** Search "Environment Variables" → System Variables → Path → add your GAMS folder (e.g. `C:\GAMS\48.2`)
    - **macOS:** Add `export PATH="/Applications/GAMS48.2:$PATH"` to `~/.zshrc`, then run `source ~/.zshrc`

    Verify: `gams` in a terminal should return the GAMS version.

---

## Installation

Two options are available. The **Windows Installer** automates the full setup (Git, Conda, Python environment, repository clone); only GAMS needs to be pre-installed. The **Manual Setup** is for macOS, Linux, or advanced Windows users who prefer to control each step.

=== "Windows Installer"

    <span style="font-size:0.78rem; color:#856404; background:#fff3cd; padding:4px 10px; border-radius:4px; display:inline-block; margin-bottom:1rem;">⚠ Beta — under development. If you encounter issues, try the Manual Setup tab or [report them here](../contributing/contributing_issues.md).</span>

    [Download epm.exe](https://github.com/ESMAP-World-Bank-Group/EPM/raw/main/installer/epm.exe){ .md-button .md-button--primary }

    <div style="font-size:0.85rem; margin-top:1.2rem;">

    **Steps**

    1. **Double-click** `epm.exe` — a terminal window opens
    2. **Choose an install folder** when prompted, or press Enter for the default (`C:\Users\you\EPM`)
    3. **Wait** a few minutes — the installer clones the repository and sets up the Python environment
    4. You see **"Installation complete!"** — you're done

    **After installation**

    - EPM is in the folder you chose
    - A **"Launch EPM Dashboard"** shortcut is on your Desktop — double-click it to start

    > **Note:** GAMS must be installed separately with a valid license. The installer will warn you if it is not detected.

    </div>

=== "Manual Setup"

    <div style="font-size:0.85rem;">

    **1. Install prerequisites**

    - **Git** — [Windows](https://git-scm.com/download/win) · [macOS](https://sourceforge.net/projects/git-osx-installer/)
    - **Python & Conda** — [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (all platforms)
    - **Code editor** *(optional)* — [VS Code](https://code.visualstudio.com/)

    **2. Clone the repository**

    ```sh
    git clone https://github.com/ESMAP-World-Bank-Group/EPM.git
    cd EPM
    git checkout -b my_country_2025
    git push -u origin my_country_2025
    ```

    **3. Set up the Python environment**

    ```sh
    conda create -n epm_env python=3.10
    conda activate epm_env
    pip install -r requirements.txt
    ```

    **4. Verify**

    ```sh
    cd epm && python epm.py --simple
    ```

    Results are written to `output/`. If something fails, see [Debugging](run_debugging.md).

    </div>

---

## How do you want to run EPM?

| Method | Best for | Go to |
|---|---|---|
| **Dashboard** | Quick runs, visual interface, no command line *(under development — errors must be debugged via command line)* | [Run from Dashboard](run_dashboard.md) |
| **Python CLI** | Scenarios, automation, Monte Carlo | [Run from Python](run_python.md) |
| **GAMS Studio** | Model development, GAMS debugging | [Run from GAMS Studio](run_gams_studio.md) |
| **Remote Server** | Heavy computations, parallel runs | [Run on Remote Server](run_remote_server.md) |
