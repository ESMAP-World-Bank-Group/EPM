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

    !!! note "Coming soon"
        The `.exe` installer is in development. Download will be available here.

        In the meantime, use the **Manual Setup** tab.

=== "Manual Setup"

    ### 1. Install prerequisites

    === "Git"
        Git is used to download and update the EPM code.

        - **Windows**: [Download Git for Windows](https://git-scm.com/download/win), with default settings
        - **macOS**: [Download Git for macOS](https://sourceforge.net/projects/git-osx-installer/)

        Verify: `git --version`

    === "Python & Conda"
        Required to use the Python interface (scenarios, Monte Carlo, automation).

        [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html) for all platforms.

        Verify: `python --version` and `conda --version`

    === "Code Editor (optional)"
        Useful for editing input files and navigating the codebase.

        - [Visual Studio Code](https://code.visualstudio.com/) (recommended)
        - [PyCharm](https://www.jetbrains.com/pycharm/)

    ### 2. Clone the repository

    ```sh
    git clone https://github.com/ESMAP-World-Bank-Group/EPM.git
    cd EPM
    ```

    Create your own working branch (replace `my_country_2025` with your project name):

    ```sh
    git checkout -b my_country_2025
    git push -u origin my_country_2025
    ```

    ### 3. Set up the Python environment

    ```sh
    conda create -n epm_env python=3.10
    conda activate epm_env
    pip install -r requirements.txt
    ```

    !!! note "Windows + Monte Carlo"
        If you plan to run Monte Carlo analysis on Windows, first install [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), then:
        ```sh
        pip install chaospy==4.3.18
        ```

    ### 4. Verify the installation

    ```sh
    cd epm
    python epm.py --simple
    ```

    Results written to `output/`. If something fails, see [Debugging](run_debugging.md).

---

## How do you want to run EPM?

| Method | Best for | Go to |
|---|---|---|
| **Dashboard** | Quick runs, visual interface, no command line | [Run from Dashboard](run_dashboard.md) |
| **Python CLI** | Scenarios, automation, Monte Carlo | [Run from Python](run_python.md) |
| **GAMS Studio** | Model development, GAMS debugging | [Run from GAMS Studio](run_gams_studio.md) |
| **Remote Server** | Heavy computations, parallel runs | [Run on Remote Server](run_remote_server.md) |
