# Prerequisites: Install Required Tools

This guide explains how to install the essential tools needed to run the EPM model on **macOS or Windows**.  
You’ll need:

- **Git** (to download and manage the code)
- **GAMS** (required to run the model)
- **Python & Conda** (optional, for enhanced functionalities via the Python API)
- **A code editor** (optional, to help navigate and edit the codebase)

---

## 1. Git (Version Control System)

Git is used to download the EPM code from the GitHub repository and manage version updates.

- **Windows**:  
  [Download Git for Windows](https://git-scm.com/download/win) and install it using default settings.

- **macOS**:  
  [Download Git for macOS](https://sourceforge.net/projects/git-osx-installer/) and follow the installer instructions.

- **Verify installation**:  
  Open Terminal or Command Prompt and run:
  ```sh
  git --version
  ```

---

## 2. GAMS (Optimization Engine)

GAMS is used to solve the energy planning model. A free version with limited capabilities is available for non-commercial use.

- **Download** (all platforms):  
  [https://www.gams.com/download/](https://www.gams.com/download/)

Ask your institution or the World Bank team if you need a full license.

We are using gamsapi version >= 48.2.0, you should run a version of GAMS that is **at least version 48.2.0.**, released on October 29, 2024.

---

### Add GAMS to Your System PATH

This step allows you to call `gams` from your terminal or command line.

#### On Windows

1. Open GAMS Studio → go to `Help > About` → copy the install path (e.g., `C:\GAMS\40.1`)
2. Search for **Environment Variables** in the Start Menu and open it.
3. In the **System Properties** window:
   - Click `Environment Variables`
   - Under **System Variables**, select `Path` and click `Edit`
   - Click `New`, then paste the GAMS path
4. Click OK and close all dialogs.
5. Open a new Command Prompt and test:
   ```sh
   gams
   ```

#### On macOS

1. Locate your GAMS installation (e.g., `/Applications/GAMS40.1`)
2. Open Terminal and run:
   ```sh
   nano ~/.zshrc
   ```
3. Add the following line:
   ```sh
   export PATH="/Applications/GAMS40.1:$PATH"
   ```
4. Save and apply changes:
   ```sh
   source ~/.zshrc
   gams
   ```

You should now be able to run GAMS from the terminal.

---

## 3. Python & Conda (Optional but Recommended)

Python is used to run EPM through its Python API, enabling advanced features such as scenario generation and Monte Carlo analysis.

- **Install Miniconda** (all platforms):  
  [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)  
  Download the installer for your operating system and follow the setup instructions.

- **Verify installation**:  
  Open Terminal or Command Prompt and run:
  ```sh
  python --version
  conda --version
  ```

---

## 4. Code Editor (Optional)

A code editor helps you navigate, compare, and edit model files efficiently—even if you don't plan to modify the Python code.

### Recommended Editors

- [Visual Studio Code](https://code.visualstudio.com/) (recommended)
- [PyCharm](https://www.jetbrains.com/pycharm/download/)

These editors also offer:

- Git integration to track changes and synchronize with the repository
- File diff tools to compare versions
- Syntax highlighting and code validation

We especially recommend using them to:

- Compare your regional/country version of the model with updates from the main EPM framework
- Merge changes when the EPM core is updated to maintain compatibility

---

By installing these tools, you'll be ready to run and explore the EPM model both through GAMS Studio and Python scripting.
