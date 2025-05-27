
# Prerequisites: Install Required Tools

This guide explains how to install the necessary tools to run the EPM model. 
It covers Git, Python/Conda, and GAMS setup for **macOS and Windows**. 

---

## 1. Git (Version Control System)

Git is used to download and manage versions of the model repository.

- **Windows**:  
  [Download Git for Windows](https://git-scm.com/download/win) and install with default settings.

- **macOS**:  
  [Download Git for macOS](https://sourceforge.net/projects/git-osx-installer/) and follow the installer instructions.

- **Verify installation**:  
  Open Terminal or Command Prompt and type:
  ```sh
  git --version
  ```

---

## 2. Python & Conda (Optional – Required for Python Scripts)

If you want to run the model using Python, install Miniconda.

- **All platforms**:  
  Download Miniconda from:  
  [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)  
  Choose the installer for your OS and follow the prompts.

- **Verify installation**:
  Open Terminal or Command Prompt and type:  
  ```sh
  python --version
  conda --version
  ```

---

## 3. GAMS (Needed for GAMS Studio Users)

- **All platforms**:  
  Download and install GAMS from:  
  [https://www.gams.com/download/](https://www.gams.com/download/)

GAMS is a commercial optimization software. You can use the free version for educational purposes, which has some limitations.
Ask your institution for a license.

---

## Adding GAMS to PATH

This allows you to run GAMS from the command line.

### Windows

1. Open GAMS Studio, go to `Help > About`, and copy the installation path (e.g., `C:\GAMS\40.1`).
2. Open the Start Menu, search for `Environment Variables`, and open it.
3. In the `System Properties` window, click `Environment Variables`.
4. Under **System Variables**, select `Path`, click `Edit`, then `New`, and paste the GAMS path.
5. Click OK to save. Reopen Command Prompt and test:
   ```sh
   gams
   ```

### macOS

1. Locate your GAMS install folder (e.g., `/Applications/GAMS40.1`).
2. Open Terminal and edit your shell config:
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

GAMS should now be accessible from the terminal.