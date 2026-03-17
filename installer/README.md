# EPM Installer

This folder contains the files needed to build a Windows installer (`.exe`) for EPM.

## Files

| File | Description |
|------|-------------|
| `installer.ps1` | PowerShell script that does the actual setup |
| `setup.iss` | Inno Setup script that wraps `installer.ps1` into a `.exe` |
| `README.md` | This file |

---

## How to build the `.exe`

### Prerequisites
Install **Inno Setup 6** (free): https://jrsoftware.org/isinfo.php

### Steps
1. Open **Inno Setup Compiler**
2. File → Open → select `setup.iss`
3. Build → Compile (or press `F9`)
4. The `.exe` will be generated in `installer/dist/EPM_Setup.exe`

---

## What the installer does (for the end user)

1. Checks for **Git** — installs it via `winget` if missing
2. Asks where to install EPM (folder picker)
3. Clones `https://github.com/ESMAP-World-Bank-Group/EPM` (branch: `main`)
4. Checks for **Conda** (Anaconda or Miniconda) — installs Miniconda if missing
5. Creates the `esmap_env` conda environment and installs `requirements.txt`
6. Creates a **"Launch EPM Dashboard"** shortcut on the Desktop
7. Warns if **GAMS** is not detected (must be installed separately)

---

## Notes

- GAMS must be installed manually with a valid license
- The installer requires an internet connection
- Admin rights are required (for Git/Miniconda installation if needed)
- To update EPM later: re-run the installer (it will `git pull` if already installed)
