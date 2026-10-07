# EPM Installer

This folder contains the files needed to build the Windows installer (`epm.exe`) that is
distributed from the [documentation site](https://esmap-world-bank-group.github.io/EPM/run/run_installation/).

## Files

| File | Description |
|------|-------------|
| `installer.ps1` | PowerShell script that performs the setup — **the single source of truth** |
| `epm.exe` | The distributed installer: `installer.ps1` compiled with PS2EXE |
| `872324.ico` | Icon embedded in `epm.exe` |
| `setup.iss` | **Deprecated** — see [About setup.iss](#about-setupiss-deprecated) |
| `README.md` | This file |

---

## ⚠️ `epm.exe` must be rebuilt after every change to `installer.ps1`

`epm.exe` embeds its **own copy** of `installer.ps1`. Editing the `.ps1` alone has **no effect
whatsoever** on users who download the `.exe` — they keep getting the old script.

The two files must always be rebuilt and committed together. A change to `installer.ps1` that
ships without a matching `epm.exe` is a silent no-op, and it is invisible in code review because
the diff looks correct.

Users download the binary straight from `main` via
`https://github.com/ESMAP-World-Bank-Group/EPM/raw/main/installer/epm.exe`, so it goes live as
soon as the commit lands. There is no release to publish.

---

## How `epm.exe` is rebuilt

The distributed executable is built with **PS2EXE**, a PowerShell module that wraps a `.ps1`
script into a standalone console executable.

### Automatically, via CI — the normal path

[`.github/workflows/build_installer.yml`](../.github/workflows/build_installer.yml) rebuilds
`epm.exe` whenever `installer.ps1` changes and opens a pull request with the new binary. Edit
the `.ps1`, push, then **merge the PR the bot opens** — that is the only manual step.

The bot opens a PR rather than pushing to `main` directly because `main` requires changes to go
through a pull request. It reuses the same `bot/rebuild-epm-exe` branch (force-pushed), so
repeated rebuilds update the existing PR instead of piling up new ones.

This needs "Allow GitHub Actions to create and approve pull requests" to be enabled — in
**organization** settings, not just repository settings. When the org disables it, the
repository checkbox is greyed out and only an org owner can unblock it.

The workflow only triggers when `installer.ps1` (or the icon) actually changes, because PS2EXE
output is not byte-reproducible — it embeds a build timestamp, so an unconditional rebuild would
produce an empty commit on every push.

It can also be run manually from the Actions tab (`workflow_dispatch`).

### Manually — fallback only

You should not normally need this. Note that PS2EXE requires PowerShell in **`FullLanguage`**
mode: it generates .NET code, which is forbidden under `ConstrainedLanguage` — the default on
corporate laptops locked down with WDAC/AppLocker. Check yours with:

```powershell
$ExecutionContext.SessionState.LanguageMode
```

If it returns `ConstrainedLanguage`, you cannot build locally; use the CI workflow instead.

#### 1. Install PS2EXE (once)

```powershell
Install-Module -Name ps2exe -Scope CurrentUser
```

#### 2. Compile

```powershell
cd installer
Invoke-ps2exe -inputFile installer.ps1 -outputFile epm.exe -iconFile 872324.ico -title "EPM Installer"
```

Do **not** pass `-noConsole` (the installer prints its progress to a terminal) and do **not**
pass `-requireAdmin` (the installer runs as the invoking user; requiring elevation causes
failures on managed corporate laptops).

#### 3. Verify before committing

Check that the new binary really embeds the current script — PS2EXE stores it as plain text,
so a string search is enough:

```powershell
if (Select-String -Path epm.exe -Pattern "conda-forge" -Quiet) {
    "OK - current script embedded"
} else {
    "STALE - rebuild did not take"
}
```

Replace `conda-forge` with any string unique to your change. Then run the `.exe` once on a
clean machine or VM before committing.

---

## What the installer does (for the end user)

1. Checks for **Git** — installs it via `winget` if missing
2. Asks where to install EPM (defaults to `%USERPROFILE%\EPM`)
3. Clones `https://github.com/ESMAP-World-Bank-Group/EPM` (branch: `main`)
4. Checks for **Conda** (Anaconda or Miniconda) — installs Miniconda if missing
5. Creates the `epm_env` conda environment **from `conda-forge`** and installs `requirements.txt`
6. Creates a **"Launch EPM Dashboard"** shortcut on the Desktop
7. Warns if **GAMS** is not detected (must be installed separately)

If the environment already exists, step 5 only refreshes the pip dependencies and the installer
continues to step 6 — so re-running `epm.exe` is a safe way to repair a partial install.

### Why `conda-forge`

Environment creation is pinned to `conda-forge` via `--override-channels -c conda-forge`.
Anaconda's default channels (`repo.anaconda.com/pkgs/*`) now require each user to explicitly
accept Anaconda's Terms of Service, which makes `conda create` fail outright. Those channels
also carry commercial licensing conditions for large organizations. `conda-forge` avoids both
problems, and the resulting environment is equivalent — `python=3.10` is the only package conda
installs; everything else comes from `pip install -r requirements.txt`.

---

## Notes

- **GAMS** must be installed manually with a valid license. Supported range: **48.2.0 to 53.x**
  (54.x is not yet supported).
- The installer requires an internet connection.
- To update EPM later, re-run the installer — it performs a `git pull` if the folder is already
  a repository.

---

## About `setup.iss` (deprecated)

`setup.iss` is an [Inno Setup](https://jrsoftware.org/isinfo.php) script that wraps
`installer.ps1` into a different executable (`dist/EPM_Setup.exe`). **It is not what ships.**

It is kept only for reference. Do not use it to rebuild the distributed installer: it produces a
differently-named binary that requests administrator elevation (`PrivilegesRequired=admin`),
which the PS2EXE build deliberately avoids.
