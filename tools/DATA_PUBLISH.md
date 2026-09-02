# Publish / fetch model data (private store)

> 📖 **Full guide (EPM docs):** `docs/run/run_data_sync.md` — *Installation & Run →
> Publish & Sync Data (DVC)*. This file is a quick cheat sheet next to the scripts.

EPM **data** (input/output) no longer lives in this public repo: it sits in a
**private store** (R2 today, WB S3 later). The repo keeps only small **`.dvc`
pointers**. This folder provides the tooling to publish and fetch it.

> **Prototype.** The current store is a Cloudflare R2 bucket (test data).
> Confidential data → wait for the Bank's S3 store.

---

## 🔧 Setup — once per machine

1. Install the dependencies (includes DVC):
   ```
   pip install -r requirements.txt
   ```
2. Add your store access keys:
   - copy `tools/.env.example` → `tools/.env`
   - paste the 4 values (provided by the store admin)
   - ⚠️ `tools/.env` is **gitignored**: never commit real keys.

That's it. The remote config (URL/endpoint) is already in the repo (`.dvc/config`,
remote `r2`) **on this branch**.

> This holds for `blacksea_2026`, which versions its own `.dvc/config`. It is **not**
> true on `main`, where the file is not versioned and you must declare the remote
> yourself (`dvc remote add` / `dvc remote modify`). See `tools/DATA_PUBLISH.md` on
> `main`. Do not assume the config travels with the repo.

---

## ⬆️ Publish (after changing data) — Windows

**Double-click `Publish.bat`** (at the repo root).

It does everything: re-hash the data (`dvc add`) → commit + push the pointers →
`dvc push` (data → store, for the server) → upload the readable copies
(inputs + `epm/output_view/`) → store, for **EPM View**.

> To show **results** in EPM View: copy the runs you want into
> `epm/output_view/<run>/<scenario>/output_csv/` before publishing (only `.csv` files
> are sent). `epm/output_view/` is gitignored (local staging area).

### Only what changed is sent

A file is re-sent only when the store does not have it, has it at a different size, or
has it older than the local copy. The store itself is the reference, read in one listing
call, so there is no local state to go stale and no first run that re-sends everything.
Publishing twice in a row sends nothing the second time, and the 4 GB of
`pDispatchComplete.csv` are not even re-read when their year slices are already current.

Options, for the command line only (double-clicking `Publish.bat` needs none of them):

```powershell
powershell -File tools/publish.ps1 -Check        # compare, send nothing, commit nothing
powershell -File tools/publish.ps1 -Force        # re-send everything
powershell -File tools/publish.ps1 -SkipResults  # inputs only
powershell -File tools/publish.ps1 -SkipData     # results only
powershell -File tools/publish.ps1 -SkipData -Only "simulations_run_*/summary.csv"
```

Failed uploads are retried three times, then listed at the end and the script exits
non-zero. Nothing is ever deleted from the store.

---

## ⬇️ Fetch code + data

- **Windows**: double-click `Sync.bat`
- **Linux server**: `bash tools/sync.sh`

= `git pull` (code + pointers) + `dvc pull` (data from the store).

---

## Onboarding a NEW model (once per model)

Move a model's data out of git and into the store (the "switchover"):
```
dvc init                                   # once per repo
dvc remote add -d store s3://<bucket>/dvcstore
dvc remote modify store endpointurl <endpoint>
# remove the folder's whitelist entry from .gitignore, keep !epm/input/<data>.dvc
git rm -r --cached epm/input/<data_folder>
dvc add epm/input/<data_folder>
```
Then publish (`Publish.bat`). After that, everyone just does *setup + publish/sync*.

---

## On the EPM View side (the app)

EPM View reads data per branch. Branches whose data sits in the private store are
listed in `R2_BRANCHES` (file `src/utils/epmFetch.js` in the `epm-data-explorer`
repo). To wire up a new region: add its branch there.

---

## Files in this folder (data publishing)

- `publish.ps1` — engine behind `Publish.bat`
- `sync.ps1` / `sync.sh` — fetching (Windows / server)
- `upload_to_r2.py`, `upload_output_view_to_r2.py` — readable-upload helpers
- `r2_sync.py` — what gets sent and what is skipped, shared by both uploaders
- `.env.example` — key template (copy to `.env`, gitignored)
