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
3. Declare the DVC remote (URL + endpoint, provided by the store admin):
   ```
   dvc remote add -d store s3://<bucket>/dvcstore
   dvc remote modify store endpointurl <endpoint>
   ```

> **`.dvc/config` is not versioned on this branch** — hence step 3. The command above
> writes to `.dvc/config`, which is local: redo it on every machine, and do not commit
> it until the final store is settled (see the *Prototype* box above: R2 today, Bank S3
> next). Some study branches version their own `.dvc/config`; do not rely on that
> from `main`.

---

## ⬆️ Publish (after changing data) — Windows

**Double-click `Publish.bat`** (at the repo root).

It does everything: re-hash the data (`dvc add`) → commit + push the pointers →
`dvc push` (data → store, for the server) → upload the readable copies
(inputs + `epm/output_view/`) → store, for **EPM View**.

> To show **results** in EPM View: copy the runs you want into
> `epm/output_view/<run>/<scenario>/output_csv/` before publishing (only `.csv` files
> are sent). `epm/output_view/` is gitignored (local staging area).

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
- `.env.example` — key template (copy to `.env`, gitignored)
