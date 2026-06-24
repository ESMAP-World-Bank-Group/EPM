# Publishing & Syncing Data (DVC)

EPM **code** is public, but the **data** (input CSVs, results) is kept in a **private
store** — not in this public repository. The repo only keeps tiny **pointers**; the real
data lives in the store and is fetched on demand.

This page explains, from scratch, how to **publish** your data and how to **get it back**
on another machine or on the server. No prior knowledge of DVC is assumed.

!!! note "Prototype"
    The store is currently a Cloudflare R2 bucket (test data). For confidential data,
    the production store will be the World Bank S3 — same workflow, different endpoint.

---

## The idea in one picture

```
YOU (laptop) — you edit inputs/outputs in EPM
        │
        │  one action (double-click Publish):
        │   ├─ git push  → GitHub : code + .dvc pointers   (no data)
        │   ├─ dvc push  → STORE  : the data               (for runs)
        │   └─ publish   → STORE  : a readable copy         (for EPM View)
        ▼
   ┌──────────────── PRIVATE STORE (R2 / S3) ────────────────┐
   └───────────────┬─────────────────────────┬──────────────┘
                   │ dvc pull                 │ gated read
            SERVER (run the model)        EPM View (display)
```

**What is DVC?** Think *"git for data"*. Git stays light because the data is replaced by a
6-line pointer file (`*.dvc` = a hash + a path, **no values**). The real data lives in the
store, moved with `dvc push` / `dvc pull`.

The store keeps **two layouts** of the same data: a hash-addressed copy for DVC (runs), and
a readable copy by path for EPM View (the browser can't speak DVC). Both are written by the
same "publish" action — you don't manage this by hand.

---

## One-time setup (per machine)

1. Install the dependencies (DVC is included):
   ```bash
   pip install -r requirements.txt
   ```
2. Add your store keys: copy `tools/.env.example` to `tools/.env` and paste the 4 values
   (ask the store admin).
   ```
   tools/.env        ← your keys   (git-ignored — NEVER commit real keys)
   tools/.env.example ← template
   ```

That's it. The remote URL/endpoint is already in the repo (`.dvc/config`).

---

## Publish your data

After you change inputs (or want to show results), **publish in one action**:

=== "Windows"
    Double-click **`Publish.bat`** at the repo root.

=== "Command line"
    ```bash
    powershell -File tools/publish.ps1   # Windows
    ```

This does everything: re-hash the data (`dvc add`) → commit + push the pointer →
`dvc push` (data to the store, for the server) → upload readable copies (inputs +
`epm/output_view/`) for EPM View.

!!! tip "Showing results in EPM View"
    Copy the runs you want to display into
    `epm/output_view/<run>/<scenario>/output_csv/` **before** publishing (only `.csv`
    files are sent). `epm/output_view/` is a local staging folder (git-ignored).

---

## Get the data (another machine / the server)

=== "Server (Linux)"
    ```bash
    bash tools/sync.sh
    ```
=== "Windows"
    Double-click **`Sync.bat`**.

This runs `git pull` (code + pointers) then `dvc pull` (data from the store) — then you can
run the model with up-to-date data.

---

## Onboarding a new model (one-time per model)

Moving a model's data out of git into the store (the "switch"):

```bash
dvc init                                          # once per repo
dvc remote add -d store s3://<bucket>/dvcstore
dvc remote modify store endpointurl <endpoint>
# in .gitignore: remove the whitelist of the data folder, add  !epm/input/<data>.dvc
git rm -r --cached epm/input/<data_folder>        # untrack (files stay on disk)
dvc add epm/input/<data_folder>                   # create the pointer
```
Then publish (`Publish.bat`). After this, everyone just does *setup + publish/sync*.

To make **EPM View** read this branch from the store, add the branch name to `R2_BRANCHES`
in `src/utils/epmFetch.js` of the `epm-data-explorer` repo.

---

## Troubleshooting

- **`Unable to locate credentials`** → keys not loaded. On the server, either
  `export AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=…`, or set them once with
  `dvc remote modify --local store access_key_id …` (stays in git-ignored `config.local`).
- **`CERTIFICATE_VERIFY_FAILED` / SSL** → a corporate TLS proxy is intercepting the
  connection. Try off-VPN, or `pip install pip-system-certs` (makes Python trust the OS
  certificate store). The AWS server is not affected.
- **`dvc` command not found** (server) → use `python -m dvc …`, or add `~/.local/bin` to
  your `PATH`.

---

## Files (in `tools/`)

| File | Role |
|------|------|
| `publish.ps1` | engine behind `Publish.bat` |
| `sync.ps1` / `sync.sh` | get code + data (Windows / server) |
| `upload_to_r2.py`, `upload_output_view_to_r2.py` | upload the readable copies |
| `.env.example` | keys template (copy to `.env`, git-ignored) |
