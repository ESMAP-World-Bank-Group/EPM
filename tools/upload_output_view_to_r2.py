"""
Upload des RESULTATS choisis (epm/output_view/) vers le data store, en LISIBLE
pour EPM View. Arrive sous {branch}/epm/output/...  (l'app lit "output").

On n'envoie QUE les .csv (les gros .gdx/logs sont ignores).

Variables d'env (via publish.ps1 -> tools/.env) :
  EPM_REPO, EPM_BRANCH, STORE_ENDPOINT, STORE_BUCKET + AWS_* (s3fs)
"""
import csv
import json
import os
import tempfile
from pathlib import Path

import s3fs

# pDispatchComplete est enorme (horaire x 16 ans) -> on le decoupe par annee
# ({name}/y{annee}.csv) pour que EPM View charge une annee a la fois (fluide).
DISPATCH_NAME = "pDispatchComplete.csv"


def upload_dispatch_split(fs, p, prefix):
    """Split {run}/{scen}/output_csv/pDispatchComplete.csv by year -> .../pDispatchComplete/y{Y}.csv.
    Returns True if split+uploaded, False if no 'y' column (caller uploads whole file)."""
    rel_dir = p.relative_to(LOCAL).parent.as_posix()  # {run}/{scen}/output_csv
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return False
        cols = [h.strip().lower() for h in header]
        if "y" not in cols:
            return False
        yi = cols.index("y")
        tmpdir = Path(tempfile.mkdtemp())
        handles = {}  # year -> (file handle, csv.writer, temp path)
        try:
            for row in reader:
                if yi >= len(row):
                    continue
                y = row[yi].strip()
                if not y:
                    continue
                if y not in handles:
                    tp = tmpdir / f"y{y}.csv"
                    fh = tp.open("w", encoding="utf-8", newline="")
                    w = csv.writer(fh)
                    w.writerow(header)
                    handles[y] = (fh, w, tp)
                handles[y][1].writerow(row)
            for y, (fh, w, tp) in handles.items():
                fh.close()
                fs.put_file(str(tp), f"{bucket}/{prefix}/{rel_dir}/pDispatchComplete/y{y}.csv")
            print(f"    {DISPATCH_NAME} -> {len(handles)} annee(s): {', '.join(sorted(handles))}")
            return True
        finally:
            for y, (fh, w, tp) in handles.items():
                if not fh.closed:
                    fh.close()
                tp.unlink(missing_ok=True)
            tmpdir.rmdir()

EPM_REPO = os.environ["EPM_REPO"]
BRANCH   = os.environ["EPM_BRANCH"]
endpoint = os.environ["STORE_ENDPOINT"]
bucket   = os.environ["STORE_BUCKET"]

LOCAL  = Path(EPM_REPO) / "epm" / "output_view"
PREFIX = f"{BRANCH}/epm/output"

if not LOCAL.is_dir():
    print("  (pas de epm/output_view -> rien a publier cote resultats)")
    raise SystemExit(0)

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint})

files = [p for p in LOCAL.rglob("*.csv") if p.is_file()]
if not files:
    print("  (output_view vide -> rien a publier)")
    raise SystemExit(0)

print(f"  {len(files)} csv -> s3://{bucket}/{PREFIX}/")
for p in files:
    rel = p.relative_to(LOCAL).as_posix()
    if p.name == DISPATCH_NAME and upload_dispatch_split(fs, p, PREFIX):
        continue  # uploaded as per-year splits, skip the giant whole file
    fs.put_file(str(p), f"{bucket}/{PREFIX}/{rel}")
print(f"  OK ({len(files)} csv)")

# A run now draws its own map and writes the pair next to input_scenarios.csv,
# so EPM View reads the layers from the run folder before falling back to the
# input one. Send them too -- they are small, and without them the fallback is
# the only thing left and it shows the zoning of whatever the input folder
# happens to ship rather than the one the run solved.
layers = [p for p in LOCAL.glob("*/*.geojson") if p.is_file()]
for p in layers:
    fs.put_file(str(p), f"{bucket}/{PREFIX}/{p.relative_to(LOCAL).as_posix()}")
print(f"  OK ({len(layers)} map layer(s))")

# --- manifest des runs (R2 public ne liste pas les dossiers -> EPM View lit ce json) ---
runs = sorted({
    p.relative_to(LOCAL).parts[0]
    for p in files
    if len(p.relative_to(LOCAL).parts) > 1
})

# EPM View also needs to know WHICH csv each run/scenario holds. The catalogue is
# not fixed -- a merged publish writes 11 files, an older run about 55 -- and a
# public bucket cannot be listed, so without this the app is left probing names.
# The answer is already here: it is the very list of files this script uploaded.
files_by_run = {}
for p in files:
    parts = p.relative_to(LOCAL).parts
    if len(parts) < 4 or parts[2] != "output_csv":
        continue                      # input_scenarios.csv, map layers, anything else
    run, scenario = parts[0], parts[1]
    files_by_run.setdefault(run, {}).setdefault(scenario, []).append("/".join(parts[3:]))
for scenarios in files_by_run.values():
    for names in scenarios.values():
        names.sort()

# "runs" stays first and unchanged: an older EPM View reads it and ignores the rest.
manifest = json.dumps({"runs": runs, "files": files_by_run}, indent=2)
fs.pipe_file(f"{bucket}/{PREFIX}/manifest.json", manifest.encode("utf-8"))
n_csv = sum(len(v) for sc in files_by_run.values() for v in sc.values())
print(f"  manifest.json -> {len(runs)} run(s), {n_csv} csv indexed: {', '.join(runs)}")
