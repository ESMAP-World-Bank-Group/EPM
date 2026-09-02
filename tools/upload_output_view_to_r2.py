"""
Upload the selected RESULTS (epm/output_view/) to the data store, READABLE for EPM View.
They land under {branch}/epm/output/...  (the app reads "output").

Only .csv files are sent (the large .gdx and logs are ignored), and among them only the
ones missing from the store or newer than their remote copy. See r2_sync.

  --only <pattern>  send only the paths that match (glob, relative to output_view).
                    To republish ONE file without walking the 8+ GB of the folder:
                      --only "simulations_run_*/npv_external.csv"
                    The manifest is still computed over ALL the runs present: filtering
                    it would drop from EPM View the runs that were not re-uploaded.
  --force           send everything again, even what is already up to date
  --check           compare local and remote without sending anything

Environment variables (via publish.ps1 -> tools/.env):
  EPM_REPO, EPM_BRANCH, STORE_ENDPOINT, STORE_BUCKET plus AWS_* (s3fs)
"""
import argparse
import csv
import fnmatch
import json
import os
import shutil
import tempfile
from pathlib import Path

import s3fs

from r2_sync import MTIME_TOLERANCE_S, add_sync_args, remote_index, report, upload_many

# pDispatchComplete is huge (hourly x 16 years), so it is split by year
# ({name}/y{year}.csv) for EPM View to load one year at a time and stay responsive.
DISPATCH_NAME = "pDispatchComplete.csv"


def dispatch_is_current(p, idx, rel_dir):
    """Are the year slices already in the store newer than the source file?

    The 36 pDispatchComplete of the folder weigh 4 GB. Without this guard every publish
    re-reads and re-splits them locally before even knowing whether it has anything to
    send. A run that changes rewrites its CSV, so its mtime, so the answer becomes no.
    """
    head = f"{rel_dir}/pDispatchComplete/"
    slices = [v for k, v in idx.items() if k.startswith(head)]
    if not slices:
        return False
    try:
        src = p.stat().st_mtime
    except OSError:
        return False
    return all(mtime + MTIME_TOLERANCE_S >= src for _, mtime in slices)


def split_dispatch_by_year(p, rel_dir):
    """Split {run}/{scen}/output_csv/pDispatchComplete.csv by year.

    Returns (temporary folder, [(path, relative key)]) to send, or None when the file has
    no 'y' column, in which case the caller sends the whole file instead. The temporary
    folder is the caller's to remove.
    """
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None
        cols = [h.strip().lower() for h in header]
        if "y" not in cols:
            return None
        yi = cols.index("y")
        tmpdir = Path(tempfile.mkdtemp())
        handles = {}  # year -> (file handle, csv.writer, path)
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
        finally:
            for fh, _, _ in handles.values():
                if not fh.closed:
                    fh.close()
        if not handles:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None
        print(f"    {DISPATCH_NAME} -> {len(handles)} year(s): {', '.join(sorted(handles))}")
        return tmpdir, [(tp, f"{rel_dir}/pDispatchComplete/y{y}.csv")
                        for y, (_, _, tp) in sorted(handles.items())]


EPM_REPO = os.environ["EPM_REPO"]
BRANCH   = os.environ["EPM_BRANCH"]
endpoint = os.environ["STORE_ENDPOINT"]
bucket   = os.environ["STORE_BUCKET"]

LOCAL  = Path(EPM_REPO) / "epm" / "output_view"
PREFIX = f"{BRANCH}/epm/output"

if not LOCAL.is_dir():
    print("  (no epm/output_view -> nothing to publish on the results side)")
    raise SystemExit(0)

ap = argparse.ArgumentParser()
ap.add_argument("--only", help="glob on the path relative to output_view (e.g. 'run_x/npv_external.csv')")
args = add_sync_args(ap).parse_args()

all_files = [p for p in LOCAL.rglob("*.csv") if p.is_file()]
if not all_files:
    print("  (output_view is empty -> nothing to publish)")
    raise SystemExit(0)

files = all_files
if args.only:
    files = [p for p in all_files if fnmatch.fnmatch(p.relative_to(LOCAL).as_posix(), args.only)]
    print(f"  --only {args.only} -> {len(files)}/{len(all_files)} csv")
    if not files:
        print("  (no file matches -> nothing to publish)")
        raise SystemExit(1)

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint})
idx = remote_index(fs, bucket, PREFIX)
print(f"  -> s3://{bucket}/{PREFIX}/  ({len(idx)} object(s) already there)")

tasks, dispatch = [], []
dispatch_skipped = 0
for p in files:
    rel = p.relative_to(LOCAL).as_posix()
    if p.name == DISPATCH_NAME:
        rel_dir = p.relative_to(LOCAL).parent.as_posix()
        if not args.force and dispatch_is_current(p, idx, rel_dir):
            dispatch_skipped += 1
            continue
        dispatch.append((p, rel, rel_dir))
        continue
    tasks.append((p, rel))

# The map layers: a run draws its own and writes it next to input_scenarios.csv, and EPM
# View reads the layers of the run folder before falling back on those of the input.
# Without them the fallback is all that is left, and it shows the zoning of the input
# folder rather than the one the run resolved. They are small, so they are sent too.
layers = [] if args.only else [p for p in LOCAL.glob("*/*.geojson") if p.is_file()]
tasks += [(p, p.relative_to(LOCAL).as_posix()) for p in layers]

if dispatch_skipped:
    print(f"  {dispatch_skipped} {DISPATCH_NAME} already split and up to date -> not re-read")

sent, skipped, failed = upload_many(fs, bucket, PREFIX, tasks, idx, jobs=args.jobs,
                                    force=args.force, check=args.check, label="csv + layers")

for p, rel, rel_dir in dispatch:
    if args.check:
        print(f"    [check] would re-split and send: {rel}")
        continue
    split = split_dispatch_by_year(p, rel_dir)
    if split is None:                      # no 'y' column, so the whole file goes
        s, _, f = upload_many(fs, bucket, PREFIX, [(p, rel)], idx, jobs=args.jobs,
                              force=True, label="csv (not split)")
        sent += s
        failed += f
        continue
    tmpdir, slices = split
    try:
        s, _, f = upload_many(fs, bucket, PREFIX, slices, idx, jobs=args.jobs,
                              force=True, label="year slice(s)")
        sent += s
        failed += f
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# --- run manifest (public R2 does not list folders, so EPM View reads this json) ---
# Always rewritten, even when nothing else is sent: it is what EPM View depends on to
# know which runs exist, and it costs nothing.
runs = sorted({
    p.relative_to(LOCAL).parts[0]
    for p in all_files
    if len(p.relative_to(LOCAL).parts) > 1
})
if args.check:
    print(f"  [check] manifest.json unchanged ({len(runs)} run(s))")
else:
    fs.pipe_file(f"{bucket}/{PREFIX}/manifest.json",
                 json.dumps({"runs": runs}, indent=2).encode("utf-8"))
    print(f"  manifest.json -> {len(runs)} run(s): {', '.join(runs)}")
    print(f"  OK ({sent} sent, {skipped} already up to date)")

raise SystemExit(report(failed))
