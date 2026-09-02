"""
Upload one INPUT folder to the data store, as a READABLE copy for EPM View.
Remote path expected by the app: {branch}/epm/input/{dataFolder}/...

Only the files missing from the store, or newer than their remote copy, are sent again.
See r2_sync, which holds the rule and the upload pool.

Environment variables (set by publish.ps1, which reads tools/.env):
  EPM_REPO, EPM_BRANCH, EPM_DATA_FOLDER, STORE_ENDPOINT, STORE_BUCKET
  plus AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (picked up by s3fs on its own)
"""
import argparse
import os
from pathlib import Path

import s3fs

from r2_sync import add_sync_args, remote_index, report, upload_many

EPM_REPO    = os.environ["EPM_REPO"]
BRANCH      = os.environ["EPM_BRANCH"]
DATA_FOLDER = os.environ["EPM_DATA_FOLDER"]
endpoint    = os.environ["STORE_ENDPOINT"]
bucket      = os.environ["STORE_BUCKET"]

LOCAL  = Path(EPM_REPO) / "epm" / "input" / DATA_FOLDER
PREFIX = f"{BRANCH}/epm/input/{DATA_FOLDER}"

if not LOCAL.is_dir():
    raise SystemExit(f"  folder not found: {LOCAL}")

args = add_sync_args(argparse.ArgumentParser()).parse_args()

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint})  # keys from AWS_* env
idx = remote_index(fs, bucket, PREFIX)

tasks = [(p, p.relative_to(LOCAL).as_posix()) for p in LOCAL.rglob("*") if p.is_file()]
print(f"  -> s3://{bucket}/{PREFIX}/")
sent, skipped, failed = upload_many(fs, bucket, PREFIX, tasks, idx, jobs=args.jobs,
                                    force=args.force, check=args.check, label="files")
if not args.check:
    print(f"  OK ({sent} sent, {skipped} already up to date)")
raise SystemExit(report(failed))
