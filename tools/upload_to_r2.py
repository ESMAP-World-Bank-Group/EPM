"""
Upload d'un dossier d'INPUTS vers le data store (copie LISIBLE pour EPM View).
Chemin distant attendu par l'app : {branch}/epm/input/{dataFolder}/...

Seuls les fichiers absents ou plus recents que leur copie distante repartent : voir
r2_sync, qui porte la regle et le pool d'envoi.

Variables d'environnement (fournies par publish.ps1, qui lit tools/.env) :
  EPM_REPO, EPM_BRANCH, EPM_DATA_FOLDER, STORE_ENDPOINT, STORE_BUCKET
  + AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (lues automatiquement par s3fs)
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
    raise SystemExit(f"  dossier introuvable : {LOCAL}")

args = add_sync_args(argparse.ArgumentParser()).parse_args()

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint})  # cles via AWS_* env
idx = remote_index(fs, bucket, PREFIX)

tasks = [(p, p.relative_to(LOCAL).as_posix()) for p in LOCAL.rglob("*") if p.is_file()]
print(f"  -> s3://{bucket}/{PREFIX}/")
sent, skipped, failed = upload_many(fs, bucket, PREFIX, tasks, idx, jobs=args.jobs,
                                    force=args.force, check=args.check, label="fichiers")
if not args.check:
    print(f"  OK ({sent} envoye(s), {skipped} deja a jour)")
raise SystemExit(report(failed))
