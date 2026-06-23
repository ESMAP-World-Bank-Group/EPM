"""
Upload d'un dossier d'INPUTS vers le data store (copie LISIBLE pour EPM View).
Chemin distant attendu par l'app : {branch}/epm/input/{dataFolder}/...

Variables d'environnement (fournies par publish.ps1, qui lit tools/.env) :
  EPM_REPO, EPM_BRANCH, EPM_DATA_FOLDER, STORE_ENDPOINT, STORE_BUCKET
  + AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (lues automatiquement par s3fs)
"""
import os
from pathlib import Path

import s3fs

EPM_REPO    = os.environ["EPM_REPO"]
BRANCH      = os.environ["EPM_BRANCH"]
DATA_FOLDER = os.environ["EPM_DATA_FOLDER"]
endpoint    = os.environ["STORE_ENDPOINT"]
bucket      = os.environ["STORE_BUCKET"]

LOCAL  = Path(EPM_REPO) / "epm" / "input" / DATA_FOLDER
PREFIX = f"{BRANCH}/epm/input/{DATA_FOLDER}"

if not LOCAL.is_dir():
    raise SystemExit(f"  dossier introuvable : {LOCAL}")

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint})  # cles via AWS_* env

files = [p for p in LOCAL.rglob("*") if p.is_file()]
print(f"  {len(files)} fichiers -> s3://{bucket}/{PREFIX}/")
for p in files:
    rel = p.relative_to(LOCAL).as_posix()
    fs.put_file(str(p), f"{bucket}/{PREFIX}/{rel}")
print(f"  OK ({len(files)} fichiers)")
