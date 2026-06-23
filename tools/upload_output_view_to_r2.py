"""
Upload des RESULTATS choisis (epm/output_view/) vers le data store, en LISIBLE
pour EPM View. Arrive sous {branch}/epm/output/...  (l'app lit "output").

On n'envoie QUE les .csv (les gros .gdx/logs sont ignores).

Variables d'env (via publish.ps1 -> tools/.env) :
  EPM_REPO, EPM_BRANCH, STORE_ENDPOINT, STORE_BUCKET + AWS_* (s3fs)
"""
import os
from pathlib import Path

import s3fs

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
    fs.put_file(str(p), f"{bucket}/{PREFIX}/{rel}")
print(f"  OK ({len(files)} csv)")
