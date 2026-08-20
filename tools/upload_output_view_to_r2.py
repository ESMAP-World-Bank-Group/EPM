"""
Upload the selected RESULTS (epm/output_view/) to the data store, in a READABLE
form for EPM View. Lands under {branch}/epm/output/...  (the app reads "output").

ONLY .csv files are sent (the large .gdx/logs are ignored).

Env variables (via publish.ps1 -> tools/.env):
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
    print("  (no epm/output_view -> nothing to publish on the results side)")
    raise SystemExit(0)

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint})

files = [p for p in LOCAL.rglob("*.csv") if p.is_file()]
if not files:
    print("  (output_view empty -> nothing to publish)")
    raise SystemExit(0)

print(f"  {len(files)} csv -> s3://{bucket}/{PREFIX}/")
for p in files:
    rel = p.relative_to(LOCAL).as_posix()
    fs.put_file(str(p), f"{bucket}/{PREFIX}/{rel}")
print(f"  OK ({len(files)} csv)")
