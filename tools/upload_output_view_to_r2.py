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

# A run now draws its own map and writes the pair next to input_scenarios.csv,
# so EPM View reads the layers from the run folder before falling back to the
# input one. Send them too -- they are small, and without them the fallback is
# the only thing left and it shows the zoning of whatever the input folder
# happens to ship rather than the one the run solved.
layers = [p for p in LOCAL.glob("*/*.geojson") if p.is_file()]
for p in layers:
    fs.put_file(str(p), f"{bucket}/{PREFIX}/{p.relative_to(LOCAL).as_posix()}")
print(f"  OK ({len(layers)} map layer(s))")
