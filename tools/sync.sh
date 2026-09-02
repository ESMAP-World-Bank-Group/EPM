#!/usr/bin/env bash
# sync.sh: server side (Linux), fetches code and data.
#   git pull   (code and pointers)
#   dvc pull   (data from the store)
set -e
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$TOOLS")"
cd "$REPO"

# load the keys when tools/.env exists (otherwise dvc reads .dvc/config.local or the AWS_* already exported)
if [ -f "$TOOLS/.env" ]; then
  set -a; . "$TOOLS/.env"; set +a
fi

echo "[1/2] git pull ..."
git pull
echo "[2/2] dvc pull ..."
python -m dvc pull -r r2
echo "OK - code and data up to date."
