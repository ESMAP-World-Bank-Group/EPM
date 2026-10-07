#!/usr/bin/env bash
# sync.sh - server (Linux): gets the latest code + data.
#   git pull   (code + pointers)
#   dvc pull   (data from the store)
set -e
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$TOOLS")"
cd "$REPO"

# load the keys if tools/.env exists (otherwise: dvc reads .dvc/config.local or the AWS_* already exported)
if [ -f "$TOOLS/.env" ]; then
  set -a; . "$TOOLS/.env"; set +a
fi

echo "[1/2] git pull ..."
git pull
echo "[2/2] dvc pull ..."
python -m dvc pull
echo "OK - code + data up to date."
