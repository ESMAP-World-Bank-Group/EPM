#!/usr/bin/env bash
# sync.sh — serveur (Linux) : recupere code + donnees.
#   git pull   (code + pointeurs)
#   dvc pull   (donnees depuis le store)
set -e
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$TOOLS")"
cd "$REPO"

# charger les cles si tools/.env existe (sinon: dvc lit .dvc/config.local ou les AWS_* deja exportes)
if [ -f "$TOOLS/.env" ]; then
  set -a; . "$TOOLS/.env"; set +a
fi

echo "[1/2] git pull ..."
git pull
echo "[2/2] dvc pull ..."
python -m dvc pull
echo "OK - code + donnees a jour."
