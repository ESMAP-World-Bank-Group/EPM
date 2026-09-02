"""
Envoi vers le data store : ce qui a change, et rien d'autre.

Les deux uploaders renvoyaient l'integralite de leur dossier a chaque publish. Cote
resultats cela faisait ~10 Go a chaque fois, dont 4 Go de pDispatchComplete relus et
redecoupes en local avant meme d'etre envoyes, que le run ait bouge ou non.

La regle appliquee ici est celle de rsync : un fichier ne repart que si le distant ne l'a
pas, n'a pas la meme taille, ou est plus vieux que le fichier local.

Deux choix expliquent le reste du fichier.

Pas de hash. Comparer les md5 serait exact, mais hasher 10 Go coute plusieurs minutes a
chaque publish : on deplacerait la lenteur au lieu de l'enlever. Taille plus date suffit
pour des fichiers qu'un run reecrit entierement.

Pas de fichier d'etat local. La verite est le bucket, lu en une requete de listing. Rien
a corrompre, rien a gitignorer, rien qui mente si on publie depuis une autre machine ou
si quelqu'un a touche au store. Et le premier publish apres cette modif est deja rapide :
il n'y a pas de passe d'amorcage a 10 Go.

En cas de doute, on envoie. Un saut demande une preuve positive que le distant est bon et
plus recent ; horloge decalee, taille differente, cle absente, listing en echec, tout cela
mene a un envoi. L'echec possible est un envoi inutile, jamais un fichier manquant.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone

# Assez de fils pour couvrir la latence aller-retour derriere le proxy, pas assez pour
# que R2 commence a repondre 503.
DEFAULT_JOBS = int(os.environ.get("EPM_UPLOAD_JOBS", "8"))

# Les horloges du poste et du store ne sont pas synchronisees a la seconde pres.
MTIME_TOLERANCE_S = 2.0

RETRIES = 3


def add_sync_args(ap):
    """Les trois options communes aux deux uploaders."""
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                    help=f"envois en parallele (defaut {DEFAULT_JOBS})")
    ap.add_argument("--force", action="store_true",
                    help="tout renvoyer, meme ce qui est deja a jour")
    ap.add_argument("--check", action="store_true",
                    help="comparer local et distant sans rien envoyer")
    return ap


def _epoch(info):
    ts = info.get("LastModified") or info.get("last_modified")
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if ts.tzinfo is None:            # R2 repond en UTC, sans le dire toujours
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()


def remote_index(fs, bucket, prefix):
    """Chemin relatif au prefixe -> (taille, date en secondes). Vide si le listing echoue."""
    root = f"{bucket}/{prefix}"
    try:
        found = fs.find(root, detail=True)
    except Exception as e:                                    # pragma: no cover
        print(f"  (listing distant indisponible : {e!r} -> tout sera renvoye)")
        return {}
    idx = {}
    head = root + "/"
    for key, info in found.items():
        if not key.startswith(head):
            continue
        idx[key[len(head):]] = (int(info.get("size") or 0), _epoch(info))
    return idx


def is_current(local, entry):
    """Le distant porte-t-il deja ce fichier, dans cette version ?"""
    if not entry:
        return False
    size, mtime = entry
    try:
        st = local.stat()
    except OSError:
        return False
    return st.st_size == size and mtime + MTIME_TOLERANCE_S >= st.st_mtime


def plan(tasks, idx, force=False):
    """Repartit les (chemin local, cle relative) entre ce qui part et ce qui reste."""
    todo, skipped = [], []
    for local, rel in tasks:
        (todo if force or not is_current(local, idx.get(rel)) else skipped).append((local, rel))
    return todo, skipped


def upload_many(fs, bucket, prefix, tasks, idx, jobs=DEFAULT_JOBS, force=False,
                check=False, label="fichiers"):
    """Envoie ce qui doit l'etre. Renvoie (envoyes, sautes, echecs)."""
    todo, skipped = plan(tasks, idx, force)
    print(f"  {len(tasks)} {label} : {len(todo)} a envoyer, {len(skipped)} deja a jour")
    if check:
        for _, rel in todo[:20]:
            print(f"    [check] a envoyer : {rel}")
        if len(todo) > 20:
            print(f"    [check] ... et {len(todo) - 20} autres")
        return 0, len(skipped), []
    if not todo:
        return 0, len(skipped), []

    def send(item):
        local, rel = item
        for attempt in range(1, RETRIES + 1):
            try:
                fs.put_file(str(local), f"{bucket}/{prefix}/{rel}")
                return None
            except Exception as e:
                if attempt == RETRIES:
                    return (rel, repr(e))
                time.sleep(2 ** attempt)   # le proxy WB coupe, ca repasse au coup d'apres

    failed = []
    step = max(1, len(todo) // 10)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for i, res in enumerate(pool.map(send, todo), 1):
            if res:
                failed.append(res)
            if i % step == 0 or i == len(todo):
                print(f"    {i}/{len(todo)}")
    return len(todo) - len(failed), len(skipped), failed


def report(failed):
    """Recapitule les echecs et donne le code de sortie. 0 quand tout est passe."""
    if not failed:
        return 0
    print(f"  ECHECS : {len(failed)} fichier(s) non envoye(s)")
    for rel, err in failed[:10]:
        print(f"    {rel} : {err}")
    if len(failed) > 10:
        print(f"    ... et {len(failed) - 10} autres")
    return 1
