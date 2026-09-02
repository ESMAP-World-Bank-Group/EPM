"""
Upload des RESULTATS choisis (epm/output_view/) vers le data store, en LISIBLE
pour EPM View. Arrive sous {branch}/epm/output/...  (l'app lit "output").

On n'envoie QUE les .csv (les gros .gdx/logs sont ignores), et parmi eux seulement
ceux qui manquent au store ou y sont plus vieux que la copie locale : voir r2_sync.

  --only <motif>   n'envoie que les chemins qui matchent (glob, relatif a output_view).
                   Pour republier UN fichier sans repasser sur les 8+ Go du dossier :
                     --only "simulations_run_*/npv_external.csv"
                   Le manifest reste calcule sur TOUS les runs presents : le filtrer
                   ferait disparaitre de EPM View les runs qu'on n'a pas reuploades.
  --force          tout renvoyer, meme ce qui est deja a jour
  --check          comparer local et distant sans rien envoyer

Variables d'env (via publish.ps1 -> tools/.env) :
  EPM_REPO, EPM_BRANCH, STORE_ENDPOINT, STORE_BUCKET + AWS_* (s3fs)
"""
import argparse
import csv
import fnmatch
import json
import os
import shutil
import tempfile
from pathlib import Path

import s3fs

from r2_sync import MTIME_TOLERANCE_S, add_sync_args, remote_index, report, upload_many

# pDispatchComplete est enorme (horaire x 16 ans) -> on le decoupe par annee
# ({name}/y{annee}.csv) pour que EPM View charge une annee a la fois (fluide).
DISPATCH_NAME = "pDispatchComplete.csv"


def dispatch_is_current(p, idx, rel_dir):
    """Les tranches annuelles deja dans le store sont-elles plus recentes que la source ?

    Les 36 pDispatchComplete du dossier pesent 4 Go : sans ce garde, chaque publish les
    relit et les redecoupe en local avant meme de savoir s'il a quelque chose a envoyer.
    Un run qui change reecrit son CSV, donc son mtime, donc la reponse devient non.
    """
    head = f"{rel_dir}/pDispatchComplete/"
    slices = [v for k, v in idx.items() if k.startswith(head)]
    if not slices:
        return False
    try:
        src = p.stat().st_mtime
    except OSError:
        return False
    return all(mtime + MTIME_TOLERANCE_S >= src for _, mtime in slices)


def split_dispatch_by_year(p, rel_dir):
    """Decoupe {run}/{scen}/output_csv/pDispatchComplete.csv par annee.

    Renvoie (dossier temporaire, [(chemin, cle relative)]) a envoyer, ou None si le
    fichier n'a pas de colonne 'y' (l'appelant enverra alors le fichier entier).
    Le dossier temporaire est a la charge de l'appelant.
    """
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None
        cols = [h.strip().lower() for h in header]
        if "y" not in cols:
            return None
        yi = cols.index("y")
        tmpdir = Path(tempfile.mkdtemp())
        handles = {}  # annee -> (file handle, csv.writer, chemin)
        try:
            for row in reader:
                if yi >= len(row):
                    continue
                y = row[yi].strip()
                if not y:
                    continue
                if y not in handles:
                    tp = tmpdir / f"y{y}.csv"
                    fh = tp.open("w", encoding="utf-8", newline="")
                    w = csv.writer(fh)
                    w.writerow(header)
                    handles[y] = (fh, w, tp)
                handles[y][1].writerow(row)
        finally:
            for fh, _, _ in handles.values():
                if not fh.closed:
                    fh.close()
        if not handles:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None
        print(f"    {DISPATCH_NAME} -> {len(handles)} annee(s) : {', '.join(sorted(handles))}")
        return tmpdir, [(tp, f"{rel_dir}/pDispatchComplete/y{y}.csv")
                        for y, (_, _, tp) in sorted(handles.items())]


EPM_REPO = os.environ["EPM_REPO"]
BRANCH   = os.environ["EPM_BRANCH"]
endpoint = os.environ["STORE_ENDPOINT"]
bucket   = os.environ["STORE_BUCKET"]

LOCAL  = Path(EPM_REPO) / "epm" / "output_view"
PREFIX = f"{BRANCH}/epm/output"

if not LOCAL.is_dir():
    print("  (pas de epm/output_view -> rien a publier cote resultats)")
    raise SystemExit(0)

ap = argparse.ArgumentParser()
ap.add_argument("--only", help="glob sur le chemin relatif a output_view (ex: 'run_x/npv_external.csv')")
args = add_sync_args(ap).parse_args()

all_files = [p for p in LOCAL.rglob("*.csv") if p.is_file()]
if not all_files:
    print("  (output_view vide -> rien a publier)")
    raise SystemExit(0)

files = all_files
if args.only:
    files = [p for p in all_files if fnmatch.fnmatch(p.relative_to(LOCAL).as_posix(), args.only)]
    print(f"  --only {args.only} -> {len(files)}/{len(all_files)} csv")
    if not files:
        print("  (aucun fichier ne matche -> rien a publier)")
        raise SystemExit(1)

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint})
idx = remote_index(fs, bucket, PREFIX)
print(f"  -> s3://{bucket}/{PREFIX}/  ({len(idx)} objet(s) deja en place)")

tasks, dispatch = [], []
dispatch_skipped = 0
for p in files:
    rel = p.relative_to(LOCAL).as_posix()
    if p.name == DISPATCH_NAME:
        rel_dir = p.relative_to(LOCAL).parent.as_posix()
        if not args.force and dispatch_is_current(p, idx, rel_dir):
            dispatch_skipped += 1
            continue
        dispatch.append((p, rel, rel_dir))
        continue
    tasks.append((p, rel))

# Les couches de carte : un run dessine la sienne et l'ecrit a cote de input_scenarios.csv,
# et EPM View lit les couches du dossier de run avant de retomber sur celles de l'input.
# Sans elles le repli est tout ce qui reste et il montre le zonage du dossier d'entree
# plutot que celui que le run a resolu. Elles sont petites, on les envoie aussi.
layers = [] if args.only else [p for p in LOCAL.glob("*/*.geojson") if p.is_file()]
tasks += [(p, p.relative_to(LOCAL).as_posix()) for p in layers]

if dispatch_skipped:
    print(f"  {dispatch_skipped} {DISPATCH_NAME} deja decoupe(s) et a jour -> non relu(s)")

sent, skipped, failed = upload_many(fs, bucket, PREFIX, tasks, idx, jobs=args.jobs,
                                    force=args.force, check=args.check, label="csv + couches")

for p, rel, rel_dir in dispatch:
    if args.check:
        print(f"    [check] a redecouper et envoyer : {rel}")
        continue
    split = split_dispatch_by_year(p, rel_dir)
    if split is None:                      # pas de colonne 'y' -> le fichier entier
        s, _, f = upload_many(fs, bucket, PREFIX, [(p, rel)], idx, jobs=args.jobs,
                              force=True, label="csv (non decoupe)")
        sent += s
        failed += f
        continue
    tmpdir, slices = split
    try:
        s, _, f = upload_many(fs, bucket, PREFIX, slices, idx, jobs=args.jobs,
                              force=True, label="tranche(s) annuelle(s)")
        sent += s
        failed += f
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# --- manifest des runs (R2 public ne liste pas les dossiers -> EPM View lit ce json) ---
# Toujours reecrit, meme quand rien d'autre ne part : c'est lui dont depend EPM View pour
# savoir quels runs existent, et il ne coute rien.
runs = sorted({
    p.relative_to(LOCAL).parts[0]
    for p in all_files
    if len(p.relative_to(LOCAL).parts) > 1
})
if args.check:
    print(f"  [check] manifest.json inchange ({len(runs)} run(s))")
else:
    fs.pipe_file(f"{bucket}/{PREFIX}/manifest.json",
                 json.dumps({"runs": runs}, indent=2).encode("utf-8"))
    print(f"  manifest.json -> {len(runs)} run(s) : {', '.join(runs)}")
    print(f"  OK ({sent} envoye(s), {skipped} deja a jour)")

raise SystemExit(report(failed))
