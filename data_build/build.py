# -*- coding: utf-8 -*-
"""Construit un dossier d'entrées EPM à partir d'un dossier de référence et d'un YAML.

Le moteur est générique : il ne contient aucun nom de pays, de zone, de saison ni de
centrale. Tout ce qui est spécifique à un déploiement vit dans le YAML de build et
dans les tables de correspondance CSV de mappings/.

    python data_build/build.py --config data_build/build_casa.yaml --check
    python data_build/build.py --config data_build/build_casa.yaml --apply

--check construit dans un dossier temporaire et compare à la cible existante, sans
rien écrire. C'est le test qui compte : la cible doit être reproductible.
--apply remplace la cible par le résultat du build.

Verbes disponibles (champ "action" d'une ressource) :

    keep          ne rien faire, le fichier est repris tel quel        (défaut)
    empty         ne garder que la ligne d'en-tête
    rows          remplacer tout le contenu par les lignes données
    drop_where    supprimer les lignes dont une colonne vaut une valeur
    drop_column   supprimer une colonne
    table         remplacer le fichier entier par une table produite par un extracteur

Un fichier absent du YAML est simplement recopié.
"""

import argparse
import csv
import filecmp
import json
import os
import shutil
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


# ── Lecture / écriture CSV ────────────────────────────────────────────────────

def read_csv(path):
    """Retourne (en-tête, lignes). utf-8-sig pour absorber un éventuel BOM."""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def write_csv(path, header, rows):
    """Écrit en UTF-8 avec des fins de ligne LF, sans ligne vide finale."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def col_index(header, name, path):
    """Indice d'une colonne, par nom, insensible à la casse et aux espaces."""
    want = str(name).strip().lower()
    for i, h in enumerate(header):
        if str(h).strip().lower() == want:
            return i
    raise KeyError("colonne '{}' absente de {} (en-tête : {})".format(name, path, header))


# ── Les verbes ────────────────────────────────────────────────────────────────

def act_keep(header, rows, spec, path):
    return header, rows


def act_empty(header, rows, spec, path):
    return header, []


def act_rows(header, rows, spec, path):
    """Remplace le contenu. Un scalaire vaut une ligne d'une seule cellule."""
    new = []
    for item in spec.get("rows", []):
        new.append([item] if not isinstance(item, (list, tuple)) else list(item))
    return header, new


def act_drop_where(header, rows, spec, path):
    i = col_index(header, spec["column"], path)
    values = spec["value"]
    if not isinstance(values, (list, tuple)):
        values = [values]
    values = {str(v).strip() for v in values}
    return header, [r for r in rows if len(r) <= i or str(r[i]).strip() not in values]


def act_drop_column(header, rows, spec, path):
    i = col_index(header, spec["column"], path)
    cut = lambda r: [c for j, c in enumerate(r) if j != i]
    return cut(header), [cut(r) for r in rows]


def act_table(header, rows, spec, path):
    """Remplace le fichier entier par une table produite hors du moteur.

    Le champ "from" designe un CSV, relatif a data_build/, ecrit par un des
    extract_*.py. Le moteur ne sait pas ce qu'il y a dedans : il verifie seulement
    que la table n'est pas vide et qu'elle a autant de colonnes partout.
    """
    src = os.path.join(HERE, str(spec["from"]).replace("/", os.sep))
    if not os.path.isfile(src):
        raise IOError("ressource '{}' : table absente ({}). Lancer l'extracteur "
                      "correspondant.".format(path, spec["from"]))
    new_header, new_rows = read_csv(src)
    if not new_rows:
        raise IOError("ressource '{}' : la table {} est vide.".format(path, spec["from"]))
    widths = {len(r) for r in new_rows} | {len(new_header)}
    if len(widths) > 1:
        raise IOError("ressource '{}' : la table {} a des lignes de largeurs "
                      "differentes ({}).".format(path, spec["from"], sorted(widths)))
    return new_header, new_rows


ACTIONS = {
    "keep": act_keep,
    "empty": act_empty,
    "rows": act_rows,
    "drop_where": act_drop_where,
    "drop_column": act_drop_column,
    "table": act_table,
}


# ── Résolution des chemins de ressources ──────────────────────────────────────

def resource_paths(datapackage, source_dir):
    """nom de ressource -> chemin relatif.

    Deux autorités, dans cet ordre. datapackage.json donne le vocabulaire commun à
    tous les déploiements, mais il est incomplet : au 2026-08, 12 paramètres lus par
    le config.csv d'Asie centrale n'y figurent pas (pReserveSeasonFlag,
    pPlanningReserveMarginZone, pContractedTrade*, etc.). Le config.csv du
    déploiement, lui, est exact par construction puisque c'est ce que GAMS lit.
    """
    known = {}
    if os.path.isfile(datapackage):
        with open(datapackage, "r", encoding="utf-8") as fh:
            pkg = json.load(fh)
        known.update({r["name"]: r["path"] for r in pkg.get("resources", []) if "name" in r})

    cfg_path = os.path.join(source_dir, "config.csv")
    if os.path.isfile(cfg_path):
        _, rows = read_csv(cfg_path)
        for r in rows:                      # metadata, paramNames, file
            if len(r) >= 3 and r[1].strip() and r[2].strip().endswith(".csv"):
                known.setdefault(r[1].strip(), r[2].strip())
    return known


def resolve(name, spec, known, source_dir):
    """Chemin relatif d'une ressource : 'path' explicite, sinon datapackage."""
    rel = spec.get("path") or known.get(name)
    if rel is None:
        raise KeyError("ressource '{}' inconnue : absente de datapackage.json et "
                       "sans champ 'path'".format(name))
    rel = rel.replace("\\", "/")
    if rel.startswith(".."):
        return None, rel   # ressource partagée, hors déploiement
    if not os.path.isfile(os.path.join(source_dir, rel.replace("/", os.sep))):
        raise IOError("ressource '{}' : fichier introuvable dans la référence ({})"
                      .format(name, rel))
    return rel, rel


# ── Build ─────────────────────────────────────────────────────────────────────

def build(cfg, out_dir):
    """Recopie la référence puis applique les transformations. Retourne le rapport."""
    source_dir = os.path.join(REPO, cfg["deployment"]["source"].replace("/", os.sep))
    datapackage = os.path.join(REPO, cfg["deployment"]["datapackage"].replace("/", os.sep))

    if not os.path.isdir(source_dir):
        raise IOError("dossier de référence introuvable : " + source_dir)

    # 1. Tout recopier, pour ne rien perdre (geojson, options cplex, etc.).
    # copyfile et non copy2 : on ne copie que le contenu. Recopier les métadonnées
    # échoue sous Windows dans cet environnement (WinError 127), et les dates de
    # fichier de la référence n'ont de toute façon aucune valeur pour le build.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(source_dir, out_dir, copy_function=shutil.copyfile)

    known = resource_paths(datapackage, source_dir)
    report = []

    # 2. Appliquer les transformations déclarées.
    for name, spec in (cfg.get("resources") or {}).items():
        spec = spec or {}
        action = spec.get("action", "keep")
        if action not in ACTIONS:
            raise KeyError("ressource '{}' : action '{}' inconnue (connues : {})"
                           .format(name, action, ", ".join(sorted(ACTIONS))))

        rel, shown = resolve(name, spec, known, source_dir)
        # Tous les champs descriptifs du YAML sont recopiés tels quels dans le
        # rapport : le moteur n'a pas à connaître le vocabulaire de suivi.
        entry = dict(spec)
        entry.pop("rows", None)
        entry.update({"resource": name, "path": shown, "action": action})

        if rel is None:                       # ressource partagée, on n'y touche pas
            entry["rows_in"] = entry["rows_out"] = "-"
            entry["action"] = "shared"
            report.append(entry)
            continue

        path = os.path.join(out_dir, rel.replace("/", os.sep))
        header, rows = read_csv(path)
        entry["rows_in"] = len(rows)

        if action != "keep":
            header, rows = ACTIONS[action](header, rows, spec, rel)
            write_csv(path, header, rows)

        entry["rows_out"] = len(rows)
        report.append(entry)

    return report


def diff(a, b):
    """Liste des fichiers qui diffèrent entre deux arborescences."""
    out = []
    for root, _, files in os.walk(a):
        for f in files:
            pa = os.path.join(root, f)
            rel = os.path.relpath(pa, a)
            pb = os.path.join(b, rel)
            if not os.path.exists(pb):
                out.append((rel, "absent de la cible"))
            elif not filecmp.cmp(pa, pb, shallow=False):
                out.append((rel, "contenu different"))
    for root, _, files in os.walk(b):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), b)
            if not os.path.exists(os.path.join(a, rel)):
                out.append((rel, "en trop dans la cible"))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--check", action="store_true",
                    help="construire a cote et comparer, sans rien ecrire")
    ap.add_argument("--apply", action="store_true",
                    help="remplacer la cible par le resultat du build")
    args = ap.parse_args()

    if not (args.check or args.apply):
        ap.error("choisir --check ou --apply")

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    target = os.path.join(REPO, cfg["deployment"]["target"].replace("/", os.sep))
    staging = os.path.join(tempfile.mkdtemp(prefix="epm_build_"), "out")

    report = build(cfg, staging)

    if args.check:
        if os.path.isdir(target):
            d = diff(staging, target)
            if d:
                print("ECART entre le build et la cible existante :")
                for rel, why in d:
                    print("   {:<45s} {}".format(rel, why))
            else:
                print("Le build reproduit la cible a l'octet pres.")
        else:
            print("Cible inexistante, rien a comparer.")
        print("(build jete : {})".format(staging))
    else:
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.move(staging, target)
        print("Ecrit : " + target)

    out = os.path.join(HERE, "build_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"config": os.path.basename(args.config),
                   "sources": cfg.get("sources", {}),
                   "resources": report}, fh, indent=2, ensure_ascii=False,
                  default=str)   # les dates du YAML sont des objets date
    print("Rapport : " + out)

    acted = sum(1 for r in report if r["action"] not in ("keep", "shared"))
    print("{} ressources declarees, {} transformees.".format(len(report), acted))


if __name__ == "__main__":
    main()
