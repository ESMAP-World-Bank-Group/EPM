# -*- coding: utf-8 -*-
"""Classeur de suivi des données, DÉDUIT du build.

    python data_build/tracker.py --config data_build/build_casa.yaml

Le classeur n'est pas une source d'information : c'est une vue. Tout ce qu'il
affiche vient de build_report.json, donc de ce que le build fait réellement, plus
des blocs "sources" et "leads" du YAML. Il ne peut donc pas diverger de la donnée,
ce qui était le défaut du suivi tenu à la main sur Black Sea.

Lance le build en mode --check si le rapport est absent ou plus vieux que le YAML.
"""

import argparse
import json
import os
import re
import subprocess
import sys

import yaml
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))

# Code couleur repris de Black Sea : vert fait, jaune partiel, rouge à faire, gris hors périmètre.
FILLS = {"G": PatternFill("solid", fgColor="C6EFCE"),
         "Y": PatternFill("solid", fgColor="FFEB9C"),
         "R": PatternFill("solid", fgColor="FFC7CE"),
         "B": PatternFill("solid", fgColor="D9D9D9")}
HDR = PatternFill("solid", fgColor="1F4E79")
SUB = PatternFill("solid", fgColor="D6E4F7")
PRIO = {"P1": PatternFill("solid", fgColor="FCE4D6"),
        "P2": PatternFill("solid", fgColor="FFF2CC"),
        "P3": PatternFill("solid", fgColor="F2F2F2")}
WHITE = Font(color="FFFFFF", bold=True, size=10)
BOLD = Font(bold=True, size=9)
SMALL = Font(size=9)
TOP = Alignment(vertical="top", wrap_text=True)
THIN = Border(*[Side(style="thin", color="BFBFBF")] * 4)


def remaining(entry):
    """Ce qu'il reste à faire, ou chaîne vide si le todo est un compte rendu."""
    txt = str(entry.get("todo", "")).strip()
    return "" if txt.upper().startswith("FAIT") else txt


def state_of(entry):
    """Avancement d'une ressource, déduit du build seul.

    Deux faits suffisent. Le build a-t-il transformé le fichier (action), et
    reste-t-il du travail déclaré (todo) ? Rien n'est saisi à la main ici, donc
    rien ne peut rester vert par oubli.
    """
    if entry.get("work") == "IGNORER":
        return "B", "hors périmètre"
    touched = entry.get("action") not in ("keep", "shared")
    # Un todo qui commence par FAIT n'est pas un reste : c'est le compte rendu de
    # ce qui a été fait, gardé dans le YAML à côté de la décision qu'il explique.
    todo = bool(remaining(entry))
    if touched and not todo:
        return "G", "terminé"
    if touched:
        # FAIT = la donnée est en place ; le todo qui subsiste est un complément
        # (variante de scénario, source meilleure attendue), pas un manque.
        if entry.get("work") == "FAIT":
            return "Y", "rempli, compléments à venir"
        return "Y", "structure adaptée, données à faire"
    if not todo:
        return "G", "hérité, conforme"
    return "R", "hérité 2020, à traiter"


def phase_of(entry):
    """Numéro de phase, lu au début du todo. Convention : 'PHASE 3. ...'."""
    m = re.match(r"\s*PHASE\s+(\d+)", str(entry.get("todo", "")), re.I)
    return "P" + m.group(1) if m else ""


def sheet(wb, title, widths, headers, first=False):
    ws = wb.active if first else wb.create_sheet()
    ws.title = title
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for i, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=i, value=h)
        c.fill, c.font, c.alignment, c.border = HDR, WHITE, TOP, THIN
    ws.freeze_panes = "A2"
    return ws


def put(ws, row, values, fill=None, font=SMALL):
    for i, v in enumerate(values, 1):
        # les scalaires plies du YAML gardent un saut de ligne final
        c = ws.cell(row=row, column=i, value=v.strip() if isinstance(v, str) else v)
        c.alignment, c.border, c.font = TOP, THIN, font
        if fill:
            c.fill = fill
    return row + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "DATA_SOURCES_casa.xlsx"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    report_path = os.path.join(HERE, "build_report.json")

    # Le rapport doit refléter le YAML courant : sinon on relance le build à vide.
    if (not os.path.isfile(report_path)
            or os.path.getmtime(report_path) < os.path.getmtime(args.config)):
        print("Rapport absent ou périmé, relance du build en --check.")
        subprocess.check_call([sys.executable, os.path.join(HERE, "build.py"),
                               "--config", args.config, "--check"])

    report = json.load(open(report_path, "r", encoding="utf-8"))
    sources = report.get("sources") or {}
    rows = report.get("resources") or []

    wb = openpyxl.Workbook()

    # ── Feuille 1 : suivi ──────────────────────────────────────────────────
    ws = sheet(wb, "Suivi",
               [13, 22, 30, 34, 9, 8, 26, 13, 6, 20, 20, 11, 60, 70],
               ["Bloc", "Ressource", "Fichier", "Contenu", "Lignes", "Phase",
                "État", "Travail", "Prio", "Sources", "Onglet source", "Millésime",
                "Ce qui a été fait", "Ce qui reste"], first=True)
    r, bloc, tally = 2, None, {"G": 0, "Y": 0, "R": 0, "B": 0}
    for e in rows:
        if e.get("group") != bloc:
            bloc = e.get("group")
            c = ws.cell(row=r, column=1, value=bloc or "(non classé)")
            c.fill, c.font, c.alignment, c.border = SUB, Font(bold=True, size=9), TOP, THIN
            for i in range(2, 15):
                ws.cell(row=r, column=i).fill = SUB
            r += 1
        code, label = state_of(e)
        tally[code] += 1
        n = e.get("rows_out")
        r = put(ws, r, [bloc, e["resource"], e.get("path", ""), e.get("what", ""),
                        n if n != "-" else "-", phase_of(e), label,
                        e.get("work", ""), e.get("priority", ""),
                        ", ".join(e.get("source") or []),
                        e.get("sheet", ""), str(e.get("vintage", "") or ""),
                        e.get("note", ""), e.get("todo", "")])
        for i in (7,):
            ws.cell(row=r - 1, column=i).fill = FILLS[code]
        p = PRIO.get(e.get("priority"))
        if p:
            ws.cell(row=r - 1, column=9).fill = p

    r += 1
    put(ws, r, ["BILAN", "{} terminé | {} partiel | {} à faire | {} hors périmètre"
                .format(tally["G"], tally["Y"], tally["R"], tally["B"])], font=BOLD)

    # ── Feuille 2 : sources ────────────────────────────────────────────────
    ws = sheet(wb, "Sources", [18, 46, 12, 20, 30, 34, 80],
               ["Clé", "Source", "Date", "Accès", "Couvre", "Où la trouver", "Ce qu'elle contient"])
    r = 2
    for key, s in sources.items():
        r = put(ws, r, [key, s.get("name", ""), str(s.get("date", "")),
                        s.get("access", ""), s.get("covers", ""),
                        s.get("where", ""), s.get("note", "")])

    # ── Feuille 3 : pistes ─────────────────────────────────────────────────
    ws = sheet(wb, "Pistes", [14, 34, 30, 20, 62, 70],
               ["Pays", "Organisme", "Adresse", "Statut", "Contenu constaté",
                "Ce qu'on en fait"])
    r = 2
    # Le statut reprend le code couleur du suivi : ce qui est testé et disponible
    # est vert, ce qui est testé et vide est rouge. Une piste non testée reste grise.
    STAT = {"DISPONIBLE": "G", "PARTIEL": "Y", "RIEN": "R"}
    for lead in (cfg.get("leads") or []):
        r = put(ws, r, list(lead))
        statut = str(lead[3]) if len(lead) > 3 else ""
        code = next((c for k, c in STAT.items() if k in statut), "B")
        ws.cell(row=r - 1, column=4).fill = FILLS[code]

    wb.save(args.out)
    print("Écrit : {}".format(args.out))
    print("{} terminé | {} partiel | {} à faire | {} hors périmètre"
          .format(tally["G"], tally["Y"], tally["R"], tally["B"]))


if __name__ == "__main__":
    main()
