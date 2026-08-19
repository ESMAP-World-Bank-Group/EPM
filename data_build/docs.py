# -*- coding: utf-8 -*-
"""Page DATA_SOURCES, DÉDUITE du build.

    python data_build/docs.py --config data_build/build_casa.yaml

Même principe que Black Sea : la documentation des sources est une PAGE, posée à
côté des données qu'elle décrit, et non un classeur tenu à la main. Ici elle est
produite à partir de build_report.json et du YAML, donc des mêmes faits que le
build lui-même : elle ne peut pas diverger de ce qui a réellement été fait.

Écrit DATA_SOURCES.html dans le dossier de déploiement, à côté des CSV décrits.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracker import phase_of, remaining, state_of   # noqa: E402  même état que le suivi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def h(text):
    """Échappe le texte : tout ce qui vient du YAML est du texte, pas du balisage."""
    return (str(text or "").strip()
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def anchor(key, prefix="src"):
    """Ancre stable : c'est elle qu'on colle dans un courriel pour pointer un point précis."""
    return prefix + "-" + re.sub(r"[^a-z0-9]+", "-", str(key).lower()).strip("-")


def horizon(deployment):
    """Horizon lu dans l'en-tête de pDemandForecast : la donnée le dit mieux que moi."""
    path = os.path.join(ROOT, deployment, "load", "pDemandForecast.csv")
    if not os.path.isfile(path):
        return "—"
    with open(path, "r", encoding="utf-8") as fh:
        cols = fh.readline().strip().split(",")
    years = [c for c in cols if c.strip().isdigit()]
    if not years:
        return "—"
    return "{}–{} · {} pas".format(years[0], years[-1], len(years))


def perimetre(deployment):
    """Zones et pays lus dans zcmap : le périmètre est un fait du modèle."""
    path = os.path.join(ROOT, deployment, "zcmap.csv")
    if not os.path.isfile(path):
        return "—"
    with open(path, "r", encoding="utf-8-sig") as fh:
        head = [c.strip() for c in fh.readline().strip().split(",")]
        rows = [r.strip().split(",") for r in fh if r.strip()]
    def col(name):
        if name not in head:
            return []
        i = head.index(name)
        return sorted({r[i].strip() for r in rows if len(r) > i and r[i].strip()})
    return "{} zones · {} pays".format(len(col("z")), len(col("c")))


CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 max-width:1280px;margin:0 auto;padding:24px 32px;color:#2c3e50;line-height:1.55;font-size:14px}
h1{border-bottom:3px solid #1a5276;padding-bottom:10px;margin-bottom:4px;font-size:1.7em}
h2{border-bottom:1px solid #d5d8dc;margin-top:44px;padding-bottom:6px;color:#1a5276;font-size:1.25em}
h3{margin-top:26px;font-size:1.02em;color:#1a5276}
p.meta{color:#888;font-size:.82em;margin:4px 0 26px}
table{border-collapse:collapse;width:100%;margin:12px 0 20px;font-size:.87em}
th{background:#2c3e50;color:#fff;padding:9px 12px;text-align:left;font-weight:600;white-space:nowrap}
td{padding:7px 12px;border-bottom:1px solid #eaecee;vertical-align:top}
tr:hover td{background:#f8f9fa}
.cat-row td{background:#eaecee;font-weight:700;color:#444;font-size:.78em;letter-spacing:.08em;
 text-transform:uppercase;padding:5px 12px}
.st{white-space:nowrap;font-weight:600;font-size:.92em}
.st-G{background:#edf7ed;color:#276327}
.st-Y{background:#fdf6e3;color:#8a5e12}
.st-R{background:#fdeeec;color:#a33228}
.st-B{background:#f4f4f4;color:#999}
code{background:#f2f3f4;padding:1px 5px;border-radius:3px;font-family:monospace;font-size:.9em}
a{color:#1a5276}
.legend{margin:10px 0 26px;font-size:.82em;color:#666}
.legend span{margin-right:20px}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin-right:5px}
.toc{background:#f8f9fa;border:1px solid #e0e0e0;border-radius:4px;padding:12px 20px;
 display:inline-block;min-width:300px}
.toc ul{margin:4px 0;padding-left:18px}
.src{border-left:3px solid #d5d8dc;padding:2px 0 2px 14px;margin:14px 0 20px}
.src .tag{font-size:.74em;color:#fff;background:#7f8c8d;border-radius:3px;padding:1px 6px;margin-left:6px}
.src .tag.open{background:#27865a}.src .tag.conf{background:#a33228}.src .tag.hyp{background:#b8860b}
.src p{margin:5px 0;color:#555}
.todo{color:#8a5e12}
.muted{color:#999}
.foot{margin-top:44px;padding-top:12px;border-top:1px solid #eaecee;font-size:.78em;color:#999}
"""

ACCES = {"open_source": ("open", "source ouverte"),
         "internal_wb": ("", "interne"),
         "client_confidential": ("conf", "confidentiel client"),
         "assumption": ("hyp", "hypothèse")}


def render(cfg, report, deployment):
    src = report.get("sources") or {}
    res = report.get("resources") or []
    today = date.today()
    out = []
    a = out.append

    a('<!DOCTYPE html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">')
    a('<meta name="viewport" content="width=device-width, initial-scale=1">')
    a('<title>Data Sources — EPM — Asie centrale 2026</title>')
    a('<style>%s</style>\n</head>\n<body>' % CSS)
    a('<h1>Data Sources — EPM — Asie centrale 2026</h1>')
    a('<p class="meta">Générée le %s · déploiement <code>%s</code> · %s · horizon %s</p>'
      % (today, h(os.path.basename(deployment)), h(perimetre(deployment)),
         h(horizon(deployment))))

    a('<div class="legend">')
    for col, txt in (("#edf7ed", "traité"), ("#fdf6e3", "partiel"),
                     ("#fdeeec", "hérité 2020, à traiter"), ("#f4f4f4", "hors périmètre")):
        a('<span><span class="sw" style="background:%s;border:1px solid #ddd"></span>%s</span>'
          % (col, txt))
    a('</div>')

    a('<div class="toc"><b>Sommaire</b><ul>'
      '<li><a href="#resources">Ressources du modèle</a></li>'
      '<li><a href="#detail">Ce qui a été fait, ressource par ressource</a></li>'
      '<li><a href="#sources">Les sources</a></li>'
      '<li><a href="#publiques">Sources publiques accessibles en ligne</a></li>'
      '</ul></div>')

    # ── Table des ressources ──────────────────────────────────────────────────
    a('<h2 id="resources">Ressources du modèle</h2>')
    a('<p class="meta">La colonne <i>Contenu</i> décrit le fichier tel qu’il vient du '
      'modèle 2020 ; l’état dit ce que le build en a fait, et le détail suit plus bas.</p>')
    a('<table><thead><tr><th>Ressource</th><th>Fichier</th><th>Contenu hérité</th>'
      '<th>Lignes</th><th>État</th><th>Phase</th><th>Millésime</th><th>Sources</th>'
      '</tr></thead><tbody>')
    groupe = None
    tally = {"G": 0, "Y": 0, "R": 0, "B": 0}
    for e in res:
        if e.get("group") != groupe:
            groupe = e.get("group")
            a('<tr class="cat-row"><td colspan="8">%s</td></tr>' % h(groupe or "non classé"))
        code, label = state_of(e)
        tally[code] += 1
        liens = " ".join('<a href="#%s">%s</a>' % (anchor(k), h(k)) for k in (e.get("source") or []))
        n = e.get("rows_out")
        nom = '<code>%s</code>' % h(e["resource"])
        if e.get("action") not in ("keep", "shared"):
            nom = '<a href="#%s">%s</a>' % (anchor(e["resource"], "res"), nom)
        a('<tr><td>%s</td><td><code>%s</code></td><td>%s</td>'
          '<td>%s</td><td class="st st-%s">%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
          % (nom, h(e.get("path", "")), h(e.get("what", "")),
             h(n) if n not in (None, "-") else '<span class="muted">—</span>',
             code, h(label), h(phase_of(e).lstrip("P")), h(e.get("vintage", "")),
             liens or '<span class="muted">—</span>'))
    a('</tbody></table>')
    a('<p class="meta">%d ressources : %d terminé · %d partiel · %d à faire · %d hors périmètre</p>'
      % (len(res), tally["G"], tally["Y"], tally["R"], tally["B"]))

    # ── Détail par ressource transformée ──────────────────────────────────────
    a('<h2 id="detail">Ce qui a été fait, ressource par ressource</h2>')
    a('<p class="meta">Seules les ressources effectivement transformées par le build '
      'apparaissent ici. Les autres sont reprises telles quelles du modèle 2020.</p>')
    for e in res:
        if e.get("action") in ("keep", "shared"):
            continue
        code, label = state_of(e)
        a('<h3 id="%s"><code>%s</code> <span class="muted">— %s</span></h3>'
          % (anchor(e["resource"], "res"), h(e["resource"]), h(label)))
        if e.get("note"):
            a('<p>%s</p>' % h(e["note"]))
        reste = remaining(e)
        if reste:
            a('<p class="todo"><b>Reste :</b> %s</p>' % h(reste))

    # ── Les sources ───────────────────────────────────────────────────────────
    a('<h2 id="sources">Les sources</h2>')
    for key, s in src.items():
        tag, texte = ACCES.get(s.get("access", ""), ("", s.get("access", "")))
        a('<div class="src" id="%s">' % anchor(key))
        a('<b>%s</b> <span class="tag %s">%s</span>' % (h(s.get("name", key)), tag, h(texte)))
        meta = [x for x in [h(s.get("date", "")), h(s.get("covers", ""))] if x]
        if meta:
            a('<p class="meta">%s</p>' % " · ".join(meta))
        if s.get("where"):
            a('<p class="meta">Emplacement : <code>%s</code></p>' % h(s["where"]))
        if s.get("note"):
            a('<p>%s</p>' % h(s["note"]))
        a('</div>')

    # ── Sources publiques en ligne ────────────────────────────────────────────
    leads = cfg.get("leads") or []
    if leads:
        a('<h2 id="publiques">Sources publiques accessibles en ligne</h2>')
        a('<table><thead><tr><th>Pays</th><th>Organisme</th><th>Adresse</th>'
          '<th>Statut</th><th>Contenu constaté</th><th>Ce qu\'on en fait</th>'
          '</tr></thead><tbody>')
        for lead in leads:
            row = list(lead) + [""] * (6 - len(lead))
            st = str(row[3])
            code = ("G" if "DISPONIBLE" in st else "Y" if "PARTIEL" in st
                    else "R" if "RIEN" in st else "B")
            adr = h(row[2])
            if "." in adr and adr != "-":
                adr = '<a href="https://%s">%s</a>' % (adr.split()[0].split("/")[0], adr)
            a('<tr><td>%s</td><td>%s</td><td><code>%s</code></td>'
              '<td class="st st-%s">%s</td><td>%s</td><td>%s</td></tr>'
              % (h(row[0]), h(row[1]), adr, code, h(row[3]), h(row[4]), h(row[5])))
        a('</tbody></table>')

    a('<div class="foot">Page produite par <code>data_build/docs.py</code> à partir de '
      '<code>build_report.json</code> et de <code>build_casa.yaml</code>. '
      'Ne pas éditer à la main : régénérer.</div>')
    a('</body>\n</html>')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    deployment = cfg["deployment"]["target"]   # relatif à la racine du dépôt
    report_path = os.path.join(HERE, "build_report.json")

    if (not os.path.isfile(report_path)
            or os.path.getmtime(report_path) < os.path.getmtime(args.config)):
        print("Rapport absent ou périmé, relance du build en --check.")
        subprocess.check_call([sys.executable, os.path.join(HERE, "build.py"),
                               "--config", args.config, "--check"])

    report = json.load(open(report_path, "r", encoding="utf-8"))
    out = args.out or os.path.join(ROOT, deployment, "DATA_SOURCES.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render(cfg, report, deployment))
    print("Écrit : {}".format(out))


if __name__ == "__main__":
    main()
