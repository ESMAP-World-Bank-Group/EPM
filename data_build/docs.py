# -*- coding: utf-8 -*-
"""DATA_SOURCES page, DERIVED from the build.

    python data_build/docs.py --config data_build/build_casa.yaml

Same principle as Black Sea: the source documentation is a PAGE, sitting next to
the data it describes, not a hand-kept workbook. Here it is produced from
build_report.json and the YAML, hence from the same facts as the build itself: it
cannot drift away from what was actually done.

Writes DATA_SOURCES.html into the deployment folder, beside the CSVs it describes.
Nothing in this file names a country, a zone or a column: the perimeter is read from
zcmap.csv and the coverage declared by each source is read from the YAML.

HOW OLD AND HOW SOLID, WHICH IS THE QUESTION THE PAGE HAD STOPPED ANSWERING. A cell of
the coverage matrix used to be painted green as soon as ANY source declared covering
that country, so a country read from the 2025 study and a country read from the 2020
model came out the same colour, which is exactly what the reader must not conclude.
The cell is now graded from the best source that covers that country for that resource,
on two facts the YAML holds beside the source itself: its date, and its grade
(primary, secondary, placeholder). Age gives the band, and the grade caps it: a
secondary source is never better than "recent" however fresh it is, a pre-filled
template that nobody has validated not being a measurement; a placeholder is never
better than "assumption" however recent its date, which is the trap this closes, the
"assumed" source being dated of today and having read as fresh data until now.
The judgement is data, not code: it lives in the YAML, one line per source.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import date

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracker import covers, phase_of, remaining, state_of   # noqa: E402  same state as the tracker

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WORKBOOK = "data_build/DATA_SOURCES_casa.xlsx"   # tracked by git, hence downloadable

# Age in years -> band, anything older falling through to the last one. Read against
# the day the page is generated, so the page ages by itself and no year is written here.
BANDS = [(3, "current"), (5, "recent"), (None, "legacy")]

# Best to worst. The order is what "the best source covering this country" means, and
# what the country cards sort on.
BAND_ORDER = ["current", "recent", "legacy", "assumed", "none"]

# A grade cannot be beaten by a date: this is the ceiling each one is held to.
GRADE_CEILING = {"primary": "current", "secondary": "recent", "placeholder": "assumed"}

BAND = {"current": ("#e2f2e7", "#1e6b46", "current"),
        "recent": ("#eff4dd", "#5d6b1e", "recent"),
        "legacy": ("#fbe9e6", "#a33228", "legacy"),
        "assumed": ("#fbf3de", "#8a6410", "assumption"),
        "none": ("#fafafa", "#999999", "no source")}


def h(text):
    """Escape text: everything coming from the YAML is text, not markup."""
    return (str(text or "").strip()
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def anchor(key, prefix="src"):
    """Stable anchor: this is what gets pasted in an email to point at one item."""
    return prefix + "-" + re.sub(r"[^a-z0-9]+", "-", str(key).lower()).strip("-")


def read_csv(path):
    """(header, rows), or ([], []) when the file is missing."""
    if not os.path.isfile(path):
        return [], []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    return (rows[0], rows[1:]) if rows else ([], [])


def horizon(deployment):
    """Horizon read from the pDemandForecast header: the data says it better than I do."""
    head, _ = read_csv(os.path.join(ROOT, deployment, "load", "pDemandForecast.csv"))
    years = [c for c in head if c.strip().isdigit()]
    if not years:
        return "-"
    return "{}-{} &middot; {} steps".format(years[0], years[-1], len(years))


def perimeter(deployment):
    """Zone -> country, read from zcmap: the perimeter is a fact of the model."""
    head, rows = read_csv(os.path.join(ROOT, deployment, "zcmap.csv"))
    head = [c.strip() for c in head]
    if "z" not in head or "c" not in head:
        return {}
    iz, ic = head.index("z"), head.index("c")
    return {r[iz].strip(): r[ic].strip() for r in rows
            if len(r) > max(iz, ic) and r[iz].strip()}


def geo_values(path, zmap, countries):
    """Countries a CSV carries data for, or None when it has no geographic key.

    A column is geographic when its distinct values are all known zones, or all known
    countries. That test uses the vocabulary of zcmap.csv only: no column name and no
    country name is written here, so the same code serves any deployment. Every
    matching column contributes, which is what is wanted for a corridor file whose two
    endpoints sit in two different columns.
    """
    head, rows = read_csv(path)
    if not head or not rows:
        return None
    zones = set(zmap)
    found, geo = set(), False
    for i in range(len(head)):
        vals = {r[i].strip() for r in rows if len(r) > i and r[i].strip()}
        if not vals:
            continue
        if vals <= zones:
            geo = True
            found |= {zmap[v] for v in vals}
        elif vals <= countries:
            geo = True
            found |= vals
    return found if geo else None


def coverage(res, deployment, source_dir, zmap, countries):
    """resource -> set of countries present in the data, or None if not geographic.

    Read from the built file, falling back to the 2020 reference when the build
    emptied it: an emptied file still describes a perimeter, and the fallback says
    which countries the rebuild will have to cover.
    """
    out = {}
    for e in res:
        rel = str(e.get("path", "")).replace("/", os.sep)
        if not rel or rel.startswith(".."):
            out[e["resource"]] = None
            continue
        got = geo_values(os.path.join(ROOT, deployment, rel), zmap, countries)
        if not got:
            got = geo_values(os.path.join(ROOT, source_dir, rel), zmap, countries)
        out[e["resource"]] = got
    return out


def source_year(s):
    """Year of a source, read off its date whatever precision that date carries."""
    m = re.match(r"(\d{4})", str(s.get("date", "")))
    return int(m.group(1)) if m else None


def source_band(s, now):
    """How current a source is: current, recent, legacy, or assumption.

    The date decides the band and the grade caps it. A source flagged ages: false has
    no meaningful vintage, combustion chemistry being the case at hand: it is not
    stale, it is simply not dated, so its grade alone answers for it.
    """
    grade = str(s.get("grade", "placeholder"))
    ceiling = GRADE_CEILING.get(grade, "assumed")
    if s.get("ages") is False:
        return ceiling
    year = source_year(s)
    if year is None:
        band = "legacy"
    else:
        age = now - year
        band = next(name for limit, name in BANDS if limit is None or age <= limit)
    return max(band, ceiling, key=BAND_ORDER.index)


def best_source(keys, src, country, now):
    """(band, year, key) of the best source covering that country for that resource."""
    best = None
    for k in keys:
        s = src.get(k) or {}
        if country not in (s.get("covers") or []):
            continue
        band = source_band(s, now)
        rank = BAND_ORDER.index(band)
        if best is None or rank < best[0]:
            best = (rank, band, source_year(s), k)
    return best[1:] if best else ("none", None, None)


def weights(deployment, zmap):
    """country -> share of the demand of the model, or {} when it cannot be read.

    Which row of the forecast carries energy is found by size and not by name: energies
    and peaks do not live in the same order of magnitude, so the heaviest row type is
    the energy one whatever the file happens to call it.
    """
    head, rows = read_csv(os.path.join(ROOT, deployment, "load", "pDemandForecast.csv"))
    if not head or not rows:
        return {}
    years = [i for i, c in enumerate(head) if c.strip().isdigit()]
    zone = next((i for i in range(len(head))
                 if {r[i].strip() for r in rows if len(r) > i} <= set(zmap)), None)
    kinds = [i for i in range(len(head)) if i not in years and i != zone]
    if not years or zone is None or not kinds:
        return {}
    first, kind = years[0], kinds[0]

    def value(r):
        try:
            return float(r[first])
        except (IndexError, ValueError):
            return 0.0

    totals = {}
    for r in rows:
        totals.setdefault(r[kind].strip(), 0.0)
        totals[r[kind].strip()] += value(r)
    if not totals:
        return {}
    energy = max(totals, key=lambda k: totals[k])
    out = {}
    for r in rows:
        if r[kind].strip() != energy:
            continue
        out[zmap.get(r[zone].strip(), "")] = out.get(zmap.get(r[zone].strip(), ""), 0.0) + value(r)
    total = sum(out.values())
    return {c: v / total for c, v in out.items() if c and total} if total else {}


# The YAML writes its notes as folded scalars, so they reach the page as one block of
# text however many subjects they cover. What separates the subjects is a convention the
# notes already follow: a new point opens on a run of capitals, PHASE 8 REBUILT IT and
# the like. That run is the title of the point, and this is what cuts the block on it.
LEAD = re.compile(r"^((?:[A-Z0-9][A-Z0-9\-,.'():]*\s+){2,})")
CUT = re.compile(r"(?<=\. )(?=(?:[A-Z0-9][A-Z0-9\-]*\s+){2,})")
SENTENCE = re.compile(r"(?<=[.;]) (?=[A-Z(])")
LIMIT = 420          # characters beyond which a point is cut again, on a full stop


def points(text):
    """Prose from the YAML, cut into readable points. One point when it is short."""
    text = " ".join(str(text or "").split())
    if not text:
        return []
    out = []
    for block in CUT.split(text):
        if len(block) <= LIMIT:
            out.append(block.strip())
            continue
        # Too long to read in one breath and carrying no title: regroup its sentences.
        current = ""
        for sentence in SENTENCE.split(block):
            if current and len(current) + len(sentence) > LIMIT:
                out.append(current.strip())
                current = ""
            current += sentence + " "
        if current.strip():
            out.append(current.strip())
    return [x for x in out if x]


def bullets(a, text, keep=1, css="pts"):
    """Write the text as points, the first ones open and the rest behind a fold."""
    pts = points(text)
    if not pts:
        return

    def item(point):
        m = LEAD.match(point)
        if not m:
            return "<li>%s</li>" % h(point)
        return "<li><b>%s</b> %s</li>" % (h(m.group(1)), h(point[m.end():]))

    a('<ul class="%s">%s</ul>' % (css, "".join(item(x) for x in pts[:keep])))
    if len(pts) > keep:
        a('<details><summary>%d more %s</summary><ul class="%s">%s</ul></details>'
          % (len(pts) - keep, "point" if len(pts) - keep == 1 else "points",
             css, "".join(item(x) for x in pts[keep:])))


def workbook_href(deployment):
    """Link to the tracking workbook: raw GitHub when the remote is known, else relative.

    The page is meant to be read online through htmlpreview, where a relative link
    would resolve outside the repository. The absolute raw link is therefore preferred,
    and it is built from the remote actually configured, never hard-coded.
    """
    rel = os.path.relpath(os.path.join(ROOT, WORKBOOK.replace("/", os.sep)),
                          os.path.join(ROOT, deployment)).replace(os.sep, "/")
    try:
        def git(*a):
            return subprocess.check_output(("git",) + a, cwd=ROOT,
                                           stderr=subprocess.DEVNULL).decode().strip()
        remote = git("remote", "get-url", "origin")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote)
        if m and branch and branch != "HEAD":
            return "https://raw.githubusercontent.com/{}/{}/{}".format(
                m.group(1), branch, WORKBOOK)
    except Exception:
        pass
    return rel


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
.dl{display:inline-block;margin:14px 0 0;padding:7px 14px;background:#1a5276;color:#fff;
 border-radius:4px;text-decoration:none;font-size:.85em;font-weight:600}
.dl:hover{background:#154360}
.src{border-left:3px solid #d5d8dc;padding:2px 0 2px 14px;margin:14px 0 20px}
.src .tag{font-size:.74em;color:#fff;background:#7f8c8d;border-radius:3px;padding:1px 6px;margin-left:6px}
.src .tag.open{background:#27865a}.src .tag.conf{background:#a33228}.src .tag.hyp{background:#b8860b}
.src p{margin:5px 0;color:#555}
.todo{color:#8a5e12}
ul.pts{margin:4px 0 8px;padding-left:18px;color:#555}
ul.pts li{margin:3px 0}
ul.pts b{color:#1a5276;font-size:.9em;letter-spacing:.02em}
ul.pts.left li{color:#8a5e12}
details{margin:2px 0 10px}
summary{cursor:pointer;color:#1a5276;font-size:.82em;user-select:none}
summary:hover{text-decoration:underline}
.muted{color:#999}
.matrix{table-layout:fixed}
.matrix th.cc{text-align:center;width:9.5%}
.matrix td.cc{text-align:center;padding:5px 4px;line-height:1.4}
.matrix td.res{white-space:nowrap}
.cc.b-current{background:#e2f2e7}
.cc.b-recent{background:#eff4dd}
.cc.b-legacy{background:#fbe9e6}
.cc.b-assumed{background:#fbf3de}
.cc.b-none{background:#fafafa}
.yr{display:block;font-size:.7em;color:#6b6b6b;margin-top:2px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px;margin:14px 0 8px}
.ctry{border:1px solid #e0e0e0;border-radius:5px;padding:12px 16px}
.ctry h3{margin:0;font-size:1.05em}
.ctry .zones{color:#888;font-size:.8em;margin:2px 0 8px}
.ctry ul{margin:3px 0 8px;padding-left:17px;font-size:.85em}
.ctry li{margin:1px 0}
.ctry .lbl{font-size:.74em;letter-spacing:.06em;text-transform:uppercase;color:#888;
 margin:10px 0 2px;font-weight:600}
.bar{display:flex;height:15px;border-radius:3px;overflow:hidden;margin:2px 0 4px}
.bar span{display:block;font-size:.68em;color:#fff;text-align:center;line-height:15px}
.barkey{font-size:.76em;color:#777}
.grade{display:inline-block;font-size:.7em;padding:0 5px;border-radius:3px;
 background:#e8eef4;color:#39536b;margin-left:4px}
.grade.primary{background:#e0f0e7;color:#1e6b46}
.grade.placeholder{background:#fbf3de;color:#8a6410}
.chip{display:inline-block;font-size:.72em;padding:0 5px;margin:1px;border-radius:3px;
 background:#e8eef4;color:#39536b;text-decoration:none;white-space:nowrap}
.chip.open{background:#e0f0e7;color:#1e6b46}
.chip.conf{background:#fbe6e3;color:#8f2c23}
.chip.hyp{background:#fbf0d8;color:#8a6410}
.mark{display:block;font-size:.85em;color:#276327;font-weight:700;line-height:1}
.foot{margin-top:44px;padding-top:12px;border-top:1px solid #eaecee;font-size:.78em;color:#999}
"""

# Access level -> (css class, label shown on the page)
ACCESS = {"open_source": ("open", "open source"),
          "internal_wb": ("", "internal"),
          "client_confidential": ("conf", "client confidential"),
          "assumption": ("hyp", "assumption")}


def render(cfg, report, deployment, source_dir):
    src = report.get("sources") or {}
    res = report.get("resources") or []
    zmap = perimeter(deployment)
    countries = sorted(set(zmap.values()))
    today = date.today()
    out = []
    a = out.append

    a('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">')
    a('<meta name="viewport" content="width=device-width, initial-scale=1">')
    a('<title>Data Sources &mdash; EPM &mdash; Central Asia 2026</title>')
    a('<style>%s</style>\n</head>\n<body>' % CSS)
    a('<h1>Data Sources &mdash; EPM &mdash; Central Asia 2026</h1>')
    a('<p class="meta">Generated on %s &middot; deployment <code>%s</code> &middot; '
      '%d zones &middot; %d countries &middot; horizon %s</p>'
      % (today, h(os.path.basename(deployment)), len(zmap), len(countries),
         horizon(deployment)))

    a('<div class="toc"><b>Contents</b><ul>'
      '<li><a href="#coverage">Country coverage</a></li>'
      '<li><a href="#countries">Country by country</a></li>'
      '<li><a href="#resources">Model resources</a></li>'
      '<li><a href="#detail">What was done, resource by resource</a></li>'
      '<li><a href="#sources">The sources</a></li>'
      '</ul></div>')
    a('<div><a class="dl" href="%s">Download the tracking workbook (.xlsx)</a></div>'
      % h(workbook_href(deployment)))

    # -- Country coverage ------------------------------------------------------
    a('<h2 id="coverage">Country coverage</h2>')
    a('<ul class="pts">'
      '<li><b>COLOUR</b> the best source covering that country for that resource, and '
      'the year under the chips is its date.</li>'
      '<li><b>GRADE</b> <i>primary</i> dedicated to that country &middot; <i>secondary</i> '
      'an earlier model, a derived product, an unvalidated template &middot; '
      '<i>placeholder</i> an assumption, or a request that never came back. Age gives '
      'the band, the grade caps it: a placeholder is never data whatever its date.</li>'
      '<li><b>THE DOT</b> rows for that country really are in the file, checked against '
      'the zones of <code>zcmap.csv</code>. A &dagger; resource has no geographic key '
      'at all.</li>'
      '<li><b>WHAT IT DOES NOT SAY</b> whether the number is right.</li>'
      '</ul>')
    a('<div class="legend">')
    for key in BAND_ORDER:
        col, _, txt = BAND[key]
        a('<span><span class="sw" style="background:%s;border:1px solid #ddd"></span>%s</span>'
          % (col, txt))
    a('</div>')
    cov = coverage(res, deployment, source_dir, zmap, set(countries))
    grid = {}
    a('<table class="matrix"><thead><tr><th>Resource</th>')
    for c in countries:
        a('<th class="cc">%s</th>' % h(c))
    a('</tr></thead><tbody>')
    group = None
    for e in res:
        if e.get("group") != group:
            group = e.get("group")
            a('<tr class="cat-row"><td colspan="%d">%s</td></tr>'
              % (len(countries) + 1, h(group or "unclassified")))
        got = cov.get(e["resource"])
        keys = e.get("source") or []
        name = '<code>%s</code>' % h(e["resource"])
        if got is None:
            name += (' <span class="muted" title="no geographic key in this file">'
                     '&dagger;</span>')
        a('<tr><td class="res">%s</td>' % name)
        for c in countries:
            chips = []
            for k in keys:
                s = src.get(k) or {}
                if c in (s.get("covers") or []):
                    tag = ACCESS.get(s.get("access", ""), ("", ""))[0]
                    chips.append('<a class="chip %s" href="#%s" title="%s">%s</a>'
                                 % (tag, anchor(k), h(s.get("name", k)), h(k)))
            band, year, _ = best_source(keys, src, c, today.year)
            grid[(e["resource"], c)] = band
            mark = ('<span class="mark" title="rows found for this country in the data">'
                    '&bull;</span>') if got and c in got else ""
            stamp = (str(year) if year else
                     ("&mdash;" if band == "none" else h(BAND[band][2])))
            body = mark + ("".join(chips) if chips
                           else '<span class="muted">&mdash;</span>')
            body += '<span class="yr" title="%s">%s</span>' % (h(BAND[band][2]), stamp)
            a('<td class="cc b-%s">%s</td>' % (band, body))
        a('</tr>')
    a('</tbody></table>')
    verified = sum(1 for v in cov.values() if v)
    a('<p class="meta">%d of the %d resources carry a geographic key and could be '
      'checked against the data; the other %d are indexed by plant, fuel, technology '
      'or season and rest on the declared layer alone.</p>'
      % (verified, len(res), len(res) - verified))

    # -- Country by country ----------------------------------------------------
    a('<h2 id="countries">Country by country</h2>')
    a('<p class="meta">The same grading read down the columns, against what each '
      'country weighs: its share of the demand in the first year of the horizon.</p>')
    share = weights(deployment, zmap)
    a('<div class="cards">')
    for c in countries:
        tally = {b: 0 for b in BAND_ORDER}
        for e in res:
            tally[grid.get((e["resource"], c), "none")] += 1
        graded = sum(v for b, v in tally.items() if b != "none")
        a('<div class="ctry">')
        head = '<h3>%s</h3>' % h(c)
        a(head)
        zones = sorted(z for z, k in zmap.items() if k == c)
        line = "%d %s: %s" % (len(zones), "zone" if len(zones) == 1 else "zones",
                              ", ".join(h(z) for z in zones))
        if c in share:
            line += " &middot; %.0f %% of the demand of the model" % (100 * share[c])
        a('<p class="zones">%s</p>' % line)

        a('<div class="bar">')
        for b in BAND_ORDER:
            if not tally[b]:
                continue
            col, ink, _ = BAND[b]
            a('<span style="background:%s;color:%s;width:%.4f%%">%d</span>'
              % (col, ink, 100.0 * tally[b] / len(res), tally[b]))
        a('</div>')
        a('<p class="barkey">%s</p>' % " &middot; ".join(
            "%d %s" % (tally[b], BAND[b][2]) for b in BAND_ORDER if tally[b]))

        # The sources that name this country, best first: this is the shelf it is
        # built from, and the first line of it is what its data is worth.
        mine = [(BAND_ORDER.index(source_band(v, today.year)), source_year(v) or 0, k, v)
                for k, v in src.items() if c in (v.get("covers") or [])]
        mine.sort(key=lambda t: (t[0], -t[1]))
        if mine:
            a('<p class="lbl">Sources naming this country</p><ul>')
            for _, _, k, v in mine:
                grade = str(v.get("grade", ""))
                a('<li><a href="#%s">%s</a> <span class="muted">%s</span>'
                  '<span class="grade %s">%s</span></li>'
                  % (anchor(k), h(v.get("name", k)), h(v.get("date", "")),
                     h(grade), h(grade or "ungraded")))
            a('</ul>')

        weak = [e for e in res
                if grid.get((e["resource"], c)) in ("legacy", "assumed")
                and e.get("priority") in ("P1", "P2")]
        if weak:
            a('<p class="lbl">Weakest inputs that matter (P1-P2)</p><ul>')
            for e in weak:
                band = grid[(e["resource"], c)]
                target = (anchor(e["resource"], "res")
                          if e.get("action") not in ("keep", "shared") else "detail")
                a('<li><a href="#%s"><code>%s</code></a> <span class="muted">%s</span></li>'
                  % (target, h(e["resource"]), h(BAND[band][2])))
            a('</ul>')
        a('<p class="barkey">%d of the %d resources have a source for this country.</p>'
          % (graded, len(res)))
        a('</div>')
    a('</div>')

    # -- Resource table --------------------------------------------------------
    a('<h2 id="resources">Model resources</h2>')
    a('<p class="meta">The <i>Inherited content</i> column describes the file as it '
      'comes from the 2020 model; the state says what the build made of it, and the '
      'detail follows further down. <i>Confidence</i> is what the build itself claims '
      'for the values it wrote, which is a different question from the age of the '
      'source they came from.</p>')
    a('<div class="legend">')
    for col, txt in (("#edf7ed", "done"), ("#fdf6e3", "partial"),
                     ("#fdeeec", "inherited 2020, to process"),
                     ("#f4f4f4", "out of scope")):
        a('<span><span class="sw" style="background:%s;border:1px solid #ddd"></span>%s</span>'
          % (col, txt))
    a('</div>')
    a('<table><thead><tr><th>Resource</th><th>File</th><th>Inherited content</th>'
      '<th>Rows</th><th>State</th><th>Phase</th><th>Vintage</th><th>Confidence</th>'
      '<th>Sources</th></tr></thead><tbody>')
    group = None
    tally = {"G": 0, "Y": 0, "R": 0, "B": 0}
    for e in res:
        if e.get("group") != group:
            group = e.get("group")
            a('<tr class="cat-row"><td colspan="9">%s</td></tr>'
              % h(group or "unclassified"))
        code, label = state_of(e)
        tally[code] += 1
        links = " ".join('<a href="#%s">%s</a>' % (anchor(k), h(k))
                         for k in (e.get("source") or []))
        n = e.get("rows_out")
        # A transformed resource links to its own detail block further down.
        cell = '<code>%s</code>' % h(e["resource"])
        if e.get("action") not in ("keep", "shared"):
            cell = '<a href="#%s">%s</a>' % (anchor(e["resource"], "res"), cell)
        a('<tr><td>%s</td><td><code>%s</code></td><td>%s</td>'
          '<td>%s</td><td class="st st-%s">%s</td><td>%s</td><td>%s</td><td>%s</td>'
          '<td>%s</td></tr>'
          % (cell, h(e.get("path", "")), h(e.get("what", "")),
             h(n) if n not in (None, "-") else '<span class="muted">&mdash;</span>',
             code, h(label), h(phase_of(e).lstrip("P")), h(e.get("vintage", "")),
             h(e.get("confidence", "")) or '<span class="muted">&mdash;</span>',
             links or '<span class="muted">&mdash;</span>'))
    a('</tbody></table>')
    a('<p class="meta">%d resources: %d done &middot; %d partial &middot; %d to do '
      '&middot; %d out of scope</p>'
      % (len(res), tally["G"], tally["Y"], tally["R"], tally["B"]))

    # -- Detail, transformed resources only ------------------------------------
    a('<h2 id="detail">What was done, resource by resource</h2>')
    a('<p class="meta">Only the resources actually transformed by the build appear '
      'here; the others are carried over as is from the 2020 model. The first point of '
      'each is open, the rest folded.</p>')
    for e in res:
        if e.get("action") in ("keep", "shared"):
            continue
        code, label = state_of(e)
        a('<h3 id="%s"><code>%s</code> <span class="muted">&mdash; %s</span></h3>'
          % (anchor(e["resource"], "res"), h(e["resource"]), h(label)))
        if e.get("note"):
            bullets(a, e["note"], keep=2)
        left = remaining(e)
        if left:
            a('<p class="lbl">Left to do</p>')
            bullets(a, left, css="pts left")

    # -- The sources -----------------------------------------------------------
    a('<h2 id="sources">The sources</h2>')
    for key, s in src.items():
        tag, tag_label = ACCESS.get(s.get("access", ""), ("", s.get("access", "")))
        a('<div class="src" id="%s">' % anchor(key))
        a('<b>%s</b> <span class="tag %s">%s</span>'
          % (h(s.get("name", key)), tag, h(tag_label)))
        grade = str(s.get("grade", ""))
        if grade:
            a('<span class="grade %s">%s</span>' % (h(grade), h(grade)))
        meta = [x for x in [h(s.get("date", "")), h(covers(s))] if x]
        if meta:
            a('<p class="meta">%s</p>' % " &middot; ".join(meta))
        if s.get("where"):
            a('<p class="meta">Location: <code>%s</code></p>' % h(s["where"]))
        if s.get("note"):
            bullets(a, s["note"])
        a('</div>')

    a('<div class="foot">Page produced by <code>data_build/docs.py</code> from '
      '<code>build_report.json</code> and <code>build_casa.yaml</code>. '
      'Do not edit by hand: regenerate. The leads on public online sources are kept '
      'in the tracking workbook.</div>')
    a('</body>\n</html>')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    deployment = cfg["deployment"]["target"]   # relative to the repo root
    source_dir = cfg["deployment"]["source"]
    report_path = os.path.join(HERE, "build_report.json")

    if (not os.path.isfile(report_path)
            or os.path.getmtime(report_path) < os.path.getmtime(args.config)):
        print("Report missing or stale, re-running the build with --check.")
        subprocess.check_call([sys.executable, os.path.join(HERE, "build.py"),
                               "--config", args.config, "--check"])

    report = json.load(open(report_path, "r", encoding="utf-8"))
    out = args.out or os.path.join(ROOT, deployment, "DATA_SOURCES.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render(cfg, report, deployment, source_dir))
    print("Written: {}".format(out))


if __name__ == "__main__":
    main()
