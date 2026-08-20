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
.muted{color:#999}
.matrix{table-layout:fixed}
.matrix th.cc{text-align:center;width:9.5%}
.matrix td.cc{text-align:center;padding:5px 4px;line-height:1.4}
.matrix td.res{white-space:nowrap}
.cc.has{background:#f4faf6}
.cc.none{background:#fafafa}
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

    a('<div class="legend">')
    for col, txt in (("#edf7ed", "done"), ("#fdf6e3", "partial"),
                     ("#fdeeec", "inherited 2020, to process"),
                     ("#f4f4f4", "out of scope")):
        a('<span><span class="sw" style="background:%s;border:1px solid #ddd"></span>%s</span>'
          % (col, txt))
    a('</div>')

    a('<div class="toc"><b>Contents</b><ul>'
      '<li><a href="#coverage">Country coverage</a></li>'
      '<li><a href="#resources">Model resources</a></li>'
      '<li><a href="#detail">What was done, resource by resource</a></li>'
      '<li><a href="#sources">The sources</a></li>'
      '</ul></div>')
    a('<div><a class="dl" href="%s">Download the tracking workbook (.xlsx)</a></div>'
      % h(workbook_href(deployment)))

    # -- Country coverage ------------------------------------------------------
    a('<h2 id="coverage">Country coverage</h2>')
    a('<p class="meta">Which source can speak for which country, resource by '
      'resource. Two layers. The chips are the <b>declared</b> coverage: the '
      'sources attached to the resource that state they can speak for that country. '
      'The dot above them is <b>verified in the data</b>: the file really carries rows '
      'for that country, found by matching column values against the zones of '
      '<code>zcmap.csv</code>. A resource marked &dagger; has no geographic key at all '
      '(it is indexed by plant, fuel or technology), so only the declared layer applies '
      'to it.</p>')
    cov = coverage(res, deployment, source_dir, zmap, set(countries))
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
            mark = ('<span class="mark" title="rows found for this country in the data">'
                    '&bull;</span>') if got and c in got else ""
            body = mark + ("".join(chips) if chips
                           else '<span class="muted">&mdash;</span>')
            a('<td class="cc %s">%s</td>' % ("has" if chips or mark else "none", body))
        a('</tr>')
    a('</tbody></table>')
    verified = sum(1 for v in cov.values() if v)
    a('<p class="meta">%d of the %d resources carry a geographic key and could be '
      'checked against the data; the other %d are indexed by plant, fuel, technology '
      'or season and rest on the declared layer alone.</p>'
      % (verified, len(res), len(res) - verified))

    # -- Resource table --------------------------------------------------------
    a('<h2 id="resources">Model resources</h2>')
    a('<p class="meta">The <i>Inherited content</i> column describes the file as it '
      'comes from the 2020 model; the state says what the build made of it, and the '
      'detail follows further down.</p>')
    a('<table><thead><tr><th>Resource</th><th>File</th><th>Inherited content</th>'
      '<th>Rows</th><th>State</th><th>Phase</th><th>Vintage</th><th>Sources</th>'
      '</tr></thead><tbody>')
    group = None
    tally = {"G": 0, "Y": 0, "R": 0, "B": 0}
    for e in res:
        if e.get("group") != group:
            group = e.get("group")
            a('<tr class="cat-row"><td colspan="8">%s</td></tr>'
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
          '<td>%s</td><td class="st st-%s">%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
          % (cell, h(e.get("path", "")), h(e.get("what", "")),
             h(n) if n not in (None, "-") else '<span class="muted">&mdash;</span>',
             code, h(label), h(phase_of(e).lstrip("P")), h(e.get("vintage", "")),
             links or '<span class="muted">&mdash;</span>'))
    a('</tbody></table>')
    a('<p class="meta">%d resources: %d done &middot; %d partial &middot; %d to do '
      '&middot; %d out of scope</p>'
      % (len(res), tally["G"], tally["Y"], tally["R"], tally["B"]))

    # -- Detail, transformed resources only ------------------------------------
    a('<h2 id="detail">What was done, resource by resource</h2>')
    a('<p class="meta">Only the resources actually transformed by the build appear '
      'here. The others are carried over as is from the 2020 model.</p>')
    for e in res:
        if e.get("action") in ("keep", "shared"):
            continue
        code, label = state_of(e)
        a('<h3 id="%s"><code>%s</code> <span class="muted">&mdash; %s</span></h3>'
          % (anchor(e["resource"], "res"), h(e["resource"]), h(label)))
        if e.get("note"):
            a('<p>%s</p>' % h(e["note"]))
        left = remaining(e)
        if left:
            a('<p class="todo"><b>Left to do:</b> %s</p>' % h(left))

    # -- The sources -----------------------------------------------------------
    a('<h2 id="sources">The sources</h2>')
    for key, s in src.items():
        tag, tag_label = ACCESS.get(s.get("access", ""), ("", s.get("access", "")))
        a('<div class="src" id="%s">' % anchor(key))
        a('<b>%s</b> <span class="tag %s">%s</span>'
          % (h(s.get("name", key)), tag, h(tag_label)))
        meta = [x for x in [h(s.get("date", "")), h(covers(s))] if x]
        if meta:
            a('<p class="meta">%s</p>' % " &middot; ".join(meta))
        if s.get("where"):
            a('<p class="meta">Location: <code>%s</code></p>' % h(s["where"]))
        if s.get("note"):
            a('<p>%s</p>' % h(s["note"]))
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
