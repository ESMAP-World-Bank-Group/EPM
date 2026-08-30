"""Refresh the model-side content of Data/calibration/calibration_review.html from an EPM run.

The calibration page is hand-written bilingual prose wrapped around sixteen charts whose
model side goes stale on every re-run. Three chart idioms live in the page:

  A  430x210  model-only stacked capacity, one per country ("capacite installee (GW)")
  B  660x300  plan / model / delta, one per country per copy (the page repeats each
              country's comparison in the snapshot block and again in its section 5)
  C  360x240  model-only stacked generation (TWh)

Idiom B carries a hand-built plan side and a peak-demand marker that no run can produce.
Both are parsed out of the existing markup and re-emitted at the new scale, so only the
model bars and the recomputed delta bars change. Everything outside the <svg> elements
this script targets is left byte-identical, so the prose survives.

Usage:
    python tools/calibration_review_update.py --run <folder with per-scenario output_csv>
    python tools/calibration_review_update.py --run ... --dry-run   # report, change nothing

The run folder is the root of an unzipped simulations_run_* archive, i.e. it holds
baseline/output_csv/pTechFuelMerged.csv.
"""

import argparse
import io
import math
import os
import re
import sys
from datetime import date

import pandas as pd

# ---------------------------------------------------------------- configuration

HTML_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Data", "calibration", "calibration_review.html",
)

MILESTONES = [2025, 2030, 2035, 2040]
COUNTRIES = ["Turkiye", "Georgia", "Azerbaijan", "Armenia"]
# The page writes Turkiye with a dotted capital; the run data does not.
PAGE_NAME = {"Turkiye": "Türkiye", "Georgia": "Georgia",
             "Azerbaijan": "Azerbaijan", "Armenia": "Armenia"}

# Stacking order, bottom to top. Must match the legend order in the page.
BUCKETS = [
    ("Coal", "#6f7680"),
    ("Gas", "#9A7040"),
    ("Oil", "#5f564e"),
    ("Nuclear", "#B78CE0"),
    ("Hydro", "#1E8AE0"),
    ("Wind", "#3FC9DA"),
    ("Solar", "#F5C518"),
    ("Geothermal", "#D07A18"),
    ("Biomass", "#49B25C"),
    ("Storage", "#8894a4"),
]
ORDER = [name for name, _ in BUCKETS]
COLOUR = dict(BUCKETS)

# The page's older charts label the grey block "Battery". It now carries pumped hydro
# as well (2.1 GW of Turkish PSH), so the model side is relabelled "Storage".
ALIAS = {"Battery": "Storage"}

COAL_FUELS = {"Coal", "DomesticCoal", "ImportedCoal", "Lignite", "Peat", "Shale"}
GAS_FUELS = {"Gas", "LNG", "Methane"}
OIL_FUELS = {"HFO", "LFO", "Diesel", "ALC"}
SOLAR_FUELS = {"Solar", "PV", "CSP"}
BIO_FUELS = {"Biomass", "Biogas", "Waste"}


def bucket_of(tech, fuel):
    """Map an EPM (tech, fuel) pair onto one of the legend buckets."""
    tech = str(tech).strip()
    fuel = str(fuel).strip()
    if tech == "ImportTransmission":
        return None  # imports are a flow, not a plant
    if tech.startswith("Storage") or tech.startswith("STO") or tech == "pvwsto":
        return "Storage"
    if fuel in COAL_FUELS:
        return "Coal"
    if fuel in GAS_FUELS:
        return "Gas"
    if fuel in OIL_FUELS:
        return "Oil"
    if fuel == "Uranium":
        return "Nuclear"
    if fuel == "Water":
        return "Hydro"
    if fuel == "Wind":
        return "Wind"
    if fuel in SOLAR_FUELS:
        return "Solar"
    if fuel == "Geothermal":
        return "Geothermal"
    if fuel in BIO_FUELS:
        return "Biomass"
    return None


# ------------------------------------------------------------------- extraction

def load_run(run_dir, scenario):
    path = os.path.join(run_dir, scenario, "output_csv", "pTechFuelMerged.csv")
    if not os.path.isfile(path):
        sys.exit("pTechFuelMerged.csv not found under %s" % path)
    df = pd.read_csv(path)
    df["bucket"] = [bucket_of(t, f) for t, f in zip(df["tech"], df["f"])]
    return df.dropna(subset=["bucket"])


def series(df, attribute, divisor):
    """{country: {year: {bucket: value}}} in the chart's unit."""
    sub = df[(df["attribute"] == attribute) & (df["y"].isin(MILESTONES))]
    grouped = sub.groupby(["c", "bucket", "y"])["value"].sum() / divisor
    out = {}
    for country in COUNTRIES:
        out[country] = {}
        for year in MILESTONES:
            row = {}
            for name in ORDER:
                v = float(grouped.get((country, name, year), 0.0))
                if v > 1e-9:
                    row[name] = v
            out[country][year] = row
    return out


# ------------------------------------------------------------------- svg helpers

NICE = [0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000, 2000]


def nice_step(top, max_lines):
    for s in NICE:
        if top / s <= max_lines:
            return s
    return NICE[-1]


def sig3(x):
    if x <= 0:
        return x
    return round(x, -int(math.floor(math.log10(x))) + 2)


def num(v):
    """Gridline labels: integers where the step is integral."""
    return ("%g" % round(v, 2)) if abs(v) < 1 else ("%.0f" % v)


def esc(name):
    return ALIAS.get(name, name)


# ------------------------------------------------------------- idiom A: 430x210

A_Y0, A_SPAN = 190.0, 162.5
A_X = [57.3, 154.3, 251.3, 348.3]
A_W = 40.6 * 0 + 50.4
A_GRID_Y = [190.0, 99.0, 8.0]
A_MIN = 0.025  # GW below this are not drawn; matches the existing charts


def svg_a(byyear, unit):
    totals = [sum(byyear[y].values()) for y in MILESTONES]
    top = max(totals) if totals else 1.0
    scale = A_SPAN / top
    p = []
    for gy in A_GRID_Y:
        v = top * (A_Y0 - gy) / A_SPAN
        p.append('<line x1="34" y1="%.1f" x2="422" y2="%.1f" stroke="#eef1f6"/>' % (gy, gy))
        p.append('<text x="30" y="%.1f" font-size="8" text-anchor="end" fill="#8a96a8">%s</text>'
                 % (gy + 3.0, num(v)))
    for i, year in enumerate(MILESTONES):
        x, cx = A_X[i], A_X[i] + A_W / 2
        cursor = A_Y0
        for name in ORDER:
            v = byyear[year].get(name, 0.0)
            if v < A_MIN:
                continue
            h = v * scale
            cursor -= h
            p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s">'
                     '<title>%s %d: %.1f %s</title></rect>'
                     % (x, cursor, A_W, h, COLOUR[name], esc(name), year, v, unit))
        p.append('<text x="%.1f" y="203.0" font-size="9" text-anchor="middle" '
                 'fill="#33445e">%d</text>' % (cx, year))
        p.append('<text x="%.1f" y="%.1f" font-size="8.5" font-weight="700" '
                 'text-anchor="middle" fill="#33445e">%.0f</text>'
                 % (cx, cursor - 3.0, totals[i]))
    return ('<svg viewBox="0 0 430 210" width="100%" height="210" style="max-width:430px" '
            'data-gen="model">' + "".join(p) + "</svg>")


# ------------------------------------------------------------- idiom C: 360x240

C_Y0, C_BAND = 206.0, 178.0
C_X = [53.8, 132.8, 211.8, 290.8]
C_W = 39.5
C_MIN = 0.005


def svg_c(byyear, unit):
    totals = [sum(byyear[y].values()) for y in MILESTONES]
    top = max(totals) if totals else 1.0
    scale = C_BAND / top
    step = nice_step(top, 6)
    p = []
    v = 0.0
    while v <= top + 1e-9:
        gy = C_Y0 - v * scale
        p.append('<line x1="34" y1="%.1f" x2="350" y2="%.1f" stroke="#eef1f6"/>' % (gy, gy))
        p.append('<text x="30" y="%.1f" font-size="7.5" fill="#9aa4b2" '
                 'text-anchor="end">%s</text>' % (gy + 2.5, num(v)))
        v += step
    for i, year in enumerate(MILESTONES):
        x, cx = C_X[i], C_X[i] + C_W / 2
        cursor = C_Y0
        for name in ORDER:
            val = byyear[year].get(name, 0.0)
            if val < C_MIN:
                continue
            h = val * scale
            cursor -= h
            p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.2f" fill="%s">'
                     '<title>%s %d: %.1f %s</title></rect>'
                     % (x, cursor, C_W, h, COLOUR[name], esc(name), year, val, unit))
        p.append('<text x="%.1f" y="%.1f" font-size="7" fill="#33445e" text-anchor="middle" '
                 'font-weight="600">%.0f</text>' % (cx, cursor - 2.0, totals[i]))
        p.append('<text x="%.1f" y="219.0" font-size="9" fill="#33445e" text-anchor="middle" '
                 'font-weight="700">%d</text>' % (cx, year))
    return ('<svg viewBox="0 0 360 240" width="100%" '
            'style="max-width:360px;font-family:inherit;display:block;margin:0 auto" '
            'data-gen="model">' + "".join(p) + "</svg>")


# ------------------------------------------------------------- idiom B: 660x300

B_YTOP, B_BAND = 27.5, 220.0
B_MX = [98.1, 250.1, 402.1, 554.1]     # model bar left edge
B_DX_OFF, B_PX_OFF = 38.8, -38.7       # delta / plan offsets from the model bar
B_W = 35.7
B_MIN = 0.004


def svg_b(model, plan, peak, unit):
    """model/plan: {year: {bucket: v}} (plan may be missing a year). peak: {year: GW}."""
    delta = {}
    for year in MILESTONES:
        if year not in plan:
            continue
        keys = set(model.get(year, {})) | set(plan[year])
        delta[year] = {k: model.get(year, {}).get(k, 0.0) - plan[year].get(k, 0.0)
                       for k in keys}

    maxpos = 0.0
    for year in MILESTONES:
        maxpos = max(maxpos, sum(model.get(year, {}).values()))
        if year in plan:
            maxpos = max(maxpos, sum(plan[year].values()))
    maxneg = 0.0
    for year, d in delta.items():
        maxneg = max(maxneg, sum(-v for v in d.values() if v < 0))
    scale = sig3(B_BAND / (maxpos + maxneg)) if maxpos + maxneg else 1.0
    zero = B_YTOP + maxpos * scale
    step = nice_step(maxpos, 6)

    p = ['<line x1="40" y1="%.1f" x2="648" y2="%.1f" stroke="#b0bcc9" stroke-width="1"/>'
         '<text x="36" y="%.1f" font-size="7.5" fill="#9aa4b2" text-anchor="end">0</text>'
         % (zero, zero, zero + 2.5)]
    k = 1
    while k * step <= maxpos:
        gy = zero - k * step * scale
        p.append('<line x1="40" y1="%.1f" x2="648" y2="%.1f" stroke="#eef1f6" stroke-width="1"/>'
                 '<text x="36" y="%.1f" font-size="7.5" fill="#9aa4b2" text-anchor="end">%s</text>'
                 % (gy, gy, gy + 2.5, num(k * step)))
        k += 1
    k = 1
    while k * step <= maxneg:
        gy = zero + k * step * scale
        p.append('<line x1="40" y1="%.1f" x2="648" y2="%.1f" stroke="#eef1f6" stroke-width="1"/>'
                 '<text x="36" y="%.1f" font-size="7.5" fill="#9aa4b2" text-anchor="end">-%s</text>'
                 % (gy, gy, gy + 2.5, num(k * step)))
        k += 1

    def stack_up(x, rows, tag, year):
        cursor = zero
        for name in ORDER:
            v = rows.get(name, 0.0)
            if v < B_MIN:
                continue
            h = v * scale
            cursor -= h
            p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.2f" fill="%s">'
                     '<title>%s %s %d: %.1f %s</title></rect>'
                     % (x, cursor, B_W, h, COLOUR[name], esc(name), tag, year, v, unit))
        return cursor

    for i, year in enumerate(MILESTONES):
        mx = B_MX[i]
        px, dx = mx + B_PX_OFF, mx + B_DX_OFF
        mcx, pcx, dcx = mx + 17.9, px + 17.8, dx + 17.9
        mrows = model.get(year, {})
        mtop = stack_up(mx, mrows, "model", year)
        ptop = stack_up(px, plan[year], "plan", year) if year in plan else None

        neg_bottom = zero
        if year in delta:
            up, down = zero, zero
            for name in ORDER:
                v = delta[year].get(name, 0.0)
                if abs(v) < B_MIN:
                    continue
                h = abs(v) * scale
                if v > 0:
                    up -= h
                    y = up
                else:
                    y = down
                    down += h
                p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.2f" fill="%s">'
                         '<title>%s Δ %d: %+.1f %s</title></rect>'
                         % (dx, y, B_W, h, COLOUR[name], esc(name), year, v, unit))
            neg_bottom = down

        p.append('<text x="%.1f" y="%.1f" font-size="7" fill="#33445e" text-anchor="middle" '
                 'font-weight="600">%.0f</text>' % (mcx, mtop - 2.0, sum(mrows.values())))
        if ptop is not None:
            p.append('<text x="%.1f" y="%.1f" font-size="7" fill="#33445e" '
                     'text-anchor="middle" font-weight="600">%.0f</text>'
                     % (pcx, ptop - 2.0, sum(plan[year].values())))

        if year in peak:
            py = zero - peak[year] * scale
            p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c0392b" '
                     'stroke-width="1.7"/>' % (mx - 2.5, py, dx - 0.5, py))
            p.append('<path d="M %.1f %.1f l 2.6 2.6 l -2.6 2.6 l -2.6 -2.6 z" fill="#c0392b"/>'
                     % (mcx, py - 2.6))
            p.append('<text x="%.1f" y="%.1f" font-size="6" fill="#c0392b" text-anchor="middle" '
                     'font-weight="700" stroke="#fff" stroke-width="1.6" '
                     'paint-order="stroke">%.0f</text>' % (mcx, py - 4.0, peak[year]))

        if year in delta:
            net = sum(delta[year].values())
            p.append('<text x="%.1f" y="%.1f" font-size="7" fill="%s" text-anchor="middle" '
                     'font-weight="700">%.1f</text>'
                     % (dcx, neg_bottom + 8.0, "#c0682a" if net < 0 else "#2e7d5b", net))

        p.append('<text x="%.1f" y="267.0" font-size="6.8" fill="#7a869c" '
                 'text-anchor="middle">Model</text>' % mcx)
        if year in plan:
            p.append('<text x="%.1f" y="267.0" font-size="6.8" fill="#7a869c" '
                     'text-anchor="middle">Plan</text>' % pcx)
            p.append('<text x="%.1f" y="267.0" font-size="6.8" fill="#7a869c" '
                     'text-anchor="middle">Δ</text>' % dcx)
        else:
            p.append('<text x="%.1f" y="267.0" font-size="5.8" fill="#b7c0cc" '
                     'text-anchor="middle" font-style="italic">plan n/a</text>' % pcx)
        p.append('<text x="%.1f" y="280.0" font-size="9.5" fill="#33445e" text-anchor="middle" '
                 'font-weight="700">%d</text>' % (mcx, year))

    return ('<svg viewBox="0 0 660 300" width="100%" '
            'style="max-width:660px;font-family:inherit;display:block;margin:0 auto" '
            'data-gen="model">' + "".join(p) + "</svg>")


# --------------------------------------------------------------- svg discovery

RE_SVG = re.compile(r'<svg[^>]*>.*?</svg>', re.S)
RE_TITLE = re.compile(r'<rect x="([\d.]+)" y="([\d.]+)" width="[\d.]+" height="([\d.]+)"'
                      r'[^>]*>\s*<title>([^<]*)</title>')
RE_TIP_B = re.compile(r'^(\w+) (model|plan|Δ) (\d{4}): ([-+]?[\d.]+) (GW|TWh)$')
RE_TIP_S = re.compile(r'^([\w ]+?) (\d{4}): ([-+]?[\d.]+) (GW|TWh)$')


def classify(svg):
    """Return (idiom, unit) or (None, None)."""
    tips = [m.group(4) for m in RE_TITLE.finditer(svg)]
    if not tips:
        return None, None
    vb = re.search(r'viewBox="([^"]+)"', svg).group(1)
    if RE_TIP_B.match(tips[0]) and vb == "0 0 660 300":
        return "B", RE_TIP_B.match(tips[0]).group(5)
    m = RE_TIP_S.match(tips[0])
    if m and vb == "0 0 430 210":
        return "A", m.group(4)
    if m and vb == "0 0 360 240":
        return "C", m.group(4)
    return None, None


RE_ANCHOR_SN = re.compile(r'<div class="(?:sn|ct)"[^>]*>(?:(?!</div>).){0,120}?'
                          r'(Türkiye|Georgia|Azerbaijan|Armenia)', re.S)
RE_ANCHOR_CT = re.compile(r'data-ct="(Turkiye|Georgia|Azerbaijan|Armenia)"')
UNPAGE = {v: k for k, v in PAGE_NAME.items()}


def country_anchors(html):
    """(offset, country) for every section heading that names a country, in order.

    Nearest-name-above is not safe here: the prose of one country's section routinely
    names its neighbours. The page's own section markers are unambiguous.
    """
    anchors = [(m.start(), UNPAGE[m.group(1)]) for m in RE_ANCHOR_SN.finditer(html)]
    anchors += [(m.start(), m.group(1)) for m in RE_ANCHOR_CT.finditer(html)]
    return sorted(anchors)


def country_of(anchors, pos):
    best = None
    for off, country in anchors:
        if off > pos:
            break
        best = country
    return best


def parse_plan_side(svg):
    """Recover the plan bars and the peak-demand markers from an idiom-B chart."""
    zero = float(re.search(r'y1="([\d.]+)"[^>]*stroke="#b0bcc9"', svg).group(1))
    grid = re.findall(r'<line x1="40" y1="([\d.]+)"[^>]*/><text[^>]*>(-?[\d.]+)</text>', svg)
    ref = [(float(y), float(v)) for y, v in grid if float(v) != 0]
    scale = (zero - ref[0][0]) / ref[0][1]

    plan, xs = {}, {}
    for m in RE_TITLE.finditer(svg):
        x, _, h, tip = float(m.group(1)), 0, float(m.group(3)), m.group(4)
        t = RE_TIP_B.match(tip)
        if not t:
            continue
        name, role, year = t.group(1), t.group(2), int(t.group(3))
        if role == "model":
            xs[year] = x
        if role != "plan":
            continue
        plan.setdefault(year, {})[ALIAS.get(name, name)] = h / scale

    peak = {}
    marks = re.findall(r'<line x1="([\d.]+)" y1="([\d.]+)"[^>]*stroke="#c0392b"', svg)
    for mx, my in marks:
        mx, my = float(mx) + 2.5, float(my)
        year = min(xs, key=lambda y: abs(xs[y] - mx)) if xs else None
        if year is not None and abs(xs[year] - mx) < 5:
            peak[year] = (zero - my) / scale
    return plan, peak


# ------------------------------------------------------------------- text edits

def sub_once(html, old, new, label, report):
    n = html.count(old)
    if n == 0:
        report.append("  SKIP  %s (already current or not found)" % label)
        return html
    report.append("  text  %s (%d)" % (label, n))
    return html.replace(old, new)


def stamp(html, run_name):
    line = ('<p class="sub" id="runstamp">Charts and calibration figures regenerated from run '
            '<b>%s</b> (baseline) on %s by tools/calibration_review_update.py. Plan side, '
            'peak-demand markers and prose are hand-written.</p>'
            % (run_name, date.today().isoformat()))
    old = re.search(r'<p class="sub" id="runstamp">.*?</p>', html, re.S)
    if old:
        return html[:old.start()] + line + html[old.end():]
    anchor = re.search(r'<p class="sub">.*?</p>', html, re.S)
    if not anchor:
        sys.exit("no subtitle to anchor the run stamp to")
    return html[:anchor.end()] + line + html[anchor.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--html", default=HTML_DEFAULT)
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--run-name", dest="run_name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = load_run(args.run, args.scenario)
    cap = series(df, "CapacityTechFuel", 1000.0)
    gen = series(df, "EnergyTechFuelComplete", 1000.0)

    html = io.open(args.html, encoding="utf-8").read()
    before = len(html)
    report = []

    # Charts are rebuilt back to front so earlier offsets stay valid.
    anchors = country_anchors(html)
    targets = []
    for m in RE_SVG.finditer(html):
        idiom, unit = classify(m.group(0))
        if idiom:
            country = country_of(anchors, m.start())
            if country is None:
                sys.exit("chart at %d sits under no country section" % m.start())
            targets.append((m.start(), m.end(), idiom, unit, m.group(0), country))
    for start, end, idiom, unit, old, country in reversed(targets):
        data = cap[country] if unit == "GW" else gen[country]
        if idiom == "A":
            new = svg_a(data, unit)
        elif idiom == "C":
            new = svg_c(data, unit)
        else:
            plan, peak = parse_plan_side(old)
            new = svg_b(data, plan, peak, unit)
        report.append("  chart %s  %-11s %-4s @%d  %d -> %d bytes"
                      % (idiom, country, unit, start, len(old), len(new)))
        html = html[:start] + new + html[end:]
    report.reverse()

    # Legend: the grey block now carries pumped hydro as well as batteries.
    for old, new, label in [
        ('<i style="background:#8894a4"></i>Battery',
         '<i style="background:#8894a4"></i>Storage', "legend Battery -> Storage"),
    ]:
        html = sub_once(html, old, new, label, report)

    html = stamp(html, args.run_name or os.path.basename(os.path.normpath(args.run)))

    print("run  : %s (%s)" % (args.run, args.scenario))
    print("page : %s" % args.html)
    print("\n".join(report))
    print("%d charts rebuilt; %d -> %d bytes" % (len(targets), before, len(html)))
    if args.dry_run:
        print("dry run: nothing written")
        return
    io.open(args.html, "w", encoding="utf-8", newline="\n").write(html)


if __name__ == "__main__":
    main()
