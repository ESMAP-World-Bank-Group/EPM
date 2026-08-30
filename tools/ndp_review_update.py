"""Refresh the model-side content of ndp_review.html from an EPM run.

The page compares national development plans against our least-cost build-out. The
plan side is hand-written and stays put; the model side goes stale every time we
re-run. This script regenerates only the model side:

  * the four "OUR MODEL (baseline)" capacity charts (GW),
  * the four "OUR MODEL (baseline)" generation charts (TWh),
  * the model-verdict column of the overview table and the matching header badges,
  * a provenance stamp naming the run the page was built from.

Everything else in the file is left byte-identical, so hand-written prose survives.
Re-running against the same page is idempotent.

Usage:
    python tools/ndp_review_update.py --run <folder with per-scenario output_csv>

The run folder is the root of an unzipped simulations_run_* archive, i.e. it holds
baseline/output_csv/pTechFuelMerged.csv.
"""

import argparse
import io
import os
import re
import sys
from datetime import date

import pandas as pd

# ---------------------------------------------------------------- configuration

HTML_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Data", "calibration", "ndp_review.html",
)

MILESTONES = [2025, 2030, 2035, 2040]
COUNTRIES = ["Turkiye", "Georgia", "Azerbaijan", "Armenia"]

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
        return None  # imports are a flow, not a plant; the plan charts have no such block
    if tech.startswith("Storage") or tech == "STO HY":
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

def load_run(run_dir, scenario="baseline"):
    path = os.path.join(run_dir, scenario, "output_csv", "pTechFuelMerged.csv")
    if not os.path.isfile(path):
        sys.exit("pTechFuelMerged.csv not found under %s" % path)
    df = pd.read_csv(path)
    df["bucket"] = [bucket_of(t, f) for t, f in zip(df["tech"], df["f"])]
    dropped = df[df["bucket"].isna() & (df["value"].abs() > 0)]
    if not dropped.empty:
        pairs = sorted(set(zip(dropped["tech"], dropped["f"])))
        print("  note: %d tech/fuel pairs carry value but map to no bucket: %s"
              % (len(pairs), pairs[:6]))
    return df.dropna(subset=["bucket"])


def series(df, attribute, divisor):
    """{country: {bucket: [v2025, v2030, v2035, v2040]}} in the chart's unit."""
    sub = df[(df["attribute"] == attribute) & (df["y"].isin(MILESTONES))]
    grouped = sub.groupby(["c", "bucket", "y"])["value"].sum() / divisor
    out = {}
    for country in COUNTRIES:
        rows = {}
        for name, _ in BUCKETS:
            vals = [float(grouped.get((country, name, y), 0.0)) for y in MILESTONES]
            if any(abs(v) > 1e-6 for v in vals):
                rows[name] = vals
        out[country] = rows
    return out


# ----------------------------------------------------------------------- charts

NICE_STEPS = [0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500]


def nice_step(top):
    for s in NICE_STEPS:
        if s * 4 >= top:
            return s
    return NICE_STEPS[-1]


def fmt(v):
    return ("%.0f" % v) if v >= 10 else ("%.1f" % v)


def svg(rows, unit):
    """Stacked bar chart in the exact idiom the hand-written page already uses."""
    totals = [sum(vals[i] for vals in rows.values()) for i in range(len(MILESTONES))]
    step = nice_step(max(totals) if totals else 1)
    top = step * 4
    y0, y_top = 203.0, 16.0
    span = y0 - y_top

    def ypos(v):
        return y0 - (v / top) * span

    parts = []
    for k in range(5):
        v = step * k
        y = ypos(v)
        parts.append('<line x1="30" y1="%.0f" x2="292" y2="%.0f" stroke="#eef1f6"/>' % (y, y))
        label = ("%g" % v) if step < 1 else ("%.0f" % v)
        parts.append('<text x="27" y="%.0f" font-size="8" fill="#9aa4b2" '
                     'text-anchor="end">%s</text>' % (y + 3, label))

    x0, width, gap = 42.4, 40.6, 65.5
    for i, year in enumerate(MILESTONES):
        x = x0 + i * gap
        cursor = y0
        for name, colour in BUCKETS:
            if name not in rows:
                continue
            v = rows[name][i]
            if v <= 1e-6:
                continue
            h = (v / top) * span
            cursor -= h
            parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"/>'
                         % (x, cursor, width, h, colour))
        parts.append('<text x="%.1f" y="%.0f" font-size="7.5" fill="#33445e" '
                     'text-anchor="middle" font-weight="600">%s</text>'
                     % (x + width / 2, cursor - 2, fmt(totals[i])))
        parts.append('<text x="%.0f" y="214" font-size="9" fill="#5a6577" '
                     'text-anchor="middle">%d</text>' % (x + width / 2, year))

    parts.append('<text x="27" y="11" font-size="8" fill="#9aa4b2">%s</text>' % unit)
    return ('<svg viewBox="0 0 300 225" width="100%" style="max-width:300px" '
            'data-gen="model">' + "".join(parts) + "</svg>")


# ------------------------------------------------------------------- html patch

def replace_chart(html, title, new_svg, occurrence):
    """Swap the <svg> that immediately follows the nth chart title."""
    needle = '>%s</div>' % title
    pos = -1
    for _ in range(occurrence + 1):
        pos = html.find(needle, pos + 1)
        if pos < 0:
            sys.exit("chart title not found (%s, occurrence %d)" % (title, occurrence))
    start = html.find("<svg", pos)
    end = html.find("</svg>", start)
    if start < 0 or end < 0:
        sys.exit("no <svg> after %s" % title)
    return html[:start] + new_svg + html[end + len("</svg>"):]


def replace_all(html, old, new, label):
    """Verdict wording appears twice: in the overview table and in the panel badge."""
    n = html.count(old)
    if n == 0:
        if new in html:
            print("  %s already current" % label)
            return html
        sys.exit("could not find text to replace for %s" % label)
    print("  %s updated (%d occurrence%s)" % (label, n, "" if n == 1 else "s"))
    return html.replace(old, new)


def relabel_storage(html):
    """The grey block now carries pumped hydro as well as batteries."""
    old = '<i style="background:#8894a4"></i>Battery'
    n = html.count(old)
    if n:
        print("  legend Battery -> Storage (%d occurrences)" % n)
    return html.replace(old, '<i style="background:#8894a4"></i>Storage')


def stamp(html, run_name):
    line = ('<p class="sub" id="runstamp">Model side generated from run <b>%s</b> '
            '(baseline) on %s by tools/ndp_review_update.py. Plan side is '
            'hand-written.</p>' % (run_name, date.today().isoformat()))
    existing = re.search(r'<p class="sub" id="runstamp">.*?</p>', html, re.S)
    if existing:
        return html[:existing.start()] + line + html[existing.end():]
    anchor = re.search(r'<p class="sub">.*?</p>', html, re.S)
    if not anchor:
        sys.exit("no subtitle to anchor the run stamp to")
    return html[:anchor.end()] + line + html[anchor.end():]


VERDICTS = [
    # (old fragment, new fragment, label)
    # Georgia: written while the hydro candidates were unbuildable (BuildLimitperYear=0).
    # With that fixed the model builds hydro and solar, so "prefers solar" is wrong.
    ("Partial — model prefers solar over planned hydro",
     "Partial — builds hydro and solar, stops short of 10.3 GW",
     "Georgia verdict"),
    # Armenia: ANPP retires in 2036 and the 300 MW candidate is never picked up.
    ("Policy (nuclear) vs least-cost — model matches USAID LCEDP",
     "Diverges — no nuclear once ANPP retires in 2036",
     "Armenia verdict (badge)"),
    ("Model agrees with USAID least-cost",
     "Diverges — no nuclear once ANPP retires in 2036",
     "Armenia verdict (table)"),
]

GEN_NOTE = ('<div class="pn" data-gen="note">Model generation is domestic output only; '
            'net imports are excluded so the bars need not meet demand. Grey covers '
            'batteries and pumped hydro.</div>')


def add_gen_note(html, occurrence):
    """Drop a caveat under the nth model generation chart, once."""
    needle = '>Generation (TWh) — OUR MODEL (baseline)</div>'
    pos = -1
    for _ in range(occurrence + 1):
        pos = html.find(needle, pos + 1)
    end = html.find("</svg>", pos) + len("</svg>")
    # tools/legend_inline.py wraps the svg and its legend in a .chartrow; the note belongs
    # under the pair, not between them.
    row = html.rfind('<div class="chartrow">', pos, end)
    if row != -1:
        depth = 0
        for m in re.finditer(r"<div\b|</div>", html[row:]):
            depth += -1 if m.group(0) == "</div>" else 1
            if depth == 0:
                end = row + m.end()
                break
    tail = html[end:end + len(GEN_NOTE)]
    if tail == GEN_NOTE:
        return html
    return html[:end] + GEN_NOTE + html[end:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="unzipped simulations_run_* folder")
    ap.add_argument("--html", default=HTML_DEFAULT)
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--run-name", dest="run_name", default=None,
                    help="name to cite in the stamp when --run points at a scratch copy")
    args = ap.parse_args()

    print("run   : %s" % args.run)
    print("page  : %s" % args.html)
    df = load_run(args.run, args.scenario)
    cap = series(df, "CapacityTechFuel", 1000.0)
    gen = series(df, "EnergyTechFuelComplete", 1000.0)

    html = io.open(args.html, encoding="utf-8").read()
    before = len(html)

    for i, country in enumerate(COUNTRIES):
        html = replace_chart(html, "Installed capacity (GW) — OUR MODEL (baseline)",
                             svg(cap[country], "GW"), i)
        html = replace_chart(html, "Generation (TWh) — OUR MODEL (baseline)",
                             svg(gen[country], "TWh"), i)
        html = add_gen_note(html, i)
        tot = sum(v[-1] for v in cap[country].values())
        print("  %-11s 2040 capacity %6.1f GW  across %d buckets"
              % (country, tot, len(cap[country])))

    for old, new, label in VERDICTS:
        html = replace_all(html, old, new, label)

    html = relabel_storage(html)
    html = stamp(html, args.run_name or os.path.basename(os.path.normpath(args.run)))

    io.open(args.html, "w", encoding="utf-8", newline="\n").write(html)
    print("written: %d -> %d bytes" % (before, len(html)))


if __name__ == "__main__":
    main()
