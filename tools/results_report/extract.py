# -*- coding: utf-8 -*-
"""Build the compact JSON cache the results report is rendered from.

An EPM run holds ~110 MB of hourly dispatch per scenario; the report needs a few
hundred kB of it.  This module streams the big files once and writes one JSON
per run holding only what the charts draw.

    python extract.py --run simulations_run_20260819_204446

Gotchas encoded here, all verified against the 2026-08-19 run:
  * pTransmissionMerged's NetImport is off by 1000x for *internal* pairs
    (Georgia->AzerbaijanMain 2035: Interchange 2915.7 GWh, NetImport -2.9157).
    Only Interchange is used for internal flows; external trade comes from
    summary.csv, which is consistent.
  * External corridor capacity is absent from the outputs entirely; it is read
    back from the input file the scenario actually used, resolved through
    input_scenarios.csv.
  * The `baseline` scenario column duplicates LC_Baseline exactly and is dropped.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # .../blacksea_2026/EPM
EPM = ROOT / "epm"
OUTVIEW = EPM / "output_view"
DATA = EPM / "input" / "data_blacksea"
REFLINES = ROOT / "pre-analysis" / "data" / "reference_lines.csv"

YEARS = [str(y) for y in range(2025, 2041)]
DISPATCH_YEARS = ["2025", "2030", "2035"]
HOURS = ["t%02d" % i for i in range(1, 25)]

# Stack order, bottom to top: baseload first, then dispatchable, then variable.
FUEL_ORDER = ["Nuclear", "Coal", "Gas", "Diesel", "Biomass", "Geothermal",
              "Reservoir", "ROR", "PSH", "Onshore Wind", "Offshore Wind", "PV",
              "Battery"]
# Dispatch-only rows that are not generation technologies.
DISPATCH_EXTRA = ["Imports", "Exports", "Storage Charge", "Unmet demand"]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- run layout

def scenario_dirs(run):
    """Scenario folders that actually carry output_csv.

    `baseline` is the config-only column epm.py always writes.  When the run
    also holds LC_Baseline the two are identical and `baseline` is dropped; in
    runs assembled from several server launches LC_Baseline is absent and
    `baseline` is the reference, so it has to stay.
    """
    out = []
    for d in sorted(run.iterdir()):
        if d.is_dir() and (d / "output_csv").is_dir():
            out.append(d.name)
    if "LC_Baseline" in out and "baseline" in out:
        out.remove("baseline")
    return out


def read_zcmap():
    df = pd.read_csv(DATA / "zcmap.csv", encoding="utf-8-sig")
    return dict(zip(df["z"], df["c"]))


def read_hours():
    """Hours of the year each (season, day type, hour) slot stands for.

    The charts need this twice: to weight a dispatch slot back into energy, and
    to label each day type with the share of the year it represents.
    """
    df = pd.read_csv(DATA / "pHours.csv")
    out = {}
    for _, r in df.iterrows():
        for h in HOURS:
            out["%s|%s|%s" % (r["q"], r["d"], h)] = float(r[h])
    return out


def input_file_for(run, scenario, param):
    """The input CSV a scenario used for `param`, as an absolute path."""
    df = pd.read_csv(run / "input_scenarios.csv")
    row = df[df["paramNames"] == param]
    if not len(row) or scenario not in row.columns:
        return None
    rel = str(row.iloc[0][scenario]).strip()
    if not rel or rel == "nan":
        return None
    return EPM / rel          # paths in input_scenarios.csv are epm-relative


# ---------------------------------------------------------------- annual side

def load_summary(run):
    """summary.csv, restricted to the report horizon.

    The file carries a repeated header row part-way through, so `year` may come
    back as float or as the literal string "year"; it is coerced before use.
    """
    df = pd.read_csv(run / "summary.csv")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()]
    df["year"] = df["year"].astype(int).astype(str)
    for c in df.columns[5:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df[df["year"].isin(YEARS)]


def year_series(sub, scen):
    """{year -> value} collapsed onto the 16-year axis, missing reads as 0."""
    v = dict(zip(sub["year"].astype(str), sub[scen]))
    return [round(float(v.get(y, 0) or 0), 4) for y in YEARS]


def build_annual(summary, scen, zones):
    """Everything the annual charts need for one scenario over one zone set."""
    zs = set(zones)
    d = summary[summary["zone"].isin(zs)]
    out = {}

    for key, attr in (("cap", "Capacity: MW"), ("gen", "Energy: GWh")):
        block = {}
        sub = d[d["attribute"] == attr]
        for fuel, g in sub.groupby("resolution"):
            s = year_series(g.groupby("year", as_index=False)[scen].sum(), scen)
            if any(abs(v) > 1e-6 for v in s):
                block[fuel] = [round(v / 1000.0, 4) for v in s]   # GW / TWh
        out[key] = {f: block[f] for f in FUEL_ORDER if f in block}

    for key, attr in (("demand", "Demand: GWh"),
                      ("unmet", "Unmet demand: GWh"),
                      ("emissions", "Emissions: MtCO2"),
                      ("surplus", "Surplus generation: GWh")):
        sub = d[d["attribute"] == attr].groupby("year", as_index=False)[scen].sum()
        s = year_series(sub, scen)
        out[key] = [round(v / (1000.0 if "GWh" in attr else 1.0), 4) for v in s]

    # Price: a weighted mean, not a sum.
    pr = d[d["attribute"] == "Price: $/MWh"]
    if len(pr):
        out["price"] = year_series(pr.groupby("year", as_index=False)[scen].mean(), scen)

    # Internal flows crossing the scope boundary, by partner zone.  Exchange
    # rows are one-directional: zone -> resolution.  A row leaving the scope is
    # an export; one entering it is an import, credited to the source zone.
    imp = defaultdict(lambda: [0.0] * 16)
    exp = defaultdict(lambda: [0.0] * 16)
    ex = summary[summary["attribute"] == "Annual Energy Exchanges: GWh"]
    for _, r in ex.iterrows():
        z, p, y = r["zone"], r["resolution"], str(r["year"])
        if y not in YEARS:
            continue
        v = float(r[scen] or 0) / 1000.0
        if abs(v) < 1e-9:
            continue
        i = YEARS.index(y)
        if z in zs and p not in zs:
            exp[p][i] += v
        elif p in zs and z not in zs:
            imp[z][i] += v

    # External partners (Russia, Romania, ...) come from summary, already netted
    # per direction and consistent with the energy balance.
    for attr, bucket in (("Annual Energy Imports External: GWh", imp),
                         ("Annual Energy Exports External: GWh", exp)):
        sub = summary[(summary["attribute"] == attr) & (summary["zone"].isin(zs))]
        for _, r in sub.iterrows():
            y = str(r["year"])
            if y not in YEARS:
                continue
            v = float(r[scen] or 0) / 1000.0
            if abs(v) > 1e-9:
                bucket[r["resolution"]][YEARS.index(y)] += v

    partners = sorted(set(imp) | set(exp))
    out["trade"] = {p: {"imp": [round(x, 4) for x in imp[p]],
                        "exp": [round(x, 4) for x in exp[p]]}
                    for p in partners}
    out["ext_partners"] = sorted(
        set(summary[summary["attribute"].str.contains("External", na=False)]
            ["resolution"].dropna()))
    return out


# ------------------------------------------------------------- corridor side

def build_corridors(run, scen, zcmap):
    """Per zone-pair: NTC, directional flow and utilisation, over 16 years.

    Internal pairs come from pTransmissionMerged.  External corridors have no
    capacity in the outputs, so their NTC is read from the input file the
    scenario used and their flow from summary.csv.
    """
    tm = pd.read_csv(run / scen / "output_csv" / "pTransmissionMerged.csv")
    tm["y"] = tm["y"].astype(str)
    tm = tm[tm["y"].isin(YEARS)]

    cor = {}

    def slot(a, b, external=False):
        key = "%s|%s" % (a, b) if a < b else "%s|%s" % (b, a)
        if key not in cor:
            lo, hi = sorted([a, b])
            cor[key] = {"a": lo, "b": hi, "external": external,
                        "ntc": [0.0] * 16,
                        "fwd": [0.0] * 16,     # lo -> hi, GWh
                        "rev": [0.0] * 16,     # hi -> lo, GWh
                        "util": [0.0] * 16}
        return cor[key]

    # InterconUtilization is directional -- z->z2 energy over NTC x 8760 h --
    # so the line's average utilisation is the sum of its two rows.  A max here
    # silently dropped the minority direction.  Capacity is the same line seen
    # from both ends and stays a max.
    for _, r in tm[tm["attribute"] == "TransmissionCapacity"].iterrows():
        c = slot(r["z"], r["uni"])
        i = YEARS.index(r["y"])
        c["ntc"][i] = max(c["ntc"][i], round(float(r["value"] or 0), 4))

    for _, r in tm[tm["attribute"] == "InterconUtilization"].iterrows():
        c = slot(r["z"], r["uni"])
        c["util"][YEARS.index(r["y"])] += round(float(r["value"] or 0), 4)

    for _, r in tm[tm["attribute"] == "Interchange"].iterrows():
        z, p, i = r["z"], r["uni"], YEARS.index(r["y"])
        c = slot(z, p)
        c["fwd" if z == c["a"] else "rev"][i] += round(float(r["value"] or 0) / 1000.0, 4)

    # External corridors: flow from summary, capacity from the scenario's input.
    summ = load_summary(run)
    for attr, direction in (("Annual Energy Imports External: GWh", "in"),
                            ("Annual Energy Exports External: GWh", "out")):
        for _, r in summ[summ["attribute"] == attr].iterrows():
            z, p, y = r["zone"], r["resolution"], str(r["year"])
            if y not in YEARS or z not in zcmap:
                continue
            v = float(r[scen] or 0) / 1000.0
            if abs(v) < 1e-9:
                continue
            c = slot(z, p, external=True)
            c["external"] = True
            i = YEARS.index(y)
            # fwd is always lo -> hi; "in" means p -> z.
            fwd = (p == c["a"]) if direction == "in" else (z == c["a"])
            c["fwd" if fwd else "rev"][i] += v

    ext_file = input_file_for(run, scen, "pExtTransferLimit")
    if ext_file and ext_file.exists():
        ex = pd.read_csv(ext_file)
        ex.columns = [str(c).strip() for c in ex.columns]
        # An external corridor gets a slot even with no flow, so the capacity
        # bars still show the link exists.
        for (z, zext), g in ex.groupby(["z", "zext"]):
            c = slot(z, zext, external=True)
            c["external"] = True
            for i, y in enumerate(YEARS):
                if y in g.columns:
                    c["ntc"][i] = round(float(pd.to_numeric(
                        g[y], errors="coerce").max() or 0), 4)

    # Utilisation for external corridors: energy moved vs what the NTC allows.
    # Flows are TWh, capacity MW, so 1e6 converts before the 8760 h comparison.
    for c in cor.values():
        if not c["external"]:
            continue
        for i in range(16):
            cap = c["ntc"][i]
            if cap > 0:
                c["util"][i] = round((c["fwd"][i] + c["rev"][i]) * 1e6
                                     / (cap * 8760.0), 4)

    return cor


def build_corridor_meta(zcmap):
    """Corridor -> project group and the physical lines behind it."""
    df = pd.read_csv(REFLINES, comment="#")
    meta = {}
    for _, r in df.iterrows():
        a, b = str(r["from_zone"]), str(r["to_zone"])
        if a == "nan" or b == "nan" or a == b:
            continue
        key = "%s|%s" % tuple(sorted([a, b]))
        m = meta.setdefault(key, {"project": None, "lines": []})
        proj = r["project"]
        proj = None if (pd.isna(proj) or not str(proj).strip()) else str(proj).strip()
        if proj and not m["project"]:
            m["project"] = proj
        kv = str(r["voltage_kv"])
        kv = "" if kv in ("nan", "?") else re.sub(r"[^0-9/+ ]", "", kv).strip()
        m["lines"].append({
            "from": str(r["from_substation"]), "to": str(r["to_substation"]),
            "kv": kv, "status": str(r["status"]),
            "entry": "" if pd.isna(r["earliest_entry"]) else str(r["earliest_entry"]),
            "project": proj or "",
        })
    for m in meta.values():
        if not m["project"]:
            m["project"] = "Existing network"
    return meta


# -------------------------------------------------------------- dispatch side

def _slot_axis(df):
    """Ordered (q, d, t) axis, seasons then day types then hours."""
    a = df[["q", "d", "t"]].drop_duplicates()
    a = a.sort_values(["q", "d", "t"], kind="mergesort")
    axis = [{"q": r.q, "d": r.d, "t": r.t} for r in a.itertuples()]
    return axis, {(x["q"], x["d"], x["t"]): i for i, x in enumerate(axis)}


def build_dispatch(run, scen, zones, years):
    """Hourly stack + demand for one zone set, streamed out of the big CSV.

    pDispatchComplete is ~110 MB per scenario, so it is read in chunks and cut
    down to the three report years and the zones in scope before anything else.

    """
    path = run / scen / "output_csv" / "pDispatchComplete.csv"
    zs, ys = set(zones), set(years)
    keep = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        chunk["y"] = chunk["y"].astype(str)
        chunk = chunk[chunk["z"].isin(zs) & chunk["y"].isin(ys)]
        if len(chunk):
            keep.append(chunk[["y", "q", "d", "t", "uni", "value"]])
    if not keep:
        return {"axis": [], "years": {}}

    df = pd.concat(keep, ignore_index=True)
    df = df.groupby(["y", "q", "d", "t", "uni"], as_index=False)["value"].sum()
    axis, idx = _slot_axis(df)
    n = len(axis)
    df["i"] = [idx[k] for k in zip(df["q"], df["d"], df["t"])]

    out = {}
    for y in years:
        sub = df[df["y"] == y]
        series = {}
        for uni, g in sub.groupby("uni"):
            v = [0.0] * n
            for i, val in zip(g["i"], g["value"]):
                v[i] = round(float(val), 2)
            if any(abs(x) > 1e-6 for x in v):
                series[uni] = v
        out[y] = series
    return {"axis": axis, "years": out}


def net_trade(block, hours, ext):
    """Collapse Imports and Exports into the net position of the scope.

    Imports and Exports are reported per zone, corridors inside the scope
    included, so a scope grouping zones that trade with each other carries its
    internal traffic in the same two series: on Turkiye the bands are ten times
    the external trade and run past demand.  There is no hourly flow per
    corridor in the outputs, so the only clean cut is the net position, which is
    what the internal flows cancel out of.  Returns True when it applied.
    """
    axis = block["axis"]
    if not axis:
        return False
    w = [hours["%s|%s|%s" % (s["q"], s["d"], s["t"])] for s in axis]
    n = len(axis)
    hit = False
    for y, series in block["years"].items():
        imp = series.get("Imports")
        exp = series.get("Exports")
        if not imp and not exp:
            continue
        imp = imp or [0.0] * n
        exp = exp or [0.0] * n
        gross = sum(w[i] * (abs(imp[i]) + abs(exp[i])) for i in range(n)) / 1e6
        if gross <= 1.6 * max(ext.get(y, 0.0), .001):
            continue
        net = [round(imp[i] + exp[i], 2) for i in range(n)]
        series["Imports"] = [v if v > 0 else 0.0 for v in net]
        series["Exports"] = [v if v < 0 else 0.0 for v in net]
        hit = True
    return hit


def build_price(run, scen, zones, years):
    """Marginal cost per time slot, averaged over the zones in scope."""
    path = run / scen / "output_csv" / "pHourlyPrice.csv"
    zs, ys = set(zones), set(years)
    keep = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        chunk["y"] = chunk["y"].astype(str)
        chunk = chunk[chunk["z"].isin(zs) & chunk["y"].isin(ys)]
        if len(chunk):
            keep.append(chunk[["y", "q", "d", "t", "value"]])
    if not keep:
        return {}
    df = pd.concat(keep, ignore_index=True)
    df = df.groupby(["y", "q", "d", "t"], as_index=False)["value"].mean()
    return {"%s|%s|%s|%s" % k: round(float(v), 2)
            for k, v in zip(zip(df["y"], df["q"], df["d"], df["t"]), df["value"])}


# ------------------------------------------------------------------ geometry

def simplify(coords, tol=0.02):
    """Douglas-Peucker on a lon/lat ring; keeps the map under a few hundred kB."""
    if len(coords) < 3:
        return coords
    def dist(p, a, b):
        (x, y), (x1, y1), (x2, y2) = p, a, b
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
        return ((x - x1 - t * dx) ** 2 + (y - y1 - t * dy) ** 2) ** 0.5
    dmax, imax = 0, 0
    for i in range(1, len(coords) - 1):
        d = dist(coords[i], coords[0], coords[-1])
        if d > dmax:
            dmax, imax = d, i
    if dmax > tol:
        return (simplify(coords[:imax + 1], tol)[:-1]
                + simplify(coords[imax:], tol))
    return [coords[0], coords[-1]]


def ring_area(ring):
    s = 0.0
    for i in range(len(ring) - 1):
        s += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return abs(s) / 2.0


def polys(geom, tol):
    """Geometry -> list of simplified outer rings, biggest first."""
    if not geom:
        return []
    t, c = geom.get("type"), geom.get("coordinates")
    raw = [c[0]] if t == "Polygon" else [p[0] for p in c] if t == "MultiPolygon" else []
    out = []
    for ring in raw:
        r = [[round(x, 3), round(y, 3)] for x, y in simplify(ring, tol)]
        if len(r) >= 4:
            out.append(r)
    out.sort(key=ring_area, reverse=True)
    return out[:6]


def centroid(rings):
    """Area-weighted centroid of the largest ring — where an arrow anchors."""
    if not rings:
        return None
    r = rings[0]
    a = sx = sy = 0.0
    for i in range(len(r) - 1):
        cr = r[i][0] * r[i + 1][1] - r[i + 1][0] * r[i][1]
        a += cr
        sx += (r[i][0] + r[i + 1][0]) * cr
        sy += (r[i][1] + r[i + 1][1]) * cr
    if abs(a) < 1e-12:
        return [round(sum(p[0] for p in r) / len(r), 3),
                round(sum(p[1] for p in r) / len(r), 3)]
    return [round(sx / (3 * a), 3), round(sy / (3 * a), 3)]


def clip_ring(ring, box):
    """Sutherland-Hodgman clip of a ring against a lon/lat rectangle.

    External countries such as Russia stretch to the Pacific; drawn whole they
    would squash the study area to a sliver and put the arrow anchor in Siberia.
    Clipping them to the map frame keeps both the shape and the anchor useful.
    """
    x0, y0, x1, y1 = box
    edges = [(lambda p: p[0] >= x0, lambda a, b: _cx(a, b, x0)),
             (lambda p: p[0] <= x1, lambda a, b: _cx(a, b, x1)),
             (lambda p: p[1] >= y0, lambda a, b: _cy(a, b, y0)),
             (lambda p: p[1] <= y1, lambda a, b: _cy(a, b, y1))]
    out = ring[:-1] if ring and ring[0] == ring[-1] else ring[:]
    for inside, cut in edges:
        if not out:
            return []
        nxt, prev = [], out[-1]
        for cur in out:
            if inside(cur):
                if not inside(prev):
                    nxt.append(cut(prev, cur))
                nxt.append(cur)
            elif inside(prev):
                nxt.append(cut(prev, cur))
            prev = cur
        out = nxt
    return out + [out[0]] if out else []


def _cx(a, b, x):
    t = (x - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
    return [x, a[1] + t * (b[1] - a[1])]


def _cy(a, b, y):
    t = (y - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
    return [a[0] + t * (b[0] - a[0]), y]


# Trading partners that exist in the topology as a pseudo zone rather than as a
# country of their own: no geometry carries them, so the map has nowhere to
# anchor the arrow.  The swap with Iran is settled at the Meghri / Marand
# crossing, which is where the arrow should end rather than in central Iran.
PSEUDO_ANCHOR = {"iran_swap": [46.35, 38.82]}


def build_geo(zcmap):
    out = {"zones": {}, "ext": {}, "centroids": {}}

    # zones.geojson also carries Romania and Bulgaria, which are neighbours
    # rather than modelled zones; zcmap is what decides.
    gj = json.loads((DATA / "zones.geojson").read_text(encoding="utf-8"))
    for f in gj.get("features", []):
        z = (f.get("properties") or {}).get("z")
        rings = polys(f.get("geometry"), 0.03) if z else []
        if rings:
            out["zones" if z in zcmap else "ext"][z] = rings
            out["centroids"][z] = centroid(rings)

    # The map frame: the modelled zones plus a margin for the neighbours.
    xs = [p[0] for r in out["zones"].values() for ring in r for p in ring]
    ys = [p[1] for r in out["zones"].values() for ring in r for p in ring]
    box = [min(xs) - 5.0, min(ys) - 4.0, max(xs) + 5.0, max(ys) + 4.0]
    out["box"] = [round(v, 3) for v in box]

    gj = json.loads((DATA / "zones_ext.geojson").read_text(encoding="utf-8"))
    for f in gj.get("features", []):
        z = (f.get("properties") or {}).get("z")
        if not z or z in out["zones"]:
            continue
        rings = []
        for ring in polys(f.get("geometry"), 0.15):
            c = clip_ring(ring, box)
            if len(c) >= 4:
                rings.append([[round(x, 3), round(y, 3)] for x, y in c])
        if rings:
            rings.sort(key=ring_area, reverse=True)
            out["ext"][z] = rings
            out["centroids"][z] = centroid(rings)

    for z, c in PSEUDO_ANCHOR.items():
        out["centroids"].setdefault(z, list(c))
    return out


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="simulations_run_20260819_204446")
    ap.add_argument("--scenarios", default="baseline,LC_Iso")
    ap.add_argument("--countries", default="Georgia",
                    help="pays dont on garde le dispatch horaire")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = OUTVIEW / a.run
    if not run.is_dir():
        sys.exit("run introuvable : %s" % run)
    scens = [s for s in a.scenarios.split(",") if s in scenario_dirs(run)]
    if not scens:
        sys.exit("aucun scenario valide parmi %s" % scenario_dirs(run))

    zcmap = read_zcmap()
    model_zones = [z for z in zcmap if z != "iran_swap"]
    countries = sorted({zcmap[z] for z in model_zones})
    scopes = {c: [z for z in model_zones if zcmap[z] == c] for c in countries}
    scopes["Region"] = model_zones

    log("run       : %s" % run.name)
    log("scenarios : %s" % ", ".join(scens))

    summary = load_summary(run)
    out = {
        "run": run.name,
        "scenarios": scens,
        "years": YEARS,
        "dispatch_years": DISPATCH_YEARS,
        "zcmap": zcmap,
        "scopes": scopes,
        "fuel_order": FUEL_ORDER,
        "dispatch_extra": DISPATCH_EXTRA,
        "hours": read_hours(),
        "annual": {}, "corridors": {}, "dispatch": {}, "price": {},
        "corridor_meta": build_corridor_meta(zcmap),
        "geo": build_geo(zcmap),
    }

    for scope, zones in scopes.items():
        out["annual"][scope] = {}
        for s in scens:
            out["annual"][scope][s] = build_annual(summary, s, zones)
        log("  annuel   %-10s ok" % scope)

    for s in scens:
        out["corridors"][s] = build_corridors(run, s, zcmap)
        log("  couloirs %-10s %d paires" % (s, len(out["corridors"][s])))

    out["dispatch_netted"] = []
    want = [c.strip() for c in a.countries.split(",") if c.strip()]
    want = [c for c in want if c in scopes] + ["Region"]
    for scope in want:
        out["dispatch"][scope], out["price"][scope] = {}, {}
        netted = False
        for s in scens:
            out["dispatch"][scope][s] = build_dispatch(run, s, scopes[scope],
                                                       DISPATCH_YEARS)
            # Gross trade of the scope against what it actually exchanges with
            # the outside, year by year, in TWh.
            tr = out["annual"][scope][s].get("trade", {})
            ext = {}
            for iy, y in enumerate(YEARS):
                if y in DISPATCH_YEARS:
                    ext[y] = sum(abs(v.get(side, [0] * len(YEARS))[iy])
                                 for v in tr.values() for side in ("imp", "exp"))
            netted |= net_trade(out["dispatch"][scope][s], out["hours"], ext)
            out["price"][scope][s] = build_price(run, s, scopes[scope],
                                                 DISPATCH_YEARS)
            log("  dispatch %-10s %-12s %d pas de temps%s"
                % (scope, s, len(out["dispatch"][scope][s]["axis"]),
                   "  (trade netted)" if netted else ""))
        if netted:
            out["dispatch_netted"].append(scope)

    dest = Path(a.out) if a.out else Path(__file__).parent / "cache" / ("%s.json" % run.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    log("ecrit : %s (%.1f Mo)" % (dest, dest.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
