# -*- coding: utf-8 -*-
"""Extract annual demand from the DeCA V5.1 assumption books into the EPM shape.

    python data_build/extract_demand.py [--deca FOLDER] [--scenario Reference]

Produces three files in data_build/extracted/:

    deca_demand_hourly.csv     hourly series aggregated to EPM zones (intermediate)
    deca_demand_annual.csv     energy and peak by country, both scenarios (audit)
    pDemandForecast.csv        the EPM resource, taken as is by the build

Method, in three steps.

1. Demand Evolution gives energy (GWh) and peak (MW) BY COUNTRY, 2023-2060, in two
   scenarios. No extrapolation is needed: the 2026-2050 horizon is fully covered.

2. Demand Profile gives 8760 hours per region. From it we derive, for each EPM
   zone, its energy share and its own peak. The intra-country split is therefore
   the one of the profile base year and stays constant over time: DeCA provides no
   regional trajectory.

3. Zonal energy = country energy x share. Zonal peak = the zone peak in the
   profile, rescaled by the growth of the country peak. We go through the
   coincidence factor observed in the profile, so that the zonal peak stays
   NON-COINCIDENT, which is what generate_demand.gms expects: it multiplies the
   normalised zone profile by that peak.

Afghanistan and Pakistan have no 2026 source. They are carried over from the 2020
model pDemandData and extended at their own growth rate.
"""

import argparse
import collections
import csv
import glob
import io
import os
import sys

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_DECA = os.path.join(os.path.dirname(REPO), "data_collection", "Mercados")

HEADER_ROW = 9          # DeCA books carry an 8-row title block
FIRST_DATA_ROW = 10


# ── Reading the books ─────────────────────────────────────────────────────────

def workbook(deca_dir, code):
    hits = glob.glob(os.path.join(deca_dir, "*{}_V5.1_Clean.xlsx".format(code)))
    if not hits:
        raise IOError("DeCA book not found for {} in {}".format(code, deca_dir))
    return openpyxl.load_workbook(hits[0], read_only=True, data_only=True)


def header_index(ws):
    hdr = next(ws.iter_rows(min_row=HEADER_ROW, max_row=HEADER_ROW, values_only=True))
    return {str(c).strip(): i for i, c in enumerate(hdr) if c not in (None, "")}


def read_profile(ws, columns):
    """Requested hourly series. The hour-index column filters the rows.

    The books carry total rows underneath the table (UZ row 8770 holds 76.7 TWh in
    the Central column): requiring a numeric hour index drops them cleanly, without
    hard-coding a row count.
    """
    idx = header_index(ws)
    hour_col = next(i for name, i in idx.items() if name.lower().replace(" ", "") in
                    ("year-hour", "hour-year"))
    missing = [c for c in columns if c not in idx]
    if missing:
        raise KeyError("columns missing from the profile: {} (available: {})"
                       .format(missing, sorted(idx)))
    out = {c: [] for c in columns}
    for row in ws.iter_rows(min_row=FIRST_DATA_ROW, values_only=True):
        if not isinstance(row[hour_col], (int, float)):
            continue
        for c in columns:
            v = row[idx[c]]
            out[c].append(float(v) if isinstance(v, (int, float)) else 0.0)
    return out


def read_evolution(ws):
    """{(scenario, 'Energy'|'Peak'): {year: value}}.

    The five books do not lay the block out the same way: KZ and TM prefix the row
    with the country code, KG, TJ and UZ do not; the labels carry footnote
    asterisks. So blocks are located by their unit, GWh or MW, and the scenario by
    the words Reference or Net Zero present in the row.
    """
    years, out = None, {}
    for row in ws.iter_rows(min_row=HEADER_ROW, values_only=True):
        cells = list(row)
        nums = [(i, c) for i, c in enumerate(cells)
                if isinstance(c, (int, float)) and 2000 < c < 2100 and float(c).is_integer()]
        labels = " ".join(str(c) for c in cells if isinstance(c, str))

        # header row of a block: a run of years
        if len(nums) >= 20:
            years = {i: int(c) for i, c in nums}
            continue
        if years is None:
            continue

        unit = "Energy" if " GWh" in " " + labels else ("Peak" if " MW" in " " + labels else None)
        if unit is None:
            continue
        scen = "Net Zero" if "net zero" in labels.lower() else (
               "Reference" if "reference" in labels.lower() else None)
        if scen is None:
            continue
        series = {y: float(cells[i]) for i, y in years.items()
                  if isinstance(cells[i], (int, float))}
        if series:
            out[(scen, unit)] = series
    return out


# ── Carrying Afghanistan / Pakistan over from the 2020 model ──────────────────

def legacy_demand(ref_dir):
    """Energy (GWh) and peak (MW) per zone and year, read from the 2020 model."""
    hours = {}
    with io.open(os.path.join(ref_dir, "pHours.csv"), encoding="utf-8-sig", newline="") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        for r in rd:                       # q, d, then one column per time block
            for i, v in enumerate(r[2:], 2):
                if v not in ("", None):
                    hours[(r[0], r[1], hdr[i])] = float(v)

    energy = collections.defaultdict(float)
    peak = collections.defaultdict(float)
    with io.open(os.path.join(ref_dir, "load", "pDemandData.csv"),
                 encoding="utf-8-sig", newline="") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)                     # z, q, d, y, then one column per block
        for r in rd:
            z, q, d, y = r[0], r[1], r[2], int(float(r[3]))
            for i, v in enumerate(r[4:], 4):
                if v in ("", None):
                    continue
                mw = float(v)
                energy[(z, y)] += mw * hours.get((q, d, hdr[i]), 0.0) / 1e3
                peak[(z, y)] = max(peak[(z, y)], mw)
    return energy, peak


def extend(series, years):
    """Extend a series at the compound rate of its last five years."""
    known = sorted(series)
    if not known:
        return {y: 0.0 for y in years}
    last, first = known[-1], known[max(0, len(known) - 6)]
    span = last - first
    rate = ((series[last] / series[first]) ** (1.0 / span) - 1.0) if span and series[first] else 0.0
    out = {}
    for y in years:
        if y in series:
            out[y] = series[y]
        elif y < known[0]:
            out[y] = series[known[0]]
        else:
            out[y] = series[last] * (1.0 + rate) ** (y - last)
    return out


# ── Assembly ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deca", default=DEFAULT_DECA)
    ap.add_argument("--scenario", default="Reference",
                    help="DeCA scenario taken as the reference (Reference | Net Zero)")
    ap.add_argument("--target", default=os.path.join(REPO, "epm", "input", "data_casa"))
    ap.add_argument("--reference", default=os.path.join(REPO, "epm", "input", "data_casa_2020"))
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "extracted")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # model years, read from y.csv: nothing is hard-coded
    with io.open(os.path.join(args.target, "y.csv"), encoding="utf-8-sig", newline="") as fh:
        years = [int(r[0]) for r in list(csv.reader(fh))[1:] if r and r[0].strip()]

    # lookup table EPM zone -> DeCA columns
    with io.open(os.path.join(HERE, "mappings", "zones_demand.csv"),
                 encoding="utf-8-sig", newline="") as fh:
        mapping = list(csv.DictReader(fh))

    books = sorted({m["workbook"] for m in mapping if m["workbook"]})
    evolution, profiles = {}, {}
    for code in books:
        wb = workbook(args.deca, code)
        evolution[code] = read_evolution(wb["Demand Evolution"])
        cols = sorted({c for m in mapping if m["workbook"] == code
                       for c in m["profile_columns"].split(";") if c})
        profiles[code] = read_profile(wb["Demand Profile"], cols)
        wb.close()

    # ---- audit log: both scenarios, by country
    with io.open(os.path.join(out_dir, "deca_demand_annual.csv"), "w",
                 encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["workbook", "scenario", "type", "year", "value"])
        for code, ev in sorted(evolution.items()):
            for (scen, unit), series in sorted(ev.items()):
                for y in sorted(series):
                    w.writerow([code, scen, unit, y, round(series[y], 3)])

    # ---- hourly series aggregated to EPM zones (intermediate deliverable)
    hourly = {}
    for m in mapping:
        if not m["workbook"]:
            continue
        cols = [c for c in m["profile_columns"].split(";") if c]
        series = profiles[m["workbook"]]
        hourly[m["z"]] = [sum(series[c][h] for c in cols) for h in range(len(series[cols[0]]))]
    with io.open(os.path.join(out_dir, "deca_demand_hourly.csv"), "w",
                 encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["z", "hour", "MW"])
        for z in sorted(hourly):
            for h, v in enumerate(hourly[z], 1):
                w.writerow([z, h, round(v, 3)])

    # ---- pDemandForecast
    legacy_energy, legacy_peak = legacy_demand(args.reference)
    rows, diag, warnings = {}, [], []

    for code in books:
        zones = [m for m in mapping if m["workbook"] == code]
        ev = evolution[code]
        try:
            e_country = ev[(args.scenario, "Energy")]
            p_country = ev[(args.scenario, "Peak")]
        except KeyError:
            raise KeyError("book {}: scenario '{}' missing (available: {})"
                           .format(code, args.scenario, sorted({s for s, _ in ev})))

        # Unit guard. A peak cannot be lower than the average load: when it is, the
        # block is in GW under an MW label. That is the case of the TJ book, which
        # reports a 3.72 MW peak for 19.6 TWh.
        mean_mw = max(e_country.values()) * 1e3 / 8760.0
        if max(p_country.values()) < mean_mw:
            p_country = {y: v * 1e3 for y, v in p_country.items()}
            warnings.append("   {}: peaks multiplied by 1000, the block was in GW "
                            "under an MW label".format(code))

        tot = {z["z"]: sum(hourly[z["z"]]) for z in zones}
        grand = sum(tot.values())
        pk = {z["z"]: max(hourly[z["z"]]) for z in zones}
        n = len(hourly[zones[0]["z"]])
        coincident = max(sum(hourly[z["z"]][h] for z in zones) for h in range(n))

        # the country peak of the profile is the yardstick; the trajectory is DeCA's
        base = min(y for y in p_country if y in e_country)
        for z in zones:
            share = tot[z["z"]] / grand
            ratio = pk[z["z"]] / coincident        # own peak / coincident peak
            rows[(z["z"], "Energy")] = {y: e_country.get(y, 0.0) * share for y in years}
            rows[(z["z"], "Peak")] = {y: p_country.get(y, 0.0) * ratio for y in years}
        diag.append("   {}  share {} | coincidence {:.2f} | profile base {:.1f} TWh vs DeCA {} {:.1f} TWh"
                    .format(code, " / ".join("{} {:.1f}%".format(z["z"], 100 * tot[z["z"]] / grand)
                                             for z in zones),
                            sum(pk.values()) / coincident, grand / 1e6, base,
                            e_country[base] / 1e3))

    for m in mapping:
        if m["workbook"]:
            continue
        e = extend({y: v for (zz, y), v in legacy_energy.items() if zz == m["z"]}, years)
        p = extend({y: v for (zz, y), v in legacy_peak.items() if zz == m["z"]}, years)
        rows[(m["z"], "Energy")], rows[(m["z"], "Peak")] = e, p

    order = [m["z"] for m in mapping]
    path = os.path.join(out_dir, "pDemandForecast.csv")
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["z", "type"] + [str(y) for y in years])
        for z in order:
            for kind in ("Energy", "Peak"):
                w.writerow([z, kind] + [round(rows[(z, kind)][y], 1) for y in years])

    print("Scenario used: {}".format(args.scenario))
    print("\n".join(diag))
    print("\n".join(warnings))
    print("Written: {}".format(path))
    for y in (years[0], years[-1]):
        tot_e = sum(rows[(z, "Energy")][y] for z in order)
        tot_p = sum(rows[(z, "Peak")][y] for z in order)
        print("   {}: {:.1f} TWh, sum of zonal peaks {:.0f} MW"
              .format(y, tot_e / 1e3, tot_p))


if __name__ == "__main__":
    main()
