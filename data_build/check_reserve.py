# -*- coding: utf-8 -*-
"""What the planning reserve asks of each zone, and what the zone already has.

    python data_build/check_reserve.py

Reads:
    <target>/reserve/pPlanningReserveMarginZone.csv   the margin, one per zone
    <target>/reserve/pReserveSeasonFlag.csv           the season firm capacity is valued in
    <target>/supply/pGenDataInput.csv                 capacity, status, start and retirement
    <target>/supply/pAvailabilityCustom.csv           the credit, season by season
    <target>/load/pDemandForecast.csv                 the zonal peak, year by year
    <reference>/reserve/pReserveSeasonFlag.csv        the same two files of the 2020 model,
    <reference>/supply/pAvailabilityCustom.csv        to measure what the season change did

Writes:
    extracted/reserve_margin_report.csv

A CHECK AND NOT A BUILD: it writes no input, it only measures. Nothing in the target
folder is touched.

WHAT IT MEASURES, AND WHY IT HAD TO BE MEASURED. base.gms:1000 asks each zone to cover
(1 + margin) times its peak with firm capacity, and base.gms:1005 values that firm
capacity in one season alone, the one flagged by pReserveSeasonFlag:

    eZonalPlanningReserveSupply(z,y).. sum(z2, vCapacityReserveFlow(z,z2,y))
        =l= sum(gzmap(g,z), pCapacityCreditPeakSeason(g,y)*vCap(g,y));

and main.gms:699 sets that credit to the availability of the unit in that season. The
2020 model flagged an artificial peak season Q5 in which 224 units out of 403 carried
an availability of 1.0 against 0.85 the rest of the year. THAT SEASON WAS REMOVED in
phase 1, being an artefact of the old time slicing, and the flag now points at Q1,
where those same units carry 0.85. The margin was never touched and still reads 0.15
everywhere, but the requirement it expresses has moved: asking for 1.15 times the peak
out of a fleet valued at nameplate is not the same thing as asking it out of a fleet
valued at 85 per cent of nameplate.

Which of the two is right is a modelling choice and this script does not make it. It
prints what each convention gives, zone by zone, so that the choice is made on numbers.

THE PEAK IS READ FROM pDemandForecast AND NOT FROM pDemandData, which is empty until
the representative days of phase 8 are built. The equation will use the smax over
pDemandData once it exists; the forecast peak is what that smax is meant to reproduce,
so the two agree by construction and the report is valid in advance.

WHAT IT DELIBERATELY DOES NOT COUNT. Candidates, whose Capacity column is a build limit
and not a plant, so the gap it prints is the gap the expansion has to close. And the
firm capacity a zone can borrow from its neighbours: base.gms:1011 lets a zone commit
capacity to another over a corridor, so a zone short on its own is not necessarily
short once the network is counted. The country totals give an idea of that pooling
where the corridors are strong.
"""

import argparse
import csv
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def read_csv(path):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        rd = csv.reader(fh)
        return next(rd), [r for r in rd if any(c.strip() for c in r)]


def write_csv(path, header, rows):
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def index(header):
    return {name.strip(): j for j, name in enumerate(header)}


def num(x):
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def flagged_season(path):
    """The season firm capacity is valued in. One is expected, more is not an error."""
    _, rows = read_csv(path)
    return [r[0].strip() for r in rows if num(r[1])]


def credits(path, season):
    """{generator: availability in that season}."""
    header, rows = read_csv(path)
    ci = index(header)
    if season not in ci:
        raise KeyError("no column " + season + " in " + path)
    return dict((r[0].strip(), num(r[ci[season]]) or 0.0) for r in rows)


def peaks(path):
    """{zone: {year: peak MW}} from the Peak rows of the demand forecast."""
    header, rows = read_csv(path)
    hi = index(header)
    ys = [(int(c), j) for c, j in hi.items() if c.isdigit()]
    out = {}
    for r in rows:
        if r[hi["type"]].strip().lower() != "peak":
            continue
        out[r[0].strip()] = dict((y, num(r[j]) or 0.0) for y, j in ys)
    return out


def main():
    ap = argparse.ArgumentParser(description="Measure the zonal planning reserve.")
    ap.add_argument("--reference", default=os.path.join("epm", "input", "data_casa_2020"))
    ap.add_argument("--target", default=os.path.join("epm", "input", "data_casa"))
    ap.add_argument("--years", default="2026,2035,2050")
    args = ap.parse_args()

    target = os.path.join(REPO, args.target)
    reference = os.path.join(REPO, args.reference)
    wanted = [int(y) for y in args.years.split(",")]

    now = flagged_season(os.path.join(target, "reserve", "pReserveSeasonFlag.csv"))
    was = flagged_season(os.path.join(reference, "reserve", "pReserveSeasonFlag.csv"))
    if len(now) != 1 or len(was) != 1:
        raise ValueError("expected exactly one flagged season on each side")
    cr_now = credits(os.path.join(target, "supply", "pAvailabilityCustom.csv"), now[0])
    cr_was = credits(os.path.join(reference, "supply", "pAvailabilityCustom.csv"), was[0])

    _, mrows = read_csv(os.path.join(target, "reserve", "pPlanningReserveMarginZone.csv"))
    margin = dict((r[0].strip(), num(r[1]) or 0.0) for r in mrows)

    _, zrows = read_csv(os.path.join(target, "zcmap.csv"))
    country = dict((r[0].strip(), r[1].strip()) for r in zrows)

    header, gens = read_csv(os.path.join(target, "supply", "pGenDataInput.csv"))
    gi = index(header)
    pk = peaks(os.path.join(target, "load", "pDemandForecast.csv"))

    rows, totals = [], {}
    for z in sorted(margin):
        for y in wanted:
            firm_now = firm_was = 0.0
            for g in gens:
                if g[gi["z"]].strip() != z:
                    continue
                if (num(g[gi["Status"]]) or 0) == 3:
                    continue          # a candidate, its Capacity is a build limit
                st, rt = num(g[gi["StYr"]]) or 0, num(g[gi["RetrYr"]]) or 9999
                if not (st <= y <= rt):
                    continue
                cap = num(g[gi["Capacity"]]) or 0.0
                name = g[0].strip()
                firm_now += cap * cr_now.get(name, 0.0)
                firm_was += cap * cr_was.get(name, 0.0)
            peak = pk.get(z, {}).get(y, 0.0)
            need = (1 + margin[z]) * peak
            rows.append([z, country.get(z, ""), y,
                         "{0:.0f}".format(peak), "{0:g}".format(margin[z]),
                         "{0:.0f}".format(need), "{0:.0f}".format(firm_now),
                         "{0:.0f}".format(firm_now - need),
                         "{0:.0f}".format(firm_was),
                         "{0:.0f}".format(firm_was - need)])
            k = (country.get(z, ""), y)
            t = totals.setdefault(k, [0.0, 0.0, 0.0])
            t[0] += need
            t[1] += firm_now
            t[2] += firm_was

    write_csv(os.path.join(HERE, "extracted", "reserve_margin_report.csv"),
              ["z", "c", "year", "peak_mw", "margin", "required_mw",
               "firm_" + now[0] + "_mw", "gap_mw",
               "firm_" + was[0] + "_2020_mw", "gap_2020_mw"], rows)

    print("firm capacity valued in {0} (2020 model: {1})".format(now[0], was[0]))
    print("%-6s %6s %10s %10s %10s %10s" % ("c", "year", "required", "firm", "gap",
                                            "gap 2020"))
    for (c, y) in sorted(totals):
        need, fn, fw = totals[(c, y)]
        print("%-6s %6d %10.0f %10.0f %10.0f %10.0f" % (c, y, need, fn, fn - need,
                                                        fw - need))


if __name__ == "__main__":
    main()
