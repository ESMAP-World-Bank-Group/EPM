# -*- coding: utf-8 -*-
"""The solar and wind profiles, on the time structure of build_demand.py.

    python data_build/build_vre.py

Reads:
    mappings/vre_profiles.csv              which DeCA rows each (zone, technology) reads
    mappings/seasons_months.csv            which months make each season
    extracted/pHours.csv                   the time structure, for its days
    <reference>/pHours.csv                 the 2020 structure, for its block widths
    <reference>/supply/pVREProfile.csv     the only intra-day shape there is
    the five DeCA AssumptionBooks, sheet RenewableProfile

Writes:
    extracted/pVREProfile.csv
    extracted/vre_report.csv

RUN AFTER build_demand.py: it reads the pHours that step writes.

WHY IT IS BUILT FROM TWO SOURCES. base.gms:812 writes the output of every solar and
wind unit as pVREgenProfile times the availability times the capacity, so this table
is the whole production of the renewable fleet, in level as well as in shape. Neither
source gives both.
  - DeCA gives the LEVEL and not the shape: RenewableProfile states a monthly
    utilization factor per plant, no hour of the day. Its own header says so, "Hourly
    profiles are also needed for all WPP and SPP power plants", a line the authors
    left for themselves.
  - The 2020 model gives the SHAPE and a level that is now five years old: its blocks
    are chronological hour-of-day groups, verifiable on the file itself, solar at zero
    in the first and last blocks and highest in the middle one.
So the day shape is taken from the 2020 profile, normalised to a mean of one, and
multiplied by the DeCA seasonal utilization factor, which is the month-length weighted
mean of the monthly factors of the season over the plants the mapping selects.

WHAT THE MAPPING SAYS. One line per (zone, technology): which workbook to read and
which rows of it, by a list of substrings matched on the DeCA name. The five books name
no zone for their existing plants, so the factor is a COUNTRY MEAN over every plant of
that technology, candidates included, high yield and low yield sites together. That is
an average site, which is what a generic candidate is. A zone whose line names no
workbook keeps its 2020 level unchanged: Afghanistan and Pakistan are outside the DeCA
perimeter.
One trap the mapping steps around: the Tajik book labels its own regions with a KZ
prefix, Khatlon and Sougd and Gorno-Badakhshan all reading "KZ ...". Matching on the
workbook rather than on the name is what keeps Tajik sites out of Kazakhstan.

WHAT IT DOES NOT DO. Every representative day of a season carries the same renewable
day, because a monthly factor holds no day to day variation: there is no cloudy day and
no calm day in this model, only an average one. That understates the backup a large
renewable fleet needs and it will have to be revisited when the fleet grows, with an
hourly source. It is written here so that it is not discovered later in the results.
"""

import argparse
import collections
import csv
import glob
import io
import os

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

BOOKS = r"C:/Users/wb590892/Documents/EPM_Models/ca_2026/data_collection/Mercados"
MONTH_LENGTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
HOURS_PER_DAY = 24


def read_csv(path):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        rd = csv.reader(fh)
        return next(rd), [r for r in rd if any(c.strip() for c in r)]


def write_csv(path, header, rows):
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def dicts(path):
    header, rows = read_csv(path)
    keys = [h.strip() for h in header]
    return [dict(zip(keys, r)) for r in rows]


def num(x):
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def seasons(path):
    out = collections.OrderedDict()
    for row in dicts(path):
        out[row["season"].strip()] = [int(m) for m in row["months"].split(",")]
    return out


def legacy_of(path):
    """season -> the season of the 2020 model it inherits its shape from.

    Our seasons and the 2020 ones share their names by accident of history, not by
    construction: the 2020 Q5 was a 130 h peak block, ours is July-September. The
    mapping says which is which so that neither set has to match the other.
    """
    out = {}
    for row in dicts(path):
        out[row["season"].strip()] = row["legacy"].strip()
    return out


def block_hours(row):
    """The width in hours of each block of the 2020 day. See build_demand.py."""
    values = [v for v in row if v is not None]
    days = sum(values) / float(HOURS_PER_DAY)
    exact = [v / days for v in values]
    base = [int(v) for v in exact]
    order = sorted(range(len(exact)), key=lambda i: exact[i] - base[i], reverse=True)
    for i in order[:HOURS_PER_DAY - sum(base)]:
        base[i] += 1
    return base


def legacy_profile(reference):
    """(zone, tech, season) -> [24 factors], the 2020 profile spread over the day."""
    widths = {}
    _, rows = read_csv(os.path.join(reference, "pHours.csv"))
    for r in rows:
        widths[r[0].strip()] = block_hours([num(c) for c in r[2:]])

    out = {}
    header, rows = read_csv(os.path.join(reference, "supply", "pVREProfile.csv"))
    for r in rows:
        z, tech, q = r[0].strip(), r[1].strip(), r[2].strip()
        if q not in widths:
            continue
        day = []
        for width, value in zip(widths[q], [num(c) or 0.0 for c in r[4:]]):
            day += [value] * width
        out[(z, tech, q)] = day
    return out


def monthly_factors(workbook_code):
    """(name, [12 utilization factors]) of every row of the RenewableProfile sheet."""
    pattern = os.path.join(BOOKS, "*- {0}_V5.1_Clean.xlsx".format(workbook_code))
    matches = glob.glob(pattern)
    if not matches:
        raise IOError("no AssumptionBook for " + workbook_code)
    book = openpyxl.load_workbook(matches[0], read_only=True, data_only=True)
    sheets = [s for s in book.sheetnames
              if s.replace(" ", "").lower() == "renewableprofile"]
    if not sheets:
        raise IOError("no RenewableProfile sheet in " + os.path.basename(matches[0]))
    sheet = book[sheets[0]]

    rows = list(sheet.iter_rows(values_only=True))
    head = next(i for i, r in enumerate(rows)
                if r and any(str(c).strip() == "January" for c in r if c))
    columns = {str(c).strip(): j for j, c in enumerate(rows[head]) if c}
    name_col = columns["Assigned Name"]
    month_cols = [columns[m] for m in MONTHS]

    out = []
    for r in rows[head + 1:]:
        if len(r) <= name_col or not r[name_col]:
            continue
        values = [num(r[j]) if j < len(r) else None for j in month_cols]
        if any(v is None for v in values):
            continue
        out.append((str(r[name_col]).strip(), values))
    book.close()
    return out


def selected(rows, match, exclude=""):
    """The rows whose name carries any of the substrings and none of the exclusions.

    Both lists are read from the mapping, case insensitive, separated by a bar. The
    exclusion exists because one Kazakh solar plant is called Arm Wind: a name is a
    weak technology tag and the mapping is where that is repaired, not the code.
    """
    wanted = [w.strip().lower() for w in match.split("|") if w.strip()]
    barred = [w.strip().lower() for w in exclude.split("|") if w.strip()]
    return [(n, v) for n, v in rows
            if any(w in n.lower() for w in wanted)
            and not any(w in n.lower() for w in barred)]


def main():
    ap = argparse.ArgumentParser(description="Build the solar and wind profiles.")
    ap.add_argument("--reference", default=os.path.join("epm", "input", "data_casa_2020"))
    args = ap.parse_args()
    reference = os.path.join(REPO, args.reference)

    quarters = seasons(os.path.join(HERE, "mappings", "seasons_months.csv"))
    shape = legacy_profile(reference)
    inherits = legacy_of(os.path.join(HERE, "mappings", "seasons_months.csv"))
    plan = dicts(os.path.join(HERE, "mappings", "vre_profiles.csv"))

    _, hours = read_csv(os.path.join(HERE, "extracted", "pHours.csv"))
    days = collections.OrderedDict()
    for r in hours:
        days.setdefault(r[0].strip(), []).append(r[1].strip())

    books, out, report, claimed = {}, [], [], {}
    for line in plan:
        z, tech = line["z"].strip(), line["tech"].strip()
        code, match = line["workbook"].strip(), line["match"].strip()
        if code and code not in books:
            books[code] = monthly_factors(code)
        picked = selected(books[code], match, line.get("exclude", "")) if code else []
        if code and not picked:
            raise SystemExit("no DeCA row matches {0} for {1} {2}".format(match, z, tech))
        if code:
            names = {n for n, _ in picked}
            claimed.setdefault((code, z), {})[tech] = names
            for other, taken in claimed[(code, z)].items():
                if other != tech and names & taken:
                    raise SystemExit("{0}: {1} is claimed by both {2} and {3}".format(
                        code, sorted(names & taken)[0], tech, other))

        for q, months in quarters.items():
            base = shape.get((z, tech, inherits[q]))
            if base is None:
                raise SystemExit("no 2020 shape for {0} {1} {2} (drawn from {3})"
                                 .format(z, tech, q, inherits[q]))
            mean = sum(base) / float(HOURS_PER_DAY)
            if picked:
                weight = sum(MONTH_LENGTH[m - 1] for m in months)
                factor = sum(sum(v[m - 1] for _, v in picked) / len(picked)
                             * MONTH_LENGTH[m - 1] for m in months) / weight
            else:
                factor = mean
            scale = (factor / mean) if mean else 0.0
            day = [v * scale for v in base]
            for d in days[q]:
                out.append([z, tech, q, d] + ["{0:.6g}".format(v) for v in day])
            report.append([z, tech, q, "deca" if picked else "casa_2020",
                           len(picked), "{0:.4f}".format(mean), "{0:.4f}".format(factor),
                           "{0:.4f}".format(max(day))])

    write_csv(os.path.join(HERE, "extracted", "pVREProfile.csv"),
              ["z", "tech", "q", "d"] + ["t{0}".format(h) for h in
                                         range(1, HOURS_PER_DAY + 1)], out)
    write_csv(os.path.join(HERE, "extracted", "vre_report.csv"),
              ["z", "tech", "q", "source", "deca_rows", "cf_2020", "cf_new", "max"],
              report)

    moved = [r for r in report if r[3] == "deca"]
    print("rows       {0} on {1} seasons x {2} days".format(
        len(out), len(quarters), len(days[list(quarters)[0]])))
    print("levels     {0} of {1} seasonal factors taken from DeCA".format(
        len(moved), len(report)))
    for tech in ("PV", "WT"):
        old = [float(r[5]) for r in moved if r[1] == tech]
        new = [float(r[6]) for r in moved if r[1] == tech]
        if old:
            print("{0:<10} mean capacity factor {1:.3f} in 2020, {2:.3f} now"
                  .format(tech, sum(old) / len(old), sum(new) / len(new)))


if __name__ == "__main__":
    main()
