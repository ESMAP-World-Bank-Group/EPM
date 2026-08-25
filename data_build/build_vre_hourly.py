"""Bring the hourly Renewables.ninja series to the level the DeCA books state.

WHY THIS STEP EXISTS. The fetch gave us a real hourly year per zone -- a cloudy day, a
still day, a day-to-day spread that no monthly utilization factor can hold. What it did
not give us is a level we believe. Renewables.ninja models a clean array with a flat 10%
loss where DeCA measures metered output with outages, soiling and curtailment in it, and
its wind runs through a Gamesa G114/2000, a low-specific-power machine that reads high by
design, on MERRA-2 cells half a degree wide that cannot see a Pamir valley. Comparing the
two over the fetched year: solar came out 22% high, a clean offset; wind came out
anywhere from a twentieth to twice the model's figure, which is not an offset at all.

So the rule, and it is the whole of this file: TAKE THE SHAPE FROM RENEWABLES.NINJA AND
THE LEVEL FROM DECA. Each zone, technology and month is multiplied by whatever factor
brings its mean onto the stated one, and nothing else about the series is touched.

AND WHERE DECA IS SILENT, TAKE THE LEVEL FROM THE TERRAIN, NOT FROM 2020. The eight
Afghan and Pakistani wind zones have no book, and the 2020 model gave all eight the same
0.333 -- one number repeated, not eight readings. The Global Wind Atlas has now been
sampled over each zone's own polygon at 250 m, and it contradicts that placeholder in
both directions: Herat, where the Wind of 120 Days blows, reaches 0.56 on its best tenth,
while the Tajik lowland of northern Afghanistan tops out near 0.18. Those zones carry
level_source = atlas_p90 in the mapping and take their height from the atlas, keeping
2020's month-to-month shape because the atlas has no calendar to offer. Zones with a DeCA
book are left alone: a measurement outranks a terrain model.

WHY THE RESCALE IS HERE AND NOT IN build_vre.py. Two consumers need the corrected series,
not one. build_vre.py reads it to build pVREProfile on the demand's own representative
days; the Poncelet clustering in pre-analysis/representative_days reads it to choose those
days in the first place. Rescaling inside either one would leave the other on
Renewables.ninja's levels, and the clustering is the one that would carry the error
furthest: it would pick its still days and its cloudy days from a wind year that is wrong
by a factor of five in the mountains.

Reads:
    mappings/vre_profiles.csv                          zone x technology -> DeCA rows
    extracted/wind_atlas.csv                           the terrain, where 2020 guessed
    mappings/seasons_months.csv                        the season set and its months
    <reference>/pHours.csv                             the 2020 block widths
    <reference>/supply/pVREProfile.csv                 the 2020 level, where DeCA is silent
    the five DeCA AssumptionBooks, sheet RenewableProfile
    <staging>/PV_casa_rninja.csv, WT_casa_rninja.csv   the fetched year, at its own level

Writes:
    <staging>/PV_casa.csv, WT_casa.csv                 the same year, at the stated level
    extracted/vre_hourly_report.csv                    what was done to each zone

THE NAMES CARRY THE LEVEL, which is what makes this safe to re-run: the fetch driver
writes the _rninja files and never the plain ones, this writes the plain ones and never
the _rninja ones. Running it twice does the same thing as running it once.

RUN AFTER pipelines/run_casa_vre_fetch.py, and BEFORE build_vre.py.
"""
import argparse
import csv
import io
import os

import build_vre as vre

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STAGING = os.path.join(REPO, "pre-analysis", "representative_days", "input")

HOURS_PER_DAY = 24

# The most a measured series is allowed to be stretched before we stop calling the result
# a measurement. Beyond it the two sources are not disagreeing about a loss, they are
# disagreeing about whether the resource is there: Renewables.ninja puts southern
# Tajikistan at a 1.6% wind capacity factor against the model's 35%, and a series
# multiplied twentyfold is an invention wearing an hourly shape. Those zones keep the
# construction build_vre.py already had, and the report says which ones they are.
MAX_STRETCH = 2.0

# Where the atlas levels are read from, and at which turbine class. Class 3 is the
# low-wind machine -- the large rotor per kilowatt these sites are actually built with,
# and the class that reads highest. Choosing 1 or 2 instead moves a zone's percentile by
# 0.05-0.09, an order of magnitude below the disagreement the atlas was brought in to
# settle, so the choice is stated here rather than argued per zone.
ATLAS = vre.ATLAS
ATLAS_CLASS = vre.ATLAS_CLASS


def month_of_season(quarters):
    """month number -> season name, from the one file that cuts the year."""
    out = {}
    for season, months in quarters.items():
        for month in months:
            out[month] = season
    return out


def stated_levels(reference, plan, quarters, inherits, atlas):
    """(zone, tech) -> (12 monthly capacity factors, where the level came from).

    DeCA states a factor per calendar month, which is the finest level it has and the one
    used directly. Where there is no DeCA book, which is the Afghan and Pakistani zones,
    the level falls back to the 2020 profile's own seasonal mean, spread over the months
    of that season. That is the order of preference the rest of the build follows: DeCA
    where it speaks, 2020 where it does not.

    AND THEN THERE IS THE ATLAS, which outranks the 2020 fallback and nothing else. A
    zone whose mapping row carries level_source = atlas_p90 keeps the shape it was just
    given and has its height set by the Global Wind Atlas instead. It is turned on for
    the eight book-less wind zones and no others, because 2020 gives all eight the same
    0.333 -- one placeholder repeated, which the atlas contradicts in both directions:
    Herat's Wind of 120 Days is understated by two thirds, and the Tajik lowland of
    northern Afghanistan is overstated by half the other way. It is deliberately NOT
    turned on where DeCA has spoken. DeCA is the client's own consultant reading metered
    plant, the atlas is a model of the terrain, and where the two are close -- and they
    are, within a few points, in six of the seven -- there is no case for overruling the
    measurement with the model.
    """
    shape = vre.legacy_profile(reference)
    belongs = month_of_season(quarters)
    books, levels = {}, {}
    for line in plan:
        z, tech = line["z"].strip(), line["tech"].strip()
        code, match = line["workbook"].strip(), line["match"].strip()
        if code:
            if code not in books:
                books[code] = vre.monthly_factors(code)
            picked = vre.selected(books[code], match, line.get("exclude", ""))
            if not picked:
                raise SystemExit(
                    "no DeCA row matches {0} for {1} {2}".format(match, z, tech))
            monthly = [sum(v[m] for _, v in picked) / len(picked) for m in range(12)]
            origin = "deca"
        else:
            monthly = []
            for month in range(1, 13):
                day = shape.get((z, tech, inherits[belongs[month]]))
                if day is None:
                    raise SystemExit("no 2020 shape for {0} {1} in month {2}".format(
                        z, tech, month))
                monthly.append(sum(day) / float(HOURS_PER_DAY))
            origin = "casa_2020"

        override = (line.get("level_source") or "").strip().lower()
        if override:
            by_month, origin = vre.onto_atlas(
                dict(enumerate(monthly)), dict(enumerate(vre.MONTH_LENGTH)),
                atlas, z, override)
            monthly = [by_month[m] for m in range(12)]
        levels[(z, tech)] = (monthly, origin)
    return levels


def read_hourly(path):
    """zone -> [(month, day, hour, value)], in file order."""
    series = {}
    with io.open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("value") or "").strip()
            if not value:
                continue
            series.setdefault(row["zone"].strip(), []).append(
                (int(row["month"]), int(row["day"]), int(row["hour"]), float(value)))
    return series


def write_hourly(path, series, order):
    with io.open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["zone", "month", "day", "hour", "value"])
        for zone in order:
            for month, day, hour, value in series[zone]:
                writer.writerow([zone, month, day, hour, "{0:.4f}".format(value)])


def rescale_zone(rows, monthly):
    """The zone's year at the stated level, or None when a month cannot get there.

    IT IS ALL TWELVE MONTHS OR NONE OF THEM. A month that cannot reach its stated factor
    even at the ceiling leaves the year somewhere between the two sources -- northern
    Kyrgyzstan came out at a 0.213 wind capacity factor against a stated 0.335, which is
    neither what Renewables.ninja measured nor what DeCA claims, and nobody reading it
    later would be able to say which it was meant to be. When every month lands, the
    annual mean is exactly the stated one by construction, and that is the only outcome
    this returns.
    """
    out, scales = [], []
    for month in range(1, 13):
        block = [r for r in rows if r[0] == month]
        if not block:
            continue
        scale = vre.fit_scale([r[3] for r in block], [1.0] * len(block),
                              monthly[month - 1], MAX_STRETCH)
        if scale is None:
            return None, month
        scales.append(scale)
        out += [(m, d, h, min(v * scale, 1.0)) for m, d, h, v in block]
    return out, scales


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference",
                        default=os.path.join("epm", "input", "data_casa_2020"))
    parser.add_argument("--staging", default=STAGING)
    parser.add_argument("--atlas", default=ATLAS,
                        help="the per-zone Global Wind Atlas sample")
    parser.add_argument("--atlas-class", default=ATLAS_CLASS,
                        help="turbine class the atlas levels are read at")
    args = parser.parse_args()
    reference = os.path.join(REPO, args.reference)

    mapping = os.path.join(HERE, "mappings", "seasons_months.csv")
    quarters = vre.seasons(mapping)
    inherits = vre.legacy_of(mapping)
    plan = vre.dicts(os.path.join(HERE, "mappings", "vre_profiles.csv"))
    levels = stated_levels(reference, plan, quarters, inherits,
                           vre.atlas_levels(args.atlas, args.atlas_class))

    report = []
    for tech in sorted({line["tech"].strip() for line in plan}):
        source = os.path.join(args.staging, "{0}_casa_rninja.csv".format(tech))
        if not os.path.exists(source):
            raise SystemExit(
                "no fetched series at {0}. Run pipelines/run_casa_vre_fetch.py first; it "
                "writes the _rninja files this step rescales.".format(source))
        fetched = read_hourly(source)

        rescaled, order = {}, []
        for zone in sorted(fetched):
            if (zone, tech) not in levels:
                print("  skip    {0} {1}: not in vre_profiles.csv".format(zone, tech))
                continue
            monthly, origin = levels[(zone, tech)]
            rows = fetched[zone]
            before = sum(v for _, _, _, v in rows) / len(rows)
            wanted = sum(monthly[m - 1] * vre.MONTH_LENGTH[m - 1]
                         for m in range(1, 13)) / 365.0

            def kept(why):
                report.append([zone, tech, origin, "kept 2020 construction", why,
                               "{0:.4f}".format(before), "{0:.4f}".format(wanted), "", ""])

            # The stretch is judged once a year, on the annual mean, so that a zone cannot
            # change method between January and July and leave a seam in its own series.
            if not before or wanted / before > MAX_STRETCH:
                kept("would need x{0:.1f}".format(wanted / before) if before
                     else "series is flat zero")
                continue

            out, scales = rescale_zone(rows, monthly)
            if out is None:
                kept("month {0} cannot reach its stated factor".format(scales))
                continue

            rescaled[zone] = out
            order.append(zone)
            after = sum(v for _, _, _, v in out) / len(out)
            report.append([zone, tech, origin, "rescaled", "",
                           "{0:.4f}".format(before), "{0:.4f}".format(wanted),
                           "{0:.4f}".format(after),
                           "x{0:.2f}-{1:.2f}".format(min(scales), max(scales))])

        target = os.path.join(args.staging, "{0}_casa.csv".format(tech))
        write_hourly(target, rescaled, order)
        print("{0:<4} {1:>2} zone(s) rescaled -> {2}".format(tech, len(order), target))

    vre.write_csv(os.path.join(HERE, "extracted", "vre_hourly_report.csv"),
                  ["z", "tech", "level_from", "action", "reason", "cf_rninja",
                   "cf_stated", "cf_after", "monthly_scale"], report)

    refused = [r for r in report if r[3].startswith("kept")]
    print("\nlevels     {0} of {1} series brought onto the stated level exactly".format(
        len(report) - len(refused), len(report)))
    if refused:
        print("refused    {0} series are too far from the stated level to be rescaled into\n"
              "           it and stay with build_vre.py's 2020 construction:".format(
                  len(refused)))
        for row in refused:
            print("             {0:<11} {1:<3} {2:.3f} against {3:.3f}, {4}".format(
                row[0], row[1], float(row[5]), float(row[6]), row[4]))


if __name__ == "__main__":
    main()
