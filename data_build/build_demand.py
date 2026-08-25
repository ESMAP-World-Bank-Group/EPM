# -*- coding: utf-8 -*-
"""The time structure of the model, and the demand that lives on it.

    python data_build/build_demand.py

Reads:
    extracted/deca_demand_hourly.csv       8760 h per zone, written by extract_demand.py
    mappings/seasons_months.csv            which months make each season
    mappings/zones_demand.csv              which zones have an hourly source
    <target>/zcmap.csv                     the zones of the model
    <reference>/pHours.csv                 the 2020 time structure, for its block widths
    <reference>/load/pDemandData.csv       the only shape Afghanistan and Pakistan have

Writes:
    extracted/pHours.csv
    extracted/pDemandProfile.csv
    extracted/demand_report.csv

WHY THIS STEP EXISTS. Until it runs the model has no demand at all: pDemandData and
pDemandProfile are both empty, so every zone is dead and nothing can be solved. It is
the step that makes the model runnable, and it decides the time structure that every
other hourly resource has to follow.

THE STRUCTURE. Four seasons, D representative days per season, 24 chronological hours,
which is the canonical EPM shape (epm/input/data_test carries 4 x 5 x 24). pHours(q,d,t)
holds the NUMBER OF DAYS the representative day stands for, repeated on each of its 24
hours, so the whole table sums to 8760 h.

The 2020 model was 5 seasons x 1 day x 7 blocks, and its fifth season was not a season
but a bag of peak hours. Phase 1 removed it. What replaces it here is a PEAK
REPRESENTATIVE DAY inside each season: d1 is the calendar day carrying the highest
system load of that season, weighted one day, and the remaining days of the season are
split by daily energy into D-1 groups, each represented by the mean day of its group.
Energy and peak are therefore both exact by construction, which is what the planning
reserve needs: base.gms takes an smax over (q,d,t) and it now has a true peak to find.

THE PEAK DAY IS CHOSEN ON THE ZONES THAT HAVE AN HOURLY SOURCE, Kazakhstan without the
West, Kyrgyzstan, Tajikistan, Turkmenistan and Uzbekistan. Afghanistan and Pakistan have
no hourly series and so cannot vote on it, which matters because Pakistan is a large
share of the demand of the model and, in the 2020 data, peaks in a different season.
The coincidence between the two blocks is therefore not modelled, it is inherited.

WHAT EACH ZONE READS.
  - the nine DeCA zones read their own 8760 h series, aggregated by extract_demand.py.
    Turkmenistan states one average day per month, 288 points, expanded here to the
    days of each month: the zone is single and only its shape is used.
  - Afghanistan and Pakistan read the 2020 pDemandData. Its blocks are chronological
    hour-of-day groups and not a duration curve, which is verifiable on the 2020
    pVREProfile: solar is 0 in t1 and t6 and highest in t3, so t1 is the night and t3
    is the middle of the day. The block widths are read off the 2020 pHours, 6-5-3-3-4-3
    hours in every season, and each block is held flat over its own hours.
    Those zones have one shape per season and no day-to-day variation. Their peak day
    is lifted by the ratio their own 2020 peak season carried over their seasonal days,
    which is the one piece of peak information the old fifth season held.

NORMALISATION. pDemandProfile is divided by the maximum of the zone over the whole
table, so its maximum is exactly 1. That is what generate_demand.gms expects: with
fUseSimplifiedDemand at 1 it multiplies the profile by the peak of pDemandForecast and
then adds a correction proportional to (max - profile) until the annual energy matches,
a correction that is null at the peak. The peak of the profile IS the peak of the
forecast, and the shape carries the rest.

WHAT THIS DOES NOT DO. It does not touch pDemandData, which stays empty on purpose:
the year by year scaling is the model's own work, not the pipeline's. And it says
nothing about the level of demand, which is pDemandForecast and was settled in phase 2.
"""

import argparse
import collections
import csv
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

MONTH_LENGTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
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


def index(header):
    return {name.strip(): j for j, name in enumerate(header)}


def num(x):
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def dicts(path):
    header, rows = read_csv(path)
    keys = [h.strip() for h in header]
    return [dict(zip(keys, r)) for r in rows]


def calendar():
    """month of each day, and the hours of each day, on a 365 day year."""
    month_of_day, day_of_hour = [], []
    for m, n in enumerate(MONTH_LENGTH, 1):
        for _ in range(n):
            day_of_hour += [len(month_of_day)] * HOURS_PER_DAY
            month_of_day.append(m)
    return month_of_day, day_of_hour


def date_of(day, month_of_day):
    """A day of the year written as month/day."""
    m = month_of_day[day]
    return "{0}/{1}".format(m, day - sum(MONTH_LENGTH[:m - 1]) + 1)


def seasons(path):
    """season -> the months it covers, in the order the file states them."""
    out = collections.OrderedDict()
    for row in dicts(path):
        out[row["season"].strip()] = [int(m) for m in row["months"].split(",")]
    months = sorted(m for ms in out.values() for m in ms)
    if months != list(range(1, 13)):
        raise ValueError("the seasons do not cover the twelve months exactly")
    return out


def legacy_of(path):
    """season -> the season of the 2020 model it inherits its shape from.

    The two sets overlap in name without overlapping in meaning: Q1, Q2 and Q4 stand
    for the same months in both, but the 2020 Q3 was one five-month summer where ours
    is two, Q3a and Q3b, and the 2020 Q5 was a 130 h peak block that has no successor
    here at all. The mapping says which feeds which so that neither set has to match
    the other.
    """
    out = {}
    for row in dicts(path):
        out[row["season"].strip()] = row["legacy"].strip()
    return out


def hourly(path):
    """zone -> [MW], one value per hour of the year.

    A series of 288 points is one average day per month and is expanded over the days
    of each month. Any other length is refused rather than guessed at.
    """
    raw = collections.defaultdict(dict)
    for row in dicts(path):
        raw[row["z"].strip()][int(row["hour"])] = float(row["MW"])
    out = {}
    for z, points in raw.items():
        n = len(points)
        series = [points[h] for h in sorted(points)]
        if n == 8760:
            out[z] = series
        elif n == 12 * HOURS_PER_DAY:
            out[z] = [series[(m - 1) * HOURS_PER_DAY + h]
                      for m, days in enumerate(MONTH_LENGTH, 1)
                      for _ in range(days)
                      for h in range(HOURS_PER_DAY)]
        else:
            raise ValueError("{0} has {1} points, neither 8760 nor a day per month"
                             .format(z, n))
    return out


def block_hours(row):
    """How many hours of the day each block of the 2020 model stands for.

    The row holds the hours a block weighs over a whole season; dividing by the number
    of days of that season gives its width inside the day. The widths are integers by
    construction but the division is not exact for the season the old peak season was
    carved out of, so they are rounded by largest remainder and forced to sum to 24.
    """
    values = [v for v in row if v is not None]
    days = sum(values) / float(HOURS_PER_DAY)
    exact = [v / days for v in values]
    base = [int(v) for v in exact]
    short = HOURS_PER_DAY - sum(base)
    order = sorted(range(len(exact)), key=lambda i: exact[i] - base[i], reverse=True)
    for i in order[:short]:
        base[i] += 1
    if sum(base) != HOURS_PER_DAY or any(b < 0 for b in base):
        raise ValueError("block widths do not make a day: " + repr(base))
    return base


def legacy_shapes(reference, inherits):
    """zone -> {season: [24 MW]} and zone -> peak uplift, out of the 2020 model.

    Each of our seasons is read from the 2020 season it inherits from, named in
    seasons_months.csv; two of ours may draw on the same one, which is what the split
    summer does. Every 2020 season is read for its maximum, including the old peak
    block that is a season for nobody: that maximum is the uplift the peak day of a
    legacy zone is given.
    """
    widths = {}
    header, rows = read_csv(os.path.join(reference, "pHours.csv"))
    for r in rows:
        widths[r[0].strip()] = block_hours([num(c) for c in r[2:]])

    header, rows = read_csv(os.path.join(reference, "load", "pDemandData.csv"))
    hi = index(header)
    blocks = [hi[c] for c in header if c.strip().startswith("t")]
    year = sorted({int(num(r[hi["y"]])) for r in rows})[0]

    days, peaks = collections.defaultdict(dict), collections.defaultdict(float)
    for r in rows:
        if int(num(r[hi["y"]])) != year:
            continue
        z, q = r[hi["z"]].strip(), r[hi["q"]].strip()
        values = [num(r[j]) or 0.0 for j in blocks]
        peaks[z] = max(peaks[z], max(values))
        drawn = [ours for ours, theirs in inherits.items() if theirs == q]
        if not drawn:
            continue
        day = []
        for width, value in zip(widths[q], values):
            day += [value] * width
        for ours in drawn:
            days[z][ours] = day

    uplift = {}
    for z in days:
        own = max(max(day) for day in days[z].values())
        uplift[z] = max(1.0, peaks[z] / own) if own else 1.0
    return days, uplift


def representative(day_load, groups):
    """Split the days of a season: the peak day alone, then the rest by energy.

    day_load holds, per day, the energy of the region and the height of the stress
    signal. The peak day is the one that stresses the most zones at once; it stands
    for itself, one day, and is what gives the reserve constraint a maximum to see.
    The others are ranked by their own energy and cut into groups of as equal a size
    as the count allows.
    """
    peak_day = max(day_load, key=lambda d: day_load[d][1])
    rest = sorted((d for d in day_load if d != peak_day),
                  key=lambda d: day_load[d][0])
    out = [[peak_day]]
    n = len(rest)
    for i in range(groups - 1):
        out.append(rest[i * n // (groups - 1):(i + 1) * n // (groups - 1)])
    return [g for g in out if g]


def medoid(system, group):
    """The day of the group that is closest to the group's own average day.

    A mean day is not a day. Averaging seventy-six days flattens every trough that
    does not fall at the same hour twice, and the model reads that flattening as a
    baseload the system does not have: the load factor of Tajikistan came out 0.15
    above the truth, and the night valley of TAJ_S ended at three per cent of the peak
    once generate_demand.gms had stretched the flattened shape back onto the annual
    energy it is given. The day kept here is a real one, the least unusual member of
    its group, so every hour of it was actually observed together on one date.
    """
    mean = [sum(system[d * HOURS_PER_DAY + h] for d in group) / float(len(group))
            for h in range(HOURS_PER_DAY)]
    return min(group, key=lambda d: sum(
        (system[d * HOURS_PER_DAY + h] - mean[h]) ** 2 for h in range(HOURS_PER_DAY)))


def solve_weights(day_load, groups, chosen):
    """How many days each representative day stands for.

    The size of each group alone would lose the energy of the season, because the
    medoid of a group is its most typical day and not the day carrying its average
    energy. The peak day always counts one. The others start from the size of their
    group and are moved the shortest distance that makes the total energy of the
    season come out right, which for two groups is the exact solution of the two
    equations and for more is the least-squares one. The number of days in the season
    is untouched by construction, the correction summing to zero.
    """
    n = [float(len(g)) for g in groups]
    e = [day_load[d][0] for d in chosen]
    w = [1.0] + n[1:]
    rest = sum(day_load[d][0] for g in groups for d in g) - e[0]
    idx = list(range(1, len(groups)))
    if not idx:
        return w
    bar = sum(e[k] for k in idx) / len(idx)
    denom = sum((e[k] - bar) ** 2 for k in idx)
    if denom <= 0:
        return w
    lam = (rest - sum(n[k] * e[k] for k in idx)) / denom
    trial = [n[k] + lam * (e[k] - bar) for k in idx]
    if min(trial) < 1.0:
        return w        # the correction would empty a day; the sizes stand
    for k, v in zip(idx, trial):
        w[k] = v
    return w


def whole(w):
    """The weights in whole days, their total unchanged and none of them empty.

    input_verification.py compares the sum of pHours to 8760 with no tolerance at all,
    so this column has to hold values that add up exactly, and the same check refuses
    a block of zero hours. Largest remainder, then a floor of one day.
    """
    base = [int(x) for x in w]
    short = int(round(sum(w))) - sum(base)
    order = sorted(range(len(w)), key=lambda i: w[i] - base[i], reverse=True)
    for i in order[:short]:
        base[i] += 1
    while min(base) < 1:
        base[base.index(min(base))] += 1
        base[base.index(max(base))] -= 1
    return base


def duration(blocks, steps=20):
    """A load duration curve, read at every 1/steps of the year.

    blocks are (value, hours) pairs, which is what a representative day is and what a
    calendar year is too, so both curves are built by this one function and compared
    point by point.
    """
    blocks = sorted(blocks, key=lambda x: -x[0])
    marks = [sum(h for _, h in blocks) * i / float(steps) for i in range(1, steps)]
    out, seen, i = [], 0.0, 0
    for v, hrs in blocks:
        seen += hrs
        while i < len(marks) and seen >= marks[i]:
            out.append(v)
            i += 1
    while i < len(marks):
        out.append(blocks[-1][0])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Build the time structure and the demand "
                                             "profile.")
    ap.add_argument("--reference", default=os.path.join("epm", "input", "data_casa_2020"))
    ap.add_argument("--target", default=os.path.join("epm", "input", "data_casa"))
    ap.add_argument("--days", type=int, default=3,
                    help="representative days per season, the first being the peak day")
    args = ap.parse_args()

    if args.days < 2:
        raise SystemExit("at least two days per season: the peak day and one other")
    reference = os.path.join(REPO, args.reference)
    target = os.path.join(REPO, args.target)

    quarters = seasons(os.path.join(HERE, "mappings", "seasons_months.csv"))
    inherits = legacy_of(os.path.join(HERE, "mappings", "seasons_months.csv"))
    month_of_day, day_of_hour = calendar()
    series = hourly(os.path.join(HERE, "extracted", "deca_demand_hourly.csv"))

    zmap = {row["z"].strip(): row for row in
            dicts(os.path.join(HERE, "mappings", "zones_demand.csv"))}
    _, zrows = read_csv(os.path.join(target, "zcmap.csv"))
    zones = [r[0].strip() for r in zrows]

    sourced = [z for z in zones
               if zmap.get(z, {}).get("method", "").strip() != "LEGACY" and z in series]
    legacy_days, uplift = legacy_shapes(reference, inherits)
    legacy = [z for z in zones if z not in sourced]
    unknown = [z for z in legacy if z not in legacy_days]
    if unknown:
        raise SystemExit("no source and no 2020 shape for " + ", ".join(unknown))

    # ---- the representative days, chosen on the zones that have an hourly source
    system = [sum(series[z][h] for z in sourced) for h in range(8760)]
    # Which day of a season is its peak day is not the same question as how much power
    # the region draws that day. Summed in MW the answer is decided by Kazakhstan and
    # Uzbekistan alone, and the peak day of every season then falls on a day when
    # southern Tajikistan is at 78 % of its own maximum and Turkmenistan at 91 %; the
    # planning reserve of EPM is written per zone, so those two would be sized against
    # a peak the model never sees. Each zone divided by its own maximum first gives
    # every zone one vote, and the day chosen is the one that stresses the most of
    # them at once. Energy is still counted in MW, below: that question is about size.
    zone_peak = dict((z, max(series[z])) for z in sourced)
    stress = [sum(series[z][h] / zone_peak[z] for z in sourced) for h in range(8760)]
    hours_rows, profile = [], collections.defaultdict(dict)
    report = []

    for q, months in quarters.items():
        days_of_q = [d for d in range(365) if month_of_day[d] in months]
        load = {}
        for d in days_of_q:
            lo, hi = d * HOURS_PER_DAY, (d + 1) * HOURS_PER_DAY
            load[d] = (sum(system[lo:hi]), max(stress[lo:hi]))
        groups = representative(load, args.days)
        chosen = [groups[0][0]] + [medoid(system, g) for g in groups[1:]]
        w = whole(solve_weights(load, groups, chosen))

        for k, (group, day, days) in enumerate(zip(groups, chosen, w), 1):
            d = "d{0}".format(k)
            hours_rows.append([q, d] + [days] * HOURS_PER_DAY)
            for z in sourced:
                profile[z][(q, d)] = list(
                    series[z][day * HOURS_PER_DAY:(day + 1) * HOURS_PER_DAY])
            for z in legacy:
                shape = legacy_days[z][q]
                factor = uplift[z] if k == 1 else 1.0
                profile[z][(q, d)] = [v * factor for v in shape]
            report.append([q, d, days, date_of(day, month_of_day), len(group),
                           "peak day" if k == 1 else "median day of the group"])

    total = sum(sum(r[2:]) for r in hours_rows)
    if total != 365 * HOURS_PER_DAY:
        raise ValueError("the time structure covers {0} h, not 8760".format(total))

    # ---- each zone reaches its own maximum on the peak day
    # One day per season is shared by every zone, so the day that stresses the region
    # is not the day on which each zone separately reaches its own annual maximum:
    # northern Tajikistan only got to 84 % of its own. Left there, the profile of that
    # zone carries a load factor of 0.70 where its own series shows 0.60, and
    # generate_demand.gms, which is given the peak and the energy of the zone and has
    # to make the profile fit both, takes the excess out of the troughs and drives the
    # winter night of TAJ_N below zero. The peak days are therefore lifted, one factor
    # per zone, until the highest hour of the year in the profile is the highest hour
    # of the year in the series. It says that a zone reaches its own maximum when the
    # region is under stress, which is an assumption about coincidence and is stated
    # in the report as peak_uplift; it is also exactly the convention the 2020 model
    # used for Afghanistan and Pakistan, applied here to measured data rather than to
    # an inherited one.
    firsts = [(q, "d1") for q in quarters]
    for z in sourced:
        crest = max(max(profile[z][k]) for k in firsts)
        lift = zone_peak[z] / crest if crest else 1.0
        for k in firsts:
            profile[z][k] = [v * lift for v in profile[z][k]]
        uplift[z] = lift

    # ---- normalisation, one zone at a time
    keys = [(q, "d{0}".format(k)) for q in quarters
            for k in range(1, args.days + 1)]
    keys = [k for k in keys if k in profile[zones[0]]]
    weight = dict(((r[0], r[1]), r[2]) for r in hours_rows)
    rows, zone_report = [], []
    for z in zones:
        top = max(max(profile[z][k]) for k in keys)
        if not top:
            raise ValueError("zone " + z + " has a flat zero profile")
        energy = sum(sum(profile[z][k]) * weight[k] for k in keys)
        for q, d in keys:
            rows.append([z, q, d] + ["{0:.6g}".format(v / top) for v in profile[z][(q, d)]])

        # THE RECONSTRUCTION TEST. Twelve days are asked to stand for three hundred
        # and sixty five, and nothing in the model will ever say how well they do it.
        # So it is measured here, against the series the days were drawn from, and
        # written beside them: the load factor the twelve days produce against the one
        # the year really has, the annual energy they carry against the real one, and
        # the largest gap between the two load duration curves read at every five per
        # cent of the year, as a percentage of the true peak. A zone with no hourly
        # series has nothing to be tested against and its columns stay empty.
        if z in sourced:
            truth = series[z]
            top_true = max(truth)
            rec = duration([(v, weight[k]) for k in keys for v in profile[z][k]])
            tru = duration([(v, 1.0) for v in truth])
            test = ["{0:.3f}".format(sum(truth) / (top_true * 8760.0)),
                    "{0:+.2f}".format((energy / sum(truth) - 1.0) * 100.0),
                    "{0:.2f}".format(max(abs(a - b) for a, b in zip(rec, tru))
                                     / top_true * 100.0)]
        else:
            test = ["", "", ""]

        zone_report.append([z, "hourly" if z in sourced else "2020 shape",
                            "{0:.0f}".format(top),
                            "{0:.3f}".format(energy / (top * 8760.0))] + test +
                           ["{0:.3f}".format(uplift.get(z, 1.0))])

    out = os.path.join(HERE, "extracted")
    write_csv(os.path.join(out, "pHours.csv"),
              ["q", "d"] + ["t{0}".format(h) for h in range(1, HOURS_PER_DAY + 1)],
              hours_rows)
    write_csv(os.path.join(out, "pDemandProfile.csv"),
              ["z", "q", "d"] + ["t{0}".format(h) for h in range(1, HOURS_PER_DAY + 1)],
              rows)
    write_csv(os.path.join(out, "demand_report.csv"),
              ["q", "d", "days_stood_for", "date", "days_in_group", "kind"], report)
    write_csv(os.path.join(out, "demand_zone_report.csv"),
              ["z", "source", "profile_peak_mw", "load_factor", "load_factor_8760",
               "energy_error_pct", "duration_curve_error_pct", "peak_uplift"],
              zone_report)

    print("structure  {0} seasons x {1} days x {2} h = {3} blocks, {4} h covered"
          .format(len(quarters), args.days, HOURS_PER_DAY,
                  len(hours_rows) * HOURS_PER_DAY, total))
    print("zones      {0}, of which {1} on their own hourly series and {2} on the "
          "2020 shape".format(len(zones), len(sourced), len(legacy)))
    for r in report:
        print("{0} {1}     {2:>3} days  {3:>6}  {4}".format(r[0], r[1], r[2], r[3], r[5]))
    tested = [r for r in zone_report if r[4]]
    if tested:
        worst = max(tested, key=lambda r: float(r[6]))
        print("test       {0} zones checked against their own 8760 h; worst duration "
              "curve error {1} % of peak, on {2}".format(len(tested), worst[6], worst[0]))
        thick = max(tested, key=lambda r: float(r[7]))
        print("           the peak days are lifted at most {0} times, on {1}, to reach "
              "the annual maximum of that zone".format(thick[7], thick[0]))


if __name__ == "__main__":
    main()
