# -*- coding: utf-8 -*-
"""How much of the measured year the 360 blocks still carry.

    python data_build/review_metrics.py

The model does not run on 8,760 hours. It runs on 5 seasons x 3 representative days x
24 chronological hours, each day weighted by the number of real days it stands for.
That reduction is the largest single simplification in the model and it is checkable,
because both ends of it are on disk: the hourly year on one side, the deployed block
tables on the other. This module replays the blocks over a full year and measures what
survived, separating two questions that a single number always confuses.

    VALUE  -- how much, and how often? Energy, the duration curve, the tails. Blind to
              the hour at which anything happens: a profile shifted six hours whole
              moves none of these.
    PROFILE -- when? The mean 24 h shape and the hour of the peaks and troughs.

A reduction can pass one and fail the other, and for wind it does exactly that, so the
two are reported apart and never averaged together.

WHAT IS COMPARED WITH WHAT. Demand is read against the DeCA metered series, in per unit
of each zone's own peak, because pDemandProfile is normalised to a maximum of one by
construction and the measured series is in megawatts: the comparison is about shape, the
level being the business of pDemandForecast. Solar and wind are read against the rescaled
Renewables.ninja year -- the hourly file the block table was cut from, already brought
onto the DeCA level -- so both sides are capacity factors and the energy error is a real
energy error. That also means the VRE half measures the reduction ALONE and not the
fetch: whether Renewables.ninja is right about Central Asia is a different question, and
extracted/vre_hourly_report.csv is where it is answered.

WHAT IT CANNOT MEASURE is a zone with no hourly reference. Turkmenistan states one
average day per month for its demand, 288 points, which is a shape and not a year, and
the seven wind zones whose fetched series was refused carry the 2020 construction
instead. Neither can be compared against a year it does not have, and both are reported
as uncovered rather than scored against a stand-in.

Reads
    extracted/deca_demand_hourly.csv                  the metered demand, MW by (zone, hour)
    <hourly>/PV_casa.csv, WT_casa.csv                 the rescaled hourly VRE year
    <deployment>/pHours.csv                           what each block stands for
    <deployment>/load/pDemandProfile.csv              the demand reduction being checked
    <deployment>/supply/pVREProfile.csv               the VRE reduction being checked

Writes
    extracted/review_metrics.json                     everything below, for the review page
"""
import argparse
import csv
import io
import json
import os
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HOURS_PER_DAY = 24
HOURS_PER_YEAR = 8760
# Where the duration curve is sampled. Ten deciles plus the two tails, because the
# tails are the whole reason a duration curve is looked at rather than a mean.
QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


def read_csv(path):
    with io.open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    return rows[0], [r for r in rows[1:] if r and any(c.strip() for c in r)]


def num(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------- the metrics
def weighted_mean(pairs):
    total = sum(w for _v, w in pairs) or 1.0
    return sum(v * w for v, w in pairs) / total


def duration(pairs, points=QUANTILES):
    """The value exceeded p of the time, for each p, on a weighted series.

    Weighted because a representative day standing for fifty-three days occupies
    fifty-three days of the curve. Sorting descending and walking the cumulative
    weight is the same curve a full year would draw, at a coarser resolution.
    """
    ordered = sorted(pairs, key=lambda vw: -vw[0])
    total = sum(w for _v, w in ordered) or 1.0
    out, seen, i = [], 0.0, 0
    for p in points:
        target = p * total
        while i < len(ordered) - 1 and seen + ordered[i][1] < target:
            seen += ordered[i][1]
            i += 1
        out.append(ordered[i][0])
    return out


def nrmse(a, b, span):
    """Root mean square difference between two curves, over the span of the first.

    Normalised by the span rather than by the mean, so a series that lives near zero
    -- solar over a whole year does -- is not handed an enormous relative error for a
    difference that is small in absolute terms.
    """
    if not span:
        return None
    n = float(len(a))
    return (sum((x - y) ** 2 for x, y in zip(a, b)) / n) ** 0.5 / span


def diurnal(pairs_by_hour):
    """The mean 24 h shape: hour of day -> weighted mean value."""
    return [weighted_mean(pairs_by_hour[h]) if pairs_by_hour[h] else 0.0
            for h in range(HOURS_PER_DAY)]


def correlation(a, b):
    """Pearson between two 24 h shapes. None when either is flat, not zero.

    A flat series has no shape to be right or wrong about, and reporting 0.0 for it
    would read as a failure where the honest answer is that the question does not
    apply. The 2020 fallback days are flat by construction.
    """
    n = float(len(a))
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5


def score(measured, model_pairs, model_by_hour):
    """One series against its reduction: value on one side, profile on the other."""
    meas_pairs = [(v, 1.0) for v in measured]
    meas_by_hour = defaultdict(list)
    for h, v in enumerate(measured):
        meas_by_hour[h % HOURS_PER_DAY].append((v, 1.0))

    m_mean, d_mean = weighted_mean(meas_pairs), weighted_mean(model_pairs)
    m_ldc, d_ldc = duration(meas_pairs), duration(model_pairs)
    span = max(measured) - min(measured)
    m_day, d_day = diurnal(meas_by_hour), diurnal(model_by_hour)

    return OrderedDict([
        ("measured_mean", m_mean),
        ("model_mean", d_mean),
        ("energy_error", (d_mean / m_mean - 1.0) if m_mean else None),
        ("measured_peak", max(measured)),
        ("model_peak", max(v for v, _w in model_pairs)),
        ("peak_error", (max(v for v, _w in model_pairs) / max(measured) - 1.0)
         if max(measured) else None),
        ("ldc_nrmse", nrmse(m_ldc, d_ldc, span)),
        ("p05", (m_ldc[QUANTILES.index(0.05)], d_ldc[QUANTILES.index(0.05)])),
        ("p50", (m_ldc[QUANTILES.index(0.50)], d_ldc[QUANTILES.index(0.50)])),
        ("p95", (m_ldc[QUANTILES.index(0.95)], d_ldc[QUANTILES.index(0.95)])),
        ("diurnal_corr", correlation(m_day, d_day)),
        ("measured_ldc", m_ldc),
        ("model_ldc", d_ldc),
        ("measured_day", m_day),
        ("model_day", d_day),
    ])


# --------------------------------------------------------------------- the sources
def block_weights(path):
    """(season, day) -> how many real days that block stands for.

    pHours states the weight once per hour, all twenty-four equal, so any one of them
    is the day's weight. A block whose hours disagree is a malformed file rather than
    a case to handle.
    """
    _head, rows = read_csv(path)
    out = OrderedDict()
    for r in rows:
        hours = [num(c) for c in r[2:2 + HOURS_PER_DAY]]
        if len(set(hours)) != 1:
            raise ValueError("block {0}/{1} has unequal hours".format(r[0], r[1]))
        out[(r[0].strip(), r[1].strip())] = hours[0]
    return out


def blocks(path, key_columns, weights):
    """A block table replayed as a weighted year, per key.

    Returns key -> (pairs, by_hour) where pairs is every hour of every block with the
    day's weight on it, and by_hour groups the same values by hour of day so the mean
    diurnal shape can be taken without replaying anything twice.
    """
    _head, rows = read_csv(path)
    pairs = defaultdict(list)
    by_hour = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = tuple(c.strip() for c in r[:key_columns])
        season, day = r[key_columns - 2].strip(), r[key_columns - 1].strip()
        w = weights.get((season, day))
        if w is None:
            continue
        for h, cell in enumerate(r[key_columns:key_columns + HOURS_PER_DAY]):
            v = num(cell)
            if v is None:
                continue
            pairs[key[:key_columns - 2]].append((v, w))
            by_hour[key[:key_columns - 2]][h].append((v, w))
    return pairs, by_hour


def measured_demand(path):
    """zone -> the metered year in megawatts, in hour order."""
    _head, rows = read_csv(path)
    series = defaultdict(dict)
    for z, hour, mw in ((r[0].strip(), num(r[1]), num(r[2])) for r in rows):
        if hour is not None and mw is not None:
            series[z][int(hour)] = mw
    return dict((z, [v[h] for h in sorted(v)]) for z, v in series.items())


def measured_vre(path):
    """zone -> the rescaled hourly year, in calendar order."""
    _head, rows = read_csv(path)
    series = defaultdict(list)
    for r in rows:
        v = num(r[4])
        if v is not None:
            series[r[0].strip()].append(v)
    return dict(series)


# --------------------------------------------------------------------- the report
def demand_metrics(data_dir, weights):
    measured = measured_demand(os.path.join(HERE, "extracted", "deca_demand_hourly.csv"))
    pairs, by_hour = blocks(os.path.join(data_dir, "load", "pDemandProfile.csv"),
                            3, weights)
    out, uncovered = OrderedDict(), []
    for key in sorted(pairs):
        z = key[0]
        series = measured.get(z)
        if not series or len(series) < HOURS_PER_YEAR:
            uncovered.append(z)
            continue
        # Both sides in per unit of their own peak. pDemandProfile already is; the
        # metered series is in megawatts and has to be put on the same footing before
        # a difference between them means anything.
        peak = max(series[:HOURS_PER_YEAR]) or 1.0
        entry = score([v / peak for v in series[:HOURS_PER_YEAR]],
                      pairs[key], by_hour[key])
        entry["peak_mw"] = peak
        out[z] = entry
    return out, sorted(set(uncovered))


def vre_metrics(data_dir, hourly_dir, weights):
    pairs, by_hour = blocks(os.path.join(data_dir, "supply", "pVREProfile.csv"),
                            4, weights)
    files = {"PV": "PV_casa.csv", "WT": "WT_casa.csv"}
    out, uncovered = OrderedDict(), []
    for tech, name in sorted(files.items()):
        measured = measured_vre(os.path.join(hourly_dir, name))
        out[tech] = OrderedDict()
        for key in sorted(k for k in pairs if k[1] == tech):
            z = key[0]
            series = measured.get(z)
            if not series or len(series) < HOURS_PER_YEAR:
                uncovered.append("{0} {1}".format(z, tech))
                continue
            out[tech][z] = score(series[:HOURS_PER_YEAR], pairs[key], by_hour[key])
    return out, sorted(set(uncovered))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "epm", "input", "data_casa"))
    ap.add_argument("--hourly-dir", default=os.path.join(
        ROOT, "pre-analysis", "representative_days", "input"))
    ap.add_argument("--out", default=os.path.join(HERE, "extracted", "review_metrics.json"))
    args = ap.parse_args()

    weights = block_weights(os.path.join(args.data_dir, "pHours.csv"))
    demand, demand_gap = demand_metrics(args.data_dir, weights)
    vre, vre_gap = vre_metrics(args.data_dir, args.hourly_dir, weights)

    payload = OrderedDict([
        ("blocks", OrderedDict(("{0}/{1}".format(*k), v) for k, v in weights.items())),
        ("demand", demand),
        ("demand_uncovered", demand_gap),
        ("vre", vre),
        ("vre_uncovered", vre_gap),
    ])
    with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(payload, indent=1))

    def summarise(label, entries):
        if not entries:
            print("{0:<14} none".format(label))
            return
        e = [abs(v["energy_error"]) for v in entries.values() if v["energy_error"] is not None]
        n = [v["ldc_nrmse"] for v in entries.values() if v["ldc_nrmse"] is not None]
        c = [v["diurnal_corr"] for v in entries.values() if v["diurnal_corr"] is not None]
        med = lambda s: sorted(s)[len(s) // 2] if s else float("nan")
        print("{0:<14} {1:>3} zones   energy {2:6.2%} (max {3:6.2%})   "
              "LDC NRMSE {4:6.2%}   diurnal corr {5:5.3f} (min {6:5.3f})".format(
                  label, len(entries), med(e), max(e) if e else 0,
                  med(n), med(c), min(c) if c else float("nan")))

    summarise("demand", demand)
    for tech in vre:
        summarise(tech, vre[tech])
    print("uncovered      demand {0} | vre {1}".format(
        ", ".join(demand_gap) or "none", ", ".join(vre_gap) or "none"))
    print("written        {0}".format(args.out))


if __name__ == "__main__":
    main()
