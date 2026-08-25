# -*- coding: utf-8 -*-
"""What the 360 blocks kept of the year they were built from.

The demand side of this model does not rest on an assumption: it rests on a measured
year. The DeCA assumption books carry an hourly metered series per country -- 2019 for
Kazakhstan, 2021 for Tajikistan, 2022 for Kyrgyzstan and Uzbekistan -- and phase 8
reduced it to 5 seasons x 3 representative days x 24 hours. That reduction is the
single largest simplification in the model, and it is checkable, because both ends of
it are on disk: the raw series in extracted/deca_demand_hourly.csv and the result in
the deployed pDemandProfile.csv.

WHAT THIS MODULE CHECKS is therefore not whether the model is right about the future.
It is whether the 360 blocks still describe the year they came from: does the load
factor survive, does the trough survive, do the peak hours survive, does the energy
land in the right season. A reduction that flattens the year does not announce itself
anywhere in the output -- it shows up as a baseload the system does not have, and as a
peaking need the model never has to meet.

WHAT IT CANNOT CHECK is generation. Calibrating a dispatch means comparing modelled
output by fuel against metered output by fuel, and no such series exists for any of the
seven countries in this study. That is a data request and not a gap this file can close;
until it is answered, "calibration" here means the demand side alone and says so.

BOTH SIDES ARE READ IN PER UNIT OF THEIR OWN PEAK, because pDemandProfile is normalised
to a maximum of exactly 1 by construction and the measured series is in megawatts. The
comparison is therefore about shape, which is all the profile table carries; the level
is the business of pDemandForecast.

Reads
    extracted/deca_demand_hourly.csv        the metered series, MW by (zone, hour)
    mappings/seasons_months.csv             which months make each season
    <deployment>/pHours.csv                 the hours each block stands for
    <deployment>/load/pDemandProfile.csv    the reduction being checked
"""

import io
import csv
import os
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
HOURS_PER_DAY = 24
MONTH_LENGTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# A zone needs a full metered year to be compared at all. Turkmenistan states one
# average day per month, 288 points, which is a shape and not a year: it cannot answer
# what the trough of a real February looked like, so it is reported as uncovered
# rather than compared against a series it does not have.
FULL_YEAR = sum(MONTH_LENGTH) * HOURS_PER_DAY


def read_csv(path):
    with io.open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    return rows[0], [r for r in rows[1:] if r and any(c.strip() for c in r)]


def num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def month_of_hour():
    """The month, 1 to 12, of each of the 8760 hours of a non-leap year."""
    out = []
    for m, n in enumerate(MONTH_LENGTH, 1):
        out += [m] * n * HOURS_PER_DAY
    return out


def season_of_month(path):
    """month -> season, from the same mapping the build used to cut the year."""
    _h, rows = read_csv(path)
    out = {}
    for r in rows:
        for m in r[1].split(","):
            out[int(m.strip())] = r[0].strip()
    return out


def measured(path):
    """The metered series per zone, in megawatts, in calendar order."""
    _h, rows = read_csv(path)
    out = defaultdict(list)
    for r in rows:
        v = num(r[2])
        if v is not None:
            out[r[0].strip()].append(v)
    return out


def modelled(hours_csv, profile_csv):
    """Per zone, the (per-unit value, hours it stands for, season) of every block.

    This is the year the model actually sees. A block is not one hour: it is the value
    of one hour of one representative day, standing for as many days as pHours gives
    that day, so the weight is what makes the reduction comparable to 8760 hours.
    """
    hh, hrows = read_csv(hours_csv)
    weight = {}
    for r in hrows:
        q, d = r[0].strip(), r[1].strip()
        for t, cell in enumerate(r[2:2 + HOURS_PER_DAY]):
            weight[(q, d, t)] = num(cell) or 0.0

    ph, prows = read_csv(profile_csv)
    at = len(ph) - HOURS_PER_DAY
    out = defaultdict(list)
    for r in prows:
        z, q, d = r[0].strip(), r[1].strip(), r[2].strip()
        for t, cell in enumerate(r[at:at + HOURS_PER_DAY]):
            v = num(cell)
            if v is not None:
                out[z].append((v, weight.get((q, d, t), 0.0), q))
    return out


def duration_curve(pairs, points=80):
    """A load duration curve sampled at `points`, from (value, hours) pairs.

    Sorted high to low and walked by cumulative hours, so the model curve and the
    measured curve are read on the same horizontal axis even though one is 360 blocks
    and the other is 8760 hours.
    """
    ordered = sorted(pairs, key=lambda p: -p[0])
    total = sum(w for _v, w in ordered) or 1.0
    out, seen, i = [], 0.0, 0
    for k in range(points):
        want = total * (k + 0.5) / points
        while i < len(ordered) - 1 and seen + ordered[i][1] < want:
            seen += ordered[i][1]
            i += 1
        out.append(ordered[i][0])
    return out


def quantile(pairs, q):
    """The value at quantile q of a weighted series, walked by cumulative hours."""
    ordered = sorted(pairs)
    total = sum(w for _v, w in ordered) or 1.0
    seen = 0.0
    for v, w in ordered:
        seen += w
        if seen >= total * q:
            return v
    return ordered[-1][0]


def stats(pairs):
    """Load factor, trough and peak-hour count of a weighted series, in per unit.

    THE TROUGH IS THE FIRST PERCENTILE AND NOT THE MINIMUM, which matters more than it
    looks. The metered series carry isolated hours at nothing at all -- and one at less
    than nothing, southern Kyrgyzstan reading -4.5 per cent of its own peak -- which are
    metering accidents and not nights. Compared on the minimum, the reduction appears to
    raise the floor of Uzbekistan by 43 points and of Kyrgyzstan by 27; compared on the
    first percentile, which is 88 hours and cannot be one bad reading, it tracks the real
    year to within two points in six zones out of seven. The minimum is kept beside it,
    with the count of hours under a tenth of peak, so the accidents stay visible as
    accidents rather than being averaged into a finding.
    """
    total = sum(w for _v, w in pairs) or 1.0
    peak = max(v for v, _w in pairs) or 1.0
    pu = [(v / peak, w) for v, w in pairs]
    return dict(
        load_factor=sum(v * w for v, w in pu) / total,
        trough=quantile(pu, 0.01),
        minimum=min(v for v, _w in pu),
        low_hours=sum(w for v, w in pu if v < 0.10),
        peak_hours=sum(w for v, w in pu if v >= 0.95),
        hours=total)


def compare(data_dir):
    """Every zone with a metered year, the model beside it, and what differs."""
    met = measured(os.path.join(HERE, "extracted", "deca_demand_hourly.csv"))
    mod = modelled(os.path.join(data_dir, "pHours.csv"),
                   os.path.join(data_dir, "load", "pDemandProfile.csv"))
    smap = season_of_month(os.path.join(HERE, "mappings", "seasons_months.csv"))
    mo = month_of_hour()

    rows, uncovered = [], []
    for z in sorted(mod):
        series = met.get(z)
        if not series or len(series) < FULL_YEAR:
            uncovered.append(z)
            continue
        mpairs = [(v, 1.0) for v in series[:FULL_YEAR]]
        dpairs = [(v, w) for v, w, _q in mod[z]]
        mstat, dstat = stats(mpairs), stats(dpairs)

        # Seasonal energy: the measured year cut by the same months the build used,
        # the model by the hours pHours gives each block. Shares, not megawatts, so
        # the two are on the same footing.
        mseason, dseason = defaultdict(float), defaultdict(float)
        for h, v in enumerate(series[:FULL_YEAR]):
            mseason[smap[mo[h]]] += v
        for v, w, q in mod[z]:
            dseason[q] += v * w
        mtot = sum(mseason.values()) or 1.0
        dtot = sum(dseason.values()) or 1.0
        seasons = OrderedDict()
        for q in sorted(set(list(mseason) + list(dseason))):
            seasons[q] = (mseason[q] / mtot, dseason[q] / dtot)

        rows.append(dict(
            zone=z, measured=mstat, model=dstat, seasons=seasons,
            peak_mw=max(series[:FULL_YEAR]),
            lf_error=dstat["load_factor"] - mstat["load_factor"],
            trough_error=dstat["trough"] - mstat["trough"],
            season_error=max(abs(a - b) for a, b in seasons.values()),
            ldc_measured=duration_curve(mpairs),
            ldc_model=duration_curve(dpairs)))
    return rows, uncovered
