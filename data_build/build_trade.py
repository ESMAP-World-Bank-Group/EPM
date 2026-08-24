# -*- coding: utf-8 -*-
"""Expand the corridor and contract tables into the EPM trade resources.

    python data_build/build_trade.py

Reads:
    mappings/seasons_months.csv   the season set, and the months behind each season
    mappings/corridors.csv        one line per directed corridor, hand-edited
    mappings/contracts.csv        one line per contracted corridor, hand-edited
    <target>/y.csv                the model years
    <reference>/trade/pContractedTradeEnergy.csv   the contracted volumes

Writes:
    extracted/pTransferLimit.csv         taken as is by the build
    extracted/pContractedTradeFlag.csv   same
    extracted/pContractedTradeEnergy.csv same, re-based on the model years

The mapping holds anchor years as columns. A value holds from its anchor until the
next one, so a corridor that comes into service in 2030 is written as 1e-09 at the
2026 anchor and its rating at the 2030 anchor. Nothing is interpolated: an
interconnection is commissioned, it does not fade in.

SEASONS ARE NEVER NAMED IN THE HAND-EDITED FILES. A corridor carries one 'all' line
applied to every season, and optionally an override written as a WINDOW OF MONTHS;
a contract carries its window of months too. Both are turned into season names here,
against whatever seasons_months.csv declares. That is what lets the season set be
re-cut -- the May-September export window of CASA-1000 sitting in one season or in
two -- without a single line of the mappings moving. In exchange a window has to be
a whole number of seasons, checked below: half a season would leave it undefined
whether the rating applies to all of it or none.

The 1e-09 convention comes from the 2020 model: the GAMS reader turns a plain 0 into
a missing record, so a closed corridor is written as a value small enough to be zero
and large enough to keep the record.
"""

import argparse
import csv
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def read_csv(path):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        rd = csv.reader(fh)
        return next(rd), [r for r in rd if any(c.strip() for c in r)]


def write_csv(path, table):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        csv.writer(fh, lineterminator="\n").writerows(table)


def months_of(text):
    return [int(m) for m in str(text).split(",") if m.strip()]


def seasons(path):
    """[(season, months, days)] in file order, checked to cover the year exactly."""
    header, rows = read_csv(path)
    i = {name.strip(): j for j, name in enumerate(header)}
    out = []
    for r in rows:
        ms = months_of(r[i["months"]])
        out.append((r[i["season"]].strip(), ms, sum(MONTH_DAYS[m - 1] for m in ms)))
    covered = sorted(m for _, ms, _ in out for m in ms)
    if covered != list(range(1, 13)):
        raise ValueError("the seasons do not cover the twelve months exactly")
    return out


def window(months, season_set, what):
    """The seasons a window of months covers, in season order.

    Raised rather than guessed: a window that takes part of a season is a mismatch
    between a contract and the way the year was cut, and the model would carry it
    silently -- a rating applied to months it was never written for, or a take-or-pay
    volume split on a boundary nobody chose.
    """
    want = set(months)
    hit, partial = [], []
    for name, ms, _days in season_set:
        inside = want.intersection(ms)
        if not inside:
            continue
        if len(inside) == len(ms):
            hit.append(name)
        else:
            partial.append(name)
    if partial:
        raise ValueError("{}: the window {} cuts season(s) {} in half. A window has to "
                         "be a whole number of seasons; either move the season "
                         "boundaries in seasons_months.csv or the window here."
                         .format(what, sorted(want), partial))
    if not hit:
        raise ValueError("{}: the window {} matches no season".format(what, sorted(want)))
    return hit


def anchors(header):
    """Year columns of the mapping, in order."""
    out = []
    for i, name in enumerate(header):
        try:
            out.append((int(str(name).strip()), i))
        except ValueError:
            continue
    if not out:
        raise KeyError("no year column in the mapping (header: {})".format(header))
    return sorted(out)


def value_at(row, cols, year):
    """The anchor in force for that year: the last one at or before it."""
    take = cols[0]
    for y, i in cols:
        if y <= year:
            take = (y, i)
    return row[take[1]].strip()


def rebase(path, index_cols, years):
    """Re-emit a year-indexed resource on the model years, holding the last value.

    The inherited trade tables stop at 2030 while the model runs to 2050. A missing
    year is read as zero by GAMS, so a corridor or a contract would simply vanish
    after 2030. Holding the last known value keeps it alive; where that amounts to
    an assumption, the mapping or the build note says so.
    """
    header, rows = read_csv(path)
    cols = anchors(header[index_cols:])
    cols = [(y, i + index_cols) for y, i in cols]
    out = [header[:index_cols] + [str(y) for y in years]]
    for r in rows:
        out.append(r[:index_cols] + [value_at(r, cols, y) for y in years])
    return out


def spread(total, days):
    """Split a seasonal energy over the seasons of its window, pro rata by days.

    pContractedTradeEnergy is read season by season -- the contracted-transfer
    equality of base.gms sums the flows of one season and matches them to one cell --
    so a window that spans two seasons carries a SHARE of the volume in each, never
    the whole of it twice. The residual of the rounding goes to the longest season so
    that the shares add back to the contracted total.
    """
    span = float(sum(days))
    parts = [round(total * d / span, 3) for d in days]
    parts[days.index(max(days))] += round(total - sum(parts), 3)
    return parts


def number(value):
    """A value written back without the noise of the float that carried it."""
    return "{:g}".format(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=os.path.join(REPO, "epm", "input", "data_casa"))
    ap.add_argument("--reference", default=os.path.join(REPO, "epm", "input", "data_casa_2020"))
    args = ap.parse_args()

    maps = os.path.join(HERE, "mappings")
    out_dir = os.path.join(HERE, "extracted")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    years = [int(r[0]) for r in read_csv(os.path.join(args.target, "y.csv"))[1] if r[0].strip()]
    season_set = seasons(os.path.join(maps, "seasons_months.csv"))
    season_names = [name for name, _ms, _d in season_set]
    season_days = dict((name, d) for name, _ms, d in season_set)

    # ---- transfer limits ---------------------------------------------------
    header, rows = read_csv(os.path.join(maps, "corridors.csv"))
    cols = anchors(header)

    base, override, order = {}, {}, []
    for r in rows:
        key = (r[0].strip(), r[1].strip())
        when = r[2].strip()
        if when.lower() == "all":
            if key in base:
                raise ValueError("corridor {} declared twice as 'all'".format(key))
            base[key] = r
            order.append(key)
        else:
            for season in window(months_of(when), season_set, "corridor {}".format(key)):
                override[(key, season)] = r
    missing = {k for (k, _s) in override} - set(base)
    if missing:
        raise ValueError("season override without an 'all' line: {}".format(sorted(missing)))

    limits = [["From", "To", "q"] + [str(y) for y in years]]
    for key in order:
        for season in season_names:
            r = override.get((key, season), base[key])
            limits.append([key[0], key[1], season] + [value_at(r, cols, y) for y in years])
    write_csv(os.path.join(out_dir, "pTransferLimit.csv"), limits)

    # ---- contracted trade --------------------------------------------------
    # The reference table is the source of the volumes, not of the seasons: its own
    # season names belong to the way the 2020 model cut the year. Only the pair and
    # the yearly total are kept, and the window declared in contracts.csv says which
    # seasons of THIS model they land in.
    ref = rebase(os.path.join(args.reference, "trade", "pContractedTradeEnergy.csv"), 3, years)
    volumes = {}
    for r in ref[1:]:
        key = (r[0].strip(), r[1].strip())
        got = volumes.setdefault(key, [0.0] * len(years))
        for i, cell in enumerate(r[3:]):
            got[i] += float(cell or 0)

    _chead, crows = read_csv(os.path.join(maps, "contracts.csv"))
    flags = [["zone1", "zone2", "q", "value"]]
    energy = [["zone1", "zone2", "q"] + [str(y) for y in years]]
    contracted = 0
    for r in crows:
        key = (r[0].strip(), r[1].strip())
        if key not in base:
            raise ValueError("contract {} is on no corridor of corridors.csv".format(key))
        seasons_hit = window(months_of(r[2]), season_set, "contract {}".format(key))
        for season in seasons_hit:
            flags.append([key[0], key[1], season, 1])
        # A leg with no volume keeps its flag and gets no record: the equality then
        # reads zero and holds the counter-flow shut, as the 2020 model did.
        if key not in volumes:
            continue
        contracted += 1
        days = [season_days[s] for s in seasons_hit]
        shares = [spread(total, days) for total in volumes[key]]
        for j, season in enumerate(seasons_hit):
            energy.append([key[0], key[1], season] + [number(sh[j]) for sh in shares])
    left = set(volumes) - {(r[0].strip(), r[1].strip()) for r in crows}
    if left:
        raise ValueError("the reference carries volumes for {} but contracts.csv does "
                         "not declare them".format(sorted(left)))

    write_csv(os.path.join(out_dir, "pContractedTradeFlag.csv"), flags)
    write_csv(os.path.join(out_dir, "pContractedTradeEnergy.csv"), energy)

    print("Seasons: {}".format(", ".join("{} ({} d)".format(n, d)
                                         for n, _m, d in season_set)))
    print("Corridors: {} directed, {} seasons, {} years {}-{}"
          .format(len(order), len(season_names), len(years), years[0], years[-1]))
    print("Season overrides: {}".format(len(override) or "none"))
    print("Contracts: {} legs flagged on {} rows, {} carrying a volume on {} rows"
          .format(len(crows), len(flags) - 1, contracted, len(energy) - 1))
    print("Written: pTransferLimit.csv, pContractedTradeFlag.csv, pContractedTradeEnergy.csv")


if __name__ == "__main__":
    main()
