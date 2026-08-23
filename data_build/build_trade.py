# -*- coding: utf-8 -*-
"""Expand the corridor table into the EPM transfer-limit resource.

    python data_build/build_trade.py

Reads:
    mappings/corridors.csv   one line per directed corridor, hand-edited
    <target>/y.csv           the model years
    <reference>/trade/pTransferLimit.csv   for the season set only

Writes:
    extracted/pTransferLimit.csv         taken as is by the build
    extracted/pContractedTradeEnergy.csv same, re-based on the model years

The mapping holds anchor years as columns. A value holds from its anchor until the
next one, so a corridor that comes into service in 2030 is written as 1e-09 at the
2026 anchor and its rating at the 2030 anchor. Nothing is interpolated: an
interconnection is commissioned, it does not fade in.

A corridor carries one 'all' line, applied to every season, and optionally one line
per season that overrides it. The 1e-09 convention comes from the 2020 model: the
GAMS reader turns a plain 0 into a missing record, so a closed corridor is written
as a value small enough to be zero and large enough to keep the record.
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


def write_csv(path, table):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        csv.writer(fh, lineterminator="\n").writerows(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=os.path.join(REPO, "epm", "input", "data_casa"))
    ap.add_argument("--reference", default=os.path.join(REPO, "epm", "input", "data_casa_2020"))
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "extracted")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    years = [int(r[0]) for r in read_csv(os.path.join(args.target, "y.csv"))[1] if r[0].strip()]

    # season set, read from the reference resource so that nothing is hard-coded
    ref_header, ref_rows = read_csv(os.path.join(args.reference, "trade", "pTransferLimit.csv"))
    seasons = sorted({r[2].strip() for r in ref_rows if r[2].strip()})

    header, rows = read_csv(os.path.join(HERE, "mappings", "corridors.csv"))
    cols = anchors(header)

    base, override, order = {}, {}, []
    for r in rows:
        key = (r[0].strip(), r[1].strip())
        season = r[2].strip()
        if season.lower() == "all":
            if key in base:
                raise ValueError("corridor {} declared twice as 'all'".format(key))
            base[key] = r
            order.append(key)
        else:
            if season not in seasons:
                raise ValueError("corridor {}: season '{}' is not one of {}"
                                 .format(key, season, seasons))
            override[(key, season)] = r
    missing = {k for (k, _s) in override} - set(base)
    if missing:
        raise ValueError("season override without an 'all' line: {}".format(sorted(missing)))

    out = os.path.join(out_dir, "pTransferLimit.csv")
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["From", "To", "q"] + [str(y) for y in years])
        for key in order:
            for season in seasons:
                r = override.get((key, season), base[key])
                w.writerow([key[0], key[1], season] + [value_at(r, cols, y) for y in years])

    # contracted trade: same years, values held from the last one known
    contracted = "pContractedTradeEnergy.csv"
    table = rebase(os.path.join(args.reference, "trade", contracted), 3, years)
    write_csv(os.path.join(out_dir, contracted), table)

    print("Corridors: {} directed, {} seasons, {} years {}-{}"
          .format(len(order), len(seasons), len(years), years[0], years[-1]))
    print("Season overrides: {}".format(len(override) or "none"))
    print("Contracted trade: {} rows re-based on the model years".format(len(table) - 1))
    print("Written: {} and {}".format(out, os.path.join(out_dir, contracted)))


if __name__ == "__main__":
    main()
