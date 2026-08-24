# -*- coding: utf-8 -*-
"""Stage a solved run into epm/output_view, the copy EPM View reads.

    python tools/stage_output_view.py                 # the newest run under epm/output
    python tools/stage_output_view.py --run <name>    # a named one
    python tools/stage_output_view.py --replace       # drop the runs already staged

A run directory holds far more than a reader needs: the GDX files, the GAMS
listing, the solver log, the PDF, and the wide summary tables the postprocessing
writes for a human. Only what EPM View asks for is copied -- the scenario CSV
tables, the map layers, the scenario list -- so epm/output_view stays a curated
view rather than a mirror. That matters more here than elsewhere: this folder is
tracked in git, so every megabyte copied is a megabyte published.

pDispatchComplete IS SPLIT PER YEAR. It is by far the largest table -- 17.7 MB
for a 13-year, 360-block run -- and the reader wants one year at a time, so
sending the whole of it down a browser connection to draw one chart is the
difference between a page that opens and a page that hangs. epmFetch.js asks for
pDispatchComplete/y{year}.csv first and falls back to the whole file, so a run
staged without the split still works, only slowly.
"""

import argparse
import csv
import io
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# What EPM View reads at the root of a run, next to the scenario folders: the map
# layers, and the scenario list it takes the scenario names from.
RUN_ROOT = ("input_scenarios.csv",)
RUN_ROOT_SUFFIXES = (".geojson",)
SPLIT_BY_YEAR = "pDispatchComplete.csv"
YEAR_COLUMN = "y"


def newest_run(output_dir):
    runs = [d for d in os.listdir(output_dir)
            if d.startswith("simulations_run_")
            and os.path.isdir(os.path.join(output_dir, d))]
    if not runs:
        raise SystemExit("no run under " + output_dir)
    return sorted(runs)[-1]


def copy_flat(src_dir, dst_dir, suffixes, names=()):
    """Copy the files of one directory named in `names` or ending in a suffix."""
    taken = 0
    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        if name not in names and not name.lower().endswith(suffixes):
            continue
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        shutil.copyfile(src, os.path.join(dst_dir, name))
        taken += 1
    return taken


def split_years(src, dst_dir):
    """One file per year, each keeping the header. Returns the years written."""
    with io.open(src, encoding="utf-8-sig", newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        if YEAR_COLUMN not in header:
            raise SystemExit("{} has no '{}' column".format(src, YEAR_COLUMN))
        at = header.index(YEAR_COLUMN)
        rows = {}
        for r in rd:
            if r:
                rows.setdefault(r[at].strip(), []).append(r)
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    for year in sorted(rows):
        out = os.path.join(dst_dir, "y{}.csv".format(year))
        with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows[year])
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run name; default is the newest one")
    ap.add_argument("--output", default=os.path.join(REPO, "epm", "output"))
    ap.add_argument("--view", default=os.path.join(REPO, "epm", "output_view"))
    ap.add_argument("--replace", action="store_true",
                    help="remove the runs already staged before copying")
    args = ap.parse_args()

    run = args.run or newest_run(args.output)
    src = os.path.join(args.output, run)
    if not os.path.isdir(src):
        raise SystemExit("no such run: " + src)

    if args.replace and os.path.isdir(args.view):
        for name in sorted(os.listdir(args.view)):
            if name.startswith("simulations_run_"):
                shutil.rmtree(os.path.join(args.view, name))
                print("removed  {}".format(name))

    dst = os.path.join(args.view, run)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    print("run      {}".format(run))
    print("root     {} file(s)".format(
        copy_flat(src, dst, RUN_ROOT_SUFFIXES, RUN_ROOT)))

    scenarios = [d for d in sorted(os.listdir(src))
                 if os.path.isdir(os.path.join(src, d, "output_csv"))]
    if not scenarios:
        raise SystemExit("no scenario with an output_csv in " + src)

    for scenario in scenarios:
        src_csv = os.path.join(src, scenario, "output_csv")
        dst_csv = os.path.join(dst, scenario, "output_csv")
        taken = copy_flat(src_csv, dst_csv, (".csv",))
        whole = os.path.join(dst_csv, SPLIT_BY_YEAR)
        years = []
        if os.path.isfile(whole):
            years = split_years(whole, os.path.join(dst_csv, "pDispatchComplete"))
            os.remove(whole)
            taken -= 1
        print("{:<9}{} table(s), dispatch split over {} year(s) {}".format(
            scenario, taken, len(years),
            "{}-{}".format(years[0], years[-1]) if years else ""))

    total = sum(os.path.getsize(os.path.join(root, f))
                for root, _d, fs in os.walk(dst) for f in fs)
    print("staged   {:.1f} MB in {}".format(total / 1048576.0, dst))


if __name__ == "__main__":
    main()
