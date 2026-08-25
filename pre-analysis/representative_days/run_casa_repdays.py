"""Driver: CASA representative-days run.

Feeds the Poncelet pipeline the hourly year CASA actually has and asks it for a set of
representative days, replacing the medoid selection build_demand.py makes on its own.

WHAT IS DIFFERENT FROM THE BLACK SEA DRIVER, and it is not style. Black Sea hard-codes
its season map and its zone list in the driver. Here both are read from the files that
already declare them -- data_build/mappings/seasons_months.csv for the seasons and the
deployed zcmap.csv for the zones -- because the CASA build has exactly one place where
the year is cut and one place where the perimeter is set, and a second copy in this file
would be a second thing to keep in step. Re-cut the year in the mapping and this driver
follows without being edited.

WHAT IS NOT READY YET. The pipeline clusters Load, PV and Wind JOINTLY: that is the whole
point of it, and it is what makes a cloudy day exist. The hourly VRE series do not exist
in this repo yet -- they come from pipelines/vre_pipeline.py against Renewables.ninja --
so this driver runs on Load alone until they land, and says so on every run rather than
letting a load-only clustering pass for the real thing.

THE ZONES ARE HALF THE SYSTEM. Only seven zones carry a full metered year in the DeCA
books, and they are 45% of 2050 demand: Pakistan and Afghanistan, the other 55%, have no
hourly series at all. Since the pipeline chooses ONE set of days shared by every zone --
the special days are picked on peak demand summed ACROSS zones -- a run made now is a run
made without the half of the system the study exists to analyse. That is why this driver
writes to its own output folder and does not deploy: deploying is deploy_casa_repdays.py,
run deliberately, once the missing years arrive.

Inputs
    ../../data_build/extracted/deca_demand_hourly.csv   z,hour,MW over 8760 h
    ../../data_build/mappings/seasons_months.csv        the season set and its months
    ../../epm/input/data_casa/zcmap.csv                 the zones in the perimeter

ENVIRONMENT. This runs in gams_env, not epm_env: the pipeline needs scikit-learn, scipy
and seaborn for the clustering and gams.transfer for the weight optimisation, and epm_env
carries none of them.

    conda run -n gams_env python run_casa_repdays.py --days 5

Usage
    python run_casa_repdays.py --days 5
    python run_casa_repdays.py --days 5 --pv PV.csv --wind Wind.csv   once VRE exists
"""
from pathlib import Path
import argparse
import calendar
import csv
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from representativedays_pipeline import run_representative_days_pipeline  # noqa: E402

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
DATA_BUILD = REPO / "data_build"
INPUT_DIR = BASE / "input"
OUTPUT_DIR = BASE / "output" / "casa"

VRE_PROFILE = REPO / "epm" / "input" / "data_casa" / "supply" / "pVREProfile.csv"

# The model names its renewable technologies in pVREProfile and those names are what
# the clustering must carry: representativedays_pipeline.py uses the key of input_files
# as the `fuel` label it writes out, so a series handed over as "Wind" would reach
# pVREProfile as "Wind" and match no plant in a model that says WT. The names are read
# from the deployed file rather than written down, and the aliases below only say which
# spelling means which resource.
SOLAR_ALIASES = ("PV", "SOLAR", "SPV", "SPP", "SOLARPV")
WIND_ALIASES = ("WT", "WIND", "WPP", "WTG")


def model_vre_labels(path):
    """(solar label, wind label) exactly as the deployed pVREProfile spells them."""
    with open(path, encoding="utf-8-sig") as handle:
        found = {row["tech"].strip() for row in csv.DictReader(handle) if row.get("tech")}
    solar = [t for t in found if t.upper() in SOLAR_ALIASES]
    wind = [t for t in found if t.upper() in WIND_ALIASES]
    if len(solar) != 1 or len(wind) != 1:
        raise SystemExit(
            "cannot tell which technology is which in {0}: found {1}. Add the spelling "
            "to SOLAR_ALIASES or WIND_ALIASES.".format(path, sorted(found)))
    return solar[0], wind[0]


HOURS_PER_DAY = 24
# A non-leap calendar, matching the 8760 h the DeCA series carry and the 365 days the
# season mapping adds up to. February 29 never appears, so none has to be dropped.
MONTH_LENGTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def seasons_map(path):
    """month number -> season name, from the one file that cuts the year.

    Returned with integer keys because the pipeline maps on the month column, and with
    the season names as they are: the pipeline carries them through as labels, so Q3a
    and Q3b reach pHours spelled the way the rest of the build spells them.
    """
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            for month in row["months"].split(","):
                out[int(month.strip())] = row["season"].strip()
    missing = set(range(1, 13)) - set(out)
    if missing:
        raise SystemExit("seasons_months.csv leaves months {0} unassigned".format(
            sorted(missing)))
    return out


def perimeter(path):
    """The zones the model optimises, from the deployed zcmap."""
    with open(path, encoding="utf-8-sig") as fh:
        return {row["z"].strip() for row in csv.DictReader(fh) if row.get("z", "").strip()}


def calendar_index():
    """Hour 1..8760 -> (month, day of month, hour of day 1..24)."""
    index = {}
    hour = 0
    for month, length in enumerate(MONTH_LENGTH, 1):
        for day in range(1, length + 1):
            for hour_of_day in range(1, HOURS_PER_DAY + 1):
                hour += 1
                index[hour] = (month, day, hour_of_day)
    return index


def extract_load(zones):
    """The metered year as the pipeline wants it: zone,month,day,hour,value.

    A zone is kept only if it carries the whole 8760 h. Turkmenistan states one average
    day per month, 288 rows, which is a shape and not a year; clustering a shape against
    real years would let an invented day compete with measured ones for a slot.
    """
    source = DATA_BUILD / "extracted" / "deca_demand_hourly.csv"
    frame = pd.read_csv(source)
    frame = frame.rename(columns={"z": "zone", "MW": "value"})

    counts = frame.groupby("zone")["value"].count()
    full = set(counts[counts == len(calendar_index())].index)
    in_perimeter = set(frame["zone"].unique()) & zones

    kept = sorted(full & in_perimeter)
    partial = sorted(in_perimeter - full)
    outside = sorted(set(frame["zone"].unique()) - zones)
    absent = sorted(zones - set(frame["zone"].unique()))

    print("[casa-repdays] zones kept      : {0}".format(", ".join(kept)))
    if partial:
        print("[casa-repdays] partial year   : {0} (dropped)".format(", ".join(partial)))
    if outside:
        print("[casa-repdays] outside zcmap  : {0} (dropped)".format(", ".join(outside)))
    if absent:
        print("[casa-repdays] no series at all: {0}".format(", ".join(absent)))
    if not kept:
        raise SystemExit("no zone carries a full year; nothing to cluster")

    index = calendar_index()
    frame = frame[frame["zone"].isin(kept)].copy()
    stamps = frame["hour"].astype(int).map(index)
    frame["month"] = stamps.str[0]
    frame["day"] = stamps.str[1]
    frame["hour"] = stamps.str[2]

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = INPUT_DIR / "Load_casa.csv"
    frame[["zone", "month", "day", "hour", "value"]].to_csv(path, index=False)
    print("[casa-repdays] Load written    : {0} ({1} rows)".format(path.name, len(frame)))
    return str(path), kept


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=5,
                        help="representative days per season")
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--special-threshold", type=float, default=0.1)
    parser.add_argument("--feature-selection", type=int, default=20,
                        help="series kept before the Poncelet optimisation")
    parser.add_argument("--pv", type=str, default=None,
                        help="hourly PV csv (zone,month,day,hour,value); omit until fetched")
    parser.add_argument("--wind", type=str, default=None,
                        help="hourly wind csv, same shape as --pv")
    args = parser.parse_args()

    months = seasons_map(DATA_BUILD / "mappings" / "seasons_months.csv")
    zones = perimeter(REPO / "epm" / "input" / "data_casa" / "zcmap.csv")
    order = []
    for month in list(range(12, 13)) + list(range(1, 12)):
        if months[month] not in order:
            order.append(months[month])
    print("[casa-repdays] seasons        : {0}".format(" ".join(order)))

    load_path, kept = extract_load(zones)
    solar_label, wind_label = model_vre_labels(VRE_PROFILE)
    input_files = {"Load": load_path}
    for label, path in ((solar_label, args.pv), (wind_label, args.wind)):
        if path:
            input_files[label] = path
    if len(input_files) > 1:
        print("[casa-repdays] vre labels     : {0}".format(
            ", ".join(k for k in input_files if k != "Load")))

    if len(input_files) == 1:
        print("[casa-repdays] WARNING: clustering on Load alone. The renewable side is\n"
              "               absent, so the days chosen cannot contain a cloudy or a\n"
              "               still one. Fetch hourly PV and wind with\n"
              "               pipelines/vre_pipeline.py and pass --pv/--wind before\n"
              "               treating any of this as final.")

    print("[casa-repdays] days/season    : {0}  ({1} seasons -> {2} day types)".format(
        args.days, len(order), args.days * len(order)))

    result = run_representative_days_pipeline(
        seasons_map=months,
        input_files=input_files,
        output_dir=OUTPUT_DIR,
        gams_main_file=str(BASE / "gams" / "OptimizationModelZone.gms"),
        year_label="casa",
        zones_to_exclude=[],
        n_representative_days=args.days,
        n_clusters=args.n_clusters,
        n_bins=10,
        feature_selection_count=args.feature_selection,
        special_day_threshold=args.special_threshold,
        verbose=True,
    )

    print("\n[casa-repdays] done. Outputs:")
    for name, path in result["paths"].items():
        print("  {0}: {1}".format(name, path))
    print("\n[casa-repdays] nothing was deployed. Check with validate_casa_repdays.py,\n"
          "               then deploy with deploy_casa_repdays.py when the missing\n"
          "               hourly years have arrived.")


if __name__ == "__main__":
    main()
