"""Driver: fetch one hourly solar and wind year for every CASA zone.

WHAT THIS IS FOR. pVREProfile currently has a staircase in it: build_vre.py takes the
SHAPE from the 2020 model's six chronological blocks and the LEVEL from DeCA's monthly
utilization factors, so every profile is six plateaus held flat across twenty-four
columns, identical across all three day types. The level is defensible -- DeCA's annual
0.154 against the model's 0.167 -- but the variance is not there at all: in 2026 to 2050
as the model currently stands, no cloudy day and no still day exists. Neither source can
supply both, and DeCA's own authors left the note saying so. This is the other route.

WHICH YEAR, AND THE HYPOTHESIS IN IT. One weather year, 2022 by default. The hourly
demand it will be clustered against is not one year at all -- the DeCA books are
Kazakhstan 2022, Tajikistan 2021, and unstated for Kyrgyzstan and Uzbekistan -- so a
single weather year cannot be made to match all of them. 2022 is chosen because it
matches the largest metered zone group and sits inside MERRA-2 coverage. What this
costs: the correlation between a windy hour and a high-load hour is real within
Kazakhstan and approximate everywhere else. Say so wherever these profiles are used.

WHERE THE COORDINATES COME FROM, and solar and wind no longer share them. No plant in the
CASA fleet carries a latitude, so the points start from the zone polygons, one per zone,
via data_build/build_vre_coords.py. Solar keeps that point: irradiance varies by a few per
cent across a zone and the centroid is as good a place to ask about as any.

WIND DOES NOT KEEP IT, AND THAT WAS THE BUG. The first fetch asked about the centroid for
both technologies and came back with mountain wind near zero -- TAJ_S at 0.016, TAJ_N at
0.106 -- which read like a failure of the reanalysis. It was not. Sampled at those exact
coordinates the Global Wind Atlas, at 250 m, says 0.010 and 0.048: the two sources agree,
r = 0.90 across the region. The centroid of a mountain zone is a valley floor, and a
valley floor is where the wind is not. So wind is fetched from data_build/wind_sites.csv
instead, the centre of the best ten-kilometre square of each zone, 29 to 936 km away --
further than a MERRA-2 cell is wide, so it is different weather and not a rescaled copy.
KAZ_N is still 1.6 million km2 and one point still speaks for a slice of it; the
difference is that it is now the slice a developer would build on.

LOCAL TIME IS ON, deliberately. The region spans UTC+4:30 to UTC+6, and the demand
series are local-time series. Fetching in local time puts solar noon in the same column
as the load it is meant to serve; fetching in UTC would slide the two apart by up to an
hour and a half across the model.

ENVIRONMENT. gams_env, like the rest of the representative-days chain.

    conda run -n gams_env python run_casa_vre_fetch.py

Usage
    python run_casa_vre_fetch.py --dry-run     # show the points, call nothing
    python run_casa_vre_fetch.py               # about six minutes, 32 API requests
    python run_casa_vre_fetch.py --only wind   # refetch wind at the atlas sites
"""
from pathlib import Path
import argparse
import csv
import sys
from math import cos, hypot, radians

import pandas as pd
from timezonefinder import TimezoneFinder

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vre_pipeline import (rninja_output_filename,  # noqa: E402
                          run_renewables_ninja_workflow)

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
POINTS = REPO / "data_build" / "extracted" / "zone_points.csv"
WIND_SITES = REPO / "data_build" / "extracted" / "wind_sites.csv"
RAW_DIR = BASE / "output" / "rninja_casa"
REPDAYS_INPUT = REPO / "pre-analysis" / "representative_days" / "input"

LABEL = "casa"
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



def read_points(path):
    if not path.exists():
        raise SystemExit(
            "no zone points at {0}: run data_build/build_vre_coords.py first".format(path))
    with open(path, encoding="utf-8-sig") as handle:
        return [row for row in csv.DictReader(handle) if row.get("z", "").strip()]


def read_optional(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as handle:
        return [row for row in csv.DictReader(handle) if row.get("z", "").strip()]


def points_by_technology(zone_points, wind_sites, wind_label):
    """{technology: {zone: (lat, lon)}}, and they are not the same points.

    SOLAR KEEPS THE ZONE POINT. Irradiance varies little across a zone -- a few per cent
    between one end of Uzbekistan and the other -- so the centroid is as good a place to
    ask about as any, and the sixteen solar series fetched from it all rescaled onto their
    stated level without complaint.

    WIND DOES NOT. Sampling the Global Wind Atlas at the exact centroids these series were
    fetched from showed the reanalysis was right all along and the point was wrong: 0.010
    at TAJ_S, 0.048 at TAJ_N, against zone P95 values of 0.393 and 0.433. The centroid of a
    mountain zone is a valley floor. data_build/build_wind_atlas.py writes the centre of
    the best ten-kilometre square of each zone instead, and that is what wind is fetched
    from -- between 29 and 936 km away from the old point, which is further than a MERRA-2
    cell is wide, so this samples different weather rather than a rescaled copy.
    """
    centroids = {row["z"]: (float(row["lat"]), float(row["lon"])) for row in zone_points}
    wind = dict(centroids)
    for row in wind_sites:
        if row["z"] in wind:
            wind[row["z"]] = (float(row["lat"]), float(row["lon"]))
    if not wind_sites:
        print("[vre] WARNING: no wind sites at {0}. Wind will be fetched from the same"
              .format(WIND_SITES))
        print("               centroids as solar, which is the arrangement the atlas")
        print("               showed to be the cause of the near-zero mountain series.")
        print("               Run data_build/build_wind_atlas.py first.")
    return {"solar": centroids, "wind": wind}


def to_repdays_shape(raw_path, target_path, year, coordinates):
    """RNinja's csv is already zone,month,day,hour,<year>; three things need fixing.

    THE CALENDAR IS UTC, whatever local_time was asked for. vre_pipeline.py requests
    local time from the API and then, at lines 480-483, re-derives month, day and hour
    from the timestamp parsed with utc=True -- which throws the offset away again. It
    shows: fetched as it comes, solar over southern Kazakhstan peaks at 07:00 and sets
    at 14:00. Across a region spanning UTC+4:30 to UTC+6 that would put every solar noon
    in the wrong column and destroy exactly the correlation with demand these profiles
    are being fetched for. The fix is here and not in vre_pipeline.py because that file
    is shared with the other regional models and moving it would move their profiles too.

    The relabelling keeps all 8760 rows: the hours that fall off one end of the local
    year arrive at the other, so every (month, day, hour) of a 365-day year still gets
    exactly one value. That is checked below rather than assumed.

    The year column then becomes `value`, and the hour becomes 1..24 to match the load
    extraction -- the clustering merges the two on (month, day, hour), so a series left
    on 0..23 would be read against the load of the hour before it.
    """
    frame = pd.read_csv(raw_path)
    year_column = [c for c in frame.columns if str(c) == str(year)]
    if not year_column:
        raise SystemExit("{0} carries no column for {1}: got {2}".format(
            raw_path.name, year, list(frame.columns)))
    frame = frame.rename(columns={year_column[0]: "value"})

    finder = TimezoneFinder()
    pieces = []
    for zone, block in frame.groupby("zone", sort=False):
        latitude, longitude = coordinates[zone]
        timezone = finder.timezone_at(lat=latitude, lng=longitude)
        if timezone is None:
            raise SystemExit("no timezone found for {0} at {1}, {2}".format(
                zone, latitude, longitude))

        block = block.copy()
        stamps = pd.date_range("{0}-01-01".format(year), periods=len(block),
                               freq="h", tz="UTC").tz_convert(timezone)
        block["month"] = stamps.month
        block["day"] = stamps.day
        block["hour"] = stamps.hour + 1

        keys = set(zip(block["month"], block["day"], block["hour"]))
        if len(keys) != len(block):
            raise SystemExit(
                "{0} in {1} does not relabel to one value per hour of the year: {2} rows "
                "give {3} distinct hours. A daylight-saving jump would do this.".format(
                    zone, timezone, len(block), len(keys)))
        pieces.append(block)
        print("  {0:<12}{1:<22}shifted {2:+.1f} h from UTC".format(
            zone, timezone, stamps[0].utcoffset().total_seconds() / 3600.0))

    frame = pd.concat(pieces)[["zone", "month", "day", "hour", "value"]]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target_path, index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the points and the request count, call nothing")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--only", choices=["solar", "wind"], default=None,
                        help="fetch one technology and leave the other file alone")
    args = parser.parse_args()

    solar_label, wind_label = model_vre_labels(VRE_PROFILE)
    techs = {"solar": solar_label, "wind": wind_label}
    print("[vre] labels      : the model spells them {0} and {1}".format(
        solar_label, wind_label))

    points = read_points(POINTS)
    locations = points_by_technology(points, read_optional(WIND_SITES), wind_label)
    if args.only:
        techs = {k: v for k, v in techs.items() if k == args.only}
        locations = {k: v for k, v in locations.items() if k == args.only}
        print("[vre] fetching    : {0} only; the other technology keeps the series it has"
              .format(args.only))

    zones = sorted({z for tech in locations for z in locations[tech]})
    requests_needed = sum(len(v) for v in locations.values())
    print("[vre] year        : {0}, local time".format(args.year))
    print("[vre] zones       : {0}".format(len(zones)))
    print("[vre] requests    : {0}, about {1:.0f} min at 6 per minute".format(
        requests_needed, requests_needed / 6.0))
    print("  {0:<12}{1:>10}{2:>10}   {3:>10}{4:>10}   {5}".format(
        "zone", "solar lat", "lon", "wind lat", "lon", "apart"))
    for zone in zones:
        solar = locations.get("solar", {}).get(zone)
        wind = locations.get("wind", {}).get(zone)
        apart = ""
        if solar and wind:
            middle = radians((solar[0] + wind[0]) / 2.0)
            apart = "{0:.0f} km".format(hypot(wind[0] - solar[0],
                                              (wind[1] - solar[1]) * cos(middle)) * 111.32)
        print("  {0:<12}{1:>10}{2:>10}   {3:>10}{4:>10}   {5}".format(
            zone,
            "{0:.4f}".format(solar[0]) if solar else "-",
            "{0:.4f}".format(solar[1]) if solar else "-",
            "{0:.4f}".format(wind[0]) if wind else "-",
            "{0:.4f}".format(wind[1]) if wind else "-", apart))

    if args.dry_run:
        print("\nDry run. Nothing fetched.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # end_year is exclusive in get_renewables_ninja: range(start_year, end_year).
    run_renewables_ninja_workflow(
        locations=locations,
        start_year=args.year,
        end_year=args.year + 1,
        dataset_label=LABEL,
        input_dir=str(RAW_DIR),
        output_dir=str(RAW_DIR),
        generate_plots=not args.no_plots,
        local_time=True,
    )

    print("\n[vre] annual mean capacity factor, from the fetched year:")
    for tech, label in techs.items():
        raw = RAW_DIR / rninja_output_filename(LABEL, tech)
        if not raw.exists():
            print("  {0}: nothing written at {1}".format(label, raw.name))
            continue
        print("[vre] relabelling {0} to local time:".format(label))
        # The timezone is read off the point the series was actually fetched from, so
        # the wind relabelling follows the wind point: KAZ_S moved 936 km east to the
        # Dzungarian Gate, and asking the solar centroid for its offset would put that
        # series in the wrong hour.
        frame = to_repdays_shape(raw, REPDAYS_INPUT / "{0}_casa_rninja.csv".format(label),
                                 args.year, locations[tech])
        means = frame.groupby("zone")["value"].mean().sort_values(ascending=False)
        print("  {0:<6}{1}".format(label, "  ".join(
            "{0} {1:.3f}".format(z, v) for z, v in means.items())))
        hours = frame.groupby("zone")["value"].count()
        short = hours[hours != 8760]
        if len(short):
            print("  {0}: NOT a full year for {1}".format(label, dict(short)))
        print("  {0:<6}written to {1}".format(
            "", REPDAYS_INPUT / "{0}_casa_rninja.csv".format(label)))

    print("\n[vre] nothing was deployed, and these files carry Renewables.ninja's own\n"
          "      levels, which the wind comparison says are not the ones to model with.\n"
          "      Next: data_build/build_vre_hourly.py, which rescales them onto the DeCA\n"
          "      levels and writes the plain PV_casa.csv / WT_casa.csv that build_vre.py\n"
          "      and run_casa_repdays.py read.")


if __name__ == "__main__":
    main()
