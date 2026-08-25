"""Ask the Global Wind Atlas what wind each CASA zone actually has.

WHY THIS STEP EXISTS. The wind level is the weakest number in the build, and it is weak
in a specific way. DeCA states a capacity factor near 0.33 for every zone it covers --
0.3351 for Kyrgyzstan, 0.3726 for Tajikistan, 0.3490 for Turkmenistan -- and the 2020
model states 0.3333 for the nine Afghan and Pakistani zones it has to speak for. A single
number repeated across a region running from the Karakum desert to the Pamir is not a
measurement of a resource; it is a planning assumption about where a developer would
build. Renewables.ninja, asked the same question, answered 0.016 for southern Tajikistan
and 0.63 for Herat. Those two sources disagree by a factor of twenty and neither of them
is looking at the terrain: MERRA-2 reanalysis runs on half-degree cells that cannot see a
valley, and DeCA's figure was never a per-zone figure at all.

The Global Wind Atlas can see the terrain. It is a microscale downscaling of ERA5 onto a
0.0025 degree grid -- roughly 250 m -- with the roughness and the orography in it, and it
publishes capacity factor directly for three IEC turbine classes. That is a level source
and only a level source: it has no hours in it. So the division of labour is unchanged
from build_vre_hourly.py, with one substitution -- SHAPE FROM RENEWABLES.NINJA, LEVEL FROM
WHOEVER IS CREDIBLE ABOUT LEVEL -- and this file exists to find out whether that is still
DeCA.

WHAT IT COMPUTES, AND WHY IT IS NOT THE ZONE MEAN. Nobody builds a wind farm at the
average square kilometre of a zone. The mean over KAZ_N would average 1.6 million km2 of
northern Kazakhstan, most of which no developer would look at twice, and it would
understate the resource as badly as the centroid point does. So the whole area-weighted
distribution is reported -- mean, median, P75, P90, P95, P99 and the best pixel -- with
the land area each percentile stands for printed beside it, because a percentile of a zone
the size of western Europe and a percentile of Karachi are not the same offer. The area
column is what makes the table arguable: P99 of KAZ_N is fifteen thousand square
kilometres and P99 of PAK_KAR is forty.

LOSSES. The Global Wind Atlas capacity factor is gross. It is a power curve evaluated
against a wind climate, and it contains no wake, no availability, no electrical loss and
no curtailment. DeCA's numbers are meant to be net. Comparing them raw would flatter the
atlas by around a sixth, so a single net-of-gross factor is applied and stated, and it is
a command-line argument because it is an assumption and not a measurement.

WHAT THIS DOES NOT DO. It does not change the model. It writes a table and prints a
verdict; deciding what the wind level should become, and rebuilding pVREProfile onto it,
is a separate and deliberate step.

Reads:
    <deployed>/zones.geojson              the zone polygons and their ISO_A3 country
    mappings/vre_profiles.csv             which zones have a DeCA book and which do not
    extracted/vre_hourly_report.csv       the stated and the Renewables.ninja levels
    the Global Wind Atlas country rasters, downloaded once and cached

Writes:
    extracted/wind_atlas.csv              the distribution per zone and IEC class

THE COUNTRY CODE COMES FROM THE GEOMETRY, not from a table written down here: every
feature of zones.geojson already carries ISO_A3, so a zone that changes country, or a
perimeter that gains one, needs no edit. Note that the Afghan zones named after their
neighbours -- NEPS_TAJ, NEPS_TKM, NEPS_UZB -- are Afghan, and the file says so.

ENVIRONMENT. gams_env, for rasterio and shapely.

    conda run -n gams_env python build_wind_atlas.py

Usage
    python build_wind_atlas.py                    # about 350 MB per IEC class, once
    python build_wind_atlas.py --classes IEC3     # one class only
    python build_wind_atlas.py --losses 0.80      # a harsher net-of-gross assumption
"""
from pathlib import Path
import argparse
import csv
import io
import json
import math
import sys

import numpy as np
import rasterio
import rasterio.mask
import requests
from shapely.geometry import shape

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ZONES = REPO / "epm" / "input" / "data_casa" / "zones.geojson"
CACHE = REPO / "pre-analysis" / "pipelines" / "output" / "gwa_casa"
TARGET = HERE / "extracted" / "wind_atlas.csv"

API = "https://globalwindatlas.info/api/gis/country/{iso}/{layer}"

# The three turbine classes the atlas publishes, from the machine built for a gale to the
# machine built for a breeze. IEC3 is the low-wind, low-specific-power class, and it is
# the generous end of the range: if a zone cannot reach a stated capacity factor even on
# IEC3 at its best sites, no turbine choice rescues that number.
CLASSES = ("IEC1", "IEC2", "IEC3")

# Net of gross. Wakes in a real array cost the better part of ten per cent, availability
# two to three, electrical collection and transformation about two, and there is always
# some curtailment. Fifteen per cent all told is the ordinary planning figure and it is
# the default here; --losses moves it.
NET_OF_GROSS = 0.85

# A degree of latitude is 111.32 km; a degree of longitude is that times the cosine of the
# latitude. Used only to turn a pixel count into an area a reader can weigh.
KM_PER_DEGREE = 111.32

PERCENTILES = (50, 75, 90, 95, 99)


def zone_polygons(path):
    """[(zone, ISO_A3, geometry)] straight from the deployed layer."""
    with io.open(path, encoding="utf-8") as handle:
        collection = json.load(handle)
    out = []
    for feature in collection["features"]:
        properties = feature["properties"]
        zone = (properties.get("z") or properties.get("zone") or "").strip()
        iso = (properties.get("ISO_A3") or "").strip()
        if not zone:
            continue
        if not iso:
            raise SystemExit(
                "{0} carries no ISO_A3, so there is no way to say which national raster "
                "covers it. Add the property or the zone cannot be looked up.".format(zone))
        out.append((zone, iso, shape(feature["geometry"])))
    return sorted(out)


def cached_raster(iso, klass, cache):
    """The national capacity-factor raster, downloaded on first use and kept."""
    cache.mkdir(parents=True, exist_ok=True)
    layer = "capacity-factor_{0}".format(klass)
    path = cache / "{0}_{1}.tif".format(iso, layer)
    if path.exists() and path.stat().st_size:
        return path
    url = API.format(iso=iso, layer=layer)
    print("  fetching {0} {1} ...".format(iso, klass), end=" ", flush=True)
    response = requests.get(url, stream=True, timeout=600)
    if response.status_code != 200:
        raise SystemExit(
            "the atlas answered {0} for {1}. The path uses ISO 3166 alpha-3, which is not "
            "always the code the model uses -- Tajikistan is TJK and Turkmenistan is TKM "
            "-- and zones.geojson is where that code comes from.".format(
                response.status_code, url))
    partial = path.with_suffix(".part")
    with io.open(str(partial), "wb") as handle:
        for chunk in response.iter_content(1 << 20):
            handle.write(chunk)
    partial.replace(path)
    print("{0:.0f} MB".format(path.stat().st_size / 1e6))
    return path


def weighted_quantile(values, weights, quantile):
    """The value below which `quantile` of the weight lies, weights being pixel areas."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    running = np.cumsum(weights)
    return float(values[np.searchsorted(running, quantile * running[-1])])


def zone_distribution(path, geometry):
    """The area-weighted spread of capacity factor over one zone polygon.

    Every pixel is weighted by the ground it covers. The grid is in degrees, so a pixel in
    northern Kazakhstan is two thirds the area of a pixel in Karachi, and an unweighted
    percentile would quietly give the north a third more say than the ground it holds.
    """
    with rasterio.open(str(path)) as source:
        try:
            block, transform = rasterio.mask.mask(
                source, [geometry.__geo_interface__], crop=True, filled=True, nodata=np.nan)
        except ValueError:
            return None  # the polygon does not overlap this raster at all
        block = block[0]

    rows = np.arange(block.shape[0])
    latitudes = transform.f + (rows + 0.5) * transform.e
    cosines = np.cos(np.radians(latitudes))[:, None]
    weights = np.broadcast_to(cosines, block.shape)

    good = np.isfinite(block)
    if not good.any():
        return None
    values = block[good].astype("float64")
    weights = weights[good].astype("float64")

    cell = abs(transform.a * transform.e) * KM_PER_DEGREE ** 2
    return dict(
        values=values,
        weights=weights,
        area_km2=float(weights.sum()) * cell,
        cell_km2=cell,
    )


def stated_levels(path):
    """(zone, tech) -> (stated level, Renewables.ninja level, where the level came from)."""
    out = {}
    if not path.exists():
        return out
    with io.open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                out[(row["z"], row["tech"])] = (
                    float(row["cf_stated"]), float(row["cf_rninja"]), row["level_from"])
            except (KeyError, ValueError):
                continue
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zones", type=Path, default=ZONES)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--classes", nargs="+", default=list(CLASSES),
                        help="IEC turbine classes to sample")
    parser.add_argument("--losses", type=float, default=NET_OF_GROSS,
                        help="net capacity factor as a fraction of the atlas gross figure")
    parser.add_argument("--tech", default="WT", help="the model's label for wind")
    args = parser.parse_args()

    zones = zone_polygons(args.zones)
    print("zones      {0} from {1}".format(len(zones), args.zones.name))
    print("classes    {0}".format(", ".join(args.classes)))
    print("losses     net = {0:.2f} x gross\n".format(args.losses))

    known = stated_levels(HERE / "extracted" / "vre_hourly_report.csv")

    rows = []
    for klass in args.classes:
        print("[{0}]".format(klass))
        for zone, iso, geometry in zones:
            raster = cached_raster(iso, klass, args.cache)
            spread = zone_distribution(raster, geometry)
            if spread is None:
                print("  {0:<11} no atlas coverage inside the polygon".format(zone))
                continue
            values, weights = spread["values"] * args.losses, spread["weights"]
            record = dict(z=zone, iso=iso, iec=klass,
                          area_km2=round(spread["area_km2"], 1),
                          mean=round(float(np.average(values, weights=weights)), 4),
                          max=round(float(values.max()), 4))
            for q in PERCENTILES:
                record["p{0}".format(q)] = round(
                    weighted_quantile(values, weights, q / 100.0), 4)
                # The land above a percentile is what makes it an offer rather than a
                # statistic: the top 1% of a zone is only useful if 1% is a lot of ground.
                record["km2_above_p{0}".format(q)] = round(
                    spread["area_km2"] * (1.0 - q / 100.0), 1)
            stated, ninja, origin = known.get((zone, args.tech), ("", "", ""))
            record.update(cf_stated=stated, cf_rninja=ninja, level_from=origin)
            rows.append(record)

    if not rows:
        raise SystemExit("nothing sampled")

    header = ["z", "iso", "iec", "area_km2", "mean", "max"]
    for q in PERCENTILES:
        header += ["p{0}".format(q), "km2_above_p{0}".format(q)]
    header += ["cf_stated", "cf_rninja", "level_from"]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with io.open(str(TARGET), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("\nwrote {0} ({1} rows)".format(TARGET, len(rows)))

    report(rows, args)


def report(rows, args):
    """Print the comparison the whole file exists to make."""
    generous = args.classes[-1]
    band = [r for r in rows if r["iec"] == generous]
    print("\nWIND CAPACITY FACTOR, net of {0:.0f}% losses, on {1} -- the most forgiving "
          "class asked for".format((1 - args.losses) * 100, generous))
    print("{0:<12}{1:>8}{2:>8}{3:>8}{4:>8}{5:>8}   {6:>8}{7:>8}   verdict".format(
        "zone", "mean", "P90", "P95", "P99", "best", "stated", "ninja"))
    print("-" * 88)
    unreachable = []
    for record in sorted(band, key=lambda r: r["z"]):
        stated = record["cf_stated"]
        if stated == "":
            verdict = "no stated level"
        elif record["p99"] >= stated:
            verdict = "reachable, best {0:.0f} km2".format(record["km2_above_p99"])
        elif record["max"] >= stated:
            verdict = "only at the single best pixel"
            unreachable.append(record)
        else:
            verdict = "UNREACHABLE anywhere in the zone"
            unreachable.append(record)
        print("{0:<12}{1:>8.3f}{2:>8.3f}{3:>8.3f}{4:>8.3f}{5:>8.3f}   {6:>8}{7:>8}   {8}"
              .format(record["z"], record["mean"], record["p90"], record["p95"],
                      record["p99"], record["max"],
                      "{0:.3f}".format(stated) if stated != "" else "-",
                      "{0:.3f}".format(record["cf_rninja"]) if record["cf_rninja"] != ""
                      else "-", verdict))
    print("-" * 88)
    if unreachable:
        print("{0} zone(s) cannot reach the level the build currently states, even on {1} "
              "at their\nbest ground. That is the finding: {2}".format(
                  len(unreachable), generous,
                  ", ".join(r["z"] for r in unreachable)))


if __name__ == "__main__":
    sys.exit(main())
