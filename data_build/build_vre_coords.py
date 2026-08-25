"""Pick one representative point per zone, for the Renewables.ninja fetch.

WHY THIS EXISTS. vre_pipeline.py wants {tech: {zone: (lat, lon)}}, and its own
extractor reads the Global Atlas Power facility workbook to get real plant
coordinates. That workbook is not in this repo, and no plant in the CASA fleet --
neither in the DeCA books nor in the 2020 model -- carries a latitude anywhere. So
the coordinates have to come from the only geometry we do have: the zone polygons.

THE HYPOTHESIS, AND IT IS A REAL ONE. A point inside a zone is not where that
zone's wind farms are. Wind especially is a local resource: the Kazakh capacity
sits in the Ereymentau and Zhanatas corridors, not at the geometric middle of a
zone the size of Western Europe. The area column below is printed so the weakness
is visible where it is worst -- read a capacity factor for KAZ_N as the resource at
one point of northern Kazakhstan, not as the fleet-weighted resource of the zone.
Replace these points with real ones as soon as a plant list with coordinates
arrives; that is what the override file is for.

WHICH POINT. The centroid when it falls inside the zone, which is the reading that
needs least explaining. When it does not -- a crescent, or a MultiPolygon whose
centroid lands between its parts -- the pole of inaccessibility of the largest part
is used instead: the point furthest from any edge, which is the most interior point
there is and is stable against the shape of the border.

OVERRIDES. mappings/zone_points_override.csv, if it exists, wins per zone. It takes
z, lat, lon and a note, and exists so that a better point can be given without
editing this file: the note is carried into the output so the source of every
coordinate stays legible.

Usage
    python build_vre_coords.py
    python build_vre_coords.py --zones ../epm/input/data_casa/zones.geojson
"""
from pathlib import Path
import argparse
import csv
import json
import math

from shapely.geometry import shape
from shapely.ops import polylabel

BASE = Path(__file__).resolve().parent
ZONES = BASE.parents[0] / "epm" / "input" / "data_casa" / "zones.geojson"
OVERRIDE = BASE / "mappings" / "zone_points_override.csv"
TARGET = BASE / "extracted" / "zone_points.csv"

# A degree of latitude is about 111 km; a degree of longitude shrinks with the
# cosine of latitude. Good enough to say which zones are large, which is all the
# area column is for -- nothing downstream computes with it.
KM_PER_DEGREE = 111.32


def approximate_area_km2(geometry):
    latitude = math.radians(geometry.centroid.y)
    return geometry.area * KM_PER_DEGREE ** 2 * math.cos(latitude)


def largest_part(geometry):
    if geometry.geom_type == "MultiPolygon":
        return max(geometry.geoms, key=lambda part: part.area)
    return geometry


def representative_point(geometry, others):
    """The centroid when it is inside, the most interior point when it is not.

    A zone whose centroid escapes it is worth naming rather than silently fixing:
    the model's own map uses raw centroids as its corridor anchors, so wherever this
    branch fires, that map is anchoring the zone somewhere the zone is not.
    """
    centroid = geometry.centroid
    if geometry.contains(centroid):
        return centroid.y, centroid.x, "centroid", ""

    part = largest_part(geometry)
    # A tolerance of a hundredth of a degree, roughly a kilometre, is far below the
    # half-degree MERRA-2 grid the fetch lands on and costs nothing to compute.
    point = polylabel(part, tolerance=0.01)

    landed = [name for name, other in others.items() if other.contains(centroid)]
    where = " and lands in {0}".format(", ".join(sorted(landed))) if landed else ""
    note = ("The centroid of this zone falls outside it{0}, so the most interior point "
            "is used instead. Note that the corridor anchors in linestring_zcmap.geojson "
            "are raw centroids and therefore carry that error.".format(where))
    return point.y, point.x, "pole of inaccessibility", note


def read_overrides(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig") as handle:
        return {row["z"].strip(): row for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zones", type=Path, default=ZONES)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()

    overrides = read_overrides(OVERRIDE)
    if overrides:
        print("[coords] overrides: {0}".format(", ".join(sorted(overrides))))

    features = json.loads(args.zones.read_text(encoding="utf-8"))["features"]
    rows = []
    for feature in features:
        properties = feature["properties"]
        zone = properties["z"]
        geometry = shape(feature["geometry"])
        area = approximate_area_km2(geometry)

        override = overrides.get(zone)
        if override:
            latitude, longitude = float(override["lat"]), float(override["lon"])
            method = "override"
            note = override.get("note", "").strip()
        else:
            others = {other["properties"]["z"]: shape(other["geometry"])
                      for other in features if other["properties"]["z"] != zone}
            latitude, longitude, method, note = representative_point(geometry, others)

        rows.append(dict(z=zone, c=properties.get("c", ""), lat=round(latitude, 4),
                         lon=round(longitude, 4), method=method,
                         area_km2=int(round(area)), note=note))

    rows.sort(key=lambda row: row["z"])
    args.target.parent.mkdir(parents=True, exist_ok=True)
    with open(args.target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    widest = max(rows, key=lambda row: row["area_km2"])
    print("[coords] {0} zones written to {1}".format(len(rows), args.target))
    print("[coords] largest zone {0}, about {1:,} km2: one point speaks least for it"
          .format(widest["z"], widest["area_km2"]))
    for row in rows:
        print("  {z:<12}{lat:>9.4f}{lon:>10.4f}  {area_km2:>9,} km2  {method}".format(**row))
    for row in rows:
        if row["note"]:
            print("[coords] {0}: {1}".format(row["z"], row["note"]))


if __name__ == "__main__":
    main()
