"""Extract the water layers of the sector briefs into data/.

Sources, downloaded once into a cache folder:
- Natural Earth 10m rivers and lakes, global and the Europe supplement
  (github.com/nvkelso/natural-earth-vector).
- HydroBASINS level 4, Europe and Middle East (data.hydrosheds.org).

Writes water_rivers.geojson, water_lakes.geojson and water_basins.geojson,
clipped to the brief frame and thinned. Stdlib only.

    python extract_water.py [--cache DIR]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from shp import read_dbf, read_shp

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
FRAME = (21.0, 34.5, 54.0, 48.5)  # a little wider than the map frame

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
HYBAS = "https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_eu_lev04_v1c.zip"
RIVER_FILES = ["ne_10m_rivers_lake_centerlines", "ne_10m_rivers_europe"]
LAKE_FILES = ["ne_10m_lakes", "ne_10m_lakes_europe"]


def fetch(url: str, cache: Path) -> bytes:
    path = cache / url.rsplit("/", 1)[1]
    if not path.exists():
        print(f"Downloading {url}")
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def touches(coords) -> bool:
    return any(FRAME[0] <= x <= FRAME[2] and FRAME[1] <= y <= FRAME[3] for x, y in coords)


def thin(coords, step: float, closed: bool = False):
    """Drop vertices closer than step degrees to the last kept one."""
    out = [coords[0]]
    for p in coords[1:-1]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= step:
            out.append(p)
    out.append(coords[-1])
    out = [[round(x, 3), round(y, 3)] for x, y in out]
    if closed and len(out) < 4:
        return None
    return out if len(out) >= 2 else None


def lines(geom):
    if geom["type"] == "LineString":
        return [geom["coordinates"]]
    if geom["type"] == "MultiLineString":
        return geom["coordinates"]
    return []


def rings(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    return []


def rivers(cache: Path):
    feats = []
    for name in RIVER_FILES:
        data = json.loads(fetch(NE + name + ".geojson", cache))
        for f in data["features"]:
            if not f["geometry"]:
                continue
            p = f["properties"]
            parts = [thin(c, 0.01) for c in lines(f["geometry"]) if touches(c)]
            parts = [c for c in parts if c]
            if not parts:
                continue
            feats.append({"type": "Feature",
                          "properties": {"name": p.get("name") or "",
                                         "rank": p.get("scalerank"),
                                         "kind": p.get("featurecla"),
                                         "src": name},
                          "geometry": {"type": "MultiLineString", "coordinates": parts}})
    return feats


def lakes(cache: Path):
    feats = []
    for name in LAKE_FILES:
        data = json.loads(fetch(NE + name + ".geojson", cache))
        for f in data["features"]:
            if not f["geometry"]:
                continue
            p = f["properties"]
            polys = []
            for poly in rings(f["geometry"]):
                if not touches(poly[0]):
                    continue
                kept = [thin(r, 0.005, closed=True) for r in poly]
                if kept[0]:
                    polys.append([r for r in kept if r])
            if polys:
                feats.append({"type": "Feature",
                              "properties": {"name": p.get("name") or "",
                                             "kind": p.get("featurecla"),
                                             "rank": p.get("scalerank")},
                              "geometry": {"type": "MultiPolygon", "coordinates": polys}})
    return feats


def basins(cache: Path):
    z = zipfile.ZipFile(io.BytesIO(fetch(HYBAS, cache)))
    rows = read_dbf(z.read("hybas_eu_lev04_v1c.dbf"))
    feats = []
    for row, (bbox, parts) in zip(rows, read_shp(z.read("hybas_eu_lev04_v1c.shp"))):
        if not bbox or bbox[2] < FRAME[0] or bbox[0] > FRAME[2] or bbox[3] < FRAME[1] \
                or bbox[1] > FRAME[3]:
            continue
        kept = [thin(r, 0.03, closed=True) for r in parts]
        kept = [r for r in kept if r]
        if not kept:
            continue
        feats.append({"type": "Feature",
                      "properties": {k: row[k] for k in ("HYBAS_ID", "MAIN_BAS", "NEXT_DOWN",
                                                         "SUB_AREA", "UP_AREA", "ENDO")},
                      "geometry": {"type": "Polygon", "coordinates": kept}})
    return feats


def write(name, feats):
    path = OUT / name
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                               separators=(",", ":")), encoding="utf8")
    print(f"Wrote {path.name}: {len(feats)} features, {path.stat().st_size // 1024} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path(tempfile.gettempdir()) / "sector_briefs_water")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    write("water_rivers.geojson", rivers(args.cache))
    write("water_lakes.geojson", lakes(args.cache))
    write("water_basins.geojson", basins(args.cache))


if __name__ == "__main__":
    main()
