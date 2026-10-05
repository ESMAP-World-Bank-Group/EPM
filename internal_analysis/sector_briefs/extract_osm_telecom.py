"""Extract traced telecom cables in the map window from the worldwide OSM GeoPackage.

OpenStreetMap maps fibre routes where volunteers or operators traced them: dense in
Turkiye, Romania and Bulgaria, nearly absent in the Caucasus. The brief uses these
routes in place of the straight ITU links they cover. Stdlib only: the GeoPackage
is SQLite, its geometries are WKB behind a short header.

Usage
    python extract_osm_telecom.py [path to worldwide.gpkg]
"""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
from pathlib import Path

from power import BBOX  # the ad hoc maps use the power window, wider than mapkit.BBOX

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "osm_telecom_cables.geojson"
GPKG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[3] / "maps" / "worldwide.gpkg"
PAD = 1.0  # degrees around the window
ENVELOPE = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def wkb_lines(b: bytes, o: int = 0):
    order = "<" if b[o] == 1 else ">"
    kind = struct.unpack(order + "I", b[o + 1:o + 5])[0] % 1000
    o += 5
    n = struct.unpack(order + "I", b[o:o + 4])[0]
    o += 4
    if kind == 2:
        pts = [list(struct.unpack(order + "dd", b[o + 16 * i:o + 16 * i + 16])) for i in range(n)]
        return [pts], o + 16 * n
    out = []
    if kind == 5:
        for _ in range(n):
            lines, o = wkb_lines(b, o)
            out += lines
    return out, o


def lines_of_gpkg(blob: bytes):
    return wkb_lines(blob, 8 + ENVELOPE[(blob[3] >> 1) & 7])[0]


def main():
    x0, y0, x1, y1 = BBOX[0] - PAD, BBOX[1] - PAD, BBOX[2] + PAD, BBOX[3] + PAD
    con = sqlite3.connect(f"file:{GPKG.as_posix()}?mode=ro", uri=True)
    ids = [r[0] for r in con.execute(
        "select id from rtree_telecom_cable_geometry "
        "where maxx >= ? and minx <= ? and maxy >= ? and miny <= ?", (x0, x1, y0, y1))]
    feats = []
    for blob, osm_id, name, operator in con.execute(
            f"select geometry, id, name, operator from telecom_cable "
            f"where fid in ({','.join(map(str, ids))})"):
        lines = [[[round(x, 5), round(y, 5)] for x, y in line]
                 for line in lines_of_gpkg(blob) if len(line) > 1]
        if lines:
            feats.append({"type": "Feature",
                          "properties": {"id": osm_id, "name": name, "operator": operator},
                          "geometry": {"type": "MultiLineString", "coordinates": lines}})
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf8")
    print(f"{len(feats)} cables -> {OUT}")


if __name__ == "__main__":
    main()
