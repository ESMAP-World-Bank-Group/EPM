"""
add_romania_zone_polygon.py
Adds the Romania polygon to epm/input/data_blacksea/zones.geojson,
sourcing the geometry from the Natural Earth file in epm/resources/.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NE_SRC = ROOT / "epm/resources/postprocess/zones.geojson"
DST    = ROOT / "epm/input/data_blacksea/zones.geojson"


def ring_area(ring):
    n = len(ring); a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += ring[i][0] * ring[j][1] - ring[j][0] * ring[i][1]
    return abs(a) / 2.0


def simplify(geom):
    """Keep largest polygon, drop holes."""
    if geom["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [geom["coordinates"][0]]}
    best = max(geom["coordinates"], key=lambda p: ring_area(p[0]))
    return {"type": "Polygon", "coordinates": [best[0]]}


ne = json.loads(NE_SRC.read_text(encoding="utf-8"))
rou_feat = next(f for f in ne["features"] if f["properties"].get("ISO_A3") == "ROU")
rou_geom = simplify(rou_feat["geometry"])
print(f"Romania Natural Earth polygon: {len(rou_geom['coordinates'][0])} pts (outer ring)")

gj = json.loads(DST.read_text(encoding="utf-8"))
before = len(gj["features"])
gj["features"] = [f for f in gj["features"] if f["properties"].get("z") != "Romania"]
gj["features"].append({
    "type": "Feature",
    "properties": {"z": "Romania", "c": "Romania", "ISO_A3": "ROU"},
    "geometry": rou_geom,
})
DST.write_text(json.dumps(gj, separators=(",", ":")), encoding="utf-8")

print(f"zones.geojson: {before} → {len(gj['features'])} features")
for f in gj["features"]:
    props = f["properties"]
    npts  = len(f["geometry"]["coordinates"][0])
    print(f"  {props['z']:25s} {props['ISO_A3']}  {npts} pts")
