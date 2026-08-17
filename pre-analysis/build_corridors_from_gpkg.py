"""
Build inter-zone corridors for the Explorer from the local OSM GeoPackage.

Why this exists
---------------
`pipelines/zone_pipeline.py` sources OSM substations/HV lines from the Overpass
API, which is unreachable from the WB network (all mirrors time out). As a
result corridors were only ever generated for the Black Sea study, where the
Overpass responses had been cached.

The zone polygons themselves already exist in the Explorer for 29 countries, so
re-clustering is neither needed nor desirable (it would move existing zones).
Only the corridors are missing: the inter-zone HV links plus their capacity.

This script reads HV lines from maps/worldwide.gpkg (global OSM extract) instead
of Overpass, and reuses the *same* capacity methodology as the Black Sea run
(transmission_capacity.build_corridors → min(thermal, stability) per line,
summed per corridor), so the numbers stay comparable across regions.

Usage:
    python pre-analysis/build_corridors_from_gpkg.py --countries ZMB
    python pre-analysis/build_corridors_from_gpkg.py --sapp
    python pre-analysis/build_corridors_from_gpkg.py --sapp --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

_BASE = Path(__file__).resolve().parent                 # pre-analysis/
sys.path.insert(0, str(_BASE))

from pipelines.transmission_capacity import (  # noqa: E402
    build_corridors,
    export_corridors_geojson,
)

# ── Paths ────────────────────────────────────────────────────────────────────
# Walk up to EPM_Models/ rather than relying on brittle parents[] indexing.
def _find_models_root() -> Path:
    for p in _BASE.parents:
        if (p / "maps" / "worldwide.gpkg").exists():
            return p
    raise SystemExit("Could not locate EPM_Models root (maps/worldwide.gpkg not found)")


MODELS_ROOT = _find_models_root()
GPKG = MODELS_ROOT / "maps" / "worldwide.gpkg"

EXPLORER_ZONES = [
    MODELS_ROOT / "regional-power-explorer" / "public" / "data" / "zones",
]

SAPP = ["AGO", "BWA", "COD", "LSO", "MWI", "MOZ", "NAM", "ZAF", "SWZ", "TZA", "ZMB", "ZWE"]

MIN_VOLTAGE_KV = 100      # same threshold as fetch_hv_lines(min_voltage_kv=100)
SNAP_DEG = 0.05           # same endpoint tolerance as zone_pipeline._point_in_zone


# ── OSM lines from the GeoPackage ────────────────────────────────────────────

def load_hv_lines(bbox) -> gpd.GeoDataFrame:
    """HV lines (>= MIN_VOLTAGE_KV) inside bbox, from the local OSM extract.

    OSM stores voltage in volts; transmission_capacity expects kV.
    """
    g = gpd.read_file(GPKG, layer="power_line", bbox=bbox, engine="pyogrio")
    if g.empty:
        return g
    g = g[g.geometry.geom_type == "LineString"].copy()
    g["voltage_kv"] = pd.to_numeric(g["max_voltage"], errors="coerce") / 1000.0
    g = g[g["voltage_kv"] >= MIN_VOLTAGE_KV].copy()
    g["n_circuits"] = pd.to_numeric(g["circuits"], errors="coerce").fillna(1).clip(lower=1)
    return g


def assign_endpoints_to_zones(lines: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, which: str):
    """Map each line endpoint to a zone_name (within, else nearest <= SNAP_DEG)."""
    pts = lines.geometry.apply(
        lambda ls: Point(ls.coords[0] if which == "start" else ls.coords[-1])
    )
    gdf = gpd.GeoDataFrame({"_i": lines.index}, geometry=list(pts), crs=zones.crs)

    joined = gpd.sjoin(gdf, zones[["zone_name", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    out = joined["zone_name"].copy()

    # Endpoints just outside a polygon (coastline/border slivers) → nearest zone.
    missing = out.isna()
    if missing.any():
        near = gpd.sjoin_nearest(
            gdf.loc[missing, ["geometry"]], zones[["zone_name", "geometry"]],
            how="left", max_distance=SNAP_DEG,
        )
        near = near[~near.index.duplicated(keep="first")]
        out.loc[missing] = near["zone_name"]
    return out


def _zones_spanned(lines: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> dict:
    """line index -> ordered list of zone_names whose polygon the line intersects.

    Endpoint testing (what zone_pipeline does) badly under-detects here: OSM
    splits a circuit into many ways, and a way usually lies wholly inside one
    zone, so neither endpoint ever "changes" zone. Testing the full geometry
    against the polygons instead catches the way that actually straddles the
    border. On SAPP this recovers COD and LSO, which endpoints missed entirely.
    """
    j = gpd.sjoin(lines[["geometry"]], zones[["zone_name", "geometry"]],
                  how="inner", predicate="intersects")
    spanned: dict = {}
    for idx, name in zip(j.index, j["zone_name"]):
        spanned.setdefault(idx, [])
        if name not in spanned[idx]:
            spanned[idx].append(name)
    return spanned


def build_interzone_lines(lines: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> list[dict]:
    """Lines intersecting two or more zones = inter-zone links."""
    if lines.empty:
        return []
    spanned = _zones_spanned(lines, zones)
    zone_geom = dict(zip(zones["zone_name"], zones.geometry))

    out = []
    for idx, names in spanned.items():
        if len(names) < 2:
            continue
        geom = lines.geometry.loc[idx]
        if len(names) > 2:
            # Rare: keep the two zones carrying the longest share of the line.
            names = sorted(
                names, key=lambda z: geom.intersection(zone_geom[z]).length, reverse=True
            )[:2]
        row = lines.loc[idx]
        out.append({
            "zone_from": names[0],
            "zone_to": names[1],
            "voltage_kv": float(row["voltage_kv"]),
            "n_circuits": int(row["n_circuits"]),
            "coords": list(geom.coords),
        })
    return out


# ── Per country/zone-count run ───────────────────────────────────────────────

def run_one(iso: str, n: int, dry_run: bool = False) -> dict | None:
    label = f"{iso}_{n}z"
    zones_path = EXPLORER_ZONES[0] / f"{label}_zones.geojson"
    if not zones_path.exists():
        print(f"  {label}: no zones file - skipped")
        return None

    zones = gpd.read_file(zones_path)
    if len(zones) < 2:
        print(f"  {label}: single zone - no inter-zone corridor by definition, skipped")
        return None

    minx, miny, maxx, maxy = zones.total_bounds
    lines = load_hv_lines((minx - 0.2, miny - 0.2, maxx + 0.2, maxy + 0.2))
    interzone = build_interzone_lines(lines, zones)
    if not interzone:
        print(f"  {label}: {len(lines)} HV lines, 0 inter-zone crossings - no corridors")
        return None

    corridors = build_corridors(interzone, zones)
    if corridors is None or corridors.empty:
        print(f"  {label}: no corridors built")
        return None

    mws = [int(m) for m in corridors["mw_osm"].fillna(0)]
    print(f"  {label}: {len(lines):>5} HV lines -> {len(interzone):>4} crossings "
          f"-> {len(corridors)} corridors  (MW: {sorted(mws, reverse=True)[:5]})")

    if not dry_run:
        for zdir in EXPLORER_ZONES:
            if zdir.exists():
                export_corridors_geojson(corridors, zones, zdir / f"{label}_corridors.geojson")
    return {"label": label, "n_corridors": len(corridors)}


def main():
    ap = argparse.ArgumentParser(description="Build Explorer corridors from local OSM GeoPackage")
    ap.add_argument("--countries", nargs="+", metavar="ISO", help="ISO_A3 codes")
    ap.add_argument("--sapp", action="store_true", help="all SAPP countries")
    ap.add_argument("--min-zones", type=int, default=2,
                    help="skip zone counts below this (1z has no corridors; default 2)")
    ap.add_argument("--dry-run", action="store_true", help="compute but do not write")
    args = ap.parse_args()

    isos = SAPP if args.sapp else (args.countries or [])
    if not isos:
        ap.error("pass --countries ISO [ISO ...] or --sapp")

    index_path = EXPLORER_ZONES[0] / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))

    print(f"GeoPackage : {GPKG}")
    print(f"Targets    : {[str(p) for p in EXPLORER_ZONES if p.exists()]}")
    print(f"Countries  : {isos}\n")

    written = []
    for iso in isos:
        counts = [n for n in index.get(iso, []) if n >= args.min_zones]
        if not counts:
            print(f"{iso}: nothing to do (index: {index.get(iso)})")
            continue
        print(f"{iso}:")
        for n in counts:
            r = run_one(iso, n, args.dry_run)
            if r:
                written.append(r["label"])

    print(f"\n{'[dry-run] would write' if args.dry_run else 'Wrote'} {len(written)} corridor files.")


if __name__ == "__main__":
    main()
