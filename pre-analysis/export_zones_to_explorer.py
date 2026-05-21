"""
Export zoning study results + load centers to EPM Explorer.

Usage:
    gams_env python pre-analysis/export_zones_to_explorer.py
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path


_BASE   = Path(__file__).resolve().parent   # pre-analysis/
_REPO   = _BASE.parent                       # EPM root  (…/black_sea_2026/EPM)
_EXPLORER = _REPO.parents[1] / "regional-power-explorer"   # …/EPM_Models/regional-power-explorer

sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE / "resolution_advisor"))

STUDY_ROOT = _BASE / "output_workflow" / "zoning_study"
ZONES_OUT  = _EXPLORER / "public" / "data" / "zones"
BSC_COUNTRIES = ["TUR", "ARM", "GEO", "AZE", "ROU", "BGR"]  # noqa: kept for reference


# ── 1. Zone polygons + topology ──────────────────────────────────────────────

def export_zones():
    import geopandas as gpd
    from shapely.ops import unary_union

    ZONES_OUT.mkdir(parents=True, exist_ok=True)

    # Load the reference land boundaries (same file used by CountryPage)
    countries_10m_path = _EXPLORER / "public" / "data" / "countries_10m.geojson"
    countries_10m = gpd.read_file(countries_10m_path) if countries_10m_path.exists() else None

    index_path = STUDY_ROOT / "index.json"
    with open(index_path, encoding="utf-8") as f:
        study_index = json.load(f)

    export_index: dict[str, list[int]] = {}

    for iso, runs in study_index.items():
        # Get the reference country boundary to clip against
        clip_geom = None
        if countries_10m is not None:
            match = countries_10m[countries_10m["ISO_A3"] == iso]
            if not match.empty:
                clip_geom = unary_union(match.geometry)

        available = []
        for run in runs:
            if run["status"] != "ok":
                continue
            n = run["n_zones"]
            label = f"{iso}_{n}z"
            src_dir = Path(run["path"]) / "epm_export" / "spatial"

            # zones.geojson + inner.geojson (shared edges only — no outer border)
            zones_src = src_dir / "zones.geojson"
            if zones_src.exists():
                gdf = gpd.read_file(zones_src)
                if clip_geom is not None:
                    gdf = gdf.clip(clip_geom)
                gdf.to_file(ZONES_OUT / f"{label}_zones.geojson", driver="GeoJSON")
                print(f"  {label}_zones.geojson")

                # Internal borders = all zone boundaries minus the outer country boundary
                # Buffer the outer boundary slightly to handle floating-point gaps
                all_bounds = unary_union([g.boundary for g in gdf.geometry])
                country_union = unary_union(gdf.geometry)
                outer_bound = country_union.boundary if country_union is not None and not country_union.is_empty else None
                if outer_bound is None:
                    inner_geom = all_bounds
                else:
                    inner_geom = all_bounds.difference(outer_bound.buffer(1e-6))
                inner_lines = list(inner_geom.geoms) if hasattr(inner_geom, 'geoms') else ([inner_geom] if not inner_geom.is_empty else [])
                inner_gdf = gpd.GeoDataFrame(geometry=inner_lines, crs=gdf.crs)
                inner_gdf.to_file(ZONES_OUT / f"{label}_inner.geojson", driver="GeoJSON")
                print(f"  {label}_inner.geojson  ({len(inner_lines)} edges)")

            # sTopology.csv -> topo.json
            topo_src = src_dir / "sTopology.csv"
            if topo_src.exists():
                with open(topo_src, encoding="utf-8") as cf:
                    topo = [{"z": r["z"], "zz": r["zz"]} for r in csv.DictReader(cf)]
                with open(ZONES_OUT / f"{label}_topo.json", "w", encoding="utf-8") as tf:
                    json.dump(topo, tf)
                print(f"  {label}_topo.json  ({len(topo)} links)")

            available.append(n)

        if available:
            export_index[iso] = sorted(available)

    with open(ZONES_OUT / "index.json", "w", encoding="utf-8") as f:
        json.dump(export_index, f, indent=2)
    print(f"\n  index.json: {export_index}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Explorer target: {_EXPLORER}\n")
    export_zones()
    print("\nDone.")
