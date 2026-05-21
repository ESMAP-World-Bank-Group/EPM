"""
Create Romania 2-zone study: Dobrogea (Constanța + Tulcea) vs. rest.

Reads Natural Earth admin-1 polygons, groups into two zones, and writes the
standard EPM zoning-study outputs under:
  pre-analysis/output_workflow/zoning_study/ROU_2z/

Outputs mirror the zone_pipeline format:
  epm_export/spatial/
    zones.geojson          zone polygons (zone_name, zone_id, ISO_A3)
    zcmap.csv              z, c columns
    sTopology.csv          single link: Dobrogea <-> Romania
  report/spatial/
    zone_map_simplified.html  interactive Folium map

Usage:
    conda activate gams_env
    cd pre-analysis
    python create_romania_2z.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

_BASE    = Path(__file__).resolve().parent          # pre-analysis/
_NE_SHP  = _BASE / "resolution_advisor" / "cache" / "natural_earth" / "ne_10m_admin_1_states_provinces.shp"
OUT_ROOT = _BASE / "output_workflow" / "zoning_study" / "ROU_2z"
EPM_DIR  = OUT_ROOT / "epm_export" / "spatial"
REP_DIR  = OUT_ROOT / "report"     / "spatial"

# Dobrogea = these two counties
DOBROGEA_HASC = {"RO.CT", "RO.TL"}   # Constanța + Tulcea
DOBROGEA_NAME = "Dobrogea"
REST_NAME     = "Romania"


def build_zones() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(_NE_SHP)
    rou = gdf[gdf["adm0_a3"] == "ROU"].copy()
    if rou.empty:
        raise RuntimeError("No Romania features found in Natural Earth admin-1 shapefile.")

    dobrogea_parts = rou[rou["code_hasc"].isin(DOBROGEA_HASC)]
    rest_parts     = rou[~rou["code_hasc"].isin(DOBROGEA_HASC)]

    if dobrogea_parts.empty:
        raise RuntimeError(f"Could not find HASC codes {DOBROGEA_HASC} in shapefile.")

    dobrogea_geom = unary_union(dobrogea_parts.geometry)
    rest_geom     = unary_union(rest_parts.geometry)

    rows = [
        {"zone_id": 0, "zone_name": DOBROGEA_NAME, "ISO_A3": "ROU", "geometry": dobrogea_geom},
        {"zone_id": 1, "zone_name": REST_NAME,      "ISO_A3": "ROU", "geometry": rest_geom},
    ]
    zones_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    print(f"  {len(dobrogea_parts)} counties -> {DOBROGEA_NAME}")
    print(f"  {len(rest_parts)} counties -> {REST_NAME}")
    return zones_gdf


def write_epm(zones_gdf: gpd.GeoDataFrame):
    EPM_DIR.mkdir(parents=True, exist_ok=True)

    # zones.geojson
    geojson_path = EPM_DIR / "zones.geojson"
    zones_gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"  zones.geojson")

    # zcmap.csv
    zcmap = zones_gdf[["zone_name", "ISO_A3"]].copy()
    zcmap.columns = ["z", "c"]
    zcmap.to_csv(EPM_DIR / "zcmap.csv", index=False)
    print(f"  zcmap.csv")

    # sTopology.csv  — single link (Dobrogea shares border with Romania)
    topo = pd.DataFrame([{"z": DOBROGEA_NAME, "zz": REST_NAME}])
    topo.to_csv(EPM_DIR / "sTopology.csv", index=False)
    print(f"  sTopology.csv")

    # pTransferLimit_estimated.csv — placeholder (no reference data)
    transfer = pd.DataFrame([{
        "z": DOBROGEA_NAME, "zz": REST_NAME,
        "pTransferLimit": 2000,
        "note": "placeholder — update with actual NTC data",
    }])
    transfer.to_csv(EPM_DIR / "pTransferLimit_estimated.csv", index=False)
    print(f"  pTransferLimit_estimated.csv (placeholder 2000 MW)")


def write_map(zones_gdf: gpd.GeoDataFrame):
    REP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import folium
    except ImportError:
        print("  [!] folium not installed — skipping map")
        return

    centroids = {
        row["zone_name"]: (row.geometry.centroid.y, row.geometry.centroid.x)
        for _, row in zones_gdf.iterrows()
    }
    center_lat = sum(v[0] for v in centroids.values()) / len(centroids)
    center_lon = sum(v[1] for v in centroids.values()) / len(centroids)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6,
                   tiles="CartoDB positron")
    colors = ["#4e79a7", "#f28e2b"]

    for i, (_, row) in enumerate(zones_gdf.iterrows()):
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda f, c=colors[i]: {
                "fillColor": c, "color": "#333",
                "weight": 1.5, "fillOpacity": 0.35,
            },
            tooltip=row["zone_name"],
        ).add_to(m)
        lat, lon = centroids[row["zone_name"]]
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:12px;font-weight:bold;color:#111;'
                     f'background:rgba(255,255,255,0.8);padding:3px 6px;'
                     f'border-radius:4px">{row["zone_name"]}</div>',
                icon_size=(140, 28), icon_anchor=(70, 14),
            ),
        ).add_to(m)

    # Link between the two zones
    z1, z2 = DOBROGEA_NAME, REST_NAME
    if z1 in centroids and z2 in centroids:
        folium.PolyLine(
            [centroids[z1], centroids[z2]],
            color="#1a5fa8", weight=3, opacity=0.8,
            tooltip=f"{z1} <-> {z2}",
        ).add_to(m)

    out = REP_DIR / "zone_map_simplified.html"
    m.save(str(out))
    print(f"  zone_map_simplified.html")


def update_index(zones_gdf: gpd.GeoDataFrame):
    index_path = OUT_ROOT.parent / "index.json"
    idx: dict = {}
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)

    entry = {
        "n_zones": 2,
        "status": "ok",
        "source": "Natural Earth admin-1, Dobrogea = Constanta+Tulcea",
        "path": str(OUT_ROOT),
        "outputs": {
            "zones.geojson": str(EPM_DIR / "zones.geojson"),
            "zcmap.csv":     str(EPM_DIR / "zcmap.csv"),
            "sTopology.csv": str(EPM_DIR / "sTopology.csv"),
        },
    }
    idx.setdefault("ROU", [])
    idx["ROU"] = [r for r in idx["ROU"] if r.get("n_zones") != 2]
    idx["ROU"].append(entry)
    idx["ROU"].sort(key=lambda r: r["n_zones"])

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)
    print(f"  index.json updated")


if __name__ == "__main__":
    print(f"\nBuilding ROU_2z (Dobrogea vs. Romania)...\n")

    print("[1/4] Building zone polygons...")
    zones_gdf = build_zones()

    print("[2/4] Writing EPM export files...")
    write_epm(zones_gdf)

    print("[3/4] Generating map...")
    write_map(zones_gdf)

    print("[4/4] Updating index.json...")
    update_index(zones_gdf)

    print(f"\nDone. Output: {OUT_ROOT}")
