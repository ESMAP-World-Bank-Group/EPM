"""Statistics for the review of the Digital team's infrastructure data.

Reads the GeoPackage the Digital team delivered on 2026-10-13 and the web files
of pull request #43 (both in data/, gitignored), plus the open layers already
downloaded for the September screening note (OpenStreetMap HV lines and
railways, the ITU catalogue as fetched then). compute() returns one dict of
figures and the classified grid geometry; build_overview.py renders it.

All distances are measured in the UTM zone of each country.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GPKG = HERE / "data" / "rpe_digital_infra_2026-10-13.gpkg"
# The web files of pull request #43, copied from the branch.
WEB = HERE / "data" / "web"
OPEN = HERE.parent / "digital_connectivity" / "data"
BASEMAP = ROOT / "epm" / "input" / "data_blacksea" / "extras" / "background_countries.geojson"

COUNTRIES = {"GEO": "Georgia", "ARM": "Armenia", "AZE": "Azerbaijan", "MDA": "Moldova",
             "UKR": "Ukraine"}
# Comparators for the ITU indicators table, study countries first.
INDICATOR_ROWS = ["TUR", "GEO", "ARM", "AZE", "MDA", "UKR", "ROU", "BGR"]
# Co-location of the HV grid with fibre, per country, as published in section 7.2
# of the September screening note (ITU catalogue as fetched then).
SEPT_SHARE = {"GEO": 31, "ARM": 53, "AZE": 36, "MDA": 23, "UKR": 14}
# Outdoor coverage as reported in the delivery README, on World Bank boundaries.
COVERAGE_PCT = {"2g": 94.4, "3g": 97.5, "4g": 96.1, "5g": 2.4}
NEAR_GRID_KM = 3.0   # same threshold as section 7.2 of the screening note
ALONG_KM = 1.0       # a fibre route "follows" a corridor when within this distance


def layer(name: str) -> gpd.GeoDataFrame:
    return gpd.read_file(GPKG, layer=name)


def km(gdf: gpd.GeoDataFrame) -> float:
    return float(gdf.length.sum() / 1000.0) if len(gdf) else 0.0


def countries() -> gpd.GeoDataFrame:
    base = gpd.read_file(BASEMAP)
    base = base[base["name"].isin(COUNTRIES.values())].copy()
    base["iso"] = base["name"].map({v: k for k, v in COUNTRIES.items()})
    return base.dissolve("iso").reset_index()[["iso", "geometry"]]


def clip_to(gdf: gpd.GeoDataFrame, poly, crs) -> gpd.GeoDataFrame:
    out = gpd.clip(gdf, poly)
    out = out[~out.geometry.is_empty & out.geometry.notna()]
    return out.to_crs(crs)


def pieces(lines: gpd.GeoDataFrame, step_m: float = 250.0) -> np.ndarray:
    """Cut every line into two point segments of at most step_m."""
    parts = shapely.get_parts(lines.geometry.values)
    parts = shapely.segmentize(parts[shapely.get_type_id(parts) == 1], step_m)
    coords, idx = shapely.get_coordinates(parts, return_index=True)
    same = idx[:-1] == idx[1:]
    return shapely.linestrings(np.stack([coords[:-1][same], coords[1:][same]], axis=1))


def near_mask(segs: np.ndarray, other: gpd.GeoDataFrame, dist_km: float) -> np.ndarray:
    hit = np.zeros(len(segs), dtype=bool)
    if len(segs) and not other.empty:
        tree = shapely.STRtree(shapely.get_parts(other.geometry.values))
        hit[np.unique(tree.query(segs, predicate="dwithin", distance=dist_km * 1000.0)[0])] = True
    return hit


def near_share(lines: gpd.GeoDataFrame, other: gpd.GeoDataFrame, dist_km: float) -> float:
    """Length of `lines` within dist_km of `other`, both in a metric CRS.

    Measured on 250 m pieces, so it is exact to about one piece per run.
    Buffering and dissolving 65,000 Orange segments does not finish.
    """
    if lines.empty or other.empty:
        return 0.0
    segs = pieces(lines)
    return float(shapely.length(segs[near_mask(segs, other, dist_km)]).sum() / 1000.0)


def classified_grid(grid: gpd.GeoDataFrame, fibre: gpd.GeoDataFrame, crs) -> dict:
    """The HV lines split into the parts with and without mapped fibre nearby,
    merged back into lines and returned in lon/lat for the maps."""
    segs = pieces(grid, step_m=500.0)
    hit = near_mask(segs, fibre, NEAR_GRID_KM)
    out = {}
    for key, mask in (("near", hit), ("far", ~hit)):
        merged = shapely.line_merge(shapely.union_all(segs[mask])) if mask.any() else None
        out[key] = gpd.GeoSeries([merged], crs=crs).to_crs(4326).iloc[0] if merged else None
    return out


def compute() -> dict:
    out: dict = {"countries": {}, "grid": {}}
    polys = countries()
    backbone = layer("itu_backbone")
    nodes = layer("itu_nodes")
    facilities = layer("facilities_itu")
    opennet = layer("opennet_backbone")
    pops = layer("opennet_pop")
    orange = layer("orange_fibre")
    hv = gpd.read_file(OPEN / "osm_hv_lines.geojson")
    rail = gpd.read_file(OPEN / "osm_railways.geojson")
    itu_sept = gpd.read_file(OPEN / "itu_fibre.geojson")

    # One clip to the five countries keeps the per country clips fast.
    frame = polys.total_bounds
    rail = rail.cx[frame[0]:frame[2], frame[1]:frame[3]]
    hv = hv.cx[frame[0]:frame[2], frame[1]:frame[3]]
    # Cable names carry the network tier: FTTH and PON are the access network.
    orange["access"] = orange["cable_name"].fillna("").str.upper().str.contains("FTTH|PON")

    for iso, name in COUNTRIES.items():
        poly = polys.loc[polys.iso == iso, "geometry"].iloc[0]
        crs = gpd.GeoSeries([poly], crs=4326).estimate_utm_crs()
        bb = backbone[backbone.iso == iso].to_crs(crs)
        row = {
            "name": name,
            "itu_km": {s: float(g["km"].sum()) for s, g in bb.groupby("status")},
            "itu_segments": int(len(bb)),
            "itu_nodes": int((nodes.iso == iso).sum()),
            "itu_operators": bb.groupby("operator")["km"].sum().sort_values(ascending=False)
                               .round(0).head(3).to_dict(),
            "itu_years": bb["golive"].dropna().str[:4].value_counts().to_dict(),
            "cross_border": int(bb["cross_border"].fillna(False).astype(bool).sum()),
            "landings": int(((facilities.iso == iso) & (facilities.kind == "landing_point")).sum()),
            "earth_stations": int(((facilities.iso == iso) & (facilities.kind == "earth_station")).sum()),
            "sept_share": SEPT_SHARE[iso],
        }
        # The ITU catalogue as fetched for the September note, same country polygon.
        sept = clip_to(itu_sept[itu_sept["status"] == "Operational"], poly, crs)
        row["itu_sept_km"] = km(sept)

        grid = clip_to(hv, poly, crs)
        rails = clip_to(rail, poly, crs)
        fibre = bb[["geometry"]]
        row["hv_km"] = km(grid)
        row["hv_near_itu_km"] = near_share(grid, bb, NEAR_GRID_KM)
        if iso == "GEO":
            on = opennet.to_crs(crs)
            fibre = pd.concat([fibre, on[["geometry"]]])
            row["hv_near_partner_km"] = near_share(grid, on, NEAR_GRID_KM)
            row["partner_km"] = float(on["km"].sum())
            row["partner_along_hv_km"] = near_share(on, grid, ALONG_KM)
            row["partner_along_rail_km"] = near_share(on, rails, ALONG_KM)
        if iso == "MDA":
            cables = orange[~orange.access].to_crs(crs)
            fibre = pd.concat([fibre, cables[["geometry"]]])
            row["hv_near_partner_km"] = near_share(grid, cables, NEAR_GRID_KM)
            row["partner_km"] = float(orange["km"].sum())
            row["partner_access_km"] = float(orange.loc[orange.access, "km"].sum())
        row["hv_near_any_km"] = near_share(grid, fibre, NEAR_GRID_KM)
        if iso != "UKR":  # the OSM extract stops near 49.8 N, north Ukraine is outside
            out["grid"][iso] = classified_grid(grid, fibre, crs)
        out["countries"][iso] = row

    out["opennet"] = {
        "km_by_status": opennet.groupby("status")["km"].sum().round(0).to_dict(),
        "pops_by_status": pops.groupby("status").size().to_dict(),
        "pops_by_type": pops.groupby("pop_type").size().to_dict(),
        "population": int(pops["population"].fillna(0).sum()),
        "households": int(pops["households"].fillna(0).sum()),
        "available": pops["available"].fillna("n/a").value_counts().to_dict(),
    }
    out["orange"] = {
        "km": float(orange["km"].sum()),
        "segments": int(len(orange)),
        "access_km": float(orange.loc[orange.access, "km"].sum()),
        "named": int(orange["cable_name"].nunique()),
        "coverage_pct": COVERAGE_PCT,
    }
    out["cables"] = layer("submarine_cables").drop(columns="geometry").to_dict("records")
    data = json.loads((WEB / "itu_indicators.json").read_text(encoding="utf8"))["countries"]
    out["indicators"] = {iso: data.get(iso, {}) for iso in INDICATOR_ROWS}
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    stats = compute()
    stats.pop("grid")
    print(json.dumps(stats, indent=1, default=str))
