"""
Export all named zoning configs to merged GeoJSON files for the Explorer.

Configs defined in zoning_configs.json: {name: {ISO: n_zones, ...}, ...}

Outputs (output_workflow/preferred/):
  blacksea_{slug}_zones_hd.geojson   -- HD zone polygons for Explorer
  blacksea_{slug}_corridors.geojson  -- inter-zone corridors for Explorer
  maps/blacksea_{slug}_map.html      -- Folium interactive map
  blacksea_configs.json              -- index written to Explorer zones/

Data sources: OpenStreetMap (substations, HV lines, power plants),
              Natural Earth (cities). Not EPM model results.

Usage:
    conda run -n gams_env python pre-analysis/export_preferred.py
"""
from __future__ import annotations
import csv
import json
import math
import sys
from pathlib import Path
from typing import Tuple

_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE / "resolution_advisor"))

from fetch.osm import fetch_substations, fetch_hv_lines, fetch_generators
from fetch.natural_earth import load_cities
from pipelines.zone_pipeline import _find_interzone_lines
from pipelines.transmission_capacity import build_corridors, _effective_mw, export_corridors_geojson

_REFERENCE_LINES     = _BASE / "data" / "reference_lines.csv"
_REFERENCE_CORRIDORS = _BASE / "data" / "reference_corridors.csv"

STUDY_ROOT      = _BASE / "output_workflow" / "zoning_study"
OUT_DIR         = _BASE / "output_workflow" / "preferred"
MAPS_DIR        = OUT_DIR / "maps"
CONFIGS_PATH    = _BASE / "zoning_configs.json"
_EXPLORER       = _BASE.parent.parents[1] / "regional-power-explorer"
_EXPLORER_ZONES = _EXPLORER / "public" / "data" / "zones"


def _slug(name: str) -> str:
    """'Turkey 9z' -> 'turkey9z'"""
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower())

_COUNTRY_COLORS = {
    "TUR": "#E8A87C",
    "ROU": "#82C0E8",
    "ARM": "#A8D8A8",
    "AZE": "#E8D87C",
    "BGR": "#C8A0E8",
    "GEO": "#E88080",
}

_FUEL_COLORS = {
    "Hydro":       "#4169E1",
    "Gas":         "#FF8C00",
    "Wind":        "#66D9E8",
    "Solar":       "#F4C430",
    "Coal":        "#696969",
    "Nuclear":     "#9370DB",
    "Oil":         "#8B4513",
    "Biomass":     "#228B22",
    "Geothermal":  "#DC143C",
    "Storage":     "#20B2AA",
    "Other":       "#A9A9A9",
}


# ── 1. Config ──────────────────────────────────────────────────────────────────

def load_all_configs() -> dict[str, dict[str, int]]:
    """Returns {name: {ISO: n_zones}} from zoning_configs.json."""
    with open(CONFIGS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── 2. Zone data ───────────────────────────────────────────────────────────────

def load_zone_data(config: dict[str, int]):
    import geopandas as gpd
    import pandas as pd

    zones_gdfs = []
    zcmap_rows: list[dict] = []
    topo_rows:  list[dict] = []

    for iso, n in config.items():
        run_dir = STUDY_ROOT / f"{iso}_{n}z" / "epm_export" / "spatial"
        if not run_dir.exists():
            print(f"  [WARN] {iso}_{n}z not found, skipping")
            continue

        gdf = gpd.read_file(run_dir / "zones.geojson")
        gdf["country"] = iso
        zones_gdfs.append(gdf)

        with open(run_dir / "zcmap.csv", encoding="utf-8") as f:
            zcmap_rows.extend(csv.DictReader(f))

        with open(run_dir / "sTopology.csv", encoding="utf-8") as f:
            topo_rows.extend(csv.DictReader(f))

        print(f"  {iso}_{n}z: {len(gdf)} zones, {len(topo_rows)} topo links so far")

    zones_gdf = gpd.GeoDataFrame(
        pd.concat(zones_gdfs, ignore_index=True), crs=zones_gdfs[0].crs
    )
    return zones_gdf, pd.DataFrame(zcmap_rows), pd.DataFrame(topo_rows)


# ── 3. OSM data ────────────────────────────────────────────────────────────────

def _country_bbox(zones_gdf, iso: str) -> Tuple[float, float, float, float]:
    b = zones_gdf[zones_gdf["country"] == iso].total_bounds  # minx, miny, maxx, maxy
    pad = 0.5
    return (b[1] - pad, b[0] - pad, b[3] + pad, b[2] + pad)  # s, w, n, e


def fetch_osm_data(zones_gdf):
    import pandas as pd

    subs_all:  list[dict] = []
    lines_all: list[dict] = []
    gens_all:  list[dict] = []

    for iso in zones_gdf["country"].unique():
        print(f"  [{iso}] querying Overpass...")
        bbox = _country_bbox(zones_gdf, iso)
        subs  = fetch_substations(bbox)
        lines = fetch_hv_lines(bbox, min_voltage_kv=100)
        gens  = fetch_generators(bbox)
        for r in subs:  r["country"] = iso
        for r in lines: r["country"] = iso
        for r in gens:  r["country"] = iso
        subs_all.extend(subs)
        lines_all.extend(lines)
        gens_all.extend(gens)
        print(f"    {len(subs)} substations · {len(lines)} HV lines · {len(gens)} plants")

    return pd.DataFrame(subs_all), lines_all, pd.DataFrame(gens_all)


# ── 4. Inter-zone corridors (cross-border + intra) ────────────────────────────

def build_preferred_corridors(zones_gdf, hvlines):
    """
    Detect inter-zone lines across the combined (multi-country) zones_gdf,
    then enrich with reference data.  Returns corridors DataFrame.
    """
    interzone = _find_interzone_lines(hvlines, zones_gdf)
    print(f"  {len(interzone)} inter-zone OSM line segments detected")
    ref_path = _REFERENCE_LINES if _REFERENCE_LINES.exists() else None
    corridors_df = build_corridors(interzone, zones_gdf, ref_path)
    print(f"  {len(corridors_df)} corridors built "
          f"({(corridors_df['mw_override'] != '').sum()} with reference override)")
    return corridors_df


# ── 4b. NTC corridor enrichment ───────────────────────────────────────────────

def apply_corridor_ntc(corridors_df, ntc_path=None):
    """
    Enrich corridors_df with existing/committed/candidate MW from reference_corridors.csv.

    Rules:
    - If ntc_path exists and a zone pair is listed → use those values (override OSM estimate).
    - If a zone pair is listed with all-zero MW → corridor is removed (e.g. closed borders).
    - If a zone pair is in ntc_path but absent from OSM → add it (e.g. submarine cables).
    - If ntc_path doesn't exist → existing_mw = OSM estimate, committed/candidate = 0.
    Returns enriched DataFrame with mw_existing, mw_committed, mw_candidate, projects columns.
    """
    import pandas as pd

    def _parse_mw(val):
        try:
            return max(0, int(round(float(str(val).replace(",", ".").strip() or "0"))))
        except (ValueError, TypeError):
            return 0

    if ntc_path is None or not Path(ntc_path).exists():
        out = corridors_df.copy()
        out["mw_existing"]  = out.apply(_effective_mw, axis=1)
        out["mw_committed"] = 0
        out["mw_candidate"] = 0
        out["projects"]     = ""
        return out

    ntc = pd.read_csv(ntc_path, comment="#")
    ntc_lookup = {}
    for _, r in ntc.iterrows():
        key = frozenset([str(r["z"]).strip().lower(), str(r["zz"]).strip().lower()])
        ntc_lookup[key] = r

    result, seen = [], set()

    for _, row in corridors_df.iterrows():
        z, zz = str(row["z"]), str(row["zz"])
        if z.lower() == zz.lower():
            continue  # skip self-loops from reference_lines intra-country entries
        key   = frozenset([z.lower(), zz.lower()])
        seen.add(key)
        nr    = row.to_dict()

        if key in ntc_lookup:
            ref = ntc_lookup[key]
            ex  = _parse_mw(ref.get("existing_mw",  0))
            com = _parse_mw(ref.get("committed_mw", 0))
            can = _parse_mw(ref.get("candidate_mw", 0))
            if ex == 0 and com == 0 and can == 0:
                continue  # explicitly zeroed → skip (e.g. closed borders)
            nr["mw_existing"]  = ex
            nr["mw_committed"] = com
            nr["mw_candidate"] = can
            proj = ref.get("projects", "")
            nr["projects"]     = "" if (proj != proj or str(proj).strip() in ("", "nan")) else str(proj).strip()
        else:
            nr["mw_existing"]  = _effective_mw(row)
            nr["mw_committed"] = 0
            nr["mw_candidate"] = 0
            nr["projects"]     = ""
        result.append(nr)

    # Add NTC-only corridors absent from OSM (e.g. planned submarine cables)
    for _, ref in ntc.iterrows():
        z, zz = str(ref["z"]).strip(), str(ref["zz"]).strip()
        key   = frozenset([z.lower(), zz.lower()])
        if key in seen:
            continue
        ex  = _parse_mw(ref.get("existing_mw",  0))
        com = _parse_mw(ref.get("committed_mw", 0))
        can = _parse_mw(ref.get("candidate_mw", 0))
        if ex + com + can == 0:
            continue
        proj = ref.get("projects", "")
        proj = "" if (proj != proj or str(proj).strip() in ("", "nan")) else str(proj).strip()
        result.append({
            "z": z, "zz": zz,
            "mw_osm": 0, "mw_override": ex, "status": "existing" if ex > 0 else "candidate",
            "mw_existing": ex, "mw_committed": com, "mw_candidate": can,
            "projects": proj,
        })

    return pd.DataFrame(result) if result else corridors_df.iloc[0:0].copy()


def export_corridors_ntc_geojson(corridors_df, zones_gdf, output_path):
    """
    Write corridors GeoJSON with mw_existing / mw_committed / mw_candidate properties.
    Falls back to mw/status for regions without NTC data (backward-compatible).
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    centroids = {}
    if zones_gdf is not None:
        for _, row in zones_gdf.iterrows():
            name = row["zone_name"]
            try:
                c = row.geometry.centroid
                centroids[name] = [round(c.x, 5), round(c.y, 5)]
            except Exception:
                pass

    has_ntc = "mw_existing" in corridors_df.columns
    features = []

    for _, row in corridors_df.iterrows():
        z1, z2 = str(row["z"]), str(row["zz"])
        if z1 not in centroids or z2 not in centroids:
            continue

        if has_ntc:
            ex  = int(row.get("mw_existing",  0) or 0)
            com = int(row.get("mw_committed", 0) or 0)
            can = int(row.get("mw_candidate", 0) or 0)
            mw  = ex or com or can
        else:
            ex = com = can = 0
            mw = _effective_mw(row)

        if mw == 0 and ex + com + can == 0:
            continue

        label_parts = []
        if ex  > 0: label_parts.append(f"{ex:,}")
        if com > 0: label_parts.append(f"+{com:,} comm.")
        if can > 0: label_parts.append(f"+{can:,} cand.")
        label = " / ".join(label_parts) + " MW" if label_parts else ""

        props = {
            "zone_a":        z1,
            "zone_b":        z2,
            "mw":            mw,
            "mw_existing":   ex,
            "mw_committed":  com,
            "mw_candidate":  can,
            "status":        str(row.get("status", "existing")),
            "projects":      str(row.get("projects", "") or ""),
            "label":         label,
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [centroids[z1], centroids[z2]]},
            "properties": props,
        })

    gj = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gj, f, separators=(",", ":"))


# ── 5. Tabular export ──────────────────────────────────────────────────────────

def export_tabular(zones_gdf, zcmap_df, topo_df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Standard version (110m-clipped outer borders)
    zones_gdf.to_file(OUT_DIR / "blacksea_preferred_zones.geojson", driver="GeoJSON")
    print(f"  blacksea_preferred_zones.geojson  ({len(zones_gdf)} zones)")

    zcmap_df.to_csv(OUT_DIR / "blacksea_preferred_zcmap.csv", index=False)
    print(f"  blacksea_preferred_zcmap.csv  ({len(zcmap_df)} rows)")

    topo_df.to_csv(OUT_DIR / "blacksea_preferred_topology.csv", index=False)
    print(f"  blacksea_preferred_topology.csv  ({len(topo_df)} links)")


def export_inner_borders(hd_gdf: "gpd.GeoDataFrame", slug: str) -> None:
    """Export only the shared (interior) zone borders to avoid double-rendering artefacts."""
    import geopandas as gpd
    from shapely.ops import unary_union

    all_bounds  = unary_union([g.boundary for g in hd_gdf.geometry])
    outer_union = unary_union(hd_gdf.geometry)
    outer_bound = (outer_union.boundary
                   if outer_union is not None and not outer_union.is_empty
                   else None)
    inner_geom  = all_bounds.difference(outer_bound.buffer(1e-6)) if outer_bound else all_bounds

    inner_gdf = gpd.GeoDataFrame(geometry=[inner_geom], crs="EPSG:4326")
    fname = f"blacksea_{slug}_inner_borders.geojson"
    inner_gdf.to_file(_EXPLORER_ZONES / fname, driver="GeoJSON")
    print(f"  inner borders -> {_EXPLORER_ZONES / fname}")


def export_tabular_hd(config: dict[str, int], slug: str) -> "gpd.GeoDataFrame | None":
    """Generate HD zones from Explorer's 10m-clipped zone files. Returns the GeoDataFrame."""
    import geopandas as gpd
    import pandas as pd

    zones_gdfs = []
    for iso, n in config.items():
        src = _EXPLORER_ZONES / f"{iso}_{n}z_zones.geojson"
        if not src.exists():
            print(f"  [WARN] {src.name} not found, falling back to pipeline output")
            src = STUDY_ROOT / f"{iso}_{n}z" / "epm_export" / "spatial" / "zones.geojson"
        if not src.exists():
            print(f"  [WARN] {iso}_{n}z not found, skipping")
            continue
        gdf = gpd.read_file(src)
        gdf["country"] = iso
        zones_gdfs.append(gdf)
        print(f"  {iso}_{n}z: loaded from {src.name}")

    if not zones_gdfs:
        print("  [WARN] No HD zones generated")
        return None

    hd_gdf = gpd.GeoDataFrame(pd.concat(zones_gdfs, ignore_index=True), crs=zones_gdfs[0].crs)
    fname = f"blacksea_{slug}_zones_hd.geojson"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hd_gdf.to_file(OUT_DIR / fname, driver="GeoJSON")
    print(f"  {fname}  ({len(hd_gdf)} zones)")

    if _EXPLORER_ZONES.exists():
        hd_gdf.to_file(_EXPLORER_ZONES / fname, driver="GeoJSON")
        print(f"  -> {_EXPLORER_ZONES / fname}")

    return hd_gdf


# ── 5. Capacity mix helpers ────────────────────────────────────────────────────

def _zone_capacity_mix(zones_gdf, gens_df) -> dict[str, dict[str, float]]:
    """Spatial-join generators to zones → {zone_id: {fuel: total_mw}}."""
    import geopandas as gpd

    if gens_df.empty:
        return {}

    gen_gdf = gpd.GeoDataFrame(
        gens_df.copy(),
        geometry=gpd.points_from_xy(gens_df["lon"], gens_df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        gen_gdf,
        zones_gdf[["zone_id", "geometry"]],
        how="left",
        predicate="within",
    )

    result: dict[str, dict[str, float]] = {}
    for zone_id, grp in joined.groupby("zone_id"):
        total_mw = grp.groupby("fuel")["capacity_mw"].sum()
        if total_mw.sum() < 1:
            # Fall back to plant counts when capacity data is missing
            total_mw = grp["fuel"].value_counts().astype(float)
        result[str(zone_id)] = total_mw.to_dict()
    return result


def _svg_donut(mix: dict[str, float], size: int = 64) -> str:
    """Return an SVG donut chart string for a fuel mix dict."""
    total = sum(mix.values())
    if total <= 0:
        return ""

    cx = cy = size / 2
    r_out = size / 2 - 2
    r_in  = r_out * 0.55

    def _arc(a0: float, a1: float) -> str:
        def _pt(a, r):
            return cx + r * math.sin(a), cy - r * math.cos(a)

        large = 1 if (a1 - a0) > math.pi else 0
        x1, y1 = _pt(a0, r_out)
        x2, y2 = _pt(a1, r_out)
        x3, y3 = _pt(a1, r_in)
        x4, y4 = _pt(a0, r_in)
        return (
            f"M {x1:.2f} {y1:.2f} "
            f"A {r_out:.2f} {r_out:.2f} 0 {large} 1 {x2:.2f} {y2:.2f} "
            f"L {x3:.2f} {y3:.2f} "
            f"A {r_in:.2f} {r_in:.2f} 0 {large} 0 {x4:.2f} {y4:.2f} Z"
        )

    paths = []
    angle = 0.0
    for fuel, val in sorted(mix.items(), key=lambda x: -x[1]):
        span = 2 * math.pi * val / total
        d = _arc(angle, angle + span)
        color = _FUEL_COLORS.get(fuel, "#A9A9A9")
        paths.append(
            f'<path d="{d}" fill="{color}" stroke="white" stroke-width="0.5"/>'
        )
        angle += span

    body = "\n  ".join(paths)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'style="filter:drop-shadow(0 1px 3px rgba(0,0,0,.45))">'
        f'\n  {body}\n</svg>'
    )


# ── 6. HTML map (Folium) ───────────────────────────────────────────────────────

def _mw_to_weight(mw: float) -> float:
    return max(1.5, min(7.0, 1.0 + mw / 250.0))


def make_html_map(zones_gdf, subs_df, hvlines, gens_df,
                  cities_df, capacity_mix, out_path, corridors_df=None):
    import folium

    m = folium.Map(
        location=[41.5, 37.0], zoom_start=5,
        tiles="CartoDB positron", prefer_canvas=True,
    )

    fg_zones    = folium.FeatureGroup(name="Zones",                        show=True)
    fg_conn_ex  = folium.FeatureGroup(name="Connections — existing",       show=True)
    fg_conn_pl  = folium.FeatureGroup(name="Connections — planned",        show=True)
    fg_hv       = folium.FeatureGroup(name="HV Grid (≥100 kV)",           show=False)
    fg_subs     = folium.FeatureGroup(name="Substations",                  show=False)
    fg_plants   = folium.FeatureGroup(name="Power Plants (OSM)",           show=True)
    fg_cities   = folium.FeatureGroup(name="Cities",                       show=True)
    fg_mix      = folium.FeatureGroup(name="Capacity Mix Donuts",          show=True)

    # Zone polygons + labels
    for _, row in zones_gdf.iterrows():
        zid     = str(row["zone_id"])
        country = row.get("country", zid[:3])
        color   = _COUNTRY_COLORS.get(country, "#999999")
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c, "color": "#333",
                "weight": 1.5, "fillOpacity": 0.3,
            },
            tooltip=zid,
        ).add_to(fg_zones)

        centroid = row.geometry.centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:11px;font-weight:700;color:#111;'
                    f'text-shadow:1px 1px 2px white,-1px -1px 2px white,'
                    f'0 0 4px white;white-space:nowrap">{zid}</div>'
                ),
                icon_size=(130, 20), icon_anchor=(65, 10),
            ),
        ).add_to(fg_zones)

    # Inter-zone corridors
    if corridors_df is not None and not corridors_df.empty:
        zone_centroids = {
            str(row["zone_id"]): (row.geometry.centroid.y, row.geometry.centroid.x)
            for _, row in zones_gdf.iterrows()
        }
        for _, row in corridors_df.iterrows():
            z1, z2 = str(row["z"]), str(row["zz"])
            if z1 not in zone_centroids or z2 not in zone_centroids:
                continue
            mw = _effective_mw(row)
            if mw == 0:
                continue
            status = str(row.get("status", "existing")).lower()
            kv = row.get("voltage_kv", "")
            kv_str = f"{int(kv)} kV · " if str(kv).isdigit() and int(kv) > 0 else ""
            src = "ref" if str(row.get("mw_override", "")).strip() not in ("", "0") else "OSM est."
            tip = f"{z1} ↔ {z2} | {kv_str}{mw:,} MW ({src})"
            pts = [zone_centroids[z1], zone_centroids[z2]]
            weight = _mw_to_weight(mw)
            if status in ("planned", "candidate", "long_term", "under_construction"):
                entry = row.get("earliest_entry", "")
                tip += f" | {status}" + (f" {entry}" if str(entry).isdigit() else "")
                folium.PolyLine(
                    pts, color="#8a6a00", weight=weight,
                    opacity=0.80, dash_array="10 6",
                    tooltip=folium.Tooltip(tip, sticky=True),
                ).add_to(fg_conn_pl)
            elif status not in ("cold_standby",):
                folium.PolyLine(
                    pts, color="#1a5fa8", weight=weight,
                    opacity=0.85,
                    tooltip=folium.Tooltip(tip, sticky=True),
                ).add_to(fg_conn_ex)

    # HV lines
    for line in hvlines:
        coords = [(lat, lon) for lon, lat in line["coords"]]
        v = line.get("voltage_kv", 0)
        color = "#d9400a" if v >= 300 else "#f0a030" if v >= 150 else "#bbb"
        folium.PolyLine(
            coords, color=color, weight=1.5, opacity=0.7,
            tooltip=f"{v} kV",
        ).add_to(fg_hv)

    # Substations
    if not subs_df.empty:
        for _, r in subs_df.iterrows():
            tip = r.get("name", "") or ""
            kv  = int(r.get("voltage_kv", 0) or 0)
            if kv:
                tip += f" {kv} kV"
            folium.CircleMarker(
                location=[r["lat"], r["lon"]], radius=4,
                color="#1a5fa8", fill=True, fill_color="#4a9de8",
                fill_opacity=0.8, weight=1,
                tooltip=tip.strip() or "substation",
            ).add_to(fg_subs)

    # Power plants
    if not gens_df.empty:
        for _, r in gens_df.iterrows():
            fuel  = r.get("fuel", "Other") or "Other"
            color = _FUEL_COLORS.get(fuel, "#A9A9A9")
            mw    = float(r.get("capacity_mw", 0) or 0)
            name  = r.get("name", "") or ""
            tip   = f"{name} [{fuel}]" + (f" · {mw:.0f} MW" if mw > 0 else "")
            folium.CircleMarker(
                location=[r["lat"], r["lon"]], radius=5,
                color="#333", fill=True, fill_color=color,
                fill_opacity=0.85, weight=0.5,
                tooltip=tip.strip(),
            ).add_to(fg_plants)

    # Cities
    if cities_df is not None and not cities_df.empty:
        for _, r in cities_df.iterrows():
            folium.Marker(
                location=[r["lat"], r["lon"]],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:10px;color:#333;font-family:sans-serif;'
                        f'text-shadow:1px 1px 1px white,-1px -1px 1px white;'
                        f'white-space:nowrap">● {r["name"]}</div>'
                    ),
                    icon_size=(110, 16), icon_anchor=(5, 8),
                ),
            ).add_to(fg_cities)

    # Capacity mix donuts
    for _, row in zones_gdf.iterrows():
        zid = str(row["zone_id"])
        mix = capacity_mix.get(zid, {})
        svg = _svg_donut(mix)
        if not svg:
            continue
        c = row.geometry.centroid
        folium.Marker(
            location=[c.y, c.x],
            icon=folium.DivIcon(
                html=svg, icon_size=(64, 64), icon_anchor=(32, 32),
            ),
        ).add_to(fg_mix)

    for fg in [fg_zones, fg_conn_ex, fg_conn_pl, fg_hv, fg_subs, fg_plants, fg_cities, fg_mix]:
        fg.add_to(m)

    # Legend panel
    country_html = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<div style="width:12px;height:12px;background:{c};'
        f'opacity:.55;border:1px solid #888"></div><span>{iso}</span></div>'
        for iso, c in _COUNTRY_COLORS.items()
    )
    fuel_html = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<div style="width:10px;height:10px;border-radius:50%;background:{c}"></div>'
        f'<span>{f}</span></div>'
        for f, c in _FUEL_COLORS.items()
        if f not in ("Other", "Storage")
    )
    legend = (
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;'
        'background:rgba(255,255,255,.93);padding:10px 14px;border-radius:8px;'
        'box-shadow:0 2px 8px rgba(0,0,0,.2);font-family:sans-serif;'
        'font-size:12px;max-width:160px">'
        f'<b>Countries</b>{country_html}'
        '<hr style="margin:6px 0">'
        f'<b>Fuel type</b>{fuel_html}'
        '<hr style="margin:6px 0">'
        '<span style="color:#888;font-size:10px">Pre-analysis only<br>'
        'Data: OSM · Natural Earth<br>Not EPM model results</span>'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"  {out_path.name}")


# ── 7. Static map (matplotlib) ─────────────────────────────────────────────────

def make_static_map(zones_gdf, gens_df, cities_df, capacity_mix, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#F0EDE8")
    ax.set_facecolor("#D6EAF8")

    zone_colors = [_COUNTRY_COLORS.get(r["country"], "#999") for _, r in zones_gdf.iterrows()]

    # Try contextily basemap; fall back to plain plot
    try:
        import contextily as cx
        zones_web = zones_gdf.to_crs(epsg=3857)
        zones_web.plot(ax=ax, color=zone_colors, alpha=0.4,
                       linewidth=1.2, edgecolor="#444")
        cx.add_basemap(ax, crs=zones_web.crs,
                       source=cx.providers.CartoDB.Positron, alpha=0.75)
        use_web = True
    except Exception:
        zones_gdf.plot(ax=ax, color=zone_colors, alpha=0.4,
                       linewidth=1.2, edgecolor="#444")
        use_web = False

    def _to_plot(lon, lat):
        if use_web:
            import pyproj
            proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            return proj.transform(lon, lat)
        return lon, lat

    # Zone labels
    for _, row in zones_gdf.iterrows():
        c = row.geometry.centroid
        cx_pt, cy_pt = _to_plot(c.x, c.y)
        ax.text(
            cx_pt, cy_pt, str(row["zone_id"]),
            fontsize=7.5, ha="center", va="center", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )

    # Capacity mix donuts (matplotlib Wedge)
    for _, row in zones_gdf.iterrows():
        zid = str(row["zone_id"])
        mix = capacity_mix.get(zid, {})
        total = sum(mix.values())
        if total <= 0:
            continue
        c = row.geometry.centroid
        cx_pt, cy_pt = _to_plot(c.x, c.y)

        # Donut radius in map units — scale roughly to ~80 km for EPSG:3857
        scale = 80_000 if use_web else 0.8
        r_out, r_in = scale, scale * 0.55

        start = 90.0
        for fuel, val in sorted(mix.items(), key=lambda x: -x[1]):
            span  = val / total * 360.0
            color = _FUEL_COLORS.get(fuel, "#A9A9A9")
            wedge = mpatches.Wedge(
                (cx_pt, cy_pt), r_out, start, start - span,
                width=r_out - r_in,
                facecolor=color, edgecolor="white", linewidth=0.5, alpha=0.92,
            )
            ax.add_patch(wedge)
            start -= span

    # Cities
    if cities_df is not None and not cities_df.empty:
        for _, r in cities_df.iterrows():
            px, py = _to_plot(r["lon"], r["lat"])
            ax.plot(px, py, "o", ms=3, color="#222", zorder=5)
            ax.text(
                px, py, f"  {r['name']}",
                fontsize=6.5, va="center", color="#333",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            )

    # Legends
    country_patches = [
        mpatches.Patch(facecolor=c, alpha=0.55, edgecolor="#666", label=iso)
        for iso, c in _COUNTRY_COLORS.items()
    ]
    fuel_patches = [
        mpatches.Patch(facecolor=c, label=f)
        for f, c in _FUEL_COLORS.items()
        if f not in ("Other", "Storage")
    ]
    leg1 = ax.legend(handles=country_patches, loc="lower left",
                     fontsize=7, title="Countries", title_fontsize=8,
                     framealpha=0.88)
    ax.add_artist(leg1)
    ax.legend(handles=fuel_patches, loc="lower right",
              fontsize=7, title="Capacity mix (OSM)", title_fontsize=8,
              framealpha=0.88, ncol=2)

    ax.set_title(
        "Black Sea Region — Preferred Zoning Configuration\n"
        "Pre-analysis only · Data: OpenStreetMap, Natural Earth · Not EPM model results",
        fontsize=10, pad=10,
    )
    ax.set_axis_off()

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_configs = load_all_configs()
    print(f"Found {len(all_configs)} zoning config(s): {list(all_configs.keys())}\n")

    # OSM data is per-country — fetch once for all countries used across configs
    all_countries = sorted({iso for cfg in all_configs.values() for iso in cfg})
    print("Fetching OSM data (cached after first run)...")
    # Build a fake zones_gdf just to get bboxes per country for OSM fetch
    import geopandas as gpd, pandas as pd
    bbox_gdfs = []
    for iso in all_countries:
        # Use 1z as reference for bbox (always exists)
        src1z = STUDY_ROOT / f"{iso}_1z" / "epm_export" / "spatial" / "zones.geojson"
        if src1z.exists():
            g = gpd.read_file(src1z); g["country"] = iso; bbox_gdfs.append(g)
    bbox_gdf = gpd.GeoDataFrame(pd.concat(bbox_gdfs, ignore_index=True), crs="EPSG:4326") if bbox_gdfs else None

    subs_df, hvlines, gens_df = fetch_osm_data(bbox_gdf) if bbox_gdf is not None else (pd.DataFrame(), [], pd.DataFrame())
    print()

    print("Loading cities (Natural Earth)...")
    cities_df = load_cities(countries=all_countries, min_pop=100_000)
    print(f"  {len(cities_df)} cities\n")

    index_entries = []

    for name, config in all_configs.items():
        slug = _slug(name)
        print(f"\n{'='*50}")
        print(f"  Config: {name!r} (slug={slug!r})")
        print(f"  Zones : {config}")
        print(f"{'='*50}")

        print("\nLoading zone data...")
        zones_gdf, zcmap_df, topo_df = load_zone_data(config)
        print(f"  Total: {len(zones_gdf)} zones across {len(config)} countries")

        print("\nComputing capacity mix per zone...")
        capacity_mix = _zone_capacity_mix(zones_gdf, gens_df)
        n_with_data = sum(1 for v in capacity_mix.values() if v)
        print(f"  {n_with_data} zones have plant data")

        print("\nBuilding inter-zone corridors...")
        corridors_df = build_preferred_corridors(zones_gdf, hvlines)

        print("\nExporting tabular data...")
        export_tabular(zones_gdf, zcmap_df, topo_df)

        print("\nExporting HD zones...")
        hd_gdf = export_tabular_hd(config, slug)
        if hd_gdf is not None and _EXPLORER_ZONES.exists():
            export_inner_borders(hd_gdf, slug)

        # Enrich corridors with NTC reference data then export
        if corridors_df is not None and len(corridors_df):
            ntc_path     = _REFERENCE_CORRIDORS if _REFERENCE_CORRIDORS.exists() else None
            corridors_ntc = apply_corridor_ntc(corridors_df, ntc_path)
            if _EXPLORER_ZONES.exists():
                corr_fname = f"blacksea_{slug}_corridors.geojson"
                corr_path  = _EXPLORER_ZONES / corr_fname
                export_corridors_ntc_geojson(corridors_ntc, hd_gdf if hd_gdf is not None else zones_gdf, corr_path)
                print(f"  corridors -> {corr_path}")

        print("\nGenerating HTML map...")
        make_html_map(
            zones_gdf, subs_df, hvlines, gens_df, cities_df, capacity_mix,
            MAPS_DIR / f"blacksea_{slug}_map.html",
            corridors_df=corridors_df,
        )

        index_entries.append({"name": name, "slug": slug, "zones": config})

    # Write config index to Explorer
    if _EXPLORER_ZONES.exists():
        idx_path = _EXPLORER_ZONES / "blacksea_configs.json"
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index_entries, f, indent=2)
        print(f"\n  blacksea_configs.json -> {idx_path}")

    print(f"\nDone. All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
