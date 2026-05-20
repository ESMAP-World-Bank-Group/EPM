"""
Zone Pipeline — generate EPM zones from open data and produce maps.

Given a list of countries and a target number of zones:
  1. Load Natural Earth boundaries + cities
  2. Fetch OSM substations + HV lines
  3. Cluster substations (weighted by voltage + nearby population) -> N zones
  4. Generate zone boundaries (Voronoi clipped to country borders) -> GeoJSON
  5. Identify inter-zone HV lines
  6. Export EPM-ready CSVs + interactive HTML maps + static PNGs

Outputs (relative to pre-analysis/output_workflow/):
  epm_export/spatial/
    zcmap.csv, zones.geojson, sTopology.csv, pTransferLimit_estimated.csv
  report/spatial/
    zone_map_detailed.html, zone_map_simplified.html,
    zone_capacity_mix.html, zone_map_simplified.png, zone_capacity_mix.png

Usage:
    conda activate esmap_env
    python pre-analysis/pipelines/zone_pipeline.py \\
        --countries TUR ROU BGR GEO ARM AZE --n_zones 12

    # or via Python:
    from pipelines.zone_pipeline import run_zone_pipeline
    run_zone_pipeline(["TUR", "ROU", "BGR", "GEO", "ARM", "AZE"], n_zones=12)
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Resolve paths
_BASE = Path(__file__).resolve().parents[1]           # pre-analysis/
_ADVISOR = _BASE / "resolution_advisor"
_OUTPUT_ROOT = _BASE / "output_workflow"

sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_ADVISOR))

# ── fuel colour palette (consistent with generators_pipeline.py style) ────────
FUEL_COLORS = {
    "Solar":       "#f4c430",
    "Wind":        "#4fc3f7",
    "Hydro":       "#1565c0",
    "Gas":         "#ef5350",
    "Coal":        "#424242",
    "Nuclear":     "#7b1fa2",
    "Oil":         "#ff8f00",
    "Biomass":     "#66bb6a",
    "Geothermal":  "#8d6e63",
    "Storage":     "#26a69a",
    "Other":       "#90a4ae",
}

VOLTAGE_COLORS = {
    400: "#d32f2f",
    220: "#f57c00",
    150: "#fbc02d",
    110: "#388e3c",
    0:   "#9e9e9e",   # unknown
}


# ── main entry point ──────────────────────────────────────────────────────────

def run_zone_pipeline(
    countries: List[str],
    n_zones: int,
    output_root: Optional[Path] = None,
    verbose: bool = True,
    boundary_mode: str = "auto",
    reference_lines_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    boundary_mode: 'auto' (try admin-1, fallback Voronoi) | 'admin' | 'voronoi'
    """
    """
    Run the full zone pipeline.
    Returns dict of output file paths.
    """
    log = print if verbose else (lambda *a: None)
    out = output_root or _OUTPUT_ROOT
    epm_dir = out / "epm_export" / "spatial"
    rep_dir = out / "report" / "spatial"
    epm_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n[zone_pipeline] Countries: {countries}, N zones: {n_zones}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log("[1/7] Loading boundaries and cities...")
    boundaries, cities = _load_geo_data(countries, log)

    log("[2/7] Fetching OSM substations and HV lines...")
    substations, hv_lines = _fetch_osm(countries, boundaries, log)

    # ── 2. Cluster -> zones ───────────────────────────────────────────────────
    log("[3/6] Clustering substations into zones...")
    zone_labels, cluster_centers, point_df = _cluster_zones(
        substations, cities, boundaries, n_zones, log
    )

    # ── 3. Zone boundaries ────────────────────────────────────────────────────
    zones_gdf = None
    if boundary_mode in ("auto", "admin"):
        log("[4/6] Generating zone boundaries (admin-1 aggregation)...")
        try:
            zones_gdf = _build_zone_boundaries_admin(
                cluster_centers, boundaries, zone_labels, point_df, countries, log
            )
        except Exception as e:
            log(f"  [admin] Unexpected error: {e} — falling back to Voronoi")
            zones_gdf = None

    if zones_gdf is None:
        if boundary_mode == "admin":
            raise RuntimeError("Admin-1 boundary mode failed and fallback is disabled.")
        log("[4/6] Generating zone boundaries (Voronoi)...")
        zones_gdf = _build_zone_boundaries(
            cluster_centers, boundaries, zone_labels, point_df, countries
        )

    # ── 4. Inter-zone lines ───────────────────────────────────────────────────
    log("[5/6] Finding inter-zone lines...")
    interzone_lines = _find_interzone_lines(hv_lines, zones_gdf)
    no_plants = pd.DataFrame(columns=["lat", "lon", "fuel", "capacity_mw", "zone_name"])

    # ── 5. Export ─────────────────────────────────────────────────────────────
    log("[6/6] Exporting EPM files and maps...")
    output_paths = {}
    epm_paths, corridors_df = _export_epm(zones_gdf, interzone_lines, countries, epm_dir, reference_lines_path)
    output_paths.update(epm_paths)
    output_paths.update(
        _generate_maps(zones_gdf, no_plants, substations, hv_lines,
                       cities, interzone_lines, rep_dir, log, corridors_df)
    )

    log("\n[zone_pipeline] Done.")
    for name, path in output_paths.items():
        log(f"  {name:<35} {path}")
    return output_paths


# ── data loading ──────────────────────────────────────────────────────────────

def _load_geo_data(countries, log):
    from fetch.natural_earth import load_boundaries, load_cities
    try:
        boundaries = load_boundaries(countries)
    except Exception as e:
        log(f"  [!] Could not load boundaries: {e}")
        boundaries = None
    try:
        cities = load_cities(countries, min_pop=50_000)
    except Exception as e:
        log(f"  [!] Could not load cities: {e}")
        cities = pd.DataFrame()
    return boundaries, cities


def _fetch_osm(countries, boundaries, log):
    from fetch.osm import fetch_substations, fetch_hv_lines
    all_subs, all_lines = [], []
    if boundaries is None:
        log("  [!] No boundaries — skipping OSM fetch")
        return all_subs, all_lines
    for iso in countries:
        subset = boundaries[boundaries["ISO_A3"] == iso]
        if subset.empty:
            continue
        b = subset.geometry.total_bounds  # minx, miny, maxx, maxy
        bbox = (b[1], b[0], b[3], b[2])  # s,w,n,e
        try:
            subs = fetch_substations(bbox)
            lines = fetch_hv_lines(bbox)
            log(f"  {iso}: {len(subs)} substations, {len(lines)} HV lines")
            all_subs.extend(subs)
            all_lines.extend(lines)
        except Exception as e:
            log(f"  [{iso}] OSM fetch failed: {e}")
    return all_subs, all_lines


# ── clustering ────────────────────────────────────────────────────────────────

def _cluster_zones(substations, cities, boundaries, n_zones, log):
    """K-means on substations weighted by nearby population."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    # Build weighted point set
    points = []  # [(lat, lon, weight)]

    if substations:
        for s in substations:
            w = max(s.get("voltage_kv", 1), 1) ** 0.5  # higher voltage = more weight
            points.append((s["lat"], s["lon"], w))

    # Add city centroids as high-weight points (load centers)
    if cities is not None and len(cities) > 0:
        for _, row in cities.iterrows():
            pop = max(float(row.get("pop", 1)), 1)
            w = (pop / 1e6) ** 0.5 * 10  # normalise
            points.append((float(row["lat"]), float(row["lon"]), w))

    if not points:
        log("  [!] No points to cluster — falling back to country centroids")
        points = _country_centroids(boundaries)

    df = pd.DataFrame(points, columns=["lat", "lon", "weight"])
    coords = df[["lat", "lon"]].values
    weights = df["weight"].values

    # Fit k-means
    n = min(n_zones, len(df))
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    km.fit(coords, sample_weight=weights)

    df["zone_id"] = km.labels_
    centers = km.cluster_centers_  # (n_zones, 2)

    log(f"  {len(df)} weighted points -> {n} clusters")
    return km.labels_, centers, df


def _country_centroids(boundaries) -> list:
    if boundaries is None:
        return [(40.0, 35.0, 1.0)]  # rough Turkey centroid as default
    centroids = []
    for _, row in boundaries.iterrows():
        c = row.geometry.centroid
        centroids.append((c.y, c.x, 1.0))
    return centroids


# ── zone boundary generation ──────────────────────────────────────────────────

def _build_zone_boundaries(centers, boundaries, labels, point_df, countries):
    """Voronoi tessellation clipped to country boundaries."""
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
        from scipy.spatial import Voronoi
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")

    if boundaries is None:
        raise RuntimeError("Cannot build zone boundaries without country boundaries")

    # Merge all country polygons into one region
    region = unary_union(boundaries.geometry)

    # Voronoi from cluster centers (lat/lon)
    n = len(centers)
    if n < 2:
        # Single zone = the whole region
        iso = countries[0] if countries else "UNK"
        return gpd.GeoDataFrame(
            [{"zone_id": 0, "zone_name": f"{iso}_1", "ISO_A3": iso,
              "geometry": region}],
            crs="EPSG:4326"
        )

    # Add far-away mirror points to close the Voronoi diagram
    pts = np.vstack([
        centers,
        [(-90, -180), (-90, 180), (90, -180), (90, 180)],
    ])
    vor = Voronoi(pts)

    rows = []
    for i, center in enumerate(centers):
        region_idx = vor.point_region[i]
        verts_idx = vor.regions[region_idx]
        if -1 in verts_idx or len(verts_idx) == 0:
            continue
        verts = vor.vertices[verts_idx]
        # Note: Voronoi uses (lat, lon) so coords are (x=lat, y=lon) — need to swap
        poly = Polygon([(v[1], v[0]) for v in verts])  # (lon, lat) for shapely
        clipped = poly.intersection(region)
        if clipped.is_empty:
            continue

        # Assign country based on which country contains the centroid
        zone_centroid_lon, zone_centroid_lat = center[1], center[0]
        iso = _point_to_country(zone_centroid_lat, zone_centroid_lon, boundaries)

        rows.append({
            "zone_id": i,
            "zone_name": f"Zone_{i+1}",
            "ISO_A3": iso or "UNK",
            "geometry": clipped,
        })

    if not rows:
        # Fallback: one zone per country
        rows = [
            {"zone_id": j, "zone_name": iso, "ISO_A3": iso, "geometry": boundaries[boundaries["ISO_A3"] == iso].geometry.iloc[0]}
            for j, iso in enumerate(countries)
        ]

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    # Auto-generate meaningful names: {ISO}_{N}
    iso_counts: dict[str, int] = defaultdict(int)
    names = []
    for _, row in gdf.iterrows():
        iso_counts[row["ISO_A3"]] += 1
        names.append(f"{row['ISO_A3']}_{iso_counts[row['ISO_A3']]}")
    gdf["zone_name"] = names

    return gdf


def _detect_exclave_components(countries, boundaries, admin1, n_zones, log):
    """
    Detect disconnected country components (exclaves like Nakhchivan).
    Returns dict: admin1_idx -> component_id (0 = main body, 1+ = exclave).
    Only applied for single-country runs where n_zones > number of exclaves.
    """
    from shapely.geometry import Point

    admin_component = {i: 0 for i in range(len(admin1))}

    if len(countries) != 1 or boundaries is None:
        return admin_component

    iso = countries[0]
    country_rows = boundaries[boundaries["ISO_A3"] == iso]
    if country_rows.empty:
        return admin_component

    try:
        country_geom = country_rows.geometry.unary_union
    except Exception:
        return admin_component

    if country_geom.geom_type != "MultiPolygon":
        return admin_component

    # Sort components by area desc (largest = main body = component 0)
    components = sorted(country_geom.geoms, key=lambda g: g.area, reverse=True)
    total_area = country_geom.area

    # Only treat components < 40% of total area as exclaves (avoids splitting islands)
    exclave_comps = [
        (comp_id + 1, comp)
        for comp_id, comp in enumerate(components[1:])
        if comp.area / total_area < 0.40
    ]
    if not exclave_comps or n_zones <= len(exclave_comps):
        return admin_component

    log(f"  [admin] {iso}: {len(exclave_comps)} exclave(s) detected — assigning dedicated zone(s)")

    for idx in range(len(admin1)):
        centroid = admin1.loc[idx, "geometry"].centroid
        for comp_id, comp in exclave_comps:
            try:
                if comp.contains(centroid) or comp.distance(centroid) < 0.3:
                    admin_component[idx] = comp_id
                    break
            except Exception:
                pass

    return admin_component


def _build_zone_boundaries_admin(centers, boundaries, labels, point_df, countries, log):
    """
    Build zone boundaries by aggregating admin-1 units to majority cluster.
    Returns a zones GeoDataFrame on success, or None to signal Voronoi fallback.
    Handles exclaves (e.g. Nakhchivan/AZE) by assigning them dedicated zones.
    """
    try:
        import geopandas as gpd
        from shapely.ops import unary_union
        from shapely.geometry import Point
    except ImportError as e:
        log(f"  [admin] Missing dependency: {e} — falling back to Voronoi")
        return None

    # Load admin-1 polygons
    try:
        from fetch.natural_earth import load_admin1
        admin1 = load_admin1(countries)
    except Exception as e:
        log(f"  [admin] Could not load admin-1: {e} — falling back to Voronoi")
        return None

    if admin1.empty:
        log(f"  [admin] No admin-1 data for {countries} — falling back to Voronoi")
        return None

    n_zones = len(centers)
    if len(admin1) < n_zones:
        log(f"  [admin] Only {len(admin1)} admin-1 units for {n_zones} zones — falling back to Voronoi")
        return None

    admin1 = admin1.reset_index(drop=True)

    # Detect exclaves and compute zone budget per component
    admin_component = _detect_exclave_components(countries, boundaries, admin1, n_zones, log)
    exclave_ids = sorted({v for v in admin_component.values() if v > 0})
    main_budget = n_zones - len(exclave_ids)  # zones allocated to main body

    # Spatial join: assign each weighted point to an admin-1 unit
    subs_gdf = gpd.GeoDataFrame(
        point_df[["lat", "lon", "zone_id"]].copy(),
        geometry=gpd.points_from_xy(point_df["lon"], point_df["lat"]),
        crs="EPSG:4326",
    )
    try:
        joined = gpd.sjoin(subs_gdf, admin1[["geometry"]], how="left", predicate="within")
    except Exception as e:
        log(f"  [admin] Spatial join failed: {e} — falling back to Voronoi")
        return None

    # Majority vote per admin-1 unit (main body only)
    admin_zone: dict[int, int] = {}
    valid = joined.dropna(subset=["index_right"])
    if not valid.empty:
        # Only vote for main-body units
        main_valid = valid[valid["index_right"].astype(int).map(
            lambda i: admin_component.get(i, 0) == 0
        )]
        if not main_valid.empty:
            votes = (
                main_valid.groupby(["index_right", "zone_id"])
                .size()
                .reset_index(name="cnt")
            )
            for admin_idx, grp in votes.groupby("index_right"):
                best_zone = int(grp.loc[grp["cnt"].idxmax(), "zone_id"])
                admin_zone[int(admin_idx)] = best_zone

    # Main body units with no points → nearest cluster center
    centers_pts = [Point(float(c[1]), float(c[0])) for c in centers]
    for idx in range(len(admin1)):
        if admin_component.get(idx, 0) == 0 and idx not in admin_zone:
            centroid = admin1.loc[idx, "geometry"].centroid
            dists = [centroid.distance(cp) for cp in centers_pts]
            admin_zone[idx] = int(np.argmin(dists))

    # Remap main body zone_ids to 0..main_budget-1
    if main_budget == 1:
        for idx in range(len(admin1)):
            if admin_component.get(idx, 0) == 0:
                admin_zone[idx] = 0
    else:
        main_used = sorted({admin_zone[i] for i in range(len(admin1))
                            if admin_component.get(i, 0) == 0 and i in admin_zone})
        remap = {old: min(new, main_budget - 1) for new, old in enumerate(main_used)}
        for idx in range(len(admin1)):
            if admin_component.get(idx, 0) == 0 and idx in admin_zone:
                admin_zone[idx] = remap.get(admin_zone[idx], 0)

    # Force exclave zones: each exclave component → dedicated zone_id
    for idx in range(len(admin1)):
        comp = admin_component.get(idx, 0)
        if comp > 0:
            admin_zone[idx] = main_budget + exclave_ids.index(comp)

    # Union admin-1 polygons per zone
    zone_polys: dict[int, list] = {}
    for admin_idx, zone_id in admin_zone.items():
        zone_polys.setdefault(zone_id, []).append(admin1.loc[admin_idx, "geometry"])

    rows = []
    iso_counts: dict[str, int] = defaultdict(int)
    total_zone_ids = list(range(main_budget)) + [main_budget + i for i in range(len(exclave_ids))]
    for zone_id in total_zone_ids:
        geoms = zone_polys.get(zone_id)
        if not geoms:
            continue
        geom = unary_union(geoms)
        if geom.is_empty:
            continue
        # Determine ISO from zone centroid
        iso = (_point_to_country(geom.centroid.y, geom.centroid.x, boundaries)
               or countries[0])
        iso_counts[iso] += 1
        rows.append({
            "zone_id": zone_id,
            "zone_name": f"{iso}_{iso_counts[iso]}",
            "ISO_A3": iso,
            "geometry": geom,
        })

    if not rows:
        log("  [admin] No zones produced — falling back to Voronoi")
        return None

    result = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    exclave_note = f" (incl. {len(exclave_ids)} exclave zone(s))" if exclave_ids else ""
    log(f"  [admin] {len(result)} zones from {len(admin1)} admin-1 units{exclave_note}")
    return result


def _point_to_country(lat, lon, boundaries) -> Optional[str]:
    try:
        from shapely.geometry import Point
        pt = Point(lon, lat)
        for _, row in boundaries.iterrows():
            if row.geometry.contains(pt):
                return row["ISO_A3"]
        # Fallback: nearest country (handles Voronoi centers that land on borders)
        min_dist = float("inf")
        nearest = None
        for _, row in boundaries.iterrows():
            try:
                d = row.geometry.distance(pt)
                if d < min_dist:
                    min_dist = d
                    nearest = row["ISO_A3"]
            except Exception:
                pass
        return nearest
    except Exception:
        pass
    return None


# ── plant assignment + inter-zone lines ───────────────────────────────────────

def _find_interzone_lines(hv_lines, zones_gdf) -> list[dict]:
    """Find HV lines that cross zone boundaries — these are inter-zone links."""
    if not hv_lines or zones_gdf is None or len(zones_gdf) == 0:
        return []
    try:
        from shapely.geometry import LineString, Point
        results = []
        for line in hv_lines:
            coords = line.get("coords", [])
            if len(coords) < 2:
                continue
            start_pt = Point(coords[0])
            end_pt = Point(coords[-1])
            zone_start = _point_in_zone(start_pt, zones_gdf)
            zone_end = _point_in_zone(end_pt, zones_gdf)
            if zone_start and zone_end and zone_start != zone_end:
                results.append({
                    "zone_from": zone_start,
                    "zone_to": zone_end,
                    "voltage_kv": line.get("voltage_kv", 0),
                    "coords": coords,
                })
        return results
    except Exception:
        return []


def _point_in_zone(pt, zones_gdf) -> Optional[str]:
    for _, row in zones_gdf.iterrows():
        try:
            if row.geometry.contains(pt) or row.geometry.distance(pt) < 0.05:
                return row["zone_name"]
        except Exception:
            pass
    return None


# ── EPM export ────────────────────────────────────────────────────────────────

def _export_epm(zones_gdf, interzone_lines, countries, epm_dir, reference_lines_path=None) -> Dict[str, Path]:
    paths = {}

    # zcmap.csv
    zcmap_path = epm_dir / "zcmap.csv"
    zcmap = zones_gdf[["zone_name", "ISO_A3"]].copy()
    zcmap.columns = ["z", "c"]
    zcmap.to_csv(zcmap_path, index=False)
    paths["zcmap.csv"] = zcmap_path

    # zones.geojson
    geojson_path = epm_dir / "zones.geojson"
    zones_gdf.to_file(geojson_path, driver="GeoJSON")
    paths["zones.geojson"] = geojson_path

    # sTopology.csv — unique zone pairs (internal links)
    topo_path = epm_dir / "sTopology.csv"
    topo_rows = []
    seen = set()
    for line in interzone_lines:
        pair = tuple(sorted([line["zone_from"], line["zone_to"]]))
        if pair not in seen:
            seen.add(pair)
            topo_rows.append({"z": pair[0], "zz": pair[1]})
    if not topo_rows:
        # At minimum, add adjacent zone pairs from zones
        topo_rows = _adjacent_zone_pairs(zones_gdf)
    pd.DataFrame(topo_rows).to_csv(topo_path, index=False)
    paths["sTopology.csv"] = topo_path

    # corridors.csv + pTransferLimit_estimated.csv + pNewTransmission_estimated.csv
    from .transmission_capacity import build_corridors, save_corridors, export_epm_csvs

    corridors_path = epm_dir / "corridors.csv"
    corridors_df = build_corridors(interzone_lines, zones_gdf, reference_lines_path)
    save_corridors(corridors_df, corridors_path)
    paths["corridors.csv"] = corridors_path

    epm_paths = export_epm_csvs(corridors_df, epm_dir)
    paths.update(epm_paths)

    return paths, corridors_df


def _adjacent_zone_pairs(zones_gdf) -> list[dict]:
    """Return pairs of geometrically adjacent zones as a fallback topology."""
    pairs = []
    seen = set()
    for i, row_i in zones_gdf.iterrows():
        for j, row_j in zones_gdf.iterrows():
            if i >= j:
                continue
            pair = tuple(sorted([row_i["zone_name"], row_j["zone_name"]]))
            if pair in seen:
                continue
            try:
                if row_i.geometry.touches(row_j.geometry) or row_i.geometry.intersects(row_j.geometry):
                    seen.add(pair)
                    pairs.append({"z": pair[0], "zz": pair[1]})
            except Exception:
                pass
    return pairs


def _voltage_to_mw_proxy(voltage_kv: int) -> int:
    """Very rough thermal capacity estimate from voltage level."""
    if voltage_kv >= 400:
        return 2000
    if voltage_kv >= 220:
        return 1000
    if voltage_kv >= 150:
        return 600
    if voltage_kv >= 110:
        return 400
    return 200


# ── map generation ────────────────────────────────────────────────────────────

def _generate_maps(zones_gdf, plants, substations, hv_lines, cities,
                   interzone_lines, rep_dir, log, corridors_df=None) -> Dict[str, Path]:
    paths = {}
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        log("  [!] folium not installed — skipping HTML maps (pip install folium)")
        folium = None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        log("  [!] matplotlib not available — skipping static maps")
        HAS_MPL = False

    if folium:
        # Detailed map
        p = rep_dir / "zone_map_detailed.html"
        _map_detailed(zones_gdf, plants, substations, hv_lines, cities, p, folium, MarkerCluster)
        paths["zone_map_detailed.html"] = p
        log(f"  zone_map_detailed.html")

        # Simplified map
        p = rep_dir / "zone_map_simplified.html"
        _map_simplified(zones_gdf, cities, interzone_lines, p, folium, corridors_df)
        paths["zone_map_simplified.html"] = p
        log(f"  zone_map_simplified.html")

        # Capacity mix map
        p = rep_dir / "zone_capacity_mix.html"
        _map_capacity_mix(zones_gdf, plants, p, folium)
        paths["zone_capacity_mix.html"] = p
        log(f"  zone_capacity_mix.html")

    if HAS_MPL:
        for png_fn, png_func, png_args in [
            ("zone_map_simplified.png", _png_simplified, (zones_gdf, cities, interzone_lines)),
            ("zone_capacity_mix.png",   _png_capacity_mix, (zones_gdf, plants)),
        ]:
            p = rep_dir / png_fn
            try:
                png_func(*png_args, p)
                paths[png_fn] = p
                log(f"  {png_fn}")
            except Exception as e:
                log(f"  [!] {png_fn} failed (non-blocking): {e}")

    return paths


# ── detailed Folium map ───────────────────────────────────────────────────────

def _map_detailed(zones_gdf, plants, substations, hv_lines, cities, out_path, folium, MarkerCluster):
    center = [zones_gdf.geometry.centroid.y.mean(), zones_gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")

    # Zone polygons
    zone_layer = folium.FeatureGroup(name="Zones", show=True)
    colors = _zone_color_palette(len(zones_gdf))
    for i, (_, row) in enumerate(zones_gdf.iterrows()):
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda f, c=colors[i]: {
                "fillColor": c, "color": "#333", "weight": 1.5,
                "fillOpacity": 0.25,
            },
            tooltip=row["zone_name"],
        ).add_to(zone_layer)
    zone_layer.add_to(m)

    # HV lines (colored by voltage)
    line_layer = folium.FeatureGroup(name="HV Lines", show=True)
    for line in hv_lines:
        v = line.get("voltage_kv", 0)
        color = _voltage_color(v)
        coords_ll = [[c[1], c[0]] for c in line["coords"]]
        folium.PolyLine(
            coords_ll, color=color, weight=1.5, opacity=0.7,
            tooltip=f"{v} kV",
        ).add_to(line_layer)
    line_layer.add_to(m)

    # Substations
    sub_layer = folium.FeatureGroup(name="Substations", show=False)
    for s in substations:
        folium.CircleMarker(
            [s["lat"], s["lon"]], radius=3,
            color="#555", fill=True, fill_color="#eee", fill_opacity=0.8,
            tooltip=f"Substation {s.get('voltage_kv', '?')} kV",
        ).add_to(sub_layer)
    sub_layer.add_to(m)

    # GPPD Plants
    if not plants.empty and "lat" in plants.columns:
        plant_layer = folium.FeatureGroup(name="Power Plants (GPPD)", show=True)
        cluster = MarkerCluster().add_to(plant_layer)
        for _, row in plants.iterrows():
            if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                continue
            fuel = str(row.get("fuel", "Other"))
            color = FUEL_COLORS.get(fuel, FUEL_COLORS["Other"])
            mw = row.get("capacity_mw", 0) or 0
            folium.CircleMarker(
                [row["lat"], row["lon"]],
                radius=max(3, min(int(mw ** 0.4), 12)),
                color=color, fill=True, fill_color=color, fill_opacity=0.75,
                tooltip=f"{row.get('plant_name', '?')} | {fuel} | {mw:.0f} MW",
            ).add_to(cluster)
        plant_layer.add_to(m)

    # Cities
    if cities is not None and len(cities) > 0:
        city_layer = folium.FeatureGroup(name="Load Centers (cities)", show=True)
        for _, row in cities.iterrows():
            pop = row.get("pop", 0) or 0
            folium.CircleMarker(
                [row["lat"], row["lon"]],
                radius=max(4, min(int((pop / 1e5) ** 0.5 * 3), 15)),
                color="#1a237e", fill=True, fill_color="#3949ab", fill_opacity=0.6,
                tooltip=f"{row.get('name', '?')} ({pop/1e3:.0f}k)",
            ).add_to(city_layer)
        city_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    _add_voltage_legend(m)
    _add_fuel_legend(m)
    m.save(str(out_path))


# ── simplified Folium map ─────────────────────────────────────────────────────

def _poly_centroid(geom) -> tuple[float, float]:
    """Return (lat, lon) centroid by averaging exterior ring vertices."""
    try:
        from shapely.geometry import mapping
        coords = geom.__geo_interface__["coordinates"]
        gtype  = geom.__geo_interface__["type"]
        pts = []
        if gtype == "Polygon":
            pts = coords[0]
        elif gtype == "MultiPolygon":
            for poly in coords:
                pts.extend(poly[0])
        if pts:
            return (sum(p[1] for p in pts) / len(pts),
                    sum(p[0] for p in pts) / len(pts))
    except Exception:
        pass
    c = geom.centroid
    return (c.y, c.x)


def _mw_to_weight(mw: float) -> float:
    """Line width in pixels, proportional to capacity."""
    return max(1.5, min(7.0, 1.0 + mw / 250.0))


def _map_simplified(zones_gdf, cities, interzone_lines, out_path, folium, corridors_df=None):
    centroids = {row["zone_name"]: _poly_centroid(row.geometry)
                 for _, row in zones_gdf.iterrows()}
    center_lat = sum(v[0] for v in centroids.values()) / max(len(centroids), 1)
    center_lon = sum(v[1] for v in centroids.values()) / max(len(centroids), 1)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6,
                   tiles="CartoDB positron")
    colors = _zone_color_palette(len(zones_gdf))

    # Zone polygons
    fg_zones = folium.FeatureGroup(name="Zones", show=True)
    for i, (_, row) in enumerate(zones_gdf.iterrows()):
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda f, c=colors[i]: {
                "fillColor": c, "color": "#333", "weight": 1.5, "fillOpacity": 0.35,
            },
            tooltip=f"{row['zone_name']} ({row.get('ISO_A3', '')})",
        ).add_to(fg_zones)
        lat, lon = centroids[row["zone_name"]]
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:bold;color:#111;'
                     f'background:rgba(255,255,255,0.78);padding:2px 5px;'
                     f'border-radius:3px;white-space:nowrap">{row["zone_name"]}</div>',
                icon_size=(130, 24), icon_anchor=(65, 12),
            ),
        ).add_to(fg_zones)
    fg_zones.add_to(m)

    # Inter-zone corridors — two toggleable layers
    fg_existing = folium.FeatureGroup(name="Connections — existing",  show=True)
    fg_planned  = folium.FeatureGroup(name="Connections — planned",   show=True)

    if corridors_df is not None and not corridors_df.empty:
        from .transmission_capacity import _effective_mw
        for _, row in corridors_df.iterrows():
            z1, z2 = str(row["z"]), str(row["zz"])
            if z1 not in centroids or z2 not in centroids:
                continue
            mw = _effective_mw(row)
            status = str(row.get("status", "existing")).lower()
            kv = row.get("voltage_kv", "")
            kv_str = f"{int(kv)} kV · " if str(kv).isdigit() and int(kv) > 0 else ""
            src = "ref" if str(row.get("mw_override", "")).strip() not in ("", "0") else "OSM est."
            tip = f"{z1} ↔ {z2} | {kv_str}{mw:,} MW ({src})"
            if status in ("planned", "candidate", "long_term", "under_construction"):
                entry = row.get("earliest_entry", "")
                tip += f" | {status}" + (f" {entry}" if str(entry).isdigit() else "")
                folium.PolyLine(
                    [centroids[z1], centroids[z2]],
                    color="#8a6a00", weight=_mw_to_weight(mw),
                    opacity=0.75, dash_array="10 6",
                    tooltip=folium.Tooltip(tip, sticky=True),
                ).add_to(fg_planned)
            elif status not in ("cold_standby",):
                folium.PolyLine(
                    [centroids[z1], centroids[z2]],
                    color="#1a5fa8", weight=_mw_to_weight(mw),
                    opacity=0.80,
                    tooltip=folium.Tooltip(tip, sticky=True),
                ).add_to(fg_existing)
    else:
        # Fallback: compute from raw OSM lines if no corridors_df
        from collections import defaultdict
        from .transmission_capacity import estimate_capacity_mw, _geodesic_km
        link_agg: dict = defaultdict(lambda: {"mw": 0, "kvs": set()})
        for link in interzone_lines:
            pair = tuple(sorted([link["zone_from"], link["zone_to"]]))
            v = float(link.get("voltage_kv") or 0)
            nc = int(link.get("n_circuits") or 1)
            coords = link.get("coords", [])
            length = _geodesic_km(coords) if len(coords) >= 2 else 0.0
            link_agg[pair]["mw"] += estimate_capacity_mw(v, length, nc)
            if v > 0:
                link_agg[pair]["kvs"].add(int(v))
        for (z1, z2), data in link_agg.items():
            if z1 not in centroids or z2 not in centroids:
                continue
            mw = data["mw"]
            kv_str = " / ".join(f"{v} kV" for v in sorted(data["kvs"], reverse=True)) or "? kV"
            folium.PolyLine(
                [centroids[z1], centroids[z2]],
                color="#1a5fa8", weight=_mw_to_weight(mw), opacity=0.80,
                tooltip=folium.Tooltip(f"{z1} ↔ {z2} | {kv_str} | ~{mw:,} MW", sticky=True),
            ).add_to(fg_existing)

    fg_existing.add_to(m)
    fg_planned.add_to(m)

    # Cities
    fg_cities = folium.FeatureGroup(name="Cities", show=True)
    if cities is not None and len(cities) > 0:
        iso_col = next((c for c in cities.columns if "iso" in c.lower()), None)
        for _, zone_row in zones_gdf.iterrows():
            iso = zone_row.get("ISO_A3", "")
            if iso_col:
                zone_cities = cities[cities[iso_col].str.upper() == iso.upper()]
            else:
                zone_cities = cities
            top = zone_cities.nlargest(3, "pop") if "pop" in zone_cities.columns else zone_cities.head(3)
            for _, city in top.iterrows():
                folium.CircleMarker(
                    [city["lat"], city["lon"]], radius=4,
                    color="#1a237e", fill=True, fill_color="#5c6bc0", fill_opacity=0.65,
                    tooltip=city.get("name", ""),
                ).add_to(fg_cities)
    fg_cities.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))


# ── capacity mix Folium map ───────────────────────────────────────────────────

def _map_capacity_mix(zones_gdf, plants, out_path, folium):
    center = [zones_gdf.geometry.centroid.y.mean(), zones_gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")
    colors = _zone_color_palette(len(zones_gdf))

    for i, (_, zone_row) in enumerate(zones_gdf.iterrows()):
        folium.GeoJson(
            zone_row.geometry.__geo_interface__,
            style_function=lambda f, c=colors[i]: {
                "fillColor": c, "color": "#444", "weight": 1.5, "fillOpacity": 0.2,
            },
        ).add_to(m)

        # Capacity mix bar as HTML tooltip
        zone_name = zone_row["zone_name"]
        if "zone_name" in plants.columns:
            zone_plants = plants[plants["zone_name"] == zone_name]
        else:
            zone_plants = pd.DataFrame()

        centroid = zone_row.geometry.centroid
        html = _capacity_bar_html(zone_name, zone_plants)
        folium.Marker(
            [centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=html,
                icon_size=(120, 80),
                icon_anchor=(60, 40),
            ),
        ).add_to(m)

    m.save(str(out_path))


def _capacity_bar_html(zone_name, plants_df) -> str:
    if plants_df.empty or "fuel" not in plants_df.columns:
        return (f'<div style="background:rgba(255,255,255,0.85);padding:4px;'
                f'border-radius:4px;font-size:9px;font-weight:bold">{zone_name}<br>no data</div>')

    mix = plants_df.groupby("fuel")["capacity_mw"].sum().sort_values(ascending=False).head(5)
    total = mix.sum()
    bars = ""
    for fuel, mw in mix.items():
        pct = mw / total * 100 if total > 0 else 0
        c = FUEL_COLORS.get(str(fuel), "#90a4ae")
        bars += (f'<div style="display:flex;align-items:center;margin:1px 0">'
                 f'<div style="width:{max(int(pct),2)}px;height:8px;background:{c};'
                 f'border-radius:2px"></div>'
                 f'<span style="font-size:7px;margin-left:2px">{fuel[:4]} {mw:.0f}MW</span></div>')
    return (f'<div style="background:rgba(255,255,255,0.9);padding:3px;'
            f'border-radius:4px;min-width:100px">'
            f'<div style="font-size:9px;font-weight:bold;margin-bottom:2px">{zone_name}</div>'
            f'{bars}</div>')


# ── static PNG maps ───────────────────────────────────────────────────────────

def _png_simplified(zones_gdf, cities, interzone_lines, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = _zone_color_palette(len(zones_gdf))

    for i, (_, row) in enumerate(zones_gdf.iterrows()):
        geom = row.geometry
        if hasattr(geom, "geoms"):  # MultiPolygon
            for g in geom.geoms:
                xs, ys = g.exterior.xy
                ax.fill(xs, ys, alpha=0.35, color=colors[i], linewidth=0)
                ax.plot(xs, ys, color="#333", linewidth=0.8)
        else:
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, alpha=0.35, color=colors[i], linewidth=0)
            ax.plot(xs, ys, color="#333", linewidth=0.8)
        c = row.geometry.centroid
        ax.text(c.x, c.y, row["zone_name"], fontsize=7, ha="center", va="center",
                fontweight="bold", color="#222")

    # Inter-zone lines
    for link in interzone_lines:
        v = link.get("voltage_kv", 0)
        coords = link["coords"]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ax.plot(xs, ys, color=_voltage_color(v), linewidth=1.5, alpha=0.8, linestyle="--")

    ax.set_title("Zone Map — Simplified", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _png_capacity_mix(zones_gdf, plants, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = _zone_color_palette(len(zones_gdf))

    for i, (_, row) in enumerate(zones_gdf.iterrows()):
        geom = row.geometry
        polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for g in polys:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, alpha=0.2, color=colors[i])
            ax.plot(xs, ys, color="#333", linewidth=0.8)

        zone_name = row["zone_name"]
        c = row.geometry.centroid
        ax.text(c.x, c.y + 0.3, zone_name, fontsize=7, ha="center",
                fontweight="bold", color="#222")

        # Small pie chart at centroid
        if "zone_name" in plants.columns and not plants.empty:
            zp = plants[plants["zone_name"] == zone_name]
            if not zp.empty and "fuel" in zp.columns:
                mix = zp.groupby("fuel")["capacity_mw"].sum()
                pie_colors = [FUEL_COLORS.get(f, "#90a4ae") for f in mix.index]
                ax2 = fig.add_axes([0, 0, 0.1, 0.1])  # placeholder, positioned below
                try:
                    ax.pie(
                        mix.values, colors=pie_colors,
                        center=(c.x, c.y - 0.6), radius=0.4,
                        wedgeprops={"linewidth": 0.3, "edgecolor": "white"},
                    )
                except Exception:
                    pass

    ax.set_title("Zone Map — Installed Capacity Mix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    # Legend
    patches = [mpatches.Patch(color=c, label=f) for f, c in FUEL_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── helpers ───────────────────────────────────────────────────────────────────

def _zone_color_palette(n: int) -> list[str]:
    """Generate n visually distinct colors."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mc
        cmap = plt.get_cmap("tab20")
        return [mc.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
    except Exception:
        base = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
        return [base[i % len(base)] for i in range(n)]


def _voltage_color(voltage_kv: int) -> str:
    for v, c in sorted(VOLTAGE_COLORS.items(), reverse=True):
        if voltage_kv >= v:
            return c
    return VOLTAGE_COLORS[0]


def _add_voltage_legend(m):
    import folium
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:8px;border-radius:6px;
                border:1px solid #ccc;font-size:11px">
    <b>HV Lines</b><br>
    <span style="color:#d32f2f">&#9644;</span> 400 kV<br>
    <span style="color:#f57c00">&#9644;</span> 220 kV<br>
    <span style="color:#fbc02d">&#9644;</span> 150 kV<br>
    <span style="color:#388e3c">&#9644;</span> 110 kV
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))


def _add_fuel_legend(m):
    import folium
    items = "".join(
        f'<span style="color:{c}">&#9632;</span> {f}<br>'
        for f, c in list(FUEL_COLORS.items())[:8]
    )
    legend_html = (
        '<div style="position:fixed;bottom:30px;right:30px;z-index:1000;'
        'background:white;padding:8px;border-radius:6px;'
        'border:1px solid #ccc;font-size:11px">'
        f'<b>Plants</b><br>{items}</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EPM Zone Pipeline")
    parser.add_argument("--countries", nargs="+", required=True, metavar="ISO",
                        help="Country ISO_A3 codes (e.g. TUR ROU BGR GEO ARM AZE)")
    parser.add_argument("--n_zones", type=int, required=True,
                        help="Total number of zones across all countries")
    parser.add_argument("--output", default=None,
                        help="Output root directory (default: pre-analysis/output_workflow/)")
    args = parser.parse_args()

    out = Path(args.output) if args.output else None
    run_zone_pipeline(args.countries, args.n_zones, output_root=out)
