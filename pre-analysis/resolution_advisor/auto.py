"""
Auto-compute flow: given a list of country ISO codes, fetch open data and
compute all CountryConfig parameters automatically.

Returns a dict of CountryConfig objects ready for the spatial recommender,
plus a provenance dict explaining the source of each parameter.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List

# Fix pyproj PROJ database path before any geo imports
sys.path.insert(0, str(Path(__file__).parent / "fetch"))
import _proj_fix  # noqa: F401

from schema import CountryConfig


def build_country_configs(
    country_isos: List[str],
    verbose: bool = True,
) -> tuple[dict[str, CountryConfig], dict[str, dict]]:
    """
    For each country ISO, fetch data and compute parameters.

    Returns:
        configs    -- {iso: CountryConfig}
        provenance -- {iso: {param: source_note}}
    """
    _log = print if verbose else (lambda *a: None)

    _log("[auto] Loading open data...")
    boundaries, cities, plants = _load_data(country_isos, _log)

    configs: dict[str, CountryConfig] = {}
    provenance: dict[str, dict] = {}

    for iso in country_isos:
        _log(f"  [{iso}] Computing parameters...")
        cfg, prov = _compute_country(iso, boundaries, cities, plants, _log)
        configs[iso] = cfg
        provenance[iso] = prov

    return configs, provenance


# -- data loading --------------------------------------------------------------

def _load_data(country_isos, log):
    from fetch.natural_earth import load_boundaries, load_cities
    from fetch.gppd import load_gppd

    log("  Loading Natural Earth boundaries...")
    try:
        boundaries = load_boundaries(country_isos)
    except Exception as e:
        log(f"  [!] Boundaries failed: {e}")
        boundaries = None

    log("  Loading Natural Earth cities...")
    try:
        cities = load_cities(country_isos, min_pop=50_000)
    except Exception as e:
        log(f"  [!] Cities failed: {e}")
        cities = None

    log("  Loading Global Power Plant Database...")
    try:
        plants = load_gppd(country_isos)
    except Exception as e:
        log(f"  [!] GPPD failed: {e}")
        plants = None

    return boundaries, cities, plants


# -- per-country computation ---------------------------------------------------

def _compute_country(iso, boundaries, cities, plants, log) -> tuple[CountryConfig, dict]:
    prov = {}

    # 1. Area
    area_km2 = _safe_area(iso, boundaries, prov, log)

    # 2. RE spread
    re_cf_spread = _safe_re_spread(iso, boundaries, prov, log)

    # 3. Distant load centers
    distant_load_centers = _safe_load_centers(iso, cities, prov, log)

    # 4. Hydro concentration
    hydro_concentration = _safe_hydro(iso, plants, cities, prov, log)

    # 5. Network bottlenecks (OSM)
    known_congestion_splits = _safe_bottlenecks(iso, boundaries, prov, log)

    # 6. Data quality -- assessed from OSM coverage
    data_quality = _assess_data_quality(iso, prov)
    prov["data_quality"] = f"assessed from OSM coverage: {data_quality}"

    cfg = CountryConfig(
        name=iso,
        area_km2=area_km2,
        n_bidding_zones=1,           # cannot automate reliably -- keep default
        known_congestion_splits=known_congestion_splits,
        re_cf_spread=re_cf_spread,
        distant_load_centers=distant_load_centers,
        hydro_concentration=hydro_concentration,
        data_quality=data_quality,
    )
    return cfg, prov


# -- individual computations with safe fallbacks -------------------------------

def _safe_area(iso, boundaries, prov, log) -> float:
    if boundaries is None:
        prov["area_km2"] = "failed -- Natural Earth not loaded"
        return 100_000.0
    try:
        from compute.area import compute_area_km2
        area = compute_area_km2(iso, boundaries)
        prov["area_km2"] = f"Natural Earth: {area:,.0f} km2"
        return area
    except Exception as e:
        prov["area_km2"] = f"error: {e}"
        return 100_000.0


def _safe_re_spread(iso, boundaries, prov, log) -> float:
    if boundaries is None:
        prov["re_cf_spread"] = "failed -- Natural Earth not loaded"
        return 0.15
    try:
        from compute.re_spread import compute_re_spread
        spread, note = compute_re_spread(iso, boundaries)
        prov["re_cf_spread"] = note
        return spread
    except Exception as e:
        prov["re_cf_spread"] = f"error: {e}"
        return 0.15


def _safe_load_centers(iso, cities, prov, log) -> bool:
    if cities is None or len(cities) == 0:
        prov["distant_load_centers"] = "failed -- city data not loaded"
        return False
    try:
        from compute.load_centers import compute_distant_load_centers
        is_distant, max_km, note = compute_distant_load_centers(iso, cities)
        prov["distant_load_centers"] = note
        return is_distant
    except Exception as e:
        prov["distant_load_centers"] = f"error: {e}"
        return False


def _safe_hydro(iso, plants, cities, prov, log) -> bool:
    if plants is None or cities is None:
        prov["hydro_concentration"] = "failed -- GPPD or cities not loaded"
        return False
    try:
        from compute.hydro_concentration import compute_hydro_concentration
        is_remote, dist_km, note = compute_hydro_concentration(iso, plants, cities)
        prov["hydro_concentration"] = note
        return is_remote
    except Exception as e:
        prov["hydro_concentration"] = f"error: {e}"
        return False


def _safe_bottlenecks(iso, boundaries, prov, log) -> int:
    if boundaries is None:
        prov["known_congestion_splits"] = "failed -- boundaries not loaded"
        return 0
    try:
        from fetch.osm import fetch_substations, fetch_hv_lines
        from compute.network_bottlenecks import compute_network_bottlenecks

        subset = boundaries[boundaries["ISO_A3"] == iso]
        if subset.empty:
            prov["known_congestion_splits"] = f"country {iso} not in boundaries"
            return 0

        bounds = subset.geometry.total_bounds  # minx, miny, maxx, maxy
        bbox = (bounds[1], bounds[0], bounds[3], bounds[2])  # s,w,n,e

        log(f"    [{iso}] Fetching OSM substations...")
        substations = fetch_substations(bbox)
        log(f"    [{iso}] Fetching OSM HV lines...")
        hv_lines = fetch_hv_lines(bbox)
        log(f"    [{iso}] {len(substations)} substations, {len(hv_lines)} HV lines")

        n, note = compute_network_bottlenecks(iso, substations, hv_lines)
        prov["known_congestion_splits"] = note
        return n

    except Exception as e:
        prov["known_congestion_splits"] = f"error: {e}"
        return 0


def _assess_data_quality(iso: str, prov: dict) -> str:
    """
    Infer data quality from OSM coverage note in provenance.
    'good'    -- OSM has enough lines for graph analysis
    'medium'  -- OSM has some lines but analysis was limited
    'limited' -- OSM too sparse or fetch failed
    """
    osm_note = prov.get("known_congestion_splits", "")
    if "sparse" in osm_note or "failed" in osm_note or "error" in osm_note:
        return "limited"
    if "too small" in osm_note or "skipped" in osm_note:
        return "medium"
    return "good"
