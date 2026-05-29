"""
Determine if hydro generation is concentrated far from load centers.

Method:
  - Find all hydro plants in the country (from GPPD)
  - Compute their geographic centroid (weighted by capacity_mw)
  - Compute the load centroid (weighted by city population)
  - If distance between the two centroids > threshold_km -> hydro is remote
"""
from __future__ import annotations
import math
from typing import Tuple


THRESHOLD_KM = 150


def compute_hydro_concentration(
    country_iso: str,
    plants_df,
    cities_df,
    threshold_km: float = THRESHOLD_KM,
) -> Tuple[bool, float, str]:
    """
    Returns (is_concentrated_remote, distance_km, note).
    """
    # Filter hydro plants
    hydro = plants_df[
        (plants_df["country_iso"] == country_iso) &
        (plants_df["fuel"].str.lower().isin(["hydro", "hydroelectric", "run-of-river"]) if "fuel" in plants_df.columns else True)
    ].copy()

    if len(hydro) == 0:
        return False, 0.0, "no hydro plants found in GPPD"

    # Weighted centroid of hydro plants
    weights = hydro["capacity_mw"].clip(lower=1.0).fillna(1.0)
    hydro_lat = float((hydro["lat"] * weights).sum() / weights.sum())
    hydro_lon = float((hydro["lon"] * weights).sum() / weights.sum())

    # Load centroid from cities
    iso_col = next(
        (c for c in ["iso_a3", "ISO_A3", "ADM0_A3"] if c in cities_df.columns), None
    )
    if iso_col and len(cities_df) > 0:
        country_cities = cities_df[cities_df[iso_col] == country_iso].copy()
        if len(country_cities) > 0:
            pop = country_cities["pop"].clip(lower=1.0).fillna(1.0) if "pop" in country_cities.columns else None
            if pop is not None:
                load_lat = float((country_cities["lat"] * pop).sum() / pop.sum())
                load_lon = float((country_cities["lon"] * pop).sum() / pop.sum())
            else:
                load_lat = float(country_cities["lat"].mean())
                load_lon = float(country_cities["lon"].mean())
        else:
            return False, 0.0, "no cities found -- cannot assess hydro distance"
    else:
        return False, 0.0, "no city data -- cannot assess hydro distance"

    dist = _haversine_km((hydro_lat, hydro_lon), (load_lat, load_lon))
    is_remote = dist >= threshold_km
    n_plants = len(hydro)
    total_mw = float(weights.sum())
    note = (
        f"{n_plants} hydro plants ({total_mw:.0f} MW total), "
        f"centroid {dist:.0f} km from load centroid "
        f"(threshold: {threshold_km} km)"
    )
    return is_remote, round(dist, 1), note


def _haversine_km(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    R = 6371.0
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))
