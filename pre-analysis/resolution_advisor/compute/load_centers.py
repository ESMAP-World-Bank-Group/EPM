"""
Determine if a country has distant load centers.

Method: take top-N cities by population, compute pairwise great-circle
distances. If the max pairwise distance exceeds threshold_km, the country
has distant load centers that justify separate zones.
"""
from __future__ import annotations
import math
from typing import Tuple


THRESHOLD_KM = 350  # cities further apart than this -> distant load centers
TOP_N_CITIES = 5


def compute_distant_load_centers(
    country_iso: str,
    cities_df,
    threshold_km: float = THRESHOLD_KM,
) -> Tuple[bool, float, str]:
    """
    Returns (is_distant, max_distance_km, note).
    """
    iso_col = next(
        (c for c in ["iso_a3", "ISO_A3", "ADM0_A3"] if c in cities_df.columns), None
    )
    if iso_col is None or cities_df.empty:
        return False, 0.0, "no city data"

    country_cities = cities_df[cities_df[iso_col] == country_iso].copy()
    if len(country_cities) == 0:
        return False, 0.0, f"no cities found for {country_iso}"

    # Sort by population descending, take top N
    if "pop" in country_cities.columns:
        country_cities = country_cities.nlargest(TOP_N_CITIES, "pop")

    coords = list(zip(country_cities["lat"], country_cities["lon"]))

    if len(coords) < 2:
        return False, 0.0, "only 1 city found"

    max_dist = 0.0
    city_names = country_cities["name"].tolist() if "name" in country_cities.columns else []

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = _haversine_km(coords[i], coords[j])
            if d > max_dist:
                max_dist = d

    is_distant = max_dist >= threshold_km
    n_cities = len(coords)
    note = (
        f"top-{n_cities} cities max separation: {max_dist:.0f} km "
        f"(threshold: {threshold_km:.0f} km)"
    )
    if city_names:
        note += f" | cities: {', '.join(city_names[:3])}"

    return is_distant, round(max_dist, 1), note


def _haversine_km(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))
