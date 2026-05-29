"""Compute country area in km2 from Natural Earth boundaries."""
from __future__ import annotations


def compute_area_km2(country_iso: str, boundaries_gdf) -> float:
    """
    Compute land area of a country in km2.
    Uses equal-area projection (EPSG:6933) for accuracy.
    """
    subset = boundaries_gdf[boundaries_gdf["ISO_A3"] == country_iso]
    if subset.empty:
        return 0.0
    try:
        projected = subset.to_crs("EPSG:6933")
        area_m2 = float(projected.geometry.area.sum())
        return round(area_m2 / 1e6, 0)
    except Exception:
        # Fallback: spherical approximation from bounding box
        bounds = subset.geometry.total_bounds  # minx, miny, maxx, maxy
        lat_span = abs(bounds[3] - bounds[1])
        lon_span = abs(bounds[2] - bounds[0])
        # 1 deg lat ~ 111 km, 1 deg lon ~ 111*cos(lat) km
        import math
        mid_lat = math.radians((bounds[1] + bounds[3]) / 2)
        return round(lat_span * 111 * lon_span * 111 * math.cos(mid_lat), 0)
