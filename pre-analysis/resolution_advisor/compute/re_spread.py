"""
Estimate RE capacity-factor spread across a country.

No API keys required -- uses a geographic proxy:
  - Solar CF is primarily driven by latitude (lower = more sun)
  - Wind CF is driven by coastal exposure and terrain
  - Countries with large N-S extent or mixed coastal/inland areas
    have higher RE heterogeneity

Returns a float in [0, 1] representing the estimated CF spread.
Flagged as 'estimated' in all outputs.
"""
from __future__ import annotations
import math


def compute_re_spread(country_iso: str, boundaries_gdf) -> tuple[float, str]:
    """
    Returns (spread_estimate, source_note).
    source_note is always 'geographic proxy (no API)'.
    """
    subset = boundaries_gdf[boundaries_gdf["ISO_A3"] == country_iso]
    if subset.empty:
        return 0.15, "default (country not found)"

    bounds = subset.geometry.total_bounds  # minx, miny, maxx, maxy
    minx, miny, maxx, maxy = bounds

    # N-S latitude span: drives solar CF variation
    lat_span_deg = abs(maxy - miny)
    # Solar CF varies ~0.02 per degree of latitude at mid-latitudes
    solar_spread = min(lat_span_deg * 0.022, 0.35)

    # E-W longitude span as proxy for terrain/coastal diversity
    lon_span_deg = abs(maxx - minx)
    # Large E-W extent -> higher chance of coastal vs. inland wind variation
    wind_spread = min(lon_span_deg * 0.008, 0.20)

    # Combined spread proxy
    spread = round(min(solar_spread + wind_spread * 0.5, 0.50), 2)

    note = (
        f"geographic proxy: lat_span={lat_span_deg:.1f}deg, "
        f"lon_span={lon_span_deg:.1f}deg -> spread~{spread:.2f}"
    )
    return spread, note
