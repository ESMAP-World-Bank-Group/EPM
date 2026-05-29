"""
Spatial resolution recommender.

For each country, computes:
  - floor_physical: minimum zones justified by physical/network drivers
  - floor_capped:   floor_physical capped by data quality
  - ceiling:        maximum zones given compute budget
  - recommended:    pragmatic pick (floor_capped, +3, +6, ceiling)
  - drivers:        list of reasons (for the report)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

from schema import AdvisorConfig, CountryConfig, Constraints


# -- thresholds --------------------------------------------------------------

RE_SPREAD_THRESHOLD = 0.25   # fraction -- above this, RE heterogeneity adds a zone
AREA_LARGE_KM2      = 500_000


# -- per-country floor --------------------------------------------------------

@dataclass
class CountryZoneResult:
    iso: str
    name: str
    floor_physical: int
    floor_capped: int
    drivers: List[str]
    data_cap_active: bool


def _country_floor(iso: str, c: CountryConfig) -> CountryZoneResult:
    drivers = []
    n = 1  # every country gets at least 1 zone

    # official market splits
    if c.n_bidding_zones > 1:
        n = max(n, c.n_bidding_zones)
        drivers.append(f"{c.n_bidding_zones} official bidding zones")

    # known network congestion
    if c.known_congestion_splits > 0:
        n += c.known_congestion_splits
        drivers.append(
            f"+{c.known_congestion_splits} documented congestion corridor(s)"
        )

    # RE resource heterogeneity
    if c.re_cf_spread > RE_SPREAD_THRESHOLD:
        n += 1
        drivers.append(
            f"RE capacity-factor spread {c.re_cf_spread:.0%} > {RE_SPREAD_THRESHOLD:.0%} threshold"
        )

    # large country
    if c.area_km2 > AREA_LARGE_KM2:
        n += 1
        drivers.append(f"area {c.area_km2/1e3:.0f} k km2 > {AREA_LARGE_KM2/1e3:.0f} k km2 threshold")

    # distant load centers
    if c.distant_load_centers:
        n += 1
        drivers.append("load centers geographically separated")

    # hydro far from load
    if c.hydro_concentration:
        n += 1
        drivers.append("major hydro resources remote from load")

    floor_physical = n
    floor_capped = min(n, c.data_cap)
    data_cap_active = floor_capped < floor_physical

    if data_cap_active:
        drivers.append(
            f"[!] data quality '{c.data_quality}' caps at {c.data_cap} zone(s) "
            f"(physical need: {floor_physical})"
        )

    if not drivers or all(d.startswith("[!]") for d in drivers):
        drivers.insert(0, "no strong driver -- single zone sufficient")

    return CountryZoneResult(
        iso=iso,
        name=c.name,
        floor_physical=floor_physical,
        floor_capped=floor_capped,
        drivers=drivers,
        data_cap_active=data_cap_active,
    )


# -- regional ceiling ---------------------------------------------------------

def _regional_ceiling(
    floor_total: int,
    cfg: AdvisorConfig,
) -> Tuple[int, str]:
    """
    Maximum total zones the compute budget can afford.
    Heuristic: N_zones * N_hours * N_years * N_scenarios <= variable_budget
    """
    n_hours_repr = 24 * 16     # assume 16 representative days as starting point
    n_years = cfg.model_use.n_periods
    n_scenarios = cfg.model_use.n_scenarios
    budget = cfg.constraints.variable_budget

    ceiling = budget // (n_hours_repr * n_years * n_scenarios)
    ceiling = max(ceiling, floor_total)   # never below floor
    ceiling = min(ceiling, 30)            # hard cap: beyond 30 zones EPM gets unwieldy

    reason = (
        f"variable_budget={cfg.constraints.variable_budget:,} "
        f"/ ({n_hours_repr}h x {n_years}y x {n_scenarios}s) = {ceiling}"
    )
    return ceiling, reason


# -- candidate N values -------------------------------------------------------

def _candidates(floor: int, ceiling: int) -> List[int]:
    """3-4 values to test for convergence."""
    pts = {floor}
    if floor + 3 <= ceiling:
        pts.add(floor + 3)
    if floor + 6 <= ceiling:
        pts.add(floor + 6)
    pts.add(ceiling)
    return sorted(pts)


# -- main entry point ---------------------------------------------------------

@dataclass
class SpatialRecommendation:
    country_results: List[CountryZoneResult]
    floor_total: int
    ceiling_total: int
    ceiling_reason: str
    candidates: List[int]
    data_caps_active: List[str]  # ISOs where data quality limited the floor

    @property
    def recommended(self) -> int:
        return self.floor_total  # conservative default; user can go up to ceiling


def recommend(cfg: AdvisorConfig) -> SpatialRecommendation:
    country_results = [
        _country_floor(iso, c)
        for iso, c in cfg.region.countries.items()
    ]

    floor_total = sum(r.floor_capped for r in country_results)
    ceiling_total, ceiling_reason = _regional_ceiling(floor_total, cfg)
    candidates = _candidates(floor_total, ceiling_total)
    data_caps_active = [r.iso for r in country_results if r.data_cap_active]

    return SpatialRecommendation(
        country_results=country_results,
        floor_total=floor_total,
        ceiling_total=ceiling_total,
        ceiling_reason=ceiling_reason,
        candidates=candidates,
        data_caps_active=data_caps_active,
    )
