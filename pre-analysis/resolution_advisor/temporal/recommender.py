"""
Temporal resolution recommender.

Outputs:
  - floor_days:       minimum representative days
  - floor_extreme:    minimum extreme days (added on top)
  - ceiling_days:     maximum given compute budget + n_zones
  - candidates:       list of N_days to test
  - drivers:          justifications
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

from schema import AdvisorConfig


# -- floor rules --------------------------------------------------------------

def _temporal_floor(cfg: AdvisorConfig) -> Tuple[int, int, List[str]]:
    """
    Returns (n_repr_days, n_extreme_days, drivers).

    Logic:
      - Baseline: 4 days (1 per season), each 24h
      - RE penetration drives the minimum upward
      - Storage relevance requires longer chronological windows
      - Hydro seasonality requires adequate seasonal coverage
      - Multi-period (many years) pushes toward lower N for tractability
    """
    mu = cfg.model_use
    drivers = []
    n = 4  # absolute minimum: 1 representative day per season

    # RE penetration
    if mu.re_penetration_target >= 0.20:
        n = max(n, 8)
        drivers.append(
            f"RE target {mu.re_penetration_target:.0%} >= 20% -> min 8 days "
            f"(need to capture weekly wind variability)"
        )
    if mu.re_penetration_target >= 0.35:
        n = max(n, 12)
        drivers.append(
            f"RE target {mu.re_penetration_target:.0%} >= 35% -> min 12 days "
            f"(need to capture low-RE / high-demand coincidence)"
        )
    if mu.re_penetration_target >= 0.50:
        n = max(n, 16)
        drivers.append(
            f"RE target {mu.re_penetration_target:.0%} >= 50% -> min 16 days"
        )

    # Storage relevance -- needs longer chronological windows
    storage_boost = {"low": 0, "medium": 2, "high": 4}
    boost = storage_boost[mu.storage_relevance]
    if boost > 0:
        n += boost
        drivers.append(
            f"storage relevance '{mu.storage_relevance}' adds {boost} days "
            f"(multi-day charge/discharge cycles must be represented)"
        )

    # Hydro seasonality
    if mu.hydro_seasonality == "high":
        n = max(n, 8)
        drivers.append(
            "high hydro seasonality -> at least 8 days "
            "(wet/dry seasons must each be covered)"
        )

    # Tractability pressure from many planning periods
    if len(mu.multi_period_years) >= 5:
        drivers.append(
            f"{len(mu.multi_period_years)} planning years -> "
            "keep N_days moderate for tractability"
        )

    # Extreme days (always added on top of representative days)
    n_extreme = 2
    if mu.re_penetration_target >= 0.30:
        n_extreme = 3
        drivers.append(
            ">=3 extreme days mandatory: peak demand, min-solar/max-load coincidence, "
            "wind drought"
        )
    else:
        drivers.append("2 extreme days: winter peak + min-RE event")

    return n, n_extreme, drivers


# -- ceiling ------------------------------------------------------------------

def _temporal_ceiling(n_repr_floor: int, n_zones: int, cfg: AdvisorConfig) -> Tuple[int, str]:
    """
    Maximum representative days given compute budget and spatial resolution.
    Same variable-budget heuristic as spatial, but solving for N_days.
    """
    budget = cfg.constraints.variable_budget
    n_years = cfg.model_use.n_periods
    n_scenarios = cfg.model_use.n_scenarios

    # solve: n_zones * (n_days * 24) * n_years * n_scenarios <= budget
    max_days = budget // (n_zones * 24 * n_years * n_scenarios)
    max_days = max(max_days, n_repr_floor)
    max_days = min(max_days, 36)   # hard cap: beyond ~36 days, gains are marginal

    reason = (
        f"variable_budget={budget:,} "
        f"/ ({n_zones} zones x 24h x {n_years}y x {n_scenarios}s) = {max_days} days"
    )
    return max_days, reason


# -- candidates ---------------------------------------------------------------

def _candidates(floor: int, ceiling: int) -> List[int]:
    pts = {floor}
    mid = (floor + ceiling) // 2
    if mid > floor:
        pts.add(mid)
    if floor + 4 <= ceiling:
        pts.add(floor + 4)
    pts.add(ceiling)
    return sorted(pts)


# -- main entry point ---------------------------------------------------------

@dataclass
class TemporalRecommendation:
    floor_repr_days: int
    floor_extreme_days: int
    ceiling_repr_days: int
    ceiling_reason: str
    candidates: List[int]
    drivers: List[str]
    total_hours_at_floor: int
    total_hours_at_ceiling: int

    @property
    def recommended_repr_days(self) -> int:
        return self.floor_repr_days

    @property
    def recommended_total_days(self) -> int:
        return self.floor_repr_days + self.floor_extreme_days


def recommend(cfg: AdvisorConfig, n_zones: int) -> TemporalRecommendation:
    floor_repr, floor_extreme, drivers = _temporal_floor(cfg)
    ceiling_repr, ceiling_reason = _temporal_ceiling(floor_repr, n_zones, cfg)
    candidates = _candidates(floor_repr, ceiling_repr)

    return TemporalRecommendation(
        floor_repr_days=floor_repr,
        floor_extreme_days=floor_extreme,
        ceiling_repr_days=ceiling_repr,
        ceiling_reason=ceiling_reason,
        candidates=candidates,
        drivers=drivers,
        total_hours_at_floor=(floor_repr + floor_extreme) * 24,
        total_hours_at_ceiling=(ceiling_repr + floor_extreme) * 24,
    )
