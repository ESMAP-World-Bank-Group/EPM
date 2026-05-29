"""
Convergence sweep runner.

Runs EPM on a grid of (n_zones x n_days) configurations and detects
at which point key outputs stabilize.

Status: PLACEHOLDER -- requires EPM inputs to be fully prepared first.
Use after completing spatial partitioning and temporal aggregation.
"""
from __future__ import annotations
from itertools import product
from typing import List
import pandas as pd


def sweep(
    spatial_candidates: List[int],
    temporal_candidates: List[int],
    epm_config: dict,
    thresholds: dict | None = None,
):
    """
    For each (n_zones, n_days) combination:
      1. Generate EPM inputs (zone map + representative days)
      2. Run EPM in simplified mode (single scenario, reduced techs)
      3. Collect key outputs

    thresholds: convergence criteria (relative change between levels)
      default: {"cost": 0.02, "investment": 0.05, "flows": 0.10}

    Returns: pd.DataFrame with one row per configuration + convergence report
    """
    if thresholds is None:
        thresholds = {"cost": 0.02, "investment": 0.05, "flows": 0.10}

    configs = list(product(spatial_candidates, temporal_candidates))
    print(f"Convergence sweep: {len(configs)} configurations to test")
    print(f"  Spatial candidates:  {spatial_candidates}")
    print(f"  Temporal candidates: {temporal_candidates}")
    print()
    print("Next steps to activate this runner:")
    print("  1. Complete spatial partitioning (spatial/zoner.py)")
    print("  2. Complete temporal aggregation (temporal/aggregator.py)")
    print("  3. Wire up EPM input generation here")
    print("  4. Call: python pre-analysis/resolution_advisor/convergence/runner.py")

    # Placeholder result structure
    rows = []
    for n_zones, n_days in configs:
        rows.append({
            "n_zones": n_zones,
            "n_days": n_days,
            "n_hours": n_days * 24,
            "status": "pending",
            "total_cost": None,
            "runtime_s": None,
        })
    return pd.DataFrame(rows)


def detect_convergence(results: pd.DataFrame, thresholds: dict) -> dict:
    """
    Identify the smallest (n_zones, n_days) where outputs stabilize.
    Stability = relative change < threshold between consecutive levels.
    """
    stable = {}
    for dim in ["n_zones", "n_days"]:
        grouped = results.groupby(dim)["total_cost"].mean().dropna()
        if len(grouped) < 2:
            stable[dim] = grouped.index[0] if len(grouped) else None
            continue
        pct_changes = grouped.pct_change().abs()
        converged = pct_changes[pct_changes < thresholds["cost"]]
        stable[dim] = converged.index[0] if len(converged) > 0 else grouped.index[-1]
    return stable
