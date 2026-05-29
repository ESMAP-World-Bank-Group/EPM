"""
Temporal period aggregation -- backends for tsam and Poncelet (Planning-tools).
Currently stubs; enable by setting ENABLED = True and installing dependencies.
"""
from __future__ import annotations

TSAM_ENABLED = False
PONCELET_ENABLED = False


def aggregate_tsam(time_series, n_periods: int, hours_per_period: int = 24,
                   method: str = "hierarchical"):
    """
    Select representative periods via tsam (RWTH Aachen).

    Requires:  pip install tsam
    Reference: Kotzur et al. 2018, Hoffmann et al. 2022
    Methods:   'hierarchical', 'k_medoids', 'k_means', 'k_maxoids'

    time_series: pd.DataFrame, hourly, columns = load/solar/wind/etc per zone
    Returns:    tsam TimeSeriesAggregation object
    """
    if not TSAM_ENABLED:
        raise NotImplementedError(
            "tsam backend not enabled. "
            "Set TSAM_ENABLED = True and run: pip install tsam"
        )
    import tsam.timeseriesaggregation as tsam_agg  # noqa: F401
    agg = tsam_agg.TimeSeriesAggregation(
        time_series,
        noTypicalPeriods=n_periods,
        hoursPerPeriod=hours_per_period,
        clusterMethod=method,
        extremePeriodMethod="new_cluster_center",
    )
    agg.createTypicalPeriods()
    return agg


def aggregate_poncelet(time_series, n_periods: int, n_bins: int = 10):
    """
    Select representative periods via MILP Poncelet (EPM Planning-tools).

    Requires: GAMS + the OptimizationModelZone.gms from pre-analysis/representative_days/
    Reference: Poncelet et al. 2017

    time_series: dict of pd.Series, one per variable (load, solar, wind)
    Returns:    dict with 'days' (list) and 'weights' (list)
    """
    if not PONCELET_ENABLED:
        raise NotImplementedError(
            "Poncelet backend not enabled. "
            "Set PONCELET_ENABLED = True and ensure GAMS + Planning-tools are available."
        )
    import sys
    sys.path.insert(0, "pre-analysis/representative_days")
    from representativedays_pipeline import launch_optim_repr_days  # noqa: F401
    return launch_optim_repr_days(time_series, n_periods, n_bins=n_bins)
