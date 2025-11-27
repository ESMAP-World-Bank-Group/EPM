# Representative Days and Special Days Algorithm

This document summarizes how the pipeline in `pre-analysis/prepare-data/representative_days/repdays_pipeline.py` constructs representative days (repr days) and special days.

## Overview

1. **Prepare inputs:** Raw hourly CSVs per tech/zone are cleaned and mapped from month/day/hour to season/day/hour (`prepare_input_timeseries` and `month_to_season`).
2. **Pick representative year:** Tech profiles are combined and a representative year is chosen by minimizing deviation from the average profile (`prepare_energy_profiles` -> `find_representative_year` with `average_profile`).
3. **Cluster days:** For each season, daily profiles are clustered (KMeans) to find centroid days and cluster probabilities (`cluster_data_new`).
4. **Select special days:** Per season, one extreme day is chosen for each tech:
   - Features: min PV, min Wind, max Load (summing all zones for the feature to get a system-wide extreme).
   - Filtering: clusters above a data-driven probability threshold are excluded (threshold = 75th percentile per season unless overridden). If that would discard the global extreme, the extreme cluster is added back; if filtering empties the set, all clusters are used.
   - Output: one special day per feature/season, with a rule label (`min_PV`, `min_Wind`, `max_Load`) and weight equal to the cluster’s day count.
5. **Optimize repr days:** A GAMS model selects representative days and weights from the remaining clustered days (`launch_optim_repr_days` + `parse_repr_days`). Special days are appended with their weights and rule labels.
6. **Export artifacts:** Repr-day tables and EPM inputs are written. A summary CSV (`representative_days_summary.csv`) lists each selected day (repr or special) with average capacity factors per tech, weight, and weight percentage, plus min/avg/max benchmarks across the full dataset.

## Key Details

- **Aggregation across zones:** Special-day extremes are computed on the summed feature across all zones, so each tech yields one special day per season for the whole system (not per country).
- **Leap-year handling:** Feb 29 is removed earlier in the load and time-series prep steps to keep consistent day counts.
- **Weights:** Repr-day weights come from the optimization (rounded). Special-day weights are derived from the size of the selected cluster (number of days). The summary export also reports each selected day’s share of total weight.
- **Robustness:** Special-day selection tolerates empty filtered sets, preserves true extremes even if they lie in high-probability clusters, and skips gracefully if inputs are incomplete (with a warning in the summary export helper).
