"""
Assessment plotting routines for scenario comparisons.

This module centralizes the logic for building base vs. counterfactual
assessment charts (cost, capacity, and energy mix) to keep the main
post-processing flow slimmer.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .plots import make_auto_yaxis_formatter
from .plots import make_stacked_barplot as _make_stacked_barplot
from .plots import make_fuel_dispatch_diff_plot
from .utils import log_warning, filter_dataframe


def _wrap_plot_function(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            label = kwargs.get("filename") or kwargs.get("title")
            log_warning(f"Failed to generate {label}: {err}")

    return wrapper


make_stacked_barplot = _wrap_plot_function(_make_stacked_barplot)


def _beautify_scenario_name(name: str) -> str:
    """
    Convert internal scenario names to readable display labels.
    Beautifies sensitivity scenarios (containing '~').
    Does NOT handle '@' - callers should split on '@' first if needed.

    Examples:
        'baseline' -> 'baseline'
        'baseline~NoBiomass' -> 'NoBiomass'
        'baseline~pDemandForecast_015' -> 'HighDemand(+15%)'
        'baseline~pDemandForecast_-015' -> 'LowDemand(-15%)'
        'baseline~TradePrice_015' -> 'HighTradePrice(+15%)'
        'baseline~TradePrice_-015' -> 'LowTradePrice(-15%)'
        'baseline~NoWind' -> 'NoWind'
    """
    import re

    # Remove 'baseline~' prefix if present (sensitivity scenarios use ~ separator)
    if name.startswith('baseline~'):
        name = name[len('baseline~'):]
    # Also handle legacy 'baseline_' prefix for backward compatibility
    elif name.startswith('baseline_'):
        name = name[len('baseline_'):]

    # Handle pDemandForecast patterns
    # Format: pDemandForecast_XX where XX represents 0.XX (e.g., 03 = 0.3 = 30%, 3 = 3.0 = 300%, 015 = 0.15 = 15%)
    match = re.match(r'pDemandForecast_(-?)(\d+)', name)
    if match:
        sign = match.group(1)
        raw_value = match.group(2)
        # Insert decimal point after first digit: "03" -> "0.3", "3" -> "3.", "015" -> "0.15"
        decimal_str = raw_value[0] + '.' + raw_value[1:] if len(raw_value) > 1 else raw_value + '.0'
        value = int(float(decimal_str) * 100)
        if sign == '-':
            return f'LowDemand(-{value}%)'
        else:
            return f'HighDemand(+{value}%)'

    # Handle TradePrice patterns
    # Format: TradePrice_XX where XX represents 0.XX (e.g., 03 = 0.3 = 30%, 3 = 3.0 = 300%, 015 = 0.15 = 15%)
    match = re.match(r'TradePrice_(-?)(\d+)', name)
    if match:
        sign = match.group(1)
        raw_value = match.group(2)
        # Insert decimal point after first digit: "03" -> "0.3", "3" -> "3.", "015" -> "0.15"
        decimal_str = raw_value[0] + '.' + raw_value[1:] if len(raw_value) > 1 else raw_value + '.0'
        value = int(float(decimal_str) * 100)
        if sign == '-':
            return f'LowTradePrice(-{value}%)'
        else:
            return f'HighTradePrice(+{value}%)'

    return name


def _compute_pairwise_differences(
    df: pd.DataFrame,
    scenario_pairs: dict[str, Iterable[str]],
    merge_cols: list[str],
) -> pd.DataFrame:
    """Return stacked differences for every base/counterfactual pair."""
    all_diffs = []
    for scenario_base, counterfactuals in scenario_pairs.items():
        for scenario_cf in counterfactuals:
            df_base = df[df["scenario"] == scenario_base].copy()
            df_cf = df[df["scenario"] == scenario_cf].copy()

            if df_base.empty or df_cf.empty:
                continue

            df_merged = pd.merge(
                df_base[merge_cols + ["value"]],
                df_cf[merge_cols + ["value"]],
                on=merge_cols,
                suffixes=("_base", "_cf"),
                how="outer",
            )
            df_merged["value_base"] = df_merged["value_base"].fillna(0)
            df_merged["value_cf"] = df_merged["value_cf"].fillna(0)
            # Project minus counterfactual (positive = project adds value)
            df_merged["value"] = df_merged["value_cf"] - df_merged["value_base"]
            # Simplified label: base scenario name (before @) for display
            # e.g., "baseline@rehabilitation" -> "baseline", "baseline_NoBiomass@rehabilitation" -> "NoBiomass"
            raw_label = scenario_cf.split('@')[0] if '@' in scenario_cf else scenario_cf
            df_merged["scenario"] = _beautify_scenario_name(raw_label)
            all_diffs.append(df_merged[merge_cols + ["scenario", "value"]])

    if not all_diffs:
        return pd.DataFrame(columns=merge_cols + ["scenario", "value"])

    return pd.concat(all_diffs, ignore_index=True)


def _plot_assessment_diffs(
    df_all_diffs: pd.DataFrame,
    scenario_pairs: dict[str, Iterable[str]],
    dict_specs: dict,
    folder: str,
    *,
    stacked_column: str,
    x_column: str,
    format_y,
    title_prefix: str,
    filename_prefix: str,
    annotate: bool = False,
    show_total_single=False,
):
    """Plot the aggregated and per-pair assessment figures."""
    if df_all_diffs.empty:
        return

    multiple_pairs = len(scenario_pairs) > 1 or any(
        len(counterfactuals) > 1 for counterfactuals in scenario_pairs.values()
    )

    # Figure with subplots by year, all scenarios on same axes
    if multiple_pairs:
        filename = os.path.join(folder, f"{filename_prefix}_AllPairs.pdf")
        make_stacked_barplot(
            df_all_diffs,
            filename,
            dict_specs["colors"],
            column_stacked=stacked_column,
            column_subplot=x_column,
            column_xaxis="scenario",
            column_value="value",
            format_y=format_y,
            rotation=45,
            annotate=annotate,
            title=f"{title_prefix} (Project - Counterfactual)",
            show_total=True,
        )

    # Individual figure only for baseline pair (no sensitivity scenarios)
    if "baseline" in scenario_pairs:
        for scenario_cf in scenario_pairs["baseline"]:
            # Scenario label is the base name (before @), beautified
            raw_label = scenario_cf.split('@')[0] if '@' in scenario_cf else scenario_cf
            scenario_label = _beautify_scenario_name(raw_label)
            df_pair = df_all_diffs[
                df_all_diffs["scenario"] == scenario_label
            ]
            if df_pair.empty:
                continue

            # Extract project name from counterfactual (after @)
            project_name = scenario_cf.split('@')[1] if '@' in scenario_cf else scenario_cf
            filename = os.path.join(
                folder, f"{filename_prefix}_{project_name}.pdf"
            )
            make_stacked_barplot(
                df_pair,
                filename,
                dict_specs["colors"],
                column_stacked=stacked_column,
                column_subplot=None,
                column_xaxis=x_column,
                column_value="value",
                format_y=format_y,
                rotation=0,
                annotate=annotate,
                title=f"{title_prefix} (Project - Counterfactual)",
                show_total=show_total_single,
            )

    # Figure with subplots per scenario pair (shows evolution over years for each pair)
    if multiple_pairs:
        filename = os.path.join(folder, f"{filename_prefix}_ByPair.pdf")
        make_stacked_barplot(
            df_all_diffs,
            filename,
            dict_specs["colors"],
            column_stacked=stacked_column,
            column_subplot="scenario",
            column_xaxis=x_column,
            column_value="value",
            format_y=format_y,
            rotation=45,
            annotate=annotate,
            title=f"{title_prefix} (Project - Counterfactual)",
            show_total=True,
        )


def _simplify_attributes(df: pd.DataFrame, new_label: str, attributes: Iterable[str]):
    """Group specific attributes together using the supplied label."""
    df_grouped = df[df["attribute"].isin(attributes)].copy()
    if df_grouped.empty:
        return df

    df_grouped = (
        df_grouped.groupby(
            [col for col in df_grouped.columns if col not in ["attribute", "value"]],
            as_index=False,
            observed=False,
        )["value"].sum()
    )
    df_grouped["attribute"] = new_label

    return pd.concat(
        [df[~df["attribute"].isin(attributes)], df_grouped], ignore_index=True
    )


def make_assessment_cost_diff(
    epm_results,
    dict_specs,
    folder,
    scenario_pairs,
    trade_attrs=None,
    reserve_attrs=None,
):
    """
    Generate cost difference bar plots for project assessment.

    Uses the same data basis as the CostSystemEvolution figure (yearly system
    costs aggregated across zones) to keep visuals consistent.
    """
    df = epm_results["pCosts"].copy()

    if reserve_attrs:
        df = _simplify_attributes(df, "Unmet reserve costs: $m", reserve_attrs)
    if trade_attrs:
        df = _simplify_attributes(df, "Trade costs: $m", trade_attrs)

    df = df.loc[df.attribute != "NPV of system cost: $m"]
    df["attribute"] = df["attribute"].str.replace(": $m", "", regex=False)

    grouping_cols = [col for col in df.columns if col not in ["value", "zone"]]
    df = df.groupby(grouping_cols, as_index=False, observed=False)["value"].sum()

    df_all_diffs = _compute_pairwise_differences(
        df, scenario_pairs, merge_cols=["year", "attribute"]
    )

    _plot_assessment_diffs(
        df_all_diffs,
        scenario_pairs,
        dict_specs,
        folder,
        stacked_column="attribute",
        x_column="year",
        format_y=make_auto_yaxis_formatter("m$"),
        title_prefix="Cost Difference (million USD)",
        filename_prefix="AssessmentCostDiff",
        annotate=False,
        show_total_single=True,
    )


def make_assessment_npv_comparison(
    epm_results,
    dict_specs,
    folder,
    scenario_pairs,
    trade_attrs=None,
    reserve_attrs=None,
):
    """
    Generate bar charts comparing NPV of system cost across scenario pairs.

    Uses the same data basis and formatting as NPVCostSystemScenarios figures.
    Creates two figures:
    - AssessmentNPVComparison_AllPairs.pdf: All pairs (including sensitivity scenarios)
    - AssessmentNPVComparison_{project}.pdf: Only baseline pairs (one per project)
    """
    df = epm_results["pNetPresentCostSystem"].copy()

    # Remove NPV row to avoid double counting (same as postprocessing.py)
    df = df.loc[df["attribute"] != "NPV of system cost: $m"]

    # Group reserve and trade attributes (same as postprocessing.py)
    if reserve_attrs:
        df = _simplify_attributes(df, "Unmet reserve costs: $m", reserve_attrs)
    if trade_attrs:
        df = _simplify_attributes(df, "Trade costs: $m", trade_attrs)

    # Remove ": $m" from attribute names
    df["attribute"] = df["attribute"].str.replace(": $m", "", regex=False)

    if df.empty:
        return

    # Compute differences for all pairs
    df_all_diffs = _compute_pairwise_differences(
        df, scenario_pairs, merge_cols=["attribute"]
    )

    if df_all_diffs.empty:
        return

    # Check if we have sensitivity scenarios
    multiple_pairs = len(scenario_pairs) > 1 or any(
        len(counterfactuals) > 1 for counterfactuals in scenario_pairs.values()
    )

    # Figure 1: All pairs (only if there are multiple pairs including sensitivities)
    if multiple_pairs:
        filename = os.path.join(folder, "AssessmentNPVComparison_AllPairs.pdf")
        make_stacked_barplot(
            df_all_diffs,
            filename,
            dict_specs["colors"],
            column_stacked="attribute",
            column_subplot=None,
            column_xaxis="scenario",
            column_value="value",
            format_y=make_auto_yaxis_formatter("m$"),
            rotation=45,
            annotate=False,
            title="NPV of System Cost: Project - Counterfactual (All Pairs)",
            show_total=True,
        )

    # Figure 2: Baseline pairs only (one per project)
    if "baseline" in scenario_pairs:
        for scenario_cf in scenario_pairs["baseline"]:
            # Get the beautified label for this pair
            raw_label = scenario_cf.split("@")[0] if "@" in scenario_cf else scenario_cf
            scenario_label = _beautify_scenario_name(raw_label)

            df_pair = df_all_diffs[df_all_diffs["scenario"] == scenario_label]
            if df_pair.empty:
                continue

            # Extract project name for filename
            project_name = scenario_cf.split("@")[1] if "@" in scenario_cf else scenario_cf
            filename = os.path.join(folder, f"AssessmentNPVComparison_{project_name}.pdf")

            make_stacked_barplot(
                df_pair,
                filename,
                dict_specs["colors"],
                column_stacked="attribute",
                column_subplot=None,
                column_xaxis="scenario",
                column_value="value",
                format_y=make_auto_yaxis_formatter("m$"),
                rotation=0,
                annotate=False,
                title=f"NPV of System Cost: {project_name} - Baseline",
                show_total=True,
            )

    # Figure 3: Subplots per scenario pair (one subplot per pair)
    if multiple_pairs:
        filename = os.path.join(folder, "AssessmentNPVComparison_ByPair.pdf")
        make_stacked_barplot(
            df_all_diffs,
            filename,
            dict_specs["colors"],
            column_stacked="attribute",
            column_subplot="scenario",
            column_xaxis=None,
            column_value="value",
            format_y=make_auto_yaxis_formatter("m$"),
            rotation=0,
            annotate=False,
            title="NPV of System Cost: Project - Counterfactual",
            show_total=True,
        )


def make_assessment_cost_template_csv(
    epm_results,
    folder,
    scenario_pairs,
    dict_specs=None,
    trade_attrs=None,
    reserve_attrs=None,
):
    """
    Export a single cost assessment CSV in wide format for investment analysis.

    Output structure (single file per project):
    ```
    EPM System Cost Comparison: Project vs Baseline (values in million USD)
    BASELINE = without project | PROJECT = with project | DIFFERENCE = PROJECT - BASELINE
    Positive difference = project INCREASES system cost | Negative = project REDUCES system cost

    Scenario   | Cost Category (M$)     | 2025 | 2030 | ... | NPV
    -----------|------------------------|------|------|-----|------
    BASELINE   | Investment costs       | 100  | 120  | ... | 450
    BASELINE   | Fixed O&M costs        | 25   | 28   | ... | 100
    BASELINE   | Variable O&M costs     | 15   | 18   | ... | 60
    BASELINE   | Fuel costs             | 80   | 85   | ... | 320
    BASELINE   | Trade costs            | 10   | 12   | ... | 40
    BASELINE   | Unmet demand costs     | 0    | 0    | ... | 0
    BASELINE   | Unmet reserve costs    | 5    | 3    | ... | 15
    BASELINE   | TOTAL                  | 235  | 266  | ... | 985
               |                        |      |      |     |
    PROJECT    | ...                    |      |      |     |
    ...
    DIFFERENCE | ...                    |      |      |     |
    ```

    Also generates a stacked bar chart showing cost differences by component.
    """
    if "baseline" not in scenario_pairs:
        return

    if "pCostsSystem" not in epm_results:
        log_warning("pCostsSystem not found in results; skipping assessment CSV export.")
        return

    # Define cost category order and mapping for cleaner names
    cost_category_order = [
        "Investment costs",
        "Fixed O&M costs",
        "Variable O&M costs",
        "Fuel costs",
        "Carbon costs",
        "Trade costs",
        "Unmet demand costs",
        "Unmet reserve costs",
    ]

    # Mapping from raw GAMS attribute names to clean category names
    # Raw names from generate_report.gms sumhdr set
    category_mapping = {
        # Investment
        "Investment costs: $m": "Investment costs",
        # O&M (note: GAMS uses "Fixed O&M" not "Fixed O&M costs")
        "Fixed O&M: $m": "Fixed O&M costs",
        "Variable O&M: $m": "Variable O&M costs",
        # Other operational
        "Startup costs: $m": "Variable O&M costs",  # Merge startup into variable O&M
        "Fuel costs: $m": "Fuel costs",
        "Spinning reserve costs: $m": "Fixed O&M costs",  # Merge into fixed O&M
        "Transmission costs: $m": "Fixed O&M costs",  # Merge into fixed O&M
        # Carbon
        "Carbon costs: $m": "Carbon costs",
        # Trade
        "Import costs with external zones: $m": "Trade costs",
        "Export revenues with external zones: $m": "Trade costs",
        "Import costs with internal zones: $m": "Trade costs",
        "Export revenues with internal zones: $m": "Trade costs",
        "Trade costs: $m": "Trade costs",  # If already aggregated
        # Unmet demand
        "Unmet demand costs: $m": "Unmet demand costs",
        "Excess generation: $m": "Unmet demand costs",
        "VRE curtailment: $m": "Unmet demand costs",
        # Unmet reserves (will be aggregated by _simplify_attributes if reserve_attrs provided)
        "Unmet country spinning reserve costs: $m": "Unmet reserve costs",
        "Unmet country planning reserve costs: $m": "Unmet reserve costs",
        "Unmet country CO2 backstop cost: $m": "Unmet reserve costs",
        "Unmet system planning reserve costs: $m": "Unmet reserve costs",
        "Unmet system spinning reserve costs: $m": "Unmet reserve costs",
        "Unmet system CO2 backstop cost: $m": "Unmet reserve costs",
        "Unmet reserve costs: $m": "Unmet reserve costs",  # If already aggregated
    }

    df_yearly = epm_results["pCostsSystem"].copy()

    if reserve_attrs:
        df_yearly = _simplify_attributes(df_yearly, "Unmet reserve costs: $m", reserve_attrs)
    if trade_attrs:
        df_yearly = _simplify_attributes(df_yearly, "Trade costs: $m", trade_attrs)

    df_yearly = df_yearly.loc[df_yearly["attribute"] != "NPV of system cost: $m"]

    # Apply category mapping
    df_yearly["attribute"] = df_yearly["attribute"].map(
        lambda x: category_mapping.get(x, x)
    )

    # Remove any remaining ": $m" suffix for unmapped attributes
    df_yearly["attribute"] = df_yearly["attribute"].str.replace(": $m", "", regex=False)

    # Aggregate by category (mapping may have grouped multiple raw attrs into one)
    df_yearly = df_yearly.groupby(
        [c for c in df_yearly.columns if c != "value"],
        as_index=False,
        observed=False,
    )["value"].sum()

    # Get NPV data for the final column
    df_npv_all = None
    if "pCostsSystem" in epm_results:
        df_npv_all = epm_results["pNetPresentCostSystem"].copy()
        df_npv_all = df_npv_all.loc[df_npv_all["attribute"] != "NPV of system cost: $m"]
        if reserve_attrs:
            df_npv_all = _simplify_attributes(df_npv_all, "Unmet reserve costs: $m", reserve_attrs)
        if trade_attrs:
            df_npv_all = _simplify_attributes(df_npv_all, "Trade costs: $m", trade_attrs)
        # Apply same category mapping
        df_npv_all["attribute"] = df_npv_all["attribute"].map(
            lambda x: category_mapping.get(x, x)
        )
        # Remove any remaining ": $m" suffix
        df_npv_all["attribute"] = df_npv_all["attribute"].str.replace(": $m", "", regex=False)
        # Aggregate by category
        df_npv_all = df_npv_all.groupby(
            [c for c in df_npv_all.columns if c != "value"],
            as_index=False,
            observed=False,
        )["value"].sum()

    for scenario_cf in scenario_pairs["baseline"]:
        df_base = df_yearly[df_yearly["scenario"] == "baseline"].copy()
        df_cf = df_yearly[df_yearly["scenario"] == scenario_cf].copy()

        if df_base.empty or df_cf.empty:
            continue

        project_name = scenario_cf.split("@")[1] if "@" in scenario_cf else scenario_cf

        # Pivot to wide format: rows=attribute, columns=year
        df_base_wide = df_base.pivot_table(
            index="attribute",
            columns="year",
            values="value",
            aggfunc="sum",
            fill_value=0,
        )
        df_cf_wide = df_cf.pivot_table(
            index="attribute",
            columns="year",
            values="value",
            aggfunc="sum",
            fill_value=0,
        )

        # Ensure both have the same columns (years)
        all_years = sorted(set(df_base_wide.columns) | set(df_cf_wide.columns))
        for yr in all_years:
            if yr not in df_base_wide.columns:
                df_base_wide[yr] = 0
            if yr not in df_cf_wide.columns:
                df_cf_wide[yr] = 0
        df_base_wide = df_base_wide[all_years]
        df_cf_wide = df_cf_wide[all_years]

        # Reindex to standard category order (only include categories present in data)
        all_categories = [c for c in cost_category_order if c in df_base_wide.index or c in df_cf_wide.index]

        df_base_wide = df_base_wide.reindex(all_categories, fill_value=0)
        df_cf_wide = df_cf_wide.reindex(all_categories, fill_value=0)

        # Compute difference (project - baseline)
        df_diff_wide = df_cf_wide - df_base_wide

        # Add NPV column if available
        if df_npv_all is not None:
            npv_base = df_npv_all[df_npv_all["scenario"] == "baseline"].copy()
            npv_cf = df_npv_all[df_npv_all["scenario"] == scenario_cf].copy()

            npv_base_dict = npv_base.groupby("attribute")["value"].sum().to_dict()
            npv_cf_dict = npv_cf.groupby("attribute")["value"].sum().to_dict()

            df_base_wide["NPV"] = df_base_wide.index.map(lambda x: npv_base_dict.get(x, 0))
            df_cf_wide["NPV"] = df_cf_wide.index.map(lambda x: npv_cf_dict.get(x, 0))
            df_diff_wide["NPV"] = df_cf_wide["NPV"] - df_base_wide["NPV"]

        # Add total row to each section
        df_base_wide.loc["TOTAL"] = df_base_wide.sum()
        df_cf_wide.loc["TOTAL"] = df_cf_wide.sum()
        df_diff_wide.loc["TOTAL"] = df_diff_wide.sum()

        # Compute Total CAPEX row (generation + storage + transmission) if available
        capex_base_row = None
        capex_cf_row = None
        capex_diff_row = None
        if "pCapexInvestmentComponent" in epm_results:
            df_capex = epm_results["pCapexInvestmentComponent"].copy()

            # Aggregate across zones and components (system-level total CAPEX)
            capex_grouping = [c for c in df_capex.columns if c not in ["value", "zone", "attribute"]]
            df_capex_agg = df_capex.groupby(
                capex_grouping, as_index=False, observed=False
            )["value"].sum()

            # Get base and project CAPEX
            capex_base = df_capex_agg[df_capex_agg["scenario"] == "baseline"].copy()
            capex_cf = df_capex_agg[df_capex_agg["scenario"] == scenario_cf].copy()

            if not capex_base.empty and not capex_cf.empty:
                # Sum by year (value is in USD, convert to M$)
                capex_base_by_year = (capex_base.groupby("year")["value"].sum() / 1e6).to_dict()
                capex_cf_by_year = (capex_cf.groupby("year")["value"].sum() / 1e6).to_dict()

                # Build CAPEX row values for each year
                capex_base_values = pd.Series({yr: capex_base_by_year.get(yr, 0) for yr in all_years})
                capex_cf_values = pd.Series({yr: capex_cf_by_year.get(yr, 0) for yr in all_years})
                capex_diff_values = capex_cf_values - capex_base_values

                # Compute NPV of CAPEX if we have discount factors
                if df_npv_all is not None and "pCapexInvestmentComponent" in epm_results:
                    # Use discounted values - sum capex * discount factor * weight
                    # For simplicity, just sum the yearly values (NPV would need proper discounting)
                    capex_base_npv = capex_base_values.sum()
                    capex_cf_npv = capex_cf_values.sum()
                else:
                    capex_base_npv = capex_base_values.sum()
                    capex_cf_npv = capex_cf_values.sum()

                # Add to dataframes as first row (before Investment costs)
                capex_base_row = capex_base_values.copy()
                capex_base_row["NPV"] = capex_base_npv
                capex_cf_row = capex_cf_values.copy()
                capex_cf_row["NPV"] = capex_cf_npv
                capex_diff_row = capex_diff_values.copy()
                capex_diff_row["NPV"] = capex_cf_npv - capex_base_npv

                # Insert as first row in each dataframe
                df_base_wide.loc["Total CAPEX"] = capex_base_row
                df_cf_wide.loc["Total CAPEX"] = capex_cf_row
                df_diff_wide.loc["Total CAPEX"] = capex_diff_row

                # Reorder to put Total CAPEX first
                new_order = ["Total CAPEX"] + [idx for idx in df_base_wide.index if idx != "Total CAPEX"]
                df_base_wide = df_base_wide.reindex(new_order)
                df_cf_wide = df_cf_wide.reindex(new_order)
                df_diff_wide = df_diff_wide.reindex(new_order)

        # Add scenario labels and reset index
        df_base_wide = df_base_wide.reset_index().rename(columns={"index": "Cost Category (M$)"})
        df_cf_wide = df_cf_wide.reset_index().rename(columns={"index": "Cost Category (M$)"})
        df_diff_wide = df_diff_wide.reset_index().rename(columns={"index": "Cost Category (M$)"})

        df_base_wide.insert(0, "Scenario", "BASELINE")
        df_cf_wide.insert(0, "Scenario", "PROJECT")
        df_diff_wide.insert(0, "Scenario", "DIFFERENCE")

        # Create empty separator row
        cols = df_base_wide.columns.tolist()
        separator = pd.DataFrame([[""] * len(cols)], columns=cols)

        # Combine data sections
        df_combined = pd.concat(
            [
                df_base_wide,
                separator,
                df_cf_wide,
                separator,
                df_diff_wide,
            ],
            ignore_index=True,
        )

        # Build metadata header lines (will be written before the CSV data)
        metadata_lines = [
            f"# EPM System Cost Comparison: {project_name} vs Baseline",
            "# Values in million USD (M$)",
            "#",
            "# Scenarios:",
            "#   BASELINE = System costs WITHOUT the project",
            "#   PROJECT  = System costs WITH the project",
            "#   DIFFERENCE = PROJECT - BASELINE",
            "#",
            "# Interpretation:",
            "#   Positive difference = project INCREASES system cost",
            "#   Negative difference = project REDUCES system cost (savings)",
            "#",
        ]

        # Get year weights from EPM results (pWeightYear) if available
        # Otherwise fall back to computing from year differences
        year_weights = {}
        if "pWeightYear" in epm_results:
            weight_df = epm_results["pWeightYear"].copy()
            year_col = "y" if "y" in weight_df.columns else "year"
            # Select baseline scenario (same across all scenarios)
            if "scenario" in weight_df.columns:
                weight_df = weight_df[weight_df["scenario"] == "baseline"]
            for _, row in weight_df.iterrows():
                year_weights[int(row[year_col])] = row["value"]

        # Fallback: compute weights from year differences
        if not year_weights:
            for i, y in enumerate(all_years):
                if i < len(all_years) - 1:
                    year_weights[y] = all_years[i + 1] - y
                else:
                    year_weights[y] = all_years[-1] - all_years[-2] if len(all_years) > 1 else 1

        # Get discount factors from EPM results (pRR) if available
        discount_factors = {}
        if "pRR" in epm_results:
            discount_df = epm_results["pRR"].copy()
            year_col = "y" if "y" in discount_df.columns else "year"
            # Select baseline scenario (same across all scenarios)
            if "scenario" in discount_df.columns:
                discount_df = discount_df[discount_df["scenario"] == "baseline"]
            for _, row in discount_df.iterrows():
                discount_factors[int(row[year_col])] = row["value"]

        # --- Compute GHG Emissions rows to add below TOTAL ---
        em_base_row = None
        em_cf_row = None
        em_diff_row = None
        if "pEmissionsZone" in epm_results:
            df_emissions = epm_results["pEmissionsZone"].copy()

            # Aggregate across zones (system-level)
            emissions_grouping = [c for c in df_emissions.columns if c not in ["value", "zone"]]
            df_emissions = df_emissions.groupby(
                emissions_grouping, as_index=False, observed=False
            )["value"].sum()

            # Get base and project emissions
            em_base = df_emissions[df_emissions["scenario"] == "baseline"].copy()
            em_cf = df_emissions[df_emissions["scenario"] == scenario_cf].copy()

            if not em_base.empty and not em_cf.empty:
                # Pivot to get emissions by year
                em_base_by_year = em_base.groupby("year")["value"].sum().to_dict()
                em_cf_by_year = em_cf.groupby("year")["value"].sum().to_dict()

                # Compute cumulative emissions (weighted sum)
                def _compute_cumulative(emissions_by_year):
                    total = 0.0
                    for y in all_years:
                        val = emissions_by_year.get(y, 0)
                        weight = year_weights.get(y, 1)
                        total += val * weight
                    return total

                cum_base = _compute_cumulative(em_base_by_year)
                cum_cf = _compute_cumulative(em_cf_by_year)

                # Build emission rows (same structure as cost rows)
                em_base_row = {"Scenario": "BASELINE", "Cost Category (M$)": "GHG Emissions (Mt CO2)"}
                em_cf_row = {"Scenario": "PROJECT", "Cost Category (M$)": "GHG Emissions (Mt CO2)"}
                em_diff_row = {"Scenario": "DIFFERENCE", "Cost Category (M$)": "GHG Emissions (Mt CO2)"}

                for yr in all_years:
                    em_base_row[yr] = em_base_by_year.get(yr, 0)
                    em_cf_row[yr] = em_cf_by_year.get(yr, 0)
                    em_diff_row[yr] = em_cf_by_year.get(yr, 0) - em_base_by_year.get(yr, 0)

                # Add cumulative column (labeled same as NPV column position)
                em_base_row["NPV"] = cum_base
                em_cf_row["NPV"] = cum_cf
                em_diff_row["NPV"] = cum_cf - cum_base

        # Insert emissions rows after TOTAL in each section
        if em_base_row is not None:
            # Find TOTAL row indices and insert emissions after each
            cols = df_combined.columns.tolist()

            # Rebuild df_combined with emissions after each TOTAL
            rows_list = df_combined.to_dict('records')
            new_rows = []
            for row in rows_list:
                new_rows.append(row)
                if row.get("Cost Category (M$)") == "TOTAL":
                    scenario = row.get("Scenario")
                    if scenario == "BASELINE":
                        new_rows.append(em_base_row)
                    elif scenario == "PROJECT":
                        new_rows.append(em_cf_row)
                    elif scenario == "DIFFERENCE":
                        new_rows.append(em_diff_row)

            df_combined = pd.DataFrame(new_rows, columns=cols)

        # Save CSV with metadata header
        filename = os.path.join(folder, f"AssessmentCostTemplate_{project_name}.csv")
        with open(filename, "w", encoding="utf-8") as f:
            # Write metadata as comment lines
            for line in metadata_lines:
                f.write(line + "\n")

            # Write year weights and discount factors first (aligned with years)
            f.write("# Model Parameters\n")
            f.write("# Year weights (pWeightYear) - number of years each model year represents\n")
            f.write("# Discount factors (pRR) - present value factor for each year\n")
            f.write("# GHG Emissions use cumulative sum (yearly value * year weight), not discounted NPV\n")
            f.write("#\n")

            # Build parameters table with same column structure
            params_rows = []
            # Year weights row
            weight_row = {"Scenario": "", "Cost Category (M$)": "Year Weight"}
            for yr in all_years:
                weight_row[yr] = year_weights.get(yr, "")
            weight_row["NPV"] = ""
            params_rows.append(weight_row)

            # Discount factors row (always include, even if empty)
            discount_row = {"Scenario": "", "Cost Category (M$)": "Discount Factor"}
            for yr in all_years:
                discount_row[yr] = discount_factors.get(yr, "")
            discount_row["NPV"] = ""
            params_rows.append(discount_row)

            df_params = pd.DataFrame(params_rows, columns=df_combined.columns.tolist())
            df_params.to_csv(f, index=False)

            f.write("\n")
            f.write("# Cost and Emissions Data\n")
            f.write("#\n")

            # Write cost data table (now includes emissions after each TOTAL)
            df_combined.to_csv(f, index=False)

        # Generate stacked bar chart for cost differences
        if dict_specs is not None:
            _make_cost_diff_stacked_bar(
                df_diff_wide,
                dict_specs,
                folder,
                project_name,
            )


def _make_cost_diff_stacked_bar(df_diff, dict_specs, folder, project_name):
    """
    Generate a stacked bar chart showing cost differences by component over years.

    Positive bars (above zero) = project increases costs
    Negative bars (below zero) = project reduces costs (savings)
    """
    # Exclude TOTAL row
    df_plot = df_diff[df_diff["Cost Category (M$)"] != "TOTAL"].copy()

    if df_plot.empty:
        return

    # Get unique categories (avoid duplicates)
    categories = df_plot["Cost Category (M$)"].unique().tolist()
    year_cols = [c for c in df_plot.columns if c not in ["Scenario", "Cost Category (M$)", "NPV"]]

    if not year_cols:
        return

    # Create figure
    _, ax = plt.subplots(figsize=(max(8, len(year_cols) * 1.2), 6))

    x = np.arange(len(year_cols))
    width = 0.6

    # Get colors from dict_specs or use defaults
    colors = dict_specs.get("colors", {})
    default_colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

    # Separate positive and negative values for proper stacking
    bottom_pos = np.zeros(len(year_cols))
    bottom_neg = np.zeros(len(year_cols))

    # Track which categories have been added to legend
    legend_added = set()

    for i, category in enumerate(categories):
        row = df_plot[df_plot["Cost Category (M$)"] == category]
        if row.empty:
            continue
        values = row[year_cols].values.flatten()
        color = colors.get(category, default_colors[i])

        # Split into positive and negative
        pos_values = np.where(values > 0, values, 0)
        neg_values = np.where(values < 0, values, 0)

        # Add label only once per category
        label = category if category not in legend_added else None

        if np.any(pos_values != 0):
            ax.bar(x, pos_values, width, bottom=bottom_pos, label=label, color=color)
            bottom_pos += pos_values
            if label:
                legend_added.add(category)
                label = None  # Don't add again for negative

        if np.any(neg_values != 0):
            ax.bar(x, neg_values, width, bottom=bottom_neg, label=label, color=color)
            bottom_neg += neg_values
            if label:
                legend_added.add(category)

    # Add zero line
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Add total annotation on each bar
    totals = df_diff[df_diff["Cost Category (M$)"] == "TOTAL"][year_cols].values.flatten()
    for i, total in enumerate(totals):
        y_pos = total if total >= 0 else total
        va = "bottom" if total >= 0 else "top"
        offset = 5 if total >= 0 else -5
        ax.annotate(
            f"{total:+,.0f}",
            xy=(i, y_pos),
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            xytext=(0, offset),
            textcoords="offset points",
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Cost Difference (M$)")
    ax.set_title(f"System Cost Difference: {project_name} vs Baseline\n(Positive = higher cost with project)")
    ax.set_xticks(x)
    ax.set_xticklabels(year_cols, rotation=45 if len(year_cols) > 6 else 0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    plt.tight_layout()
    filename = os.path.join(folder, f"AssessmentCostDiffStacked_{project_name}.pdf")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def make_assessment_capacity_diff(
    epm_results, dict_specs, folder, scenario_pairs
):
    """
    Generate new capacity difference bar plots for project assessment.
    """
    df = epm_results["pNewCapacityTechFuel"].copy()
    grouping_cols = [col for col in df.columns if col not in ["value", "zone"]]
    df = df.groupby(grouping_cols, as_index=False, observed=False)["value"].sum()

    df_all_diffs = _compute_pairwise_differences(
        df, scenario_pairs, merge_cols=["year", "fuel"]
    )
    


    _plot_assessment_diffs(
        df_all_diffs,
        scenario_pairs,
        dict_specs,
        folder,
        stacked_column="fuel",
        x_column="year",
        format_y=make_auto_yaxis_formatter("MW"),
        title_prefix="New Capacity Difference (MW)",
        filename_prefix="AssessmentNewCapacityDiff",
        annotate=False,
        show_total_single=True,
    )

    # Cumulative evolution over time (running sum by pair)
    if not df_all_diffs.empty:
        # Create complete grid of scenario/fuel/year to ensure cumsum carries forward
        all_scenarios = df_all_diffs["scenario"].unique()
        all_fuels = df_all_diffs["fuel"].unique()
        all_years = sorted(df_all_diffs["year"].unique())

        full_index = pd.MultiIndex.from_product(
            [all_scenarios, all_fuels, all_years],
            names=["scenario", "fuel", "year"]
        )
        df_full = (
            df_all_diffs.set_index(["scenario", "fuel", "year"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

        df_cumulative_year = (
            df_full.sort_values("year")
            .groupby(["scenario", "fuel"], observed=False)
            .apply(lambda grp: grp.assign(value=grp["value"].cumsum()))
            .reset_index(drop=True)
        )
        _plot_assessment_diffs(
            df_cumulative_year,
            scenario_pairs,
            dict_specs,
            folder,
            stacked_column="fuel",
            x_column="year",
            format_y=make_auto_yaxis_formatter("MW"),
            title_prefix="Cumulative New Capacity Difference (MW)",
            filename_prefix="AssessmentCumulativeNewCapacityDiff",
            annotate=False,
            show_total_single=True,
        )


def make_assessment_energy_mix_diff(
    epm_results, dict_specs, folder, scenario_pairs
):
    """
    Generate energy mix difference bar plots for project assessment.

    Mirrors the EnergyMixSystemEvolution figure by using system-level annual
    generation by fuel (excluding imports/exports).
    """
    df_energyfuel = epm_results["pEnergyTechFuel"].copy()

    df_exchange = epm_results["pEnergyBalance"].copy()
    df_exchange = df_exchange.loc[
        df_exchange["attribute"].isin(
            ["Unmet demand: GWh", "Exports exchange: GWh", "Imports exchange: GWh"]
        )
    ]
    df_exchange = df_exchange.replace(
        {
            "Unmet demand: GWh": "Unmet demand",
            "Exports exchange: GWh": "Exports",
            "Imports exchange: GWh": "Imports",
        }
    )
    df_exchange["value"] = df_exchange.apply(
        lambda row: -row["value"] if row["attribute"] == "Exports" else row["value"],
        axis=1,
    )
    df_exchange.rename(columns={"attribute": "fuel"}, inplace=True)

    df = pd.concat([df_energyfuel, df_exchange], ignore_index=True)

    grouping_cols = [col for col in df.columns if col not in ["value", "zone"]]
    df = df.groupby(grouping_cols, as_index=False, observed=False)["value"].sum()

    df_all_diffs = _compute_pairwise_differences(
        df, scenario_pairs, merge_cols=["year", "fuel"]
    )

    _plot_assessment_diffs(
        df_all_diffs,
        scenario_pairs,
        dict_specs,
        folder,
        stacked_column="fuel",
        x_column="year",
        format_y=make_auto_yaxis_formatter("GWh"),
        title_prefix="Energy Mix Difference (GWh)",
        filename_prefix="AssessmentEnergyMixDiff",
        annotate=False,
        show_total_single=True,
    )


def make_assessment_heatmap(
    epm_results,
    folder,
    scenario_pairs,
    zone_list=None,
    year=None,
):
    """
    Build a heatmap showing differences between project and counterfactual scenarios.

    Each column represents a scenario pair (project - counterfactual).
    No reference column needed since all values are already differences.

    The summary includes (mirroring make_heatmap_plot):
        1. NPV of system cost
        2. Total installed capacity in the final model year
        3. Cumulative new capacity additions by fuel
        4. Transmission capacity in final year
        5. New transmission capacity in final year
        6. Cumulative CAPEX by component
        7. Cumulative CO2 emissions

    Parameters
    ----------
    epm_results : dict
        Dictionary with post-processed EPM outputs (DataFrames).
    folder : str
        Where to save the heatmap.
    scenario_pairs : dict
        Mapping of base scenarios to their counterfactual scenarios.
    zone_list : iterable, optional
        Optional subset of zones to include when aggregating system metrics.
    year : int, optional
        Year to use for annual metrics. Defaults to the maximum year found.
    """

    def _get_dataframe(key: str):
        if key not in epm_results:
            return None
        df = epm_results[key]
        if not isinstance(df, pd.DataFrame):
            return None
        return df.copy()

    def _filter_zone(df: pd.DataFrame) -> pd.DataFrame:
        if zone_list is None or 'zone' not in df.columns:
            return df
        return df[df['zone'].isin(zone_list)]

    def _resolve_year(df: pd.DataFrame) -> int:
        if 'year' not in df.columns:
            raise KeyError("Expected a 'year' column when computing annual aggregates.")
        available_years = df['year'].dropna().unique()
        if year is not None:
            if year not in available_years:
                raise ValueError(f"Requested year {year} not available.")
            return int(year)
        return int(max(available_years))

    def _compute_pair_diff(df: pd.DataFrame, base: str, cf: str, scale: float = 1.0) -> float:
        """Compute difference between counterfactual and base scenario."""
        df_base = df[df['scenario'] == base]
        df_cf = df[df['scenario'] == cf]
        val_base = df_base['value'].sum() if not df_base.empty else 0
        val_cf = df_cf['value'].sum() if not df_cf.empty else 0
        return (val_cf - val_base) * scale

    # Build list of scenario pairs with simplified labels
    pair_labels = []
    pair_info = []  # (base, counterfactual)
    for scenario_base, counterfactuals in scenario_pairs.items():
        for scenario_cf in counterfactuals:
            # Label is the base scenario name (before @), beautified
            raw_label = scenario_cf.split('@')[0] if '@' in scenario_cf else scenario_cf
            label = _beautify_scenario_name(raw_label)
            pair_labels.append(label)
            pair_info.append((scenario_base, scenario_cf))

    if not pair_info:
        log_warning("No scenario pairs found for assessment heatmap.")
        return

    rows = []

    # 1. NPV of system cost difference
    costs_system = _get_dataframe('pNetPresentCostSystem')
    if costs_system is not None:
        npv_label = 'NPV of system cost: $m'
        reserve_attrs = [
            "Unmet country spinning reserve costs: $m",
            "Unmet country planning reserve costs: $m",
            "Unmet country CO2 backstop cost: $m",
            "Unmet system planning reserve costs: $m",
            "Unmet system spinning reserve costs: $m"
        ]

        if 'attribute' in costs_system.columns:
            costs_npv = costs_system[costs_system['attribute'] == npv_label]
        else:
            costs_npv = costs_system
        diffs = [_compute_pair_diff(costs_npv, base, cf) for base, cf in pair_info]
        rows.append(('NPV of system cost (M$)', diffs))

        # NPV breakdown by component
        if 'attribute' in costs_system.columns:
            cost_components = costs_system[costs_system['attribute'] != npv_label].copy()

            # Merge reserve cost attributes together
            reserve_mask = cost_components['attribute'].isin(reserve_attrs)
            if reserve_mask.any():
                reserve_df = cost_components[reserve_mask].copy()
                reserve_df = reserve_df.groupby(
                    [col for col in reserve_df.columns if col not in ['attribute', 'value']],
                    as_index=False,
                    observed=False
                )['value'].sum()
                reserve_df['attribute'] = 'Unmet reserve costs: $m'
                cost_components = pd.concat(
                    [cost_components[~reserve_mask], reserve_df],
                    ignore_index=True
                )

            components = sorted(cost_components['attribute'].unique())
            for comp in components:
                comp_df = cost_components[cost_components['attribute'] == comp]
                diffs = [_compute_pair_diff(comp_df, base, cf) for base, cf in pair_info]
                # Clean up component name: remove ": $m" suffix
                comp_name = comp.replace(': $m', '')
                rows.append((f'NPV - {comp_name} (M$)', diffs))

    # 2. Total installed capacity in final year
    capacity_all = _get_dataframe('pCapacityTechFuel')
    if capacity_all is not None:
        final_year = _resolve_year(capacity_all)
        capacity = _filter_zone(capacity_all)
        capacity_final = capacity[capacity['year'] == final_year]
        diffs = [_compute_pair_diff(capacity_final, base, cf) for base, cf in pair_info]
        rows.append((f'Capacity - Total {final_year} (MW)', diffs))

    # 3. Cumulative new capacity by fuel
    new_capacity_all = _get_dataframe('pNewCapacityTechFuel')
    if new_capacity_all is not None:
        final_year = _resolve_year(new_capacity_all)
        new_capacity = _filter_zone(new_capacity_all)
        new_capacity_cum = new_capacity[new_capacity['year'] <= final_year]

        # Total cumulative new capacity
        diffs = [_compute_pair_diff(new_capacity_cum, base, cf) for base, cf in pair_info]
        rows.append((f'New capacity - Total {final_year} (MW)', diffs))

        # By fuel
        if 'fuel' in new_capacity_cum.columns:
            fuels = sorted(new_capacity_cum['fuel'].unique())
            for fuel in fuels:
                fuel_df = new_capacity_cum[new_capacity_cum['fuel'] == fuel]
                diffs = [_compute_pair_diff(fuel_df, base, cf) for base, cf in pair_info]
                rows.append((f'New capacity - {fuel} (MW)', diffs))

    # 4. Transmission capacity in final year
    if 'pTransmissionCapacity' in epm_results:
        transmission_all = _get_dataframe('pTransmissionCapacity')
        if transmission_all is not None:
            trans_year = _resolve_year(transmission_all)
            transmission = transmission_all.copy()
            if zone_list is not None and {'zone', 'z2'}.issubset(transmission.columns):
                transmission = transmission[
                    transmission['zone'].isin(zone_list) | transmission['z2'].isin(zone_list)
                ]
            transmission_final = transmission[transmission['year'] == trans_year]
            diffs = [_compute_pair_diff(transmission_final, base, cf) for base, cf in pair_info]
            rows.append((f'Transmission capacity {trans_year} (MW)', diffs))

    # 5. New transmission capacity in final year
    if 'pNewTransmissionCapacity' in epm_results:
        new_trans_all = _get_dataframe('pNewTransmissionCapacity')
        if new_trans_all is not None:
            new_trans_year = _resolve_year(new_trans_all)
            new_trans = new_trans_all.copy()
            if zone_list is not None and {'zone', 'z2'}.issubset(new_trans.columns):
                new_trans = new_trans[
                    new_trans['zone'].isin(zone_list) | new_trans['z2'].isin(zone_list)
                ]
            new_trans_cum = new_trans[new_trans['year'] <= new_trans_year]
            diffs = [_compute_pair_diff(new_trans_cum, base, cf) for base, cf in pair_info]
            rows.append((f'New transmission capacity {new_trans_year} (MW)', diffs))

    # 6. Cumulative CAPEX by component
    capex_all = _get_dataframe('pCapexInvestmentComponent')
    if capex_all is not None:
        capex_year = _resolve_year(capex_all)
        capex = _filter_zone(capex_all)
        capex_cum = capex[capex['year'] <= capex_year]

        # Total CAPEX
        diffs = [_compute_pair_diff(capex_cum, base, cf, scale=1e-6) for base, cf in pair_info]
        rows.append((f'Cumulative CAPEX - Total {capex_year} (M$)', diffs))

        # By component
        if 'attribute' in capex_cum.columns:
            components = sorted(capex_cum['attribute'].unique())
            for comp in components:
                comp_df = capex_cum[capex_cum['attribute'] == comp]
                diffs = [_compute_pair_diff(comp_df, base, cf, scale=1e-6) for base, cf in pair_info]
                rows.append((f'CAPEX - {comp} (M$)', diffs))

    # 7. Cumulative CO2 emissions
    emissions_all = _get_dataframe('pEmissionsZone')
    if emissions_all is not None:
        emissions_year = _resolve_year(emissions_all)
        emissions = _filter_zone(emissions_all)
        emissions_cum = emissions[emissions['year'] <= emissions_year]
        diffs = [_compute_pair_diff(emissions_cum, base, cf) for base, cf in pair_info]
        rows.append((f'Cumulative CO2 emissions {emissions_year} (Mt)', diffs))

        # 8. NPV of emission value at $75/tCO2 with 5% discount rate
        # Compute year-by-year differences and apply NPV discounting
        discount_rate = 0.05
        carbon_price = 75  # $/tCO2

        def _compute_npv_emission_value(df: pd.DataFrame, base: str, cf: str) -> float:
            """Compute NPV of emission difference between scenarios."""
            df_base = df[df['scenario'] == base].copy()
            df_cf = df[df['scenario'] == cf].copy()

            if df_base.empty and df_cf.empty:
                return 0.0

            # Aggregate by year
            base_by_year = df_base.groupby('year')['value'].sum() if not df_base.empty else pd.Series(dtype=float)
            cf_by_year = df_cf.groupby('year')['value'].sum() if not df_cf.empty else pd.Series(dtype=float)

            # Get all years and sort
            all_years = sorted(set(base_by_year.index) | set(cf_by_year.index))
            if not all_years:
                return 0.0

            start_year = min(all_years)

            # Compute year weights (years each model year represents)
            year_weights = {}
            for i, y in enumerate(all_years):
                if i < len(all_years) - 1:
                    year_weights[y] = all_years[i + 1] - y
                else:
                    year_weights[y] = all_years[-1] - all_years[-2] if len(all_years) > 1 else 1

            npv = 0.0
            for y in all_years:
                val_base = base_by_year.get(y, 0)
                val_cf = cf_by_year.get(y, 0)
                diff = val_cf - val_base  # Mt CO2

                # Value in M$ (Mt * $/t = M$)
                value = diff * carbon_price

                # Apply discounting for each year in the period
                weight = year_weights[y]
                for offset in range(weight):
                    discount_factor = 1 / ((1 + discount_rate) ** (y - start_year + offset))
                    npv += value * discount_factor / weight

            return npv

        diffs_npv = [_compute_npv_emission_value(emissions, base, cf) for base, cf in pair_info]
        rows.append(('NPV Emission value (M$, $75/tCO2, 5%)', diffs_npv))

    if not rows:
        log_warning("No data available to build the assessment heatmap.")
        return

    # Build DataFrame
    data = pd.DataFrame(
        {label: [row[1][i] for row in rows] for i, label in enumerate(pair_labels)},
        index=[row[0] for row in rows]
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(pair_labels) * 2), max(6, len(rows) * 0.4)))

    # Normalize colors per row for better visualization
    data_normalized = data.copy()
    for idx in data.index:
        row = data.loc[idx]
        row_min, row_max = row.min(), row.max()
        if not np.isclose(row_max, row_min):
            data_normalized.loc[idx] = (row - row_min) / (row_max - row_min)
        else:
            data_normalized.loc[idx] = 0.5

    # Create annotations with formatted values
    def _format_value(x):
        if x == 0 or np.isclose(x, 0):
            return "0"
        elif abs(x) < 10:
            return f"{x:+,.1f}"
        else:
            return f"{x:+,.0f}"
    annotations = data.map(_format_value)

    sns.heatmap(
        data_normalized,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        annot=annotations,
        fmt="",
        linewidths=0.5,
        ax=ax,
        cbar=False,
        center=0.5
    )

    ax.set_title("Assessment: Project - Counterfactual", fontsize=12, fontweight='bold')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    # Bold important rows (totals and key metrics)
    bold_markers = (
        'NPV of system cost',
        'Capacity - Total',
        'New capacity - Total',
        'Transmission capacity',
        'New transmission capacity',
        'Cumulative CAPEX - Total',
        'Cumulative CO2 emissions',
        'NPV Emission value',
    )
    for label in ax.get_yticklabels():
        text = label.get_text()
        if any(marker in text for marker in bold_markers):
            label.set_fontweight('bold')

    ax.set_xlabel("")
    ax.set_ylabel("")

    filename = os.path.join(folder, "AssessmentHeatmap.pdf")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def make_assessment_dispatch_diff(epm_results, dict_specs, folder, scenario_base, scenario_cf):
    """
    Generate dispatch difference plots between base and counterfactual scenarios.

    Computes the difference in dispatch values (counterfactual - base) and produces
    stacked area plots showing how generation changes between scenarios.
    Plots are generated for the system level at the max load day and max load season.

    Parameters
    ----------
    epm_results : dict[str, pandas.DataFrame]
        Model result tables keyed by result name (e.g., ``'pDispatch'``).
    dict_specs : dict
        Plotting specifications, including a ``'colors'`` mapping per fuel.
    folder : str
        Destination directory for the generated figures.
    scenario_base : str
        The base scenario name (counterfactual, without the project).
    scenario_cf : str
        The counterfactual scenario name (with the project, contains @).
    """
    def aggregate_to_system(df):
        if df is None or df.empty:
            return df

        df_system = df.copy()
        grouping_columns = [col for col in df_system.columns if col not in {'value', 'zone'}]

        if grouping_columns:
            df_system = df_system.groupby(
                grouping_columns,
                as_index=False,
                observed=False
            )['value'].sum()
        else:
            df_system = pd.DataFrame({'value': [df_system['value'].sum()]})

        df_system.insert(0, 'zone', 'System')

        ordered_cols = ['zone'] + [col for col in grouping_columns if col in df_system.columns] + ['value']
        return df_system[ordered_cols]

    def compute_diff(df_base, df_cf, merge_cols):
        """Compute difference between counterfactual and base dataframes."""
        if df_base is None or df_base.empty or df_cf is None or df_cf.empty:
            return None

        df_merged = pd.merge(
            df_cf,
            df_base,
            on=merge_cols,
            suffixes=('', '_base'),
            how='outer'
        )
        df_merged['value'] = df_merged['value'].fillna(0)
        df_merged['value_base'] = df_merged['value_base'].fillna(0)
        df_merged['value'] = df_merged['value'] - df_merged['value_base']
        df_merged = df_merged.drop(columns=['value_base'])

        df_merged['scenario'] = 'diff'

        return df_merged

    # Extract project name for compact filename
    project_name = scenario_cf.split('@')[1] if '@' in scenario_cf else scenario_cf
    # For sensitivity scenarios, include the base scenario info (beautified)
    raw_base_label = scenario_cf.split('@')[0] if '@' in scenario_cf else scenario_base
    base_label = _beautify_scenario_name(raw_base_label)
    if base_label != 'baseline':
        file_label = f"{base_label}_{project_name}"
    else:
        file_label = project_name

    # Get dispatch generation data
    dispatch_generation = epm_results['pDispatchTechFuel']
    dispatch_components = filter_dataframe(
        epm_results['pDispatch'],
        {'attribute': ['Unmet demand', 'Exports', 'Imports', 'Storage Charge']}
    )
    demand_df = filter_dataframe(epm_results['pDispatch'], {'attribute': ['Demand']})

    # Aggregate to system level
    dispatch_generation_system = aggregate_to_system(dispatch_generation)
    dispatch_components_system = aggregate_to_system(dispatch_components)
    demand_system = aggregate_to_system(demand_df)

    if dispatch_generation_system is None or dispatch_generation_system.empty:
        return

    # Filter for the two scenarios
    gen_base = dispatch_generation_system[dispatch_generation_system['scenario'] == scenario_base].copy()
    gen_cf = dispatch_generation_system[dispatch_generation_system['scenario'] == scenario_cf].copy()

    comp_base = dispatch_components_system[dispatch_components_system['scenario'] == scenario_base].copy()
    comp_cf = dispatch_components_system[dispatch_components_system['scenario'] == scenario_cf].copy()

    if gen_base.empty or gen_cf.empty:
        return

    # Remove scenario column for merging
    gen_base = gen_base.drop(columns=['scenario'])
    gen_cf = gen_cf.drop(columns=['scenario'])
    comp_base = comp_base.drop(columns=['scenario']) if not comp_base.empty else comp_base
    comp_cf = comp_cf.drop(columns=['scenario']) if not comp_cf.empty else comp_cf

    # Determine merge columns (all columns except value)
    merge_cols_gen = [col for col in gen_base.columns if col != 'value']
    merge_cols_comp = [col for col in comp_base.columns if col != 'value'] if not comp_base.empty else []

    # Compute differences
    gen_diff = compute_diff(gen_base, gen_cf, merge_cols_gen)
    comp_diff = compute_diff(comp_base, comp_cf, merge_cols_comp) if merge_cols_comp else None

    if gen_diff is None or gen_diff.empty:
        return

    # Prepare dataframes for plotting
    dfs_to_plot_area = {
        'pDispatchPlant': gen_diff,
        'pDispatch': comp_diff
    }

    # Use demand from base scenario to determine time periods
    demand_scenario = demand_system[demand_system['scenario'] == scenario_base]
    available_years = sorted(demand_scenario['year'].unique())
    if not available_years:
        return

    years_to_plot = [available_years[0], available_years[-1]] if len(available_years) > 1 else [available_years[0]]

    for year in years_to_plot:
        df_year = demand_scenario[demand_scenario['year'] == year]
        if df_year.empty:
            continue

        season_totals = df_year.groupby('season', observed=False)['value'].sum()
        if season_totals.empty:
            continue

        max_load_season = season_totals.idxmax()
        min_load_season = season_totals.idxmin()
        extreme_seasons = [max_load_season] if max_load_season == min_load_season else [min_load_season, max_load_season]

        # Max load day plot
        df_extreme_seasons = df_year[df_year['season'].isin(extreme_seasons)]
        if not df_extreme_seasons.empty:
            day_totals = df_extreme_seasons.groupby('day', observed=False)['value'].sum()
            if not day_totals.empty:
                max_load_day = day_totals.idxmax()
                filename = os.path.join(folder, f'AssessmentDispatchDiff_{file_label}_day_{year}.pdf')
                select_time = {'season': extreme_seasons, 'day': [max_load_day]}
                make_fuel_dispatch_diff_plot(
                    dfs_to_plot_area,
                    dict_specs['colors'],
                    zone='System',
                    year=year,
                    scenario='diff',
                    fuel_grouping=None,
                    select_time=select_time,
                    filename=filename,
                    legend_loc='bottom',
                    title=f'Dispatch Difference (Project - Counterfactual)'
                )

        # Max load season plot
        filename = os.path.join(folder, f'AssessmentDispatchDiff_{file_label}_season_{year}.pdf')
        select_time = {'season': [max_load_season]}
        make_fuel_dispatch_diff_plot(
            dfs_to_plot_area,
            dict_specs['colors'],
            zone='System',
            year=year,
            scenario='diff',
            fuel_grouping=None,
            select_time=select_time,
            filename=filename,
            legend_loc='bottom',
            title=f'Dispatch Difference (Project - Counterfactual)'
        )
