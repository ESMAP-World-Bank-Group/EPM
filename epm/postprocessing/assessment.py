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
from .utils import log_warning


def _wrap_plot_function(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            label = kwargs.get("filename") or kwargs.get("title")
            log_warning(f"Failed to generate {label}: {err}")

    return wrapper


make_stacked_barplot = _wrap_plot_function(_make_stacked_barplot)


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
            # e.g., "baseline@rehabilitation" -> "baseline", "baseline_NoBiomass@rehabilitation" -> "baseline_NoBiomass"
            df_merged["scenario"] = scenario_cf.split('@')[0] if '@' in scenario_cf else scenario_cf
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
            # Scenario label is the base name (before @)
            scenario_label = scenario_cf.split('@')[0] if '@' in scenario_cf else scenario_cf
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
    df = epm_results["pYearlyCostsZone"].copy()

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
        df_cumulative_year = (
            df_all_diffs.sort_values("year")
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
    df = df[~df["fuel"].isin(["Imports", "Exports"])]

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

    # Build list of scenario pairs with simplified labels
    pair_labels = []
    pair_info = []  # (base, counterfactual)
    for scenario_base, counterfactuals in scenario_pairs.items():
        for scenario_cf in counterfactuals:
            # Label is the base scenario name (before @)
            label = scenario_cf.split('@')[0] if '@' in scenario_cf else scenario_cf
            pair_labels.append(label)
            pair_info.append((scenario_base, scenario_cf))

    if not pair_info:
        log_warning("No scenario pairs found for assessment heatmap.")
        return

    rows = []

    # 1. NPV of system cost difference
    costs_system = _get_dataframe('pCostsSystem')
    if costs_system is not None:
        npv_label = 'NPV of system cost: $m'
        costs_npv = costs_system[costs_system['attribute'] == npv_label] if 'attribute' in costs_system.columns else costs_system
        diffs = []
        for base, cf in pair_info:
            df_base = costs_npv[costs_npv['scenario'] == base]
            df_cf = costs_npv[costs_npv['scenario'] == cf]
            val_base = df_base['value'].sum() if not df_base.empty else 0
            val_cf = df_cf['value'].sum() if not df_cf.empty else 0
            diffs.append(val_cf - val_base)
        rows.append(('NPV of system cost (M$)', diffs))

    # 2. Cumulative new capacity difference
    new_capacity_all = _get_dataframe('pNewCapacityTechFuel')
    if new_capacity_all is not None:
        final_year = _resolve_year(new_capacity_all)
        new_capacity = _filter_zone(new_capacity_all)
        # Cumulative up to final year
        new_capacity_cum = new_capacity[new_capacity['year'] <= final_year]
        diffs = []
        for base, cf in pair_info:
            df_base = new_capacity_cum[new_capacity_cum['scenario'] == base]
            df_cf = new_capacity_cum[new_capacity_cum['scenario'] == cf]
            val_base = df_base['value'].sum() if not df_base.empty else 0
            val_cf = df_cf['value'].sum() if not df_cf.empty else 0
            diffs.append(val_cf - val_base)
        rows.append((f'Cumulative new capacity {final_year} (MW)', diffs))

    # 3. Cumulative CAPEX difference
    capex_all = _get_dataframe('pCapexInvestmentComponent')
    if capex_all is not None:
        capex_year = _resolve_year(capex_all)
        capex = _filter_zone(capex_all)
        capex_cum = capex[capex['year'] <= capex_year]
        diffs = []
        for base, cf in pair_info:
            df_base = capex_cum[capex_cum['scenario'] == base]
            df_cf = capex_cum[capex_cum['scenario'] == cf]
            val_base = df_base['value'].sum() / 1e6 if not df_base.empty else 0  # to M$
            val_cf = df_cf['value'].sum() / 1e6 if not df_cf.empty else 0
            diffs.append(val_cf - val_base)
        rows.append((f'Cumulative CAPEX {capex_year} (M$)', diffs))

    # 4. Cumulative energy by fuel differences
    energy_all = _get_dataframe('pEnergyTechFuel')
    if energy_all is not None:
        energy_year = _resolve_year(energy_all)
        energy = _filter_zone(energy_all)
        energy_cum = energy[energy['year'] <= energy_year]

        # Get unique fuels
        fuels = energy_cum['fuel'].unique() if 'fuel' in energy_cum.columns else []
        for fuel in sorted(fuels):
            energy_fuel = energy_cum[energy_cum['fuel'] == fuel]
            diffs = []
            for base, cf in pair_info:
                df_base = energy_fuel[energy_fuel['scenario'] == base]
                df_cf = energy_fuel[energy_fuel['scenario'] == cf]
                val_base = df_base['value'].sum() if not df_base.empty else 0
                val_cf = df_cf['value'].sum() if not df_cf.empty else 0
                diffs.append(val_cf - val_base)
            rows.append((f'Cumulative energy - {fuel} (GWh)', diffs))

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
    annotations = data.map(lambda x: f"{x:+,.0f}" if abs(x) >= 1 else f"{x:+,.2f}")

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
    ax.set_xlabel("")
    ax.set_ylabel("")

    filename = os.path.join(folder, "AssessmentHeatmap.pdf")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
