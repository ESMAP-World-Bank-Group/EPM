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

    Examples:
        'baseline' -> 'baseline'
        'baseline_NoBiomass' -> 'NoBiomass'
        'baseline_pDemandForecast_015' -> 'HighDemand(+15%)'
        'baseline_pDemandForecast_-015' -> 'LowDemand(-15%)'
        'baseline_NoWind' -> 'NoWind'
    """
    import re

    # Remove 'baseline_' prefix if present
    if name.startswith('baseline_'):
        name = name[len('baseline_'):]

    # Handle pDemandForecast patterns
    match = re.match(r'pDemandForecast_(-?)0?(\d+)', name)
    if match:
        sign = match.group(1)
        value = match.group(2)
        if sign == '-':
            return f'LowDemand(-{value}%)'
        else:
            return f'HighDemand(+{value}%)'

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
    costs_system = _get_dataframe('pCostsSystem')
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
    if 'pAnnualTransmissionCapacity' in epm_results:
        transmission_all = _get_dataframe('pAnnualTransmissionCapacity')
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
