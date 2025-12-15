"""
Assessment plotting routines for scenario comparisons.

This module centralizes the logic for building base vs. counterfactual
assessment charts (cost, capacity, and energy mix) to keep the main
post-processing flow slimmer.
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd

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
            # Counterfactual minus baseline as requested
            df_merged["value"] = df_merged["value_cf"] - df_merged["value_base"]
            df_merged["scenario"] = f"{scenario_cf} minus {scenario_base}"
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
    show_total_multi=False,
    show_total_single=False,
):
    """Plot the aggregated and per-pair assessment figures."""
    if df_all_diffs.empty:
        return

    multiple_pairs = len(scenario_pairs) > 1 or any(
        len(counterfactuals) > 1 for counterfactuals in scenario_pairs.values()
    )

    if multiple_pairs:
        filename = os.path.join(folder, f"{filename_prefix}_AllPairs.pdf")
        make_stacked_barplot(
            df_all_diffs,
            filename,
            dict_specs["colors"],
            column_stacked=stacked_column,
            column_subplot=None,
            column_xaxis=x_column,
            column_value="value",
            format_y=format_y,
            rotation=0,
            annotate=annotate,
            title=f"{title_prefix}: Counterfactual minus Baseline",
            show_total=show_total_multi,
        )

    for scenario_base, counterfactuals in scenario_pairs.items():
        for scenario_cf in counterfactuals:
            df_pair = df_all_diffs[
                df_all_diffs["scenario"] == f"{scenario_cf} minus {scenario_base}"
            ]
            if df_pair.empty:
                continue

            filename = os.path.join(
                folder, f"{filename_prefix}_{scenario_base}_vs_{scenario_cf}.pdf"
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
                title=f"{title_prefix}: {scenario_cf} minus {scenario_base}",
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
        show_total_multi=False,
        show_total_single=True,
    )


def make_assessment_capacity_diff(
    epm_results, dict_specs, folder, scenario_pairs
):
    """
    Generate capacity difference bar plots for project assessment.
    """
    df = epm_results["pCapacityTechFuel"].copy()
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
        title_prefix="Capacity Difference (MW)",
        filename_prefix="AssessmentCapacityDiff",
        annotate=False,
        show_total_multi=False,
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
        filename = os.path.join(folder, "AssessmentCapacityDiff_Cumulative.pdf")
        make_stacked_barplot(
            df_cumulative_year,
            filename,
            dict_specs["colors"],
            column_stacked="fuel",
            column_subplot="scenario",
            column_xaxis="year",
            column_value="value",
            format_y=make_auto_yaxis_formatter("MW"),
            rotation=0,
            annotate=False,
            title="Cumulative Capacity Difference by Year (MW)",
            show_total=True,
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
        show_total_multi=False,
        show_total_single=True,
    )
