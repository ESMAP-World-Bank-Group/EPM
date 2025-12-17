"""Quick plots + summary tables from the OWID energy dataset for selected countries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from load_pipeline import require_file, resolve_country_name

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "dataset" / "owid-energy-data.csv"
LEGEND_RECT_RIGHT = 0.92


@dataclass(frozen=True)
class OwidEnergyOutputs:
    """Container for output paths emitted by the OWID energy workflow."""

    population_fig: Path
    gdp_fig: Path
    electricity_fig: Path
    electricity_per_capita_fig: Path
    summary_csv: Path
    latest_csv: Path

    @property
    def all_targets(self) -> List[Path]:
        return [
            self.population_fig,
            self.gdp_fig,
            self.electricity_fig,
            self.electricity_per_capita_fig,
            self.summary_csv,
            self.latest_csv,
        ]


def _resolve_dataset_path(dataset_path: Path | str | None) -> Path:
    """Best-effort resolution of the OWID CSV relative to the repo/dataset folder."""
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET
    path = Path(dataset_path)
    if not path.is_absolute():
        candidate = SCRIPT_DIR / path
        if candidate.exists():
            path = candidate
        else:
            dataset_candidate = SCRIPT_DIR / "dataset" / path.name
            if dataset_candidate.exists():
                path = dataset_candidate
    return require_file(path, hint="Place owid-energy-data.csv under pre-analysis/dataset/.")


def _normalize_countries(df: pd.DataFrame, countries: Iterable[str], verbose: bool = True) -> List[str]:
    """Resolve requested country names against OWID country labels with fuzzy matching."""
    available = sorted(df["country"].dropna().unique())
    resolved: List[str] = []
    for country in countries:
        match = resolve_country_name(country, available, verbose=verbose, allow_missing=False)
        resolved.append(match)
    return resolved


def _filter_years(df: pd.DataFrame, start_year: Optional[int], end_year: Optional[int]) -> pd.DataFrame:
    """Restrict the dataframe to an optional year window."""
    data = df.copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    if start_year is not None:
        data = data[data["year"] >= int(start_year)]
    if end_year is not None:
        data = data[data["year"] <= int(end_year)]
    return data.dropna(subset=["year"])


def _metric_palette(countries: Sequence[str]) -> Dict[str, Tuple[float, float, float, float]]:
    """Assign consistent colors to countries across all plots."""
    cmap = plt.get_cmap("tab10")
    return {country: cmap(idx % cmap.N) for idx, country in enumerate(countries)}


def _plot_metric(
    df: pd.DataFrame,
    column: str,
    label: str,
    ylabel: str,
    output_path: Path,
    colors: Dict[str, Tuple[float, float, float, float]],
    scale: float = 1.0,
    fmt: str = ".1f",
    verbose: bool = False,
    unit: Optional[str] = None,
) -> Path:
    """Render a single-metric time series per country."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for country, group in df.groupby("country"):
        if group[column].notna().sum() == 0:
            continue
        series = group.sort_values("year")
        ax.plot(
            series["year"],
            series[column] / scale,
            label=country,
            color=colors.get(country),
            linewidth=2.0,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    resolved_title = f"{label} ({unit})" if unit else label
    ax.set_title(resolved_title)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, LEGEND_RECT_RIGHT, 1))
    else:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    if verbose:
        print(f"[owid-energy] Saved {label} plot to {output_path} using column '{column}'")
    return output_path


def _plot_electricity_demand(
    df: pd.DataFrame,
    output_path: Path,
    colors: Dict[str, Tuple[float, float, float, float]],
    verbose: bool = False,
) -> Path:
    """Plot total electricity demand (single axis)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for country, group in df.groupby("country"):
        ordered = group.sort_values("year")
        color = colors.get(country)
        if ordered["electricity_demand_twh"].notna().any():
            ax.plot(
                ordered["year"],
                ordered["electricity_demand_twh"],
                label=country,
                color=color,
                linewidth=2.0,
            )[0]
    ax.set_xlabel("Year")
    ax.set_ylabel("Electricity demand (TWh)")
    ax.set_ylim(bottom=0)
    ax.set_title("Electricity demand (TWh)")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, LEGEND_RECT_RIGHT, 1))
    else:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    if verbose:
        print("[owid-energy] Saved electricity demand plot to "
              f"{output_path} using column 'electricity_demand_twh'")
    return output_path


def _plot_per_capita_energy(
    df: pd.DataFrame,
    output_path: Path,
    colors: Dict[str, Tuple[float, float, float, float]],
    verbose: bool = False,
) -> Path:
    """Plot per-capita electricity demand plus total energy per capita."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    seen_countries: List[str] = []
    for country, group in df.groupby("country"):
        ordered = group.sort_values("year")
        color = colors.get(country)
        has_electricity = ordered["electricity_demand_per_capita_kwh"].notna().any()
        has_energy = ordered["energy_per_capita_kwh"].notna().any()
        if not has_electricity and not has_energy:
            continue
        if country not in seen_countries:
            seen_countries.append(country)
        if has_electricity:
            ax.plot(
                ordered["year"],
                ordered["electricity_demand_per_capita_kwh"],
                color=color,
                linestyle="-",
                linewidth=2.0,
            )
        if has_energy:
            ax.plot(
                ordered["year"],
                ordered["energy_per_capita_kwh"],
                color=color,
                linestyle="--",
                linewidth=1.8,
            )

    ax.set_xlabel("Year")
    ax.set_ylabel("Per-capita demand (kWh/person)")
    ax.set_ylim(bottom=0)
    ax.set_title("Per-capita electricity demand vs. total energy (kWh/person)")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)

    metric_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.0),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8),
    ]
    metric_labels = ["Electricity demand per capita", "Total energy per capita"]

    country_handles = [
        Line2D([0], [0], color=colors.get(country, "#000000"), linewidth=2.0)
        for country in seen_countries
    ]
    country_labels = seen_countries

    combined_handles = country_handles + metric_handles
    combined_labels = [f"{label}" for label in country_labels] + metric_labels
    if combined_handles:
        ax.legend(
            combined_handles,
            combined_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, LEGEND_RECT_RIGHT, 1))
    else:
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    if verbose:
        print(
            "[owid-energy] Saved per-capita demand/energy plot to "
            f"{output_path} using columns 'electricity_demand_per_capita_kwh' and 'energy_per_capita_kwh'"
        )
    return output_path


DEFAULT_REQUIRED_COLS = [
    "population",
    "gdp",
    "electricity_demand",
    "electricity_demand_per_capita",
    "energy_per_capita",
]


def _prepare_summary_table(
    df: pd.DataFrame, required_cols: Optional[Sequence[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build cleaned summary tables (full history + latest year per country)."""
    required_cols = list(required_cols or DEFAULT_REQUIRED_COLS)
    missing = [col for col in required_cols if col not in df.columns]
    # Soft-fail: add missing columns as NA so downstream plots still render.
    if missing:
        print(f"[owid-energy] Missing columns in dataset: {', '.join(missing)}; filling with NA.")
        for col in missing:
            df[col] = pd.NA

    subset = df[["country", "year", *required_cols]].copy()
    subset = subset.dropna(subset=["country", "year"], how="any")
    for col in required_cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    summary = subset.copy()
    summary["population_millions"] = summary["population"] / 1e6
    summary["gdp_billions_usd"] = summary["gdp"] / 1e9
    summary["electricity_demand_twh"] = summary["electricity_demand"]
    summary["electricity_demand_per_capita_kwh"] = summary["electricity_demand_per_capita"]
    summary["energy_per_capita_kwh"] = summary["energy_per_capita"]
    summary = summary[
        [
            "country",
            "year",
            "population_millions",
            "gdp_billions_usd",
            "electricity_demand_twh",
            "electricity_demand_per_capita_kwh",
            "energy_per_capita_kwh",
        ]
    ].dropna(how="all")

    latest_rows: List[pd.Series] = []
    for country, group in summary.groupby("country"):
        ordered = group.sort_values("year")
        latest_rows.append(ordered.iloc[-1])
    latest = pd.DataFrame(latest_rows)

    return summary, latest


def build_owid_energy_outputs(
    *,
    dataset_path: Path | str | None,
    countries: Iterable[str],
    output_dir: Path | str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    basename: str = "owid_energy",
    verbose: bool = True,
    required_columns: Optional[Sequence[str]] = None,
) -> OwidEnergyOutputs:
    """Main entry point used by Snakemake to generate OWID figures + tables."""
    required_columns = list(required_columns or DEFAULT_REQUIRED_COLS)
    dataset = _resolve_dataset_path(dataset_path)
    if verbose:
        print(f"[owid-energy] Using dataset at {dataset}")

    df = pd.read_csv(dataset)
    if verbose:
        present = [col for col in required_columns if col in df.columns]
        missing = [col for col in required_columns if col not in df.columns]
        print(f"[owid-energy] Required columns present: {', '.join(present) or 'none'}")
        if missing:
            print(f"[owid-energy] Missing columns will be filled with NA: {', '.join(missing)}")
    resolved_countries = _normalize_countries(df, countries, verbose=verbose)
    filtered = df[df["country"].isin(resolved_countries)].copy()
    filtered = _filter_years(filtered, start_year, end_year)

    if filtered.empty:
        print("[owid-energy] No rows found for requested countries; writing empty outputs.")
        filtered = pd.DataFrame(columns=["country", "year", *required_columns])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = OwidEnergyOutputs(
        population_fig=output_dir / f"{basename}_population.pdf",
        gdp_fig=output_dir / f"{basename}_gdp.pdf",
        electricity_fig=output_dir / f"{basename}_electricity.pdf",
        electricity_per_capita_fig=output_dir / f"{basename}_electricity_per_capita.pdf",
        summary_csv=output_dir / f"{basename}_summary.csv",
        latest_csv=output_dir / f"{basename}_latest.csv",
    )

    summary_table, latest_table = _prepare_summary_table(filtered, required_columns)
    summary_table.to_csv(outputs.summary_csv, index=False)
    latest_table.to_csv(outputs.latest_csv, index=False)
    if verbose:
        print(f"[owid-energy] Wrote summary table to {outputs.summary_csv}")
        print(f"[owid-energy] Wrote latest-year table to {outputs.latest_csv}")

    palette = _metric_palette(resolved_countries)
    _plot_metric(
        summary_table,
        column="population_millions",
        label="Population",
        ylabel="Population (millions)",
        output_path=outputs.population_fig,
        colors=palette,
        verbose=verbose,
        unit="millions",
    )
    _plot_metric(
        summary_table,
        column="gdp_billions_usd",
        label="GDP (real)",
        ylabel="GDP (billion USD)",
        output_path=outputs.gdp_fig,
        colors=palette,
        verbose=verbose,
        unit="billion USD",
    )
    _plot_electricity_demand(
        summary_table,
        output_path=outputs.electricity_fig,
        colors=palette,
        verbose=verbose,
    )
    _plot_per_capita_energy(
        summary_table,
        output_path=outputs.electricity_per_capita_fig,
        colors=palette,
        verbose=verbose,
    )

    return outputs


if __name__ == "__main__":  # pragma: no cover - convenience manual run
    build_owid_energy_outputs(
        dataset_path=None,
        countries=["Bosnia and Herzegovina", "Croatia"],
        output_dir=SCRIPT_DIR / "output_standalone" / "owid_energy",
        start_year=None,
        end_year=None,
        verbose=True,
    )
