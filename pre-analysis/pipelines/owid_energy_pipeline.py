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

from .load_pipeline import require_file, resolve_country_name

BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "dataset" / "owid-energy-data.csv"
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
            dataset_candidate = BASE_DIR / "dataset" / path.name
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


# ═══════════════════════════════════════════════════════════════════════════════
# EPM DEMAND FORECAST — fetch OWID from web + produce pDemandForecast rows
# ═══════════════════════════════════════════════════════════════════════════════

import io
import math
import urllib.request

OWID_URL = (
    "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
)
_OWID_CACHE_DIR = BASE_DIR / "cache" / "owid"
_OWID_CACHE_FILE = _OWID_CACHE_DIR / "owid-energy-data.csv"

# ISO-3 → OWID country name (OWID uses full names, not ISO codes directly)
_ISO3_TO_OWID: Dict[str, str] = {
    "AZE": "Azerbaijan",
    "ARM": "Armenia",
    "GEO": "Georgia",
    "TUR": "Turkey",
    "ROU": "Romania",
    "BGR": "Bulgaria",
    "RUS": "Russia",
    "IRN": "Iran",
    "UKR": "Ukraine",
    "MDA": "Moldova",
    # add more as needed
}


def fetch_owid_dataset(
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download (or return cached) OWID energy CSV from GitHub.

    The file (~10 MB) is cached locally and reused unless *force* is True.
    Falls back to the pre-analysis/dataset/ folder if it exists.
    """
    cache_dir = Path(cache_dir) if cache_dir else _OWID_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "owid-energy-data.csv"

    # Local cache hit
    if dest.exists() and not force:
        print(f"  [owid] Using cached dataset: {dest}")
        return dest

    # Also check legacy dataset/ folder used by existing owid_energy_pipeline
    legacy = BASE_DIR / "dataset" / "owid-energy-data.csv"
    if legacy.exists() and not force:
        print(f"  [owid] Using existing dataset: {legacy}")
        return legacy

    print(f"  [owid] Downloading OWID energy data (~10 MB)...")
    try:
        urllib.request.urlretrieve(OWID_URL, dest)
        print(f"  [owid] Saved {dest.stat().st_size / 1e6:.1f} MB -> {dest.name}")
        return dest
    except Exception as e:
        print(f"  [owid] Download failed: {e}")
        raise RuntimeError(
            "OWID dataset unavailable. Check internet or place owid-energy-data.csv "
            f"in {cache_dir}."
        ) from e


def load_owid_demand(
    iso3_codes: List[str],
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Load OWID electricity demand for given ISO-3 country codes.

    Returns a DataFrame with columns: iso3, year, demand_twh, generation_twh,
    net_imports_twh — one row per (country, year).
    """
    path = fetch_owid_dataset(cache_dir=cache_dir, force=force)
    df = pd.read_csv(path, low_memory=False)

    # Build reverse map: OWID country name → iso3
    owid_names = {v: k for k, v in _ISO3_TO_OWID.items()}
    targets = {_ISO3_TO_OWID[iso]: iso for iso in iso3_codes if iso in _ISO3_TO_OWID}

    if not targets:
        raise ValueError(f"None of {iso3_codes} found in _ISO3_TO_OWID mapping.")

    sub = df[df["country"].isin(targets.keys())].copy()
    sub["iso3"] = sub["country"].map(owid_names)
    sub["year"] = pd.to_numeric(sub["year"], errors="coerce").astype("Int64")
    sub["demand_twh"]     = pd.to_numeric(sub.get("electricity_demand",     pd.NA), errors="coerce")
    sub["generation_twh"] = pd.to_numeric(sub.get("electricity_generation", pd.NA), errors="coerce")
    sub["net_imports_twh"]= pd.to_numeric(sub.get("net_elec_imports",       pd.NA), errors="coerce")

    return (
        sub[["iso3", "country", "year", "demand_twh", "generation_twh", "net_imports_twh"]]
        .dropna(subset=["year", "demand_twh"])
        .sort_values(["iso3", "year"])
        .reset_index(drop=True)
    )


def get_demand_forecast(
    owid_df: pd.DataFrame,
    iso3: str,
    model_years: List[int],
    load_factor: float = 0.58,
    growth_rate: Optional[float] = None,
    cagr_window: int = 5,
) -> Dict[str, Dict[int, float]]:
    """Compute pDemandForecast (Peak MW + Energy GWh) for one country.

    Args:
        owid_df:     Output of load_owid_demand().
        iso3:        ISO-3 country code.
        model_years: List of years to produce forecasts for (e.g. 2024-2053).
        load_factor: Annual load factor used to estimate peak from energy
                     (default 0.58 = 58% → typical Caucasus/MENA).
        growth_rate: Fixed annual growth rate. If None, computed from OWID
                     CAGR over the last *cagr_window* years of data.
        cagr_window: Number of historical years used to compute CAGR.

    Returns:
        {"Peak": {year: MW, ...}, "Energy": {year: GWh, ...}}
    """
    cdf = owid_df[owid_df["iso3"] == iso3].sort_values("year")
    if cdf.empty:
        raise ValueError(f"No OWID demand data for {iso3}.")

    # Last available data point as anchor
    anchor_row  = cdf.dropna(subset=["demand_twh"]).iloc[-1]
    anchor_year = int(anchor_row["year"])
    anchor_twh  = float(anchor_row["demand_twh"])

    # Compute CAGR from historical data if growth_rate not provided
    if growth_rate is None:
        window = cdf.dropna(subset=["demand_twh"]).tail(cagr_window + 1)
        if len(window) >= 2:
            start_twh = float(window.iloc[0]["demand_twh"])
            end_twh   = float(window.iloc[-1]["demand_twh"])
            n_years   = int(window.iloc[-1]["year"]) - int(window.iloc[0]["year"])
            growth_rate = (end_twh / start_twh) ** (1 / n_years) - 1 if n_years > 0 else 0.02
        else:
            growth_rate = 0.02  # fallback 2%/yr
        print(f"  [owid] {iso3}: anchor={anchor_year} {anchor_twh:.1f} TWh, "
              f"CAGR={growth_rate*100:.1f}%/yr (last {cagr_window} yrs)")

    peak_mw: Dict[int, float] = {}
    energy_gwh: Dict[int, float] = {}

    for yr in model_years:
        delta  = yr - anchor_year
        twh    = anchor_twh * ((1 + growth_rate) ** delta)
        gwh    = round(twh * 1000, 2)
        # Peak = average demand / load_factor
        avg_mw = twh * 1e6 / 8760          # TWh → average MW
        peak   = round(avg_mw / load_factor, 2)
        energy_gwh[yr] = gwh
        peak_mw[yr]    = peak

    return {"Peak": peak_mw, "Energy": energy_gwh}


if __name__ == "__main__":  # pragma: no cover - convenience manual run
    build_owid_energy_outputs(
        dataset_path=None,
        countries=["Bosnia and Herzegovina", "Croatia"],
        output_dir=SCRIPT_DIR / "output_standalone" / "owid_energy",
        start_year=None,
        end_year=None,
        verbose=True,
    )
