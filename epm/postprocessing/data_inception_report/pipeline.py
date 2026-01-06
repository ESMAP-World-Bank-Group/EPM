"""Build a simple, end-to-end data inception report pipeline.

The module is intentionally straightforward: a handful of small functions with
explicit inputs/outputs and minimal magic. Each function returns data instead
of mutating global state, which keeps the orchestrator easy to follow and
robust to partial failures.
"""

from __future__ import annotations
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

# Allow running as a script by adding the project root to PYTHONPATH.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from epm.postprocessing import plots, maps
from epm.postprocessing.utils import read_plot_specs


REQUIRED_CONFIG_COLUMNS = [
    "supply_path",
    "price_path",
    "demand_forecast_path",
    "demand_profile_path",
    "zcmap_path",
    "pvreprofile_path",
    "ptransferlimit_path",
    "pexttransferlimit_path",
]

OUTPUT_SUBFOLDERS = ["reports", "figures", "logs"]
VERBOSE = True
MW_TO_GWH = 8.76
DISABLE_FLOAT_TEX = _PROJECT_ROOT / "pre-analysis" / "disable_float.tex"

# === Pipeline orchestration ===


def run_data_inception_report(data_folder: str) -> Dict[str, str]:
    """Run the full data inception pipeline for every case in the config file."""

    base_input, config_path, base_output = _resolve_paths(data_folder)
    ensure_output_tree(base_output)

    _log(f"Looking for configuration at {config_path}")
    config_df = load_config(config_path)
    validated_config, config_errors = validate_config(config_df)
    if validated_config.empty:
        _log("No valid configuration rows found; aborting pipeline.")
        return {"markdown": "", "docx": "", "pdf": ""}

    case_row = validated_config.iloc[0]

    datasets, dataset_errors, years = _load_case_inputs(base_input, case_row)
    case_log = [msg for msg in dataset_errors if msg]

    figures = generate_plots(
        "baseline",
        {
            "supply": datasets["supply"],
            "price": datasets["price"],
            "demand_forecast": datasets["demand_forecast"],
            "demand_profile": datasets["demand_profile"],
            "transfer": datasets["transfer"],
            "availability_custom": datasets["availability_custom"],
            "availability_default": datasets["availability_default"],
            "capex_trajectories": datasets["capex_trajectories"],
        },
        base_output / "figures",
        base_input,
        years,
    )

    # Generate tables for the report
    tables = _generate_tables(datasets, years)

    cases: List[Dict] = [
        {
            "case_id": "baseline",
            "scenario": "Baseline scenario",
            "country": "",
            "figures": figures,
            "tables": tables,
            "log": case_log,
        }
    ]

    report_context = {
        "data_folder": data_folder,
        "generation_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "config_errors": config_errors,
        "cases": cases,
    }

    report_name = f"data_inception_{data_folder}"
    markdown_path = base_output / "reports" / f"{report_name}.md"
    docx_path = base_output / "reports" / f"{report_name}.docx"
    pdf_path = base_output / "reports" / f"{report_name}.pdf"

    render_markdown_report(markdown_path, report_context)
    convert_markdown_to_docx(markdown_path, docx_path)
    convert_markdown_to_pdf(markdown_path, pdf_path)

    return {"markdown": str(markdown_path), "docx": str(docx_path), "pdf": str(pdf_path)}


def _resolve_paths(data_folder: str) -> Tuple[Path, Path, Path]:
    """Return common input/output paths for a given data folder."""

    base_input = Path("epm") / "input" / data_folder
    config_path = base_input / "config.csv"
    base_output = Path(__file__).resolve().parent / "output"
    return base_input, config_path, base_output


def _load_case_inputs(
    base_input: Path, config_row: pd.Series
) -> Tuple[Dict[str, Optional[pd.DataFrame]], List[Optional[str]], List[int]]:
    """Load all baseline inputs in one place to keep the orchestrator lean."""

    datasets: Dict[str, Optional[pd.DataFrame]] = {}
    errors: List[Optional[str]] = []

    # Core datasets listed in the driver CSV.
    path_lookup = {
        "capex_trajectories": "pcapextrajectoriesdefault_path",
        "gen_defaults": "pgendatainputdefault_path",
        "supply": "supply_path",
        "price": "price_path",
        "demand_forecast": "demand_forecast_path",
        "demand_profile": "demand_profile_path",
        "zcmap": "zcmap_path",
        "vre_profile": "pvreprofile_path",
        "settings": "psettings_path",
        "transfer": "ptransferlimit_path",
        "ext_transfer": "pexttransferlimit_path",
        "availability_custom": "pavailability_path",
        "availability_default": "pavailabilitydefault_path",
    }

    for label, column in path_lookup.items():
        df, err = load_dataset(base_input, config_row.get(column))
        datasets[label] = df
        if err:
            _log(f"Warning: failed to load {label}: {err}")
        errors.append(err)

    # Model years share the same loader; keep error handling consistent.
    years_df, years_error = load_dataset(base_input, "y.csv")
    if years_df is not None and not years_df.empty:
        years_series = pd.to_numeric(years_df.iloc[:, 0], errors="coerce").dropna().astype(int)
        years = sorted(years_series.unique().tolist())
    else:
        years = []
    if not years:
        _log("Warning: no model years found; defaulting to [0].")
        years = [0]
    else:
        _log(f"Model years detected: {years}")

    errors.append(years_error)

    # Filter all datasets to zones defined in zcmap
    zcmap_df = datasets.get("zcmap")
    if zcmap_df is not None and "zone" in zcmap_df.columns:
        valid_zones = set(zcmap_df["zone"].dropna().astype(str).str.strip())
        _log(f"Filtering data to zcmap zones: {sorted(valid_zones)}")

        zone_columns = {"zone", "from", "to"}  # case-insensitive matching
        for key, df in datasets.items():
            if df is None or key == "zcmap":
                continue
            cols_to_filter = [c for c in df.columns if c.lower() in zone_columns]
            if cols_to_filter:
                before = len(df)
                mask = pd.Series(True, index=df.index)
                for col in cols_to_filter:
                    mask &= df[col].astype(str).str.strip().isin(valid_zones)
                datasets[key] = df[mask]
                _log(f"  {key}: {before} -> {len(datasets[key])} rows")

    return datasets, errors, years


def _generate_tables(
    datasets: Dict[str, Optional[pd.DataFrame]], years: List[int]
) -> Dict[str, str]:
    """Generate markdown tables for the report."""

    tables: Dict[str, str] = {}

    # Supply table (unstacked by fuel)
    supply_summary, _ = analyze_supply(datasets.get("supply"), years)
    if not supply_summary.empty and "Status" not in supply_summary.columns:
        tables["supply"] = _df_to_markdown(supply_summary)

    # Price table with both $/MMBtu and ($/MWh)
    price_summary, _ = analyze_prices(datasets.get("price"), years)
    if not price_summary.empty and "Status" not in price_summary.columns:
        price_display = _format_price_table(price_summary)
        tables["price"] = _df_to_markdown(price_display, round_nums=False)

    # Demand forecast table
    demand_summary, _ = summarize_demand_forecast(datasets.get("demand_forecast"), years)
    if not demand_summary.empty and "Status" not in demand_summary.columns:
        tables["demand_forecast"] = _df_to_markdown(demand_summary)

    # Generator defaults table (average across zones)
    gen_defaults_summary = analyze_gen_defaults(datasets.get("gen_defaults"))
    if not gen_defaults_summary.empty and "Status" not in gen_defaults_summary.columns:
        tables["gen_defaults"] = _df_to_markdown(gen_defaults_summary)

    # Settings table (for appendix)
    settings_df = datasets.get("settings")
    if settings_df is not None and not settings_df.empty:
        settings_table = build_psettings_table(settings_df)
        if not settings_table.empty:
            tables["settings"] = _df_to_markdown(settings_table, round_nums=False)

    return tables


def _format_price_table(price_df: pd.DataFrame) -> pd.DataFrame:
    """Format price table with both $/MMBtu and ($/MWh) values."""

    MMBTU_TO_MWH = 3.412  # 1 MWh = 3.412 MMBtu
    df = price_df.copy()
    for col in df.columns:
        if col == "Fuel":
            continue
        df[col] = df[col].apply(
            lambda x: f"{int(round(x))} ({int(round(x * MMBTU_TO_MWH))})" if pd.notna(x) else ""
        )
    return df


def _df_to_markdown(df: pd.DataFrame, round_nums: bool = True) -> str:
    """Convert a DataFrame to a markdown table with rounded numeric values."""

    df = df.copy()
    if round_nums:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype(object)
    # Replace NaN with empty string without triggering FutureWarning
    for col in df.columns:
        df[col] = df[col].apply(lambda x: "" if pd.isna(x) else x)
    # Convert floats like 123.0 to int strings like "123"
    for col in df.columns:
        df[col] = df[col].apply(lambda x: int(x) if isinstance(x, float) and x == x else x)
    return df.to_markdown(index=False)


# === Configuration + data loading ===


def ensure_output_tree(base_output: Path) -> None:
    """Create the mandatory post-processing folders if they are missing."""

    for sub in OUTPUT_SUBFOLDERS:
        (base_output / sub).mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path) -> pd.DataFrame:
    """Load the driver configuration file as a DataFrame."""

    if not config_path.exists():
        _log("Config file not found; returning empty DataFrame.")
        return pd.DataFrame(columns=REQUIRED_CONFIG_COLUMNS)
    try:
        return pd.read_csv(config_path)
    except Exception:
        _log("Failed to read config file; returning empty DataFrame.")
        return pd.DataFrame(columns=REQUIRED_CONFIG_COLUMNS)


def validate_config(config_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Validate the configuration and collect any blocking errors."""

    errors: List[str] = []
    missing_cols = [col for col in REQUIRED_CONFIG_COLUMNS if col not in config_df.columns]
    if missing_cols:
        # Try to convert legacy parameter-style config to a single-case driver.
        if {"paramNames", "file"}.issubset(set(config_df.columns)):
            derived = _derive_case_from_parameter_config(config_df)
            _log("Converted parameter-style config.csv into a single-case driver.")
            return derived, errors

        errors.append(f"Missing columns: {', '.join(missing_cols)}")
        _log(f"Validation failed: missing columns {missing_cols}")
        return pd.DataFrame(columns=REQUIRED_CONFIG_COLUMNS), errors

    empty_rows = config_df[REQUIRED_CONFIG_COLUMNS].isnull().any(axis=1)
    if empty_rows.any():
        errors.append("Rows with missing required fields were dropped.")
        _log(f"Validation: dropped {empty_rows.sum()} rows with missing required fields")
        config_df = config_df.loc[~empty_rows].copy()

    return config_df, errors


def build_psettings_table(settings_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Format the pSettings input into a compact table for reporting."""

    if settings_df is None or settings_df.empty:
        return pd.DataFrame([{"Parameter": "No pSettings data found.", "Value": ""}])

    df = settings_df.copy()
    # Normalise column names for flexible input (e.g., pSettingsHeader vs Abbreviation).
    normalized_cols = {col: str(col).strip().lower() for col in df.columns}
    df = df.rename(columns=normalized_cols)

    def _pick_column(candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None

    param_col = _pick_column(["parameter", "parameters", "description", "name", "setting"])
    abbrev_col = _pick_column(["abbreviation", "abbr", "psettingsheader", "settingheader"])
    value_col = _pick_column(["value", "val"])

    ordered_cols: List[str] = []
    if param_col:
        ordered_cols.append(param_col)
    if value_col:
        ordered_cols.append(value_col)
    elif abbrev_col:
        # Fallback: use abbreviation as value if no explicit value column exists.
        ordered_cols.append(abbrev_col)
    if not ordered_cols:
        ordered_cols = list(df.columns[:2])

    tidy = df[ordered_cols].copy()
    tidy.columns = ["Parameter", "Value"][: len(tidy.columns)]
    tidy = tidy.dropna(how="all")
    tidy = tidy.replace({np.nan: ""})

    if "Parameter" in tidy.columns:
        tidy["Parameter"] = tidy["Parameter"].astype(str).str.strip()
    if "Value" in tidy.columns:
        def _clean_value(val: object) -> object:
            if pd.isna(val):
                return ""
            return str(val).strip()

        tidy["Value"] = tidy["Value"].apply(_clean_value)

    # Drop empty spacer rows that sometimes appear in pSettings.
    tidy = tidy[~((tidy.get("Parameter", "") == "") & (tidy.get("Value", "") == ""))]

    if tidy.empty:
        return pd.DataFrame([{"Parameter": "pSettings file is present but contains no data.", "Value": ""}])

    return tidy.reset_index(drop=True)


def load_dataset(base_input: Path, path_value: Union[str, Path, None]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load a dataset from an exact config path (relative to base_input if needed)."""

    if not path_value:
        return None, "No path provided."
    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        path_obj = base_input / path_obj
    if not path_obj.exists():
        return None, f"File not found: {path_obj}"
    try:
        _log(f"Loading data from {path_obj}")
        df = pd.read_csv(path_obj)
        return df, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Could not read dataset ({path_obj}): {exc}"


# === Analytics ===


def analyze_supply(df: Optional[pd.DataFrame], years: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize capacity in the first planning year by zone/tech-fuel (with total)."""

    if df is None or df.empty:
        empty = pd.DataFrame([{"Status": "no supply data"}])
        return empty, empty

    if "Capacity" not in df.columns:
        note = pd.DataFrame([{"Status": "missing Capacity column"}])
        return note, note

    supply_df = df.copy()
    for column in ["zone", "fuel", "tech"]:
        if column in supply_df.columns:
            supply_df[column] = supply_df[column].astype(str).str.strip()

    # Filter to active units when status is present and normalise fuel names.
    if "Status" in supply_df.columns:
        supply_df = supply_df[supply_df["Status"].isin([1, 2])]
    _, _, techfuel_mapping = _build_fuel_colors()

    # Create tech-fuel merged label, then apply techfuel mapping
    if "tech" in supply_df.columns and "fuel" in supply_df.columns:
        # Create techfuel key (tech-fuel with hyphen) for mapping
        supply_df["techfuel_key"] = supply_df["tech"] + "-" + supply_df["fuel"]
        # Apply techfuel mapping to get Processing name
        if techfuel_mapping:
            supply_df["tech_fuel"] = supply_df["techfuel_key"].map(techfuel_mapping).fillna(supply_df["techfuel_key"])
        else:
            supply_df["tech_fuel"] = supply_df["techfuel_key"]
    elif "fuel" in supply_df.columns:
        supply_df["tech_fuel"] = supply_df["fuel"]
    else:
        supply_df["tech_fuel"] = "Unknown"

    supply_df["Capacity"] = pd.to_numeric(supply_df.get("Capacity"), errors="coerce").fillna(0)

    clean_years = sorted({int(y) for y in years if y is not None}) if years else []
    first_year = clean_years[0] if clean_years else None
    last_year = clean_years[-1] if clean_years else first_year

    # If planning years are missing, infer a first year from start-year data.
    if first_year is None:
        inferred = pd.to_numeric(supply_df.get("StYr"), errors="coerce").dropna()
        if not inferred.empty:
            first_year = int(inferred.min())
            last_year = int(inferred.max()) if pd.notna(inferred.max()) else first_year

    supply_df["start_year"] = pd.to_numeric(supply_df.get("StYr"), errors="coerce").fillna(first_year or 0)
    supply_df["retire_year"] = pd.to_numeric(supply_df.get("RetrYr"), errors="coerce").fillna(last_year or (first_year or 0))

    if first_year is not None:
        active = supply_df[(supply_df["start_year"] <= first_year) & (supply_df["retire_year"] >= first_year)]
    else:
        active = supply_df

    if active.empty:
        empty = pd.DataFrame([{"Status": "no active capacity in first year"}])
        return empty, empty

    summary = (
        active.groupby(["zone", "tech_fuel"], observed=False)["Capacity"]
        .sum()
        .reset_index()
        .rename(columns={"zone": "Zone", "tech_fuel": "Technology", "Capacity": "Cap_MW"})
    )

    # Wide table with tech-fuel as columns; keep blanks (NaN) when a zone lacks a technology.
    summary_wide = summary.pivot(index="Zone", columns="Technology", values="Cap_MW")
    tech_order = sorted(summary_wide.columns)
    summary_wide = summary_wide.reindex(columns=tech_order)
    summary_wide["Total"] = summary_wide.sum(axis=1, skipna=True)

    totals_row = pd.DataFrame(summary_wide.sum(axis=0, skipna=True)).T
    totals_row.index = ["All"]

    summary_df = pd.concat([summary_wide, totals_row])
    summary_df = summary_df.reset_index().rename(columns={"index": "Zone"})
    summary_df = summary_df.replace(0, np.nan)

    detail_df = (
        active[["zone", "fuel", "Capacity", "start_year", "retire_year"]]
        .rename(columns={"zone": "Zone", "fuel": "Fuel", "Capacity": "Cap_MW", "start_year": "StYr", "retire_year": "RetYr"})
        .sort_values(["Zone", "Fuel"])
        .reset_index(drop=True)
    )

    return summary_df, detail_df


def analyze_prices(df: Optional[pd.DataFrame], years: Optional[List[int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize fuel prices using only valid planning years."""

    if df is None or df.empty:
        empty = pd.DataFrame([{"Status": "no price data"}])
        return empty, empty

    price_long = _reshape_price(df)
    if price_long.empty:
        note = pd.DataFrame([{"Status": "missing fuel/year columns for prices"}])
        return note, note

    if years:
        valid_years = {int(y) for y in years if pd.notna(y)}
        price_long = price_long[price_long["year"].isin(valid_years)]
        if price_long.empty:
            note = pd.DataFrame([{"Status": "no price data for model years"}])
            return note, note
    price_long = price_long.rename(columns={"fuel": "Fuel", "year": "Year", "price": "USD_per_MMBtu"})
    price_long = price_long.sort_values(["Fuel", "Year"]).reset_index(drop=True)

    summary_wide = price_long.pivot(index="Fuel", columns="Year", values="USD_per_MMBtu")
    year_order = sorted(summary_wide.columns)
    if years:
        preferred_order = [int(y) for y in years if y in summary_wide.columns]
        if preferred_order:
            year_order = preferred_order
    summary_wide = summary_wide.reindex(columns=year_order).reset_index()
    summary_wide.columns = ["Fuel"] + [str(c) for c in summary_wide.columns[1:]]

    return summary_wide, price_long


def analyze_gen_defaults(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Summarize generator default parameters averaged across zones.

    Returns a table with columns: Technology, CAPEX (USD/kW), FOM (USD/kW-yr), VOM (USD/MWh), Heat Rate (MMBtu/MWh).
    Values are averaged across all zones for each tech-fuel combination.
    """

    if df is None or df.empty:
        return pd.DataFrame([{"Status": "no generator defaults data"}])

    gen_df = df.copy()
    gen_df.columns = [str(c).strip() for c in gen_df.columns]

    # Identify required columns (case-insensitive)
    col_map = {}
    for col in gen_df.columns:
        col_lower = col.lower()
        if col_lower == "tech":
            col_map["tech"] = col
        elif col_lower == "fuel":
            col_map["fuel"] = col
        elif col_lower == "capex":
            col_map["capex"] = col
        elif col_lower == "fompermw":
            col_map["fom"] = col
        elif col_lower == "vom":
            col_map["vom"] = col
        elif col_lower == "heatrate":
            col_map["heatrate"] = col

    if "tech" not in col_map:
        return pd.DataFrame([{"Status": "missing tech column in generator defaults"}])

    # Get techfuel mapping for renaming
    _, _, techfuel_mapping = _build_fuel_colors()

    # Create tech-fuel label, then apply techfuel mapping
    if "fuel" in col_map:
        # Create techfuel key (tech-fuel with hyphen) for mapping
        gen_df["techfuel_key"] = gen_df[col_map["tech"]].astype(str).str.strip() + "-" + gen_df[col_map["fuel"]].astype(str).str.strip()
        # Apply techfuel mapping to get Processing name
        if techfuel_mapping:
            gen_df["technology"] = gen_df["techfuel_key"].map(techfuel_mapping).fillna(gen_df["techfuel_key"])
        else:
            gen_df["technology"] = gen_df["techfuel_key"]
    else:
        gen_df["technology"] = gen_df[col_map["tech"]].astype(str).str.strip()

    # Convert numeric columns
    numeric_cols = {}
    if "capex" in col_map:
        gen_df["capex_val"] = pd.to_numeric(gen_df[col_map["capex"]], errors="coerce")
        numeric_cols["capex_val"] = "CAPEX (USD/kW)"
    if "fom" in col_map:
        gen_df["fom_val"] = pd.to_numeric(gen_df[col_map["fom"]], errors="coerce")
        numeric_cols["fom_val"] = "FOM (USD/kW-yr)"
    if "vom" in col_map:
        gen_df["vom_val"] = pd.to_numeric(gen_df[col_map["vom"]], errors="coerce")
        numeric_cols["vom_val"] = "VOM (USD/MWh)"
    if "heatrate" in col_map:
        gen_df["heatrate_val"] = pd.to_numeric(gen_df[col_map["heatrate"]], errors="coerce")
        numeric_cols["heatrate_val"] = "Heat Rate (MMBtu/MWh)"

    if not numeric_cols:
        return pd.DataFrame([{"Status": "no numeric columns found in generator defaults"}])

    # Average across all zones for each technology
    agg_cols = list(numeric_cols.keys())
    averaged = gen_df.groupby("technology", as_index=False)[agg_cols].mean()

    # Rename columns to human-readable names
    averaged = averaged.rename(columns={"technology": "Technology"})
    averaged = averaged.rename(columns=numeric_cols)

    # Sort by technology name
    averaged = averaged.sort_values("Technology").reset_index(drop=True)

    # Filter out rows where all numeric values are zero or NaN
    value_cols = list(numeric_cols.values())
    mask = averaged[value_cols].fillna(0).sum(axis=1) > 0
    averaged = averaged[mask]

    if averaged.empty:
        return pd.DataFrame([{"Status": "no valid generator default data"}])

    return averaged


def summarize_demand_forecast(df: Optional[pd.DataFrame], years: Optional[List[int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Provide a compact demand forecast table (year, zone, scenario) filtered to planning years."""

    if df is None or df.empty:
        empty = pd.DataFrame([{"Status": "no demand forecast"}])
        return empty, empty

    normalized = _normalize_demand_forecast(df)
    if normalized.empty:
        empty = pd.DataFrame([{"Status": "could not normalize forecast"}])
        return empty, empty

    preferred_type = "Energy" if "Energy" in normalized["type"].unique() else normalized["type"].dropna().unique()[0]
    subset = normalized[normalized["type"] == preferred_type].copy()
    if subset.empty:
        subset = normalized.copy()

    subset["year"] = pd.to_numeric(subset.get("year"), errors="coerce")
    subset = subset.dropna(subset=["year"])
    subset["year"] = subset["year"].astype(int)
    if years:
        valid_years = {int(y) for y in years if pd.notna(y)}
        subset = subset[subset["year"].isin(valid_years)]
        if subset.empty:
            empty = pd.DataFrame([{"Status": "no demand forecast for model years"}])
            return empty, empty
    subset["demand"] = pd.to_numeric(subset.get("demand"), errors="coerce").fillna(0)
    subset["scenario"] = "Baseline"
    subset["zone"] = subset.get("zone", "Unknown").fillna("Unknown")

    grouped = (
        subset.groupby(["zone", "year"], as_index=False)["demand"]
        .sum()
        .rename(columns={"zone": "Zone", "year": "Year", "demand": "Demand"})
    )
    pivot = grouped.pivot_table(index="Zone", columns="Year", values="Demand", fill_value=0)
    year_order = sorted(pivot.columns)
    if years:
        preferred_order = [int(y) for y in years if y in pivot.columns]
        if preferred_order:
            year_order = preferred_order
    pivot = pivot.reindex(columns=year_order)
    summary_df = pivot.reset_index()
    summary_df.columns = ["Zone"] + [str(col) for col in summary_df.columns[1:]]

    detail_df = subset.rename(columns={"zone": "Zone", "scenario": "Scen", "year": "Year", "type": "Type", "demand": "Demand"})
    detail_df = detail_df[["Zone", "Scen", "Type", "Year", "Demand"]].sort_values(["Zone", "Year", "Scen"]).reset_index(drop=True)

    return summary_df, detail_df


# === Plot generation ===


def generate_plots(
    case_id: str, datasets: Dict[str, Optional[pd.DataFrame]], output_dir: Path, input_root: Path, years: List[int]
) -> Dict[str, str]:
    """Create figures for each dataset using dedicated helpers."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: Dict[str, str] = {}

    figure_paths.update(_generate_demand_profile_plots(case_id, datasets.get("demand_profile"), output_dir))
    figure_paths.update(_generate_price_plot(case_id, datasets.get("price"), output_dir))
    figure_paths.update(
        _generate_supply_plot(
            case_id,
            datasets.get("supply"),
            output_dir,
            datasets.get("availability_custom"),
            datasets.get("availability_default"),
            years,
            datasets.get("demand_forecast"),
        )
    )
    figure_paths.update(
        _generate_availability_plot(
            case_id,
            datasets.get("availability_custom"),
            datasets.get("supply"),
            output_dir,
            input_root,
        )
    )
    figure_paths.update(_generate_demand_forecast_plot(case_id, datasets.get("demand_forecast"), output_dir))
    figure_paths.update(_generate_transfer_capacity_map(case_id, datasets.get("transfer"), output_dir))
    figure_paths.update(_generate_capex_trajectories_plot(case_id, datasets.get("capex_trajectories"), output_dir, years))

    return figure_paths


def _generate_transfer_capacity_map(case_id: str, transfer_df: Optional[pd.DataFrame], output_dir: Path) -> Dict[str, str]:
    """Plot existing transfer capacity for the first year (mean across seasons)."""

    if transfer_df is None or transfer_df.empty:
        _log("Warning: transfer data missing or empty; skipping transfer map.")
        return {}

    year_cols: List[int] = []
    for col in transfer_df.columns:
        if str(col).isdigit() and col not in {"q"}:
            try:
                year_cols.append(int(col))
            except ValueError:
                continue

    if not year_cols:
        _log("Warning: no year columns found in transfer data; skipping transfer map.")
        return {}

    first_year = min(year_cols)
    year_col = str(first_year)

    if not {"From", "To", "q"}.issubset(set(transfer_df.columns)):
        _log("Warning: transfer data missing required columns From/To/q; skipping transfer map.")
        return {}

    averaged = (
        transfer_df.assign(**{year_col: pd.to_numeric(transfer_df[year_col], errors="coerce")})
        .groupby(["From", "To"], as_index=False)[year_col]
        .mean()
        .rename(columns={"From": "zone_from", "To": "zone_to", year_col: "value"})
    ).dropna(subset=["value"])

    if averaged.empty:
        _log("Warning: transfer data empty after averaging; skipping transfer map.")
        return {}

    try:
        # The static mapping assets (geojson, zone mapping, colors) live under
        # the postprocessing package, not the project root. Using parents[1]
        # keeps the path anchored to ``epm/postprocessing`` regardless of where
        # the pipeline is invoked from.
        specs_root = Path(__file__).resolve().parents[1]
        dict_specs = read_plot_specs(folder=str(specs_root))
        zone_map, geojson_to_epm = maps.get_json_data(dict_specs=dict_specs)
        zone_map, centers = maps.create_zonemap(zone_map, map_geojson_to_epm=geojson_to_epm)
    except Exception as exc:  # pragma: no cover - defensive
        _log(
            f"Warning: could not build zone map for transfer plot using static assets in {specs_root}: {exc}"
        )
        return {}

    fig_path = output_dir / f"{case_id}_transfer_capacity_map.pdf"
    maps.make_interconnection_map(
        zone_map,
        averaged,
        centers,
        title=f"Existing transfer capacity ({first_year} avg seasons)",
        min_display_value=1,
        filename=str(fig_path),
        show_arrows=True,
        arrow_offset_ratio=0.3,
        arrow_size=18,
    )

    return {"transfer_capacity_map": str(fig_path)}


def _generate_price_plot(case_id: str, price_df: Optional[pd.DataFrame], output_dir: Path) -> Dict[str, str]:
    """Generate price plot with verbose logging."""

    if price_df is None or price_df.empty:
        _log("Warning: price data missing or empty; skipping price plot.")
        return {}

    fig_path = output_dir / f"{case_id}_price.pdf"
    _log(f"Generating price plot for {case_id} at {fig_path}")
    try:
        price_long = _reshape_price(price_df)
        if price_long.empty:
            _log("Warning: no numeric price columns found; skipping price plot.")
            return {}
        plots.make_line_plot(
            price_long,
            filename=str(fig_path),
            column_xaxis="year",
            y_column="price",
            preserve_x_spacing=True,
            series_column="fuel",
            legend=True,
            xlabel="Year",
            ylabel="Price (USD/MMBtu)",
            title=f"Fuel price trajectories - {case_id}",
            format_y=lambda y, _: f"{y:.0f}",
        )
        return {"price": str(fig_path)}
    except Exception as exc:  # pragma: no cover - defensive
        _log(f"Warning: skipping price plot for {case_id}: {exc}")
        return {}


def _generate_capex_trajectories_plot(
    case_id: str,
    capex_df: Optional[pd.DataFrame],
    output_dir: Path,
    years: List[int],
) -> Dict[str, str]:
    """Plot CAPEX trajectories averaged across zones, one line per technology."""

    if capex_df is None or capex_df.empty:
        _log("Warning: CAPEX trajectories data missing or empty; skipping CAPEX plot.")
        return {}

    df = capex_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Identify tech and fuel columns
    tech_col = _find_column(df, {"tech", "technology"})
    fuel_col = _find_column(df, {"fuel"})

    if tech_col is None:
        _log("Warning: CAPEX trajectories missing tech column; skipping CAPEX plot.")
        return {}

    # Create merged tech-fuel label
    if fuel_col:
        df["technology"] = df[tech_col].astype(str).str.strip() + "-" + df[fuel_col].astype(str).str.strip()
    else:
        df["technology"] = df[tech_col].astype(str).str.strip()

    # Find year columns
    year_cols = [c for c in df.columns if c.isdigit()]
    if not year_cols:
        _log("Warning: CAPEX trajectories missing year columns; skipping CAPEX plot.")
        return {}

    # Filter to planning years if provided
    if years:
        valid_year_cols = [c for c in year_cols if int(c) in years]
        if valid_year_cols:
            year_cols = valid_year_cols

    # Average across all zones for each technology
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    averaged = df.groupby("technology", as_index=False)[year_cols].mean()

    # Reshape to long format for plotting
    capex_long = averaged.melt(
        id_vars=["technology"],
        value_vars=year_cols,
        var_name="year",
        value_name="multiplier",
    )
    capex_long["year"] = capex_long["year"].astype(int)
    capex_long = capex_long.dropna(subset=["multiplier"])
    capex_long = capex_long[capex_long["multiplier"] > 0]  # Skip zero trajectories

    if capex_long.empty:
        _log("Warning: no valid CAPEX trajectory data after processing; skipping CAPEX plot.")
        return {}

    fig_path = output_dir / f"{case_id}_capex_trajectories.pdf"
    _log(f"Generating CAPEX trajectories plot for {case_id} at {fig_path}")

    try:
        plots.make_line_plot(
            capex_long,
            filename=str(fig_path),
            column_xaxis="year",
            y_column="multiplier",
            series_column="technology",
            legend=True,
            xlabel="Year",
            ylabel="CAPEX multiplier",
            title="CAPEX trajectories by technology (average across zones)",
            format_y=lambda y, _: f"{y:.2f}",
        )
        return {"capex_trajectories": str(fig_path)}
    except Exception as exc:  # pragma: no cover - defensive
        _log(f"Warning: skipping CAPEX trajectories plot for {case_id}: {exc}")
        return {}


def _generate_supply_plot(
    case_id: str,
    supply_df: Optional[pd.DataFrame],
    output_dir: Path,
    availability_custom: Optional[pd.DataFrame],
    availability_default: Optional[pd.DataFrame],
    years: List[int],
    demand_forecast_df: Optional[pd.DataFrame],
) -> Dict[str, str]:
    """Generate supply plots aligned with the supply-demand balance notebook."""

    if supply_df is None or supply_df.empty:
        _log("Warning: supply data missing or empty; skipping supply plot.")
        return {}

    if "Capacity" not in supply_df.columns or "zone" not in supply_df.columns:
        _log("Warning: supply data missing Capacity or zone columns; skipping supply plots.")
        return {}

    if not years:
        years = [0]
    years = sorted({int(y) for y in years})

    paths: Dict[str, str] = {}
    color_lookup, _, techfuel_mapping = _build_fuel_colors()
    supply_df = supply_df.copy()

    # Clean basic columns and drop inactive statuses.
    for column in ["zone", "tech", "fuel", "gen"]:
        if column in supply_df.columns:
            supply_df[column] = supply_df[column].astype(str).str.strip()
    if "Status" in supply_df.columns:
        supply_df = supply_df[supply_df["Status"].isin([1, 2])]

    # Apply capacity factors and create tech-fuel merged label.
    supply_df["capacity_factor"] = _map_capacity_factor(supply_df, availability_custom, availability_default)
    if "tech" in supply_df.columns and "fuel" in supply_df.columns:
        # Create techfuel key for mapping
        supply_df["tech_fuel"] = supply_df["tech"] + "-" + supply_df["fuel"]
        # Apply techfuel mapping to get Processing name
        supply_df["tech_fuel"] = supply_df["tech_fuel"].map(techfuel_mapping).fillna(supply_df["tech_fuel"])
    else:
        supply_df["tech_fuel"] = supply_df["fuel"]

    # Ensure every category has a color to avoid missing legends.
    missing_categories = set(supply_df["tech_fuel"].unique()).difference(color_lookup)
    for category in missing_categories:
        color_lookup[category] = "#999999"
    # Build tech_fuel ordering based on fuel ordering
    tech_fuel_ordering = sorted(supply_df["tech_fuel"].unique())

    # Build demand overlays from the forecast to match the notebook figures.
    peak_demand, energy_demand = _build_demand_overlays(demand_forecast_df, years)
    valid_zones = set()
    for overlay in (peak_demand, energy_demand):
        if not overlay.empty:
            valid_zones.update(overlay["zone"].unique().tolist())
    if valid_zones and "zone" in supply_df.columns:
        supply_df = supply_df[supply_df["zone"].isin(valid_zones)]

    if supply_df.empty:
        _log("Warning: no supply records after filtering; skipping supply plots.")
        return paths

    # Determine active years per generator.
    min_year, max_year = min(years), max(years)
    supply_df["start_year"] = pd.to_numeric(supply_df.get("StYr"), errors="coerce").fillna(min_year)
    supply_df["retire_year"] = pd.to_numeric(supply_df.get("RetrYr"), errors="coerce").fillna(max_year)

    def compute_active_years(row: pd.Series) -> List[int]:
        return [year for year in years if year >= row["start_year"] and year <= row["retire_year"]]

    supply_df["active_years"] = supply_df.apply(compute_active_years, axis=1)
    supply_df = supply_df[supply_df["active_years"].map(len) > 0]

    if supply_df.empty:
        _log("Warning: no active generators for the planning years; skipping supply plots.")
        return paths

    generation_yearly = supply_df.explode("active_years").rename(columns={"active_years": "year"})
    generation_yearly["year"] = generation_yearly["year"].astype(int)

    # Capacity mix
    cap_df = (
        generation_yearly.groupby(["zone", "tech_fuel", "year"], observed=False)["Capacity"]
        .sum()
        .reset_index()
        .rename(columns={"tech_fuel": "technology", "Capacity": "value"})
    )
    if valid_zones:
        cap_df = cap_df[cap_df["zone"].isin(valid_zones)]
    path_cap = output_dir / f"{case_id}_supply_capacity.pdf"
    _log(f"Generating capacity mix plot for {case_id} at {path_cap}")
    plots.make_stacked_barplot(
        cap_df,
        filename=str(path_cap),
        dict_colors=color_lookup,
        overlay_df=peak_demand if not peak_demand.empty else None,
        legend_label="Peak demand trajectory",
        column_stacked="technology",
        column_subplot="zone",
        column_xaxis="year",
        column_value="value",
        annotate=False,
        show_total=False,
        rotation=45,
        order_scenarios=years,
        order_stacked=tech_fuel_ordering,
        format_y=lambda value, _: f"{value:,.0f} MW",
        title="Capacity mix by technology (MW)",
    )
    paths["supply_capacity"] = str(path_cap)

    # Energy mix using capacity factor
    energy_df = (
        generation_yearly.assign(energy_gwh=lambda df: df["Capacity"] * df["capacity_factor"] * MW_TO_GWH)
        .groupby(["zone", "tech_fuel", "year"], observed=False)["energy_gwh"]
        .sum()
        .reset_index()
        .rename(columns={"tech_fuel": "technology", "energy_gwh": "value"})
    )
    if valid_zones:
        energy_df = energy_df[energy_df["zone"].isin(valid_zones)]
    path_energy = output_dir / f"{case_id}_supply_energy.pdf"
    _log(f"Generating potential energy mix plot for {case_id} at {path_energy}")
    plots.make_stacked_barplot(
        energy_df,
        filename=str(path_energy),
        dict_colors=color_lookup,
        overlay_df=energy_demand if not energy_demand.empty else None,
        legend_label="Energy demand trajectory",
        column_stacked="technology",
        column_subplot="zone",
        column_xaxis="year",
        column_value="value",
        annotate=False,
        show_total=False,
        rotation=45,
        order_scenarios=years,
        order_stacked=tech_fuel_ordering,
        format_y=lambda value, _: f"{value:,.0f} GWh",
        title="Potential energy mix by technology (GWh)",
    )
    paths["supply_energy"] = str(path_energy)

    # System-level stacks to mirror the notebook.
    system_capacity = (
        cap_df.groupby(["technology", "year"], observed=False)["value"].sum().reset_index().assign(zone="System")
    )
    system_peak = (
        peak_demand.groupby(["year"], observed=False)["value"].sum().reset_index().assign(zone="System")
        if not peak_demand.empty
        else pd.DataFrame()
    )
    path_system_cap = output_dir / f"{case_id}_supply_capacity_system.pdf"
    plots.make_stacked_barplot(
        system_capacity,
        filename=str(path_system_cap),
        dict_colors=color_lookup,
        overlay_df=system_peak if not system_peak.empty else None,
        legend_label="System peak demand trajectory",
        column_stacked="technology",
        column_subplot="zone",
        column_xaxis="year",
        column_value="value",
        annotate=False,
        order_scenarios=years,
        order_stacked=tech_fuel_ordering,
        format_y=lambda value, _: f"{value:,.0f} MW",
        title="System capacity mix by technology (MW)",
    )
    paths["supply_capacity_system"] = str(path_system_cap)

    system_energy = (
        energy_df.groupby(["technology", "year"], observed=False)["value"].sum().reset_index().assign(zone="System")
    )
    system_energy_demand = (
        energy_demand.groupby(["year"], observed=False)["value"].sum().reset_index().assign(zone="System")
        if not energy_demand.empty
        else pd.DataFrame()
    )
    path_system_energy = output_dir / f"{case_id}_supply_energy_system.pdf"
    plots.make_stacked_barplot(
        system_energy,
        filename=str(path_system_energy),
        dict_colors=color_lookup,
        overlay_df=system_energy_demand if not system_energy_demand.empty else None,
        legend_label="System energy demand trajectory",
        column_stacked="technology",
        column_subplot="zone",
        column_xaxis="year",
        column_value="value",
        annotate=False,
        order_scenarios=years,
        order_stacked=tech_fuel_ordering,
        format_y=lambda value, _: f"{value:,.0f} GWh",
        title="System potential energy mix by technology (GWh)",
    )
    paths["supply_energy_system"] = str(path_system_energy)

    return paths


def _generate_availability_plot(
    case_id: str,
    availability_df: Optional[pd.DataFrame],
    supply_df: Optional[pd.DataFrame],
    output_dir: Path,
    input_root: Path,
) -> Dict[str, str]:
    """Plot hydrology availability factors by generator on a single heatmap."""

    fallback_availability = input_root / "supply" / "pAvailabilityCustom.csv"
    df_availability = availability_df
    if (df_availability is None or df_availability.empty) and fallback_availability.exists():
        try:
            df_availability = pd.read_csv(fallback_availability)
            _log(f"Loaded availability from fallback path {fallback_availability}")
        except Exception as exc:  # pragma: no cover - defensive
            _log(f"Warning: could not read fallback availability file {fallback_availability}: {exc}")
            df_availability = None

    if df_availability is None or df_availability.empty:
        _log("Warning: availability data missing or empty; skipping availability plot.")
        return {}

    # Standardise column names and reshape to long format.
    df_availability = df_availability.copy()
    df_availability.columns = [str(col).strip() for col in df_availability.columns]
    gen_col = _find_column(df_availability, {"gen", "generator", "plant"})
    if gen_col is None:
        _log("Warning: availability data missing generator column; skipping availability plot.")
        return {}
    if gen_col != "gen":
        df_availability = df_availability.rename(columns={gen_col: "gen"})
    df_availability["gen"] = df_availability["gen"].astype(str).str.strip()

    value_cols = [c for c in df_availability.columns if c != "gen"]
    if not value_cols:
        _log("Warning: availability data has no seasonal columns; skipping availability plot.")
        return {}

    availability_long = df_availability.melt(
        id_vars="gen", value_vars=value_cols, var_name="period", value_name="availability"
    )
    availability_long["availability"] = pd.to_numeric(availability_long["availability"], errors="coerce")
    availability_long = availability_long.dropna(subset=["availability"])

    if availability_long.empty:
        _log("Warning: availability data empty after cleaning; skipping availability plot.")
        return {}

    # Bring in zone information from pGenDataInput.
    df_supply = supply_df
    fallback_supply = input_root / "supply" / "pGenDataInput.csv"
    if (df_supply is None or df_supply.empty) and fallback_supply.exists():
        try:
            df_supply = pd.read_csv(fallback_supply)
            _log(f"Loaded supply data from fallback path {fallback_supply}")
        except Exception as exc:  # pragma: no cover - defensive
            _log(f"Warning: could not read fallback supply file {fallback_supply}: {exc}")
            df_supply = None

    if df_supply is None or df_supply.empty:
        _log("Warning: supply data missing; cannot map availability to zones.")
        return {}

    df_supply = df_supply.copy()
    df_supply.columns = [str(col).strip() for col in df_supply.columns]
    supply_gen_col = _find_column(df_supply, {"gen", "generator", "plant"})
    supply_zone_col = _find_column(df_supply, {"zone", "region"})
    supply_fuel_col = _find_column(df_supply, {"fuel"})
    if supply_gen_col is None or supply_zone_col is None or supply_fuel_col is None:
        _log("Warning: supply data missing generator, zone, or fuel columns; skipping availability plot.")
        return {}
    if supply_gen_col != "gen":
        df_supply = df_supply.rename(columns={supply_gen_col: "gen"})
    if supply_zone_col != "zone":
        df_supply = df_supply.rename(columns={supply_zone_col: "zone"})
    if supply_fuel_col != "fuel":
        df_supply = df_supply.rename(columns={supply_fuel_col: "fuel"})
    df_supply["gen"] = df_supply["gen"].astype(str).str.strip()
    df_supply["zone"] = df_supply["zone"].astype(str).str.strip()
    df_supply["fuel"] = df_supply["fuel"].astype(str).str.strip()

    # Keep only relevant plants (active status with non-zero capacity when available).
    if "Status" in df_supply.columns:
        df_supply = df_supply[df_supply["Status"].isin([1, 2])]
    if "Capacity" in df_supply.columns:
        df_supply["Capacity"] = pd.to_numeric(df_supply["Capacity"], errors="coerce")
        df_supply = df_supply[df_supply["Capacity"] > 0]
    if df_supply.empty:
        _log("Warning: no relevant supply records after filtering; skipping availability plot.")
        return {}

    df_supply = df_supply[df_supply["fuel"].str.casefold() == "water".casefold()]
    if df_supply.empty:
        _log("Warning: no hydrology supply records after filtering fuel == 'Water'; skipping availability plot.")
        return {}

    merged = availability_long.merge(df_supply[["gen", "zone", "fuel"]].dropna(), on="gen", how="left")
    merged = merged.dropna(subset=["zone", "fuel"])

    if merged.empty:
        _log("Warning: no availability records matched to zones; skipping availability plot.")
        return {}

    period_order = _order_availability_periods(value_cols)
    fig_path = output_dir / f"{case_id}_availability_custom.pdf"
    _log(f"Generating availability plot for {case_id} at {fig_path}")
    merged["gen_label"] = merged["gen"] + " (" + merged["zone"] + ")"
    gen_order = (
        merged[["zone", "gen_label"]]
        .drop_duplicates()
        .sort_values(["zone", "gen_label"])["gen_label"]
        .tolist()
    )
    plots.heatmap_plot(
        merged,
        filename=str(fig_path),
        x_column="period",
        y_column="gen_label",
        value_column="availability",
        align_axes=True,
        x_order=period_order,
        y_order=gen_order,
        cmap="YlGnBu",
        title="Custom availability by plant and season",
        unit="%",
        vmin=0,
        vmax=1,
        fonttick=11,
    )

    return {"availability_custom": str(fig_path)}


def _build_demand_overlays(demand_forecast_df: Optional[pd.DataFrame], years: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return peak and energy demand overlays filtered to the planning years."""

    if demand_forecast_df is None or demand_forecast_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    normalized = _normalize_demand_forecast(demand_forecast_df)
    required = {"zone", "year", "demand", "type"}
    if not required.issubset(set(normalized.columns)):
        return pd.DataFrame(), pd.DataFrame()

    normalized = normalized.copy()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce").astype("Int64")
    normalized["demand"] = pd.to_numeric(normalized["demand"], errors="coerce")
    normalized = normalized.dropna(subset=["year", "demand"])
    normalized["year"] = normalized["year"].astype(int)
    if years:
        normalized = normalized[normalized["year"].isin(years)]
    normalized["type"] = normalized["type"].astype(str).str.lower().str.strip()

    peak = (
        normalized[normalized["type"] == "peak"]
        .groupby(["zone", "year"], observed=False)["demand"]
        .sum()
        .reset_index()
        .rename(columns={"demand": "value"})
    )
    energy = (
        normalized[normalized["type"] == "energy"]
        .groupby(["zone", "year"], observed=False)["demand"]
        .sum()
        .reset_index()
        .rename(columns={"demand": "value"})
    )

    return peak, energy


def _generate_demand_forecast_plot(case_id: str, demand_forecast_df: Optional[pd.DataFrame], output_dir: Path) -> Dict[str, str]:
    """Generate demand forecast plots with verbose logging."""

    if demand_forecast_df is None or demand_forecast_df.empty:
        _log("Warning: demand forecast missing or empty; skipping demand forecast plots.")
        return {}

    try:
        normalized = _normalize_demand_forecast(demand_forecast_df)
        return _generate_demand_forecast_plots(case_id, normalized, output_dir)
    except Exception as exc:  # pragma: no cover - defensive
        _log(f"Warning: skipping demand forecast plots for {case_id}: {exc}")
        return {}


# === Reporting (markdown + conversion) ===


def render_markdown_report(markdown_path: Path, context: Dict) -> None:
    """Render the Markdown report using a Jinja2 template when available."""

    template_path = Path(__file__).resolve().parent / "templates" / "report.md.j2"
    template_context = _prepare_template_context(context)

    if template_path.exists():
        try:
            content = _render_with_template(template_path, template_context)
            markdown_path.write_text(content, encoding="utf-8")
            return
        except Exception:
            # Fall back to the inline builder below if template rendering fails.
            pass

    markdown_path.write_text("# Data Inception Report\nTemplate rendering failed or template missing.", encoding="utf-8")


def convert_markdown_to_docx(markdown_path: Path, output_path: Path) -> None:
    """Convert Markdown to DOCX using pandoc when available."""

    _convert_with_pandoc(markdown_path, output_path, "docx")


def convert_markdown_to_pdf(markdown_path: Path, output_path: Path) -> None:
    """Convert Markdown to PDF using pandoc when available."""

    _convert_with_pandoc(markdown_path, output_path, "pdf")


def _prepare_template_context(context: Dict) -> Dict:
    """Convert raw objects into template-friendly strings."""

    cases = []
    for case in context.get("cases", []):
        cases.append(
            {
                "id": case["case_id"],
                "scenario": case.get("scenario", ""),
                "country": case.get("country", ""),
                "figures": case.get("figures", {}),
                "tables": case.get("tables", {}),
                "log": case.get("log", []),
            }
        )

    return {
        "data_folder": context.get("data_folder", ""),
        "generation_date": context.get("generation_date", ""),
        "cases": cases,
    }


def _render_with_template(template_path: Path, context: Dict) -> str:
    """Render a Jinja2 template with the provided context."""

    import jinja2

    _log(f"Rendering report from template {template_path}")
    env = jinja2.Environment(autoescape=False, loader=jinja2.FileSystemLoader(str(template_path.parent)))
    template = env.get_template(template_path.name)
    return template.render(**context)


def _generate_demand_forecast_plots(case_id: str, demand_forecast: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    """Generate demand forecast plots inspired by the load notebook."""

    required_cols = {"year", "scenario", "demand", "type", "zone"}
    if not required_cols.issubset(set(demand_forecast.columns)):
        missing = required_cols.difference(set(demand_forecast.columns))
        _log(f"Warning: demand_forecast missing required columns {missing}; skipping forecast plots.")
        return {}

    figure_paths: Dict[str, str] = {}
    year_order = (
        pd.to_numeric(demand_forecast["year"], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    scenario_order = list(dict.fromkeys(demand_forecast["scenario"].dropna()))

    palette = plots.plt.get_cmap("tab20", max(len(scenario_order), 1))
    scenario_colors = {scenario: palette(idx) for idx, scenario in enumerate(scenario_order)}
    tick_formatter = lambda y, _: f"{y:,.0f}"

    type_configs = {
        "Energy": {
            "ylabel": "Demand (GWh)",
            "total_title": "Total demand generation",
            "total_filename": output_dir / f"{case_id}_energy_demand_total.pdf",
            "zone_title": "Demand generation by zone",
            "zone_filename": output_dir / f"{case_id}_energy_demand_by_zone.pdf",
        },
        "Peak": {
            "ylabel": "Demand (MW)",
            "total_title": "Total demand peak",
            "total_filename": output_dir / f"{case_id}_peak_demand_total.pdf",
            "zone_title": "Demand peak by zone",
            "zone_filename": output_dir / f"{case_id}_peak_demand_by_zone.pdf",
        },
    }

    for demand_type, cfg in type_configs.items():
        subset = demand_forecast[demand_forecast["type"] == demand_type].copy()
        if subset.empty:
            _log(f"Warning: no records found for {demand_type} demand; skipping.")
            continue
        subset["year"] = subset["year"].astype(int)
        subset["demand"] = pd.to_numeric(subset["demand"], errors="coerce")
        subset = subset.dropna(subset=["demand"])

        total = subset.groupby(["scenario", "year"], as_index=False)["demand"].sum()
        total["year"] = total["year"].astype(int)
        plots.make_line_plot(
            df=total,
            filename=str(cfg["total_filename"]),
            column_xaxis="year",
            y_column="demand",
            series_column="scenario",
            dict_colors=scenario_colors,
            format_y=lambda y, _: f"{y:.0f}",
            xlabel="Year",
            ylabel=cfg["ylabel"],
            title=cfg["total_title"],
            ymin=0,
        )
        figure_paths[f"{demand_type.lower()}_total"] = str(cfg["total_filename"])

        zone_order = sorted(subset["zone"].dropna().unique())
        if zone_order:
            plots.make_line_plot(
                df=subset,
                filename=str(cfg["zone_filename"]),
                column_xaxis="year",
                y_column="demand",
                series_column="scenario",
                column_subplot="zone",
                select_subplots=zone_order,
                dict_colors=scenario_colors,
                order_index=year_order,
                format_y=lambda y, _: f"{y:.0f}",
                xlabel="Year",
                ylabel=cfg["ylabel"],
                title=cfg["zone_title"] if len(zone_order) == 1 else None,
                ymin=0,
                rotation=45,
            )
            figure_paths[f"{demand_type.lower()}_zone"] = str(cfg["zone_filename"])

    return figure_paths


def _convert_with_pandoc(markdown_path: Path, output_path: Path, fmt: str) -> None:
    """Use pandoc to convert the markdown file, failing gracefully if absent."""

    fmt_lower = fmt.lower()
    if fmt_lower == "pdf":
        # Mirror the pre-analysis pipeline invocation that works across envs.
        command = [
            "pandoc",
            "--from=markdown",
            "--to=pdf",
            "-V",
            "geometry:margin=1.5cm",
            "-V",
            "float-placement=H",
        ]
        if DISABLE_FLOAT_TEX.exists():
            command += ["-H", str(DISABLE_FLOAT_TEX)]
        else:
            _log(f"Warning: {DISABLE_FLOAT_TEX} not found; skipping float override header.")
        command += ["--output", str(output_path), str(markdown_path)]
    else:
        from_arg = "--from=gfm" if fmt_lower == "docx" else "--from=markdown"
        command = ["pandoc", from_arg, f"--to={fmt_lower}", "--output", str(output_path), str(markdown_path)]

    try:
        _log(f"Converting {markdown_path} to {fmt.upper()} at {output_path}")
        # Use the project root as working dir so relative assets resolve consistently.
        subprocess.run(command, check=True, capture_output=True, cwd=_PROJECT_ROOT)
    except FileNotFoundError:
        note = f"pandoc not available; skipping {fmt.upper()} generation."
        _log(note)
        output_path.write_text(note, encoding="utf-8")
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        message = f"pandoc failed: {exc.stderr.decode(errors='ignore')}"
        _log(message)
        output_path.write_text(message, encoding="utf-8")


def _derive_case_from_parameter_config(config_df: pd.DataFrame) -> pd.DataFrame:
    """Map the parameter-style config.csv to a single-case driver table."""

    lookup = {row["paramNames"]: row.get("file", "") for _, row in config_df.iterrows() if row.get("paramNames")}

    supply_path = lookup.get("pGenDataInput") or lookup.get("pGenDataInputDefault")
    price_path = lookup.get("pFuelPrice")

    case = {
        "case_id": "baseline",
        "scenario": "Baseline scenario",
        "country": "",
        "supply_path": supply_path or "",
        "price_path": price_path or "",
        "demand_forecast_path": lookup.get("pDemandForecast", ""),
        "demand_profile_path": lookup.get("pDemandProfile", ""),
        "zcmap_path": lookup.get("zcmap", ""),
        "pvreprofile_path": lookup.get("pVREProfile", ""),
        "psettings_path": lookup.get("pSettings", ""),
        "ptransferlimit_path": lookup.get("pTransferLimit", ""),
        "pexttransferlimit_path": lookup.get("pExtTransferLimit", ""),
        "pavailability_path": lookup.get("pAvailability", ""),
        "pavailabilitydefault_path": lookup.get("pAvailabilityDefault", ""),
        "pcapextrajectoriesdefault_path": lookup.get("pCapexTrajectoriesDefault", ""),
        "pgendatainputdefault_path": lookup.get("pGenDataInputDefault", ""),
    }

    columns = REQUIRED_CONFIG_COLUMNS + ["pavailability_path", "pavailabilitydefault_path", "pcapextrajectoriesdefault_path", "pgendatainputdefault_path"]
    return pd.DataFrame([case], columns=columns)


def _normalize_demand_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize demand forecast column names for plotting."""

    renamed = {c: c.lower() for c in df.columns}
    df = df.rename(columns=renamed)

    # Rename known aliases.
    alt_names = {
        "zone_name": "zone",
        "region": "zone",
        "scenario_name": "scenario",
        "value": "demand",
        "demand_mw": "demand",
        "year_code": "year",
        "type_name": "type",
    }
    for old, new in alt_names.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # If the forecast is wide (year columns), reshape to long.
    if "year" not in df.columns:
        year_cols = [c for c in df.columns if str(c).isdigit()]
        if year_cols and "zone" in df.columns:
            df = df.melt(id_vars=[c for c in df.columns if c not in year_cols], value_vars=year_cols, var_name="year", value_name="demand")

    if "scenario" not in df.columns:
        df["scenario"] = "Baseline"
    if "type" not in df.columns:
        df["type"] = "Energy"

    return df


def _reshape_price(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide price table (years as columns) to long format with averages by fuel."""

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    id_cols = [c for c in ["zone", "fuel"] if c in df.columns]
    year_cols = [c for c in df.columns if c.isdigit()]
    if not year_cols or "fuel" not in df.columns:
        return pd.DataFrame()

    long_df = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="year", value_name="price")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype(int)
    long_df["price"] = pd.to_numeric(long_df["price"], errors="coerce")
    long_df = long_df.dropna(subset=["price"])
    grouped = long_df.groupby(["fuel", "year"], as_index=False)["price"].mean()
    return grouped


def _build_fuel_colors() -> Tuple[Dict[str, str], List[str], Dict[str, str]]:
    """Build fuel color lookup, ordering, and mapping from resources tables."""

    # Resources tables live in epm/resources/postprocess
    resources_dir = Path(__file__).resolve().parent.parent.parent / "resources" / "postprocess"
    colors_path = resources_dir / "colors.csv"
    techfuel_path = resources_dir / "pTechFuelProcessing.csv"

    def load_table(path: Path, required: List[str], comment: str = "#") -> pd.DataFrame:
        if not path.exists():
            _log(f"Resources table not found: {path.name} at {path.resolve()} (using empty frame).")
            return pd.DataFrame(columns=required)

        try:
            df = pd.read_csv(path, comment=comment).dropna(how="all")
            _log(f"Loaded resources table {path.name} from {path.resolve()} with {len(df)} rows.")
            return df
        except Exception as exc:  # keep behavior but surface context
            _log(f"Failed to load resources table {path.name} at {path.resolve()}: {exc}")
            raise

    colors = load_table(colors_path, ["Processing", "Color"])
    techfuel = load_table(techfuel_path, ["tech", "fuel", "Processing", "Color"])

    # Build base color lookup from colors.csv
    color_lookup = {}
    if {"Processing", "Color"}.issubset(colors.columns):
        subset = colors[["Processing", "Color"]].dropna()
        subset["Processing"] = subset["Processing"].astype(str).str.strip()
        subset["Color"] = subset["Color"].astype(str).str.strip()
        subset = subset[(subset["Processing"] != "") & (subset["Color"] != "")]
        color_lookup = subset.set_index("Processing")["Color"].to_dict()

    # Build techfuel mapping and colors from pTechFuelProcessing.csv
    techfuel_mapping = {}
    techfuel_order = []
    if {"tech", "fuel", "Processing", "Color"}.issubset(techfuel.columns):
        techfuel["tech"] = techfuel["tech"].astype(str).str.strip()
        techfuel["fuel"] = techfuel["fuel"].astype(str).str.strip()
        techfuel["Processing"] = techfuel["Processing"].astype(str).str.strip()
        techfuel["Color"] = techfuel["Color"].astype(str).str.strip()
        techfuel["techfuel_key"] = techfuel["tech"] + "-" + techfuel["fuel"]
        techfuel_mapping = techfuel.set_index("techfuel_key")["Processing"].to_dict()
        techfuel_order = list(dict.fromkeys(techfuel["Processing"]))
        # Merge techfuel colors into color_lookup
        techfuel_colors = techfuel.set_index("Processing")["Color"].to_dict()
        color_lookup.update(techfuel_colors)

    return color_lookup, techfuel_order, techfuel_mapping


def _normalize_column_label(label: str) -> str:
    """Return a lowercased, trimmed column label without BOM artifacts."""

    return str(label).replace("\ufeff", "").strip().lower()


def _find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Locate the first column matching any candidate label (case-insensitive)."""

    normalized = {_normalize_column_label(col): col for col in frame.columns}
    for candidate in candidates:
        key = _normalize_column_label(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _order_availability_periods(columns: Iterable[str]) -> List[str]:
    """Order availability columns by numeric suffix when possible (e.g., m1..m12)."""

    unique_cols = list(dict.fromkeys([str(col) for col in columns]))

    def _sort_key(label: str) -> Tuple[int, Union[int, str]]:
        normalized = _normalize_column_label(label)
        digits = "".join(ch for ch in normalized if ch.isdigit())
        if digits.isdigit():
            return (0, int(digits))
        return (1, normalized)

    return sorted(unique_cols, key=_sort_key)


def _tidy_availability(frame: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Return averaged capacity factors for the provided grouping columns."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=group_cols + ["capacity_factor"])
    # If expected grouping columns are missing, return empty to fall back gracefully.
    if any(col not in frame.columns for col in group_cols):
        return pd.DataFrame(columns=group_cols + ["capacity_factor"])
    data = frame.copy()
    for column in group_cols:
        if column in data.columns:
            data[column] = data[column].astype(str).str.strip()
    value_cols = [c for c in data.columns if c not in group_cols]
    for column in value_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not value_cols:
        data["capacity_factor"] = np.nan
    else:
        data["capacity_factor"] = data[value_cols].mean(axis=1, skipna=True)
    return data[group_cols + ["capacity_factor"]].dropna(subset=["capacity_factor"])


def _map_capacity_factor(
    supply_df: pd.DataFrame, availability_custom: Optional[pd.DataFrame], availability_default: Optional[pd.DataFrame]
) -> pd.Series:
    """Map capacity factors using pre-loaded availability tables with fallbacks."""

    availability_custom = _tidy_availability(availability_custom, ["gen"])
    availability_default = _tidy_availability(availability_default, ["zone", "tech", "fuel"])

    custom_cf = availability_custom.set_index("gen")["capacity_factor"].to_dict()
    default_cf = availability_default.set_index(["zone", "tech", "fuel"])["capacity_factor"].to_dict()
    default_cf_tf = availability_default.groupby(["tech", "fuel"], observed=False)["capacity_factor"].mean().to_dict()
    default_cf_f = availability_default.groupby(["fuel"], observed=False)["capacity_factor"].mean().to_dict()
    global_cf = availability_default["capacity_factor"].mean() if not availability_default.empty else np.nan

    def capacity_factor_lookup(row: pd.Series) -> float:
        generator = row.get("gen", "")
        if generator in custom_cf:
            return custom_cf[generator]

        zone, tech, fuel = row.get("zone"), row.get("tech"), row.get("fuel")
        key_zone = (zone, tech, fuel)
        if key_zone in default_cf:
            return default_cf[key_zone]

        key_tf = (tech, fuel)
        if key_tf in default_cf_tf:
            return default_cf_tf[key_tf]

        if fuel in default_cf_f:
            return default_cf_f[fuel]

        return float(global_cf) if pd.notna(global_cf) else 0.5

    return supply_df.apply(capacity_factor_lookup, axis=1)


def _generate_demand_profile_plots(case_id: str, profile_df: Optional[pd.DataFrame], output_dir: Path) -> Dict[str, str]:
    """Generate demand profile plots: hourly for year, average by month, average by day."""

    if profile_df is None or profile_df.empty:
        _log("Warning: demand profile missing or empty; skipping demand profile plots.")
        return {}

    hour_cols = [c for c in profile_df.columns if str(c).lower().startswith("t")]
    if not hour_cols:
        _log("Warning: demand profile missing hourly columns; skipping demand profile plots.")
        return {}

    figure_paths: Dict[str, str] = {}

    # Average per hour (1-24) for each zone.
    hourly_records = []
    for zone, subset in profile_df.groupby("zone"):
        avg = subset[hour_cols].mean()
        for col, val in avg.items():
            hour = int(col.strip("t").strip())
            hourly_records.append({"zone": zone, "hour": hour, "value": val})

    hourly_df = pd.DataFrame(hourly_records)
    if not hourly_df.empty:
        hourly_df = hourly_df.sort_values(["zone", "hour"])
        path_hourly = output_dir / f"{case_id}_demand_profile_hourly_avg.pdf"
        plots.make_line_plot(
            hourly_df,
            filename=str(path_hourly),
            column_xaxis="hour",
            y_column="value",
            column_subplot="zone",
            legend=False,
            xlabel=None,
            ylabel="Normalized load (p.u.)",
            title="Average load by hour (zones)",
            format_y=lambda y, _: f"{y:.0f}",
        )
        figure_paths["demand_profile_hourly_avg"] = str(path_hourly)

    # Average per season for each zone.
    season_order = ["Q1", "Q2", "Q3", "Q4"]
    season_records = []
    for (zone, season), subset in profile_df.groupby(["zone", "season"], dropna=False):
        if season is None:
            continue
        val = subset[hour_cols].mean().mean()
        season_records.append({"zone": zone, "season": season, "value": val})

    season_df = pd.DataFrame(season_records)
    if not season_df.empty:
        season_df["season"] = pd.Categorical(season_df["season"], categories=season_order, ordered=True)
        season_df = season_df.sort_values(["zone", "season"])
        path_season = output_dir / f"{case_id}_demand_profile_season_avg.pdf"
        plots.make_line_plot(
            season_df,
            filename=str(path_season),
            column_xaxis="season",
            y_column="value",
            column_subplot="zone",
            legend=False,
            xlabel=None,
            ylabel="Normalized load (p.u.)",
            title="Average load by season (zones)",
            format_y=lambda y, _: f"{y:.0f}",
        )

    return figure_paths


def _log(message: str) -> None:
    """Lightweight logger for the pipeline."""

    if VERBOSE:
        print(f"[data-inception] {message}")


if __name__ == "__main__":
    # Example CLI usage for manual runs
    import argparse

    parser = argparse.ArgumentParser(description="Run data inception report pipeline.")
    parser.add_argument(
        "data_folder",
        nargs="?",
        default="data_westbalkans",
        help="Folder name under epm/input containing config.csv (default: data_test)",
    )
    args = parser.parse_args()
    outputs = run_data_inception_report(args.data_folder)
    print("Report generated:", outputs)
