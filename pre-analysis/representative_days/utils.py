"""
Utility functions for handling time series data in the representative days pipeline.

This module provides functions to validate, clean, and process time series data
for the Electricity Planning Model (EPM), focusing on month/day/hour data.
"""

import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "_log_nan_time_index",
    "_validate_time_columns",
    "month_to_season",
    "_warn_incomplete_year",
    "_drop_feb29",
    "_ensure_normalized_series",
    "load_and_clean_timeseries",
]

# Number of days in each month (non-leap year)
NB_DAYS = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], index=range(1, 13))


def _log_nan_time_index(df: pd.DataFrame, time_cols: Iterable[str], value_cols: Iterable[str], label: str) -> None:
    """Log the time indices of NaN values to help locate data issues."""
    # Ensure value_cols is a list
    value_cols = [value_cols] if isinstance(value_cols, str) else list(value_cols)
    # Create mask for rows with any NaN in value columns
    mask = df[value_cols].isna().any(axis=1)
    if not mask.any():
        return

    # Columns to show in preview: time columns that exist plus value columns
    cols_to_show = [col for col in time_cols if col in df.columns]
    preview_cols: List[str] = cols_to_show + value_cols
    # Get first 15 rows with NaN values
    preview = df.loc[mask, preview_cols].head(15)
    logger.warning(
        "NaN values found in %s (%d rows). First rows:\n%s",
        label,
        mask.sum(),
        preview.to_string(index=False),
    )


def _validate_time_columns(df: pd.DataFrame, time_cols: Iterable[str], name: str = "", verbose: bool = True) -> pd.DataFrame:
    """Ensure time columns are numeric integers and in plausible ranges. Returns a copy with ints."""
    df_checked = df.copy()
    label = f"[{name}] " if name else ""
    for col in time_cols:
        if col not in df_checked.columns:
            raise ValueError(f"{label}Missing required column {col}")
        # Convert to numeric, coercing errors to NaN
        numeric = pd.to_numeric(df_checked[col], errors="coerce")
        # Check for non-numeric values
        non_numeric = df_checked[numeric.isna()]
        if not non_numeric.empty:
            raise ValueError(f"{label}Non-numeric values in {col} (showing first 5):\n{non_numeric.head()}")
        # Check for non-integer values
        non_int = df_checked[(numeric % 1 != 0)]
        if not non_int.empty:
            raise ValueError(f"{label}Non-integer values in {col} (showing first 5):\n{non_int.head()}")
        # Convert to int
        df_checked[col] = numeric.astype(int)

    # Basic plausibility checks for specific columns
    if "month" in df_checked.columns and ((df_checked["month"] < 1) | (df_checked["month"] > 12)).any():
        bad = df_checked[(df_checked["month"] < 1) | (df_checked["month"] > 12)]
        raise ValueError(f"{label}Month values out of range 1-12 (showing first 5):\n{bad.head()}")
    if "day" in df_checked.columns and ((df_checked["day"] < 1) | (df_checked["day"] > 31)).any():
        bad = df_checked[(df_checked["day"] < 1) | (df_checked["day"] > 31)]
        raise ValueError(f"{label}Day values out of range 1-31 (showing first 5):\n{bad.head()}")
    if "hour" in df_checked.columns and ((df_checked["hour"] < 0) | (df_checked["hour"] > 23)).any():
        bad = df_checked[(df_checked["hour"] < 0) | (df_checked["hour"] > 23)]
        raise ValueError(f"{label}Hour values out of range 0-23 (showing first 5):\n{bad.head()}")
    return df_checked


def month_to_season(data: pd.DataFrame, seasons_map: dict, other_columns=None) -> pd.DataFrame:
    """Convert month numbers to season identifiers and renumber days inside each season."""
    other_columns = other_columns or []
    df = data.copy()
    # Allow using "season" as a proxy for month
    df = df.rename(columns={"season": "month"})

    required_cols = set(["month", "day", "hour"]).union(set(other_columns))
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Map months to seasons using the provided seasons_map
    df["season"] = df["month"].map(seasons_map)
    if df["season"].isna().any():
        missing_months = df.loc[df["season"].isna(), "month"].unique()
        raise ValueError(f"Months {missing_months} do not exist in seasons_map")

    # Remove Feb 29th if present (leap year handling)
    df = df[~((df["month"] == 2) & (df["day"] == 29))]

    # Renumber days sequentially within each season (1,2,3...) based on 24-hour blocks
    df["season_day"] = df.groupby(other_columns + ["season"]).cumcount() // 24 + 1
    df = df.drop(columns=["day"]).rename(columns={"season_day": "day"})
    # Set index and reset to reorder columns
    df = df.set_index(other_columns + ["season", "day", "hour"]).reset_index().drop(columns=["month"])
    df = df.sort_values(by=other_columns + ["season", "day", "hour"])
    return df


def _warn_incomplete_year(df: pd.DataFrame, name: str = "", verbose: bool = True, raise_on_missing: bool = True) -> None:
    """Warn (or raise) if a zone is missing any month/day/hour combinations over the year."""
    if not verbose and not raise_on_missing:
        return
    required = {"zone", "month", "day", "hour"}
    if not required.issubset(df.columns):
        return

    label = f"[{name}] " if name else ""
    expected_hours = set(range(24))
    # Group by zone to check each zone separately
    for zone, g_zone in df.groupby("zone"):
        missing = []
        for month in range(1, 13):
            days_in_month = int(NB_DAYS.get(month, 0))
            g_month = g_zone[g_zone["month"] == month]
            if g_month.empty:
                # Missing entire month
                for day in range(1, days_in_month + 1):
                    missing.extend((month, day, h) for h in expected_hours)
                continue
            for day in range(1, days_in_month + 1):
                g_day = g_month[g_month["day"] == day]
                if g_day.empty:
                    # Missing entire day
                    missing.extend((month, day, h) for h in expected_hours)
                    continue
                # Check for missing hours
                hours = set(g_day["hour"])
                if hours != expected_hours:
                    missing.extend((month, day, h) for h in sorted(expected_hours - hours))

        if missing:
            # Format a sample of missing entries
            sample = missing[:6]
            formatted = "; ".join(f"m{m} d{d} h{h}" for m, d, h in sample)
            extra = "" if len(missing) <= len(sample) else f" ... (+{len(missing) - len(sample)} more)"
            message = f"{label}Incomplete year for zone {zone}: missing {len(missing)} entries: {formatted}{extra}"
            if raise_on_missing:
                raise ValueError(message + " Please clean the input data before running the pipeline.")
            if verbose:
                print(message)


def _drop_feb29(df: pd.DataFrame, name: str = "", verbose: bool = True) -> pd.DataFrame:
    """Remove Feb 29 entries if present."""
    if {"month", "day"}.issubset(df.columns):
        # Create mask for Feb 29
        mask = (df["month"] == 2) & (df["day"] == 29)
        if mask.any():
            count = int(mask.sum())
            zones = df.loc[mask, "zone"].unique() if "zone" in df.columns else None
            if verbose:
                zone_msg = f" for zones {', '.join(map(str, zones))}" if zones is not None else ""
                label = f"[{name}] " if name else ""
                print(f"{label}Dropping {count} Feb 29 entries{zone_msg}.")
            # Remove the rows
            df = df.loc[~mask].copy()
    return df


def _ensure_normalized_series(df: pd.DataFrame, value_col: str, name: str = "", verbose: bool = True) -> pd.DataFrame:
    """Normalize a value column to [0,1] if it is not already; warn when scaling."""
    df_checked = df.copy()
    label = f"[{name}] " if name else ""
    max_val = df_checked[value_col].max(skipna=True)
    min_val = df_checked[value_col].min(skipna=True)
    if pd.isna(max_val) or max_val == 0:
        if verbose:
            print(f"{label}Cannot normalize '{value_col}' because max is 0 or NaN.")
        return df_checked
    # Check if already in [0,1]
    if max_val <= 1 and min_val >= 0:
        return df_checked
    if verbose:
        print(f"{label}Normalizing '{value_col}' to [0,1] (max={max_val:.4g}).")
    # Normalize by dividing by max
    df_checked[value_col] = df_checked[value_col] / max_val
    return df_checked


def load_and_clean_timeseries(
    input_path: Union[str, Path],
    zones_to_exclude=None,
    value_column: str = "value",
    rename_value_to: Union[int, str, None] = None,
    normalize: bool = True,
    drop_feb_29: bool = True,
    check_complete_year: bool = False,
    verbose: bool = True,
    require_zone: bool = True,
) -> Tuple[pd.DataFrame, Union[str, int]]:
    """Load and standardize a raw month/day/hour time-series CSV.

    Returns a cleaned DataFrame and the name of the value column after any renaming.
    """
    zones_to_exclude = zones_to_exclude or []
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load the CSV
    df = pd.read_csv(input_path)

    # Allow "season" as alias for "month"
    if "month" not in df.columns and "season" in df.columns:
        df = df.rename(columns={"season": "month"})

    # Validate time columns
    df = _validate_time_columns(df, time_cols=("month", "day", "hour"), name=f"{input_path}", verbose=False)

    required_columns = {"month", "day", "hour"}
    if require_zone:
        required_columns.add("zone")

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    # Determine the value column
    candidate_value_cols = [c for c in df.columns if c not in required_columns]
    if value_column in df.columns:
        value_col = value_column
    elif len(candidate_value_cols) == 1:
        value_col = candidate_value_cols[0]
    else:
        # Heuristic: if columns look like years, pick the latest one; otherwise use the first non-index column.
        year_like = []
        for col in candidate_value_cols:
            try:
                year_like.append(int(col))
            except Exception:
                year_like.append(None)
        if any(y is not None for y in year_like):
            # Use the candidate with the max numeric year
            value_col = candidate_value_cols[int(year_like.index(max(y for y in year_like if y is not None)))]
        else:
            value_col = candidate_value_cols[0]

    # Rename value column if requested
    if rename_value_to is not None:
        df = df.rename(columns={value_col: rename_value_to})
        value_col = rename_value_to

    # Exclude specified zones
    if "zone" in df.columns:
        df = df[~df["zone"].isin(zones_to_exclude)]

    # Adjust hour indexing if needed (1-based to 0-based)
    if df["hour"].min() == 1 and df["hour"].max() <= 24:
        df["hour"] = df["hour"] - 1

    # Drop Feb 29 if requested
    if drop_feb_29:
        df = _drop_feb29(df, name=str(input_path), verbose=verbose)

    # Normalize if requested
    if normalize:
        df = _ensure_normalized_series(df, value_col=value_col, name=str(input_path), verbose=verbose)

    # Check for complete year if requested
    if check_complete_year:
        _warn_incomplete_year(df, name=str(input_path), verbose=verbose)

    # Log any NaN values
    _log_nan_time_index(
        df,
        time_cols=("month", "day", "hour"),
        value_cols=(value_col,),
        label=f"{input_path.name} cleaned",
    )

    return df, value_col
