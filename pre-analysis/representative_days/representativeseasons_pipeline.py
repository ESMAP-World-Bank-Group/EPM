"""Pipeline utilities to derive contiguous seasons from monthly time series."""

import calendar
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Prefer installed package import; fallback to local when run directly
try:
    from representative_days.utils import _ensure_normalized_series, load_and_clean_timeseries  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    from utils import _ensure_normalized_series, load_and_clean_timeseries  # type: ignore


def derive_contiguous_seasons(
    month_features: np.ndarray,
    K: int = 2,
    random_state: int = 0,
    verbose: bool = True,
) -> Dict[int, int]:
    """Cluster months then smooth to produce a simple month→season mapping."""
    if month_features.shape[0] != 12:
        raise ValueError(f"Expected 12 months, got {month_features.shape[0]}")

    if verbose:
        print(f"[repr-seasons] Clustering months into {K} seasons (features shape: {month_features.shape})")

    labels = KMeans(n_clusters=K, n_init="auto", random_state=random_state).fit_predict(month_features)

    # Step 2: enforce continuity by smoothing isolated months
    labels = labels.copy()
    for i in range(12):
        prev_lab = labels[(i - 1) % 12]
        next_lab = labels[(i + 1) % 12]
        if labels[i] != prev_lab and labels[i] != next_lab:
            labels[i] = prev_lab
            if verbose:
                print(f"[repr-seasons] Month {i + 1} looked lonely; aligning it with month {(i % 12) + 1}")

    season_ids: Dict[int, int] = {}
    seasons_map: Dict[int, int] = {}
    next_id = 1
    for month, lab in enumerate(labels, start=1):
        if lab not in season_ids:
            season_ids[lab] = next_id
            next_id += 1
        seasons_map[month] = season_ids[lab]

    if verbose:
        print(f"[repr-seasons] Final month→season map: {seasons_map}")
        _log_season_ranges(seasons_map)

    return seasons_map


def _log_season_ranges(seasons_map: Dict[int, int]) -> None:
    """Print human-friendly month ranges for each season."""
    seasons = sorted(set(seasons_map.values()))
    print("[repr-seasons] Season ranges:")
    for season in seasons:
        months = [m for m, s in seasons_map.items() if s == season]
        ranges = _format_month_ranges(months)
        print(f"  Season {season}: {ranges}")


def _format_month_ranges(months: List[int]) -> str:
    """Convert a list of month numbers into compact ranges like 'Jan–Mar, Oct'."""
    if not months:
        return ""
    months_sorted = sorted(months)
    # Handle wrap-around by checking if Jan belongs with Dec
    if months_sorted == list(range(1, 13)):
        return "Jan–Dec"
    ranges = []
    start = prev = months_sorted[0]
    for m in months_sorted[1:]:
        if m == prev + 1:
            prev = m
            continue
        ranges.append((start, prev))
        start = prev = m
    ranges.append((start, prev))
    def _fmt_pair(lo: int, hi: int) -> str:
        lo_label = calendar.month_abbr[lo]
        hi_label = calendar.month_abbr[hi]
        return lo_label if lo == hi else f"{lo_label}–{hi_label}"
    return ", ".join(_fmt_pair(lo, hi) for lo, hi in ranges)


def _try_load_monthly_table(
    input_path: str,
    value_column: str = "value",
    verbose: bool = True,
) -> Optional[pd.Series]:
    """Load pre-aggregated monthly tables (month + value[, zone]) if present.

    Returns a 12-length Series indexed by month or None when the file does not look like
    a monthly table (i.e., it already has day/hour columns).
    """
    path = Path(input_path)
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    month_col = cols_lower.get("month") or cols_lower.get("season")
    if month_col is None or {"day", "hour"}.intersection(cols_lower):
        return None

    candidate_value_cols = [c for c in df.columns if c not in {month_col, "zone"}]
    if value_column in df.columns:
        value_col = value_column
    elif len(candidate_value_cols) == 1:
        value_col = candidate_value_cols[0]
    elif candidate_value_cols:
        # Prefer the first numeric-looking candidate
        numeric_candidates = [
            c for c in candidate_value_cols if pd.to_numeric(df[c], errors="coerce").notna().any()
        ]
        value_col = numeric_candidates[0] if numeric_candidates else candidate_value_cols[0]
    else:
        return None

    df = df.rename(columns={month_col: "month", value_col: value_column})
    value_col = value_column
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["month", value_col])
    df = df[(df["month"] >= 1) & (df["month"] <= 12)]
    if df.empty:
        return None

    df = _ensure_normalized_series(df, value_col=value_col, name=str(path), verbose=verbose)
    monthly = df.groupby("month")[value_col].mean().reindex(range(1, 13)).fillna(0)

    if verbose:
        print(
            "[repr-seasons] Detected monthly table in "
            f"{path.name}; using column '{value_col}' (range {monthly.min():.3f}→{monthly.max():.3f})"
        )
    return monthly


def _load_monthly_features(
    input_files: Dict[str, str],
    value_column: str = "value",
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Compute per-month feature vectors from raw tech/load inputs."""
    features = []
    feature_labels: List[str] = []
    if verbose:
        print(f"[repr-seasons] Loading monthly features from {len(input_files)} inputs (value column: {value_column})")
    for tech, path in sorted(input_files.items()):
        monthly = _try_load_monthly_table(path, value_column=value_column, verbose=verbose)
        if monthly is not None:
            features.append(monthly.values)
            feature_labels.append(tech)
            continue
        df, val_col = load_and_clean_timeseries(
            input_path=path,
            value_column=value_column,
            rename_value_to=value_column,
            normalize=True,
            drop_feb_29=True,
            check_complete_year=False,
            verbose=verbose,
            require_zone=False,
        )
        monthly = df.groupby("month")[val_col].mean().reindex(range(1, 13)).fillna(0)
        features.append(monthly.values)
        feature_labels.append(tech)
        if verbose:
            print(f"[repr-seasons] {tech}: monthly mean range {monthly.min():.3f}→{monthly.max():.3f}")

    # Stack to shape (12, n_features)
    stacked = np.stack(features, axis=1)
    if verbose:
        print(f"[repr-seasons] Built feature matrix with shape {stacked.shape}")
    return stacked, feature_labels


def _plot_feature_heatmap(
    month_features: np.ndarray,
    feature_labels: List[str],
    seasons_map: Optional[Dict[int, int]],
    path: Path,
    show: bool = False,
) -> None:
    """Save (and optionally display) a heatmap of monthly normalized features."""
    months = [calendar.month_abbr[m] for m in range(1, 13)]
    if seasons_map:
        month_labels = [f"{calendar.month_abbr[m]}\n(S{seasons_map.get(m, '?')})" for m in range(1, 13)]
    else:
        month_labels = months
    data = pd.DataFrame(month_features.T, index=feature_labels, columns=months)
    fig, ax = plt.subplots(figsize=(1.4 * len(months) + 2, 0.6 * len(feature_labels) + 3))
    im = ax.imshow(data.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(month_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels)
    ax.set_title("Normalized monthly features")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized value (dark = higher)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _plot_season_means_heatmap(
    month_features: np.ndarray,
    feature_labels: List[str],
    seasons_map: Dict[int, int],
    path: Path,
    show: bool = False,
) -> None:
    """Plot average value per season (rows=features, cols=seasons)."""
    seasons = sorted(set(seasons_map.values()))
    season_means = []
    for season in seasons:
        months_in_season = [m for m, s in seasons_map.items() if s == season]
        indices = [m - 1 for m in months_in_season]
        season_means.append(month_features[indices].mean(axis=0))
    df = pd.DataFrame(season_means, index=[f"Season {s}" for s in seasons], columns=feature_labels)
    fig, ax = plt.subplots(figsize=(1.2 * len(seasons) + 2, 0.5 * len(feature_labels) + 3))
    im = ax.imshow(df.values.T, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels([f"S{s}" for s in seasons])
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels)
    ax.set_title("Average feature value by season")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Mean normalized value")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def run_representative_seasons(
    input_files: Dict[str, str],
    K: int = 2,
    random_state: int = 0,
    value_column: str = "value",
    output_path: Optional[Path] = None,
    diagnostics_dir: Optional[Path] = None,
    show_plots: bool = False,
    verbose: bool = True,
) -> Dict[int, int]:
    """Derive a contiguous seasons map from provided tech/load time series."""
    if verbose:
        print("[repr-seasons] Starting representative seasons pipeline")
    month_features, feature_labels = _load_monthly_features(
        input_files, value_column=value_column, verbose=verbose
    )
    seasons_map = derive_contiguous_seasons(month_features, K=K, random_state=random_state, verbose=verbose)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sorted(seasons_map.items()), columns=["month", "season"]).to_csv(output_path, index=False)
        if verbose:
            print(f"[repr-seasons] Wrote season mapping to {output_path}")

    diag_dir = diagnostics_dir or (output_path.parent if output_path else None)
    if diag_dir is not None:
        diag_dir = Path(diag_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)
        _plot_feature_heatmap(
            month_features,
            feature_labels,
            seasons_map,
            diag_dir / "monthly_features_heatmap.png",
            show_plots,
        )
        _plot_season_means_heatmap(
            month_features, feature_labels, seasons_map, diag_dir / "season_means_heatmap.png", show_plots
        )

    return seasons_map


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    sample_input_dir = base / "input"
    sample_output = base / "output" / "derived_seasons.csv"
    sample_input_dir.mkdir(parents=True, exist_ok=True)
    sample_output.parent.mkdir(parents=True, exist_ok=True)

    sample_files = {
        "PV": sample_input_dir / "vre_rninja_solar.csv",
        "Wind": sample_input_dir / "vre_rninja_wind.csv",
        "Load": sample_input_dir / "load_profiles.csv",
        "Precipitation": sample_input_dir / "monthly_precipitation.csv",
    }

    print("[repr-seasons] Deriving seasons from sample inputs (if present)...")
    try:
        seasons = run_representative_seasons(sample_files, K=2, random_state=0, output_path=sample_output, verbose=True)
        print(f"[repr-seasons] Seasons map written to {sample_output}: {seasons}")
    except Exception as exc:
        print(f"[repr-seasons] Skipped sample run: {exc}")
