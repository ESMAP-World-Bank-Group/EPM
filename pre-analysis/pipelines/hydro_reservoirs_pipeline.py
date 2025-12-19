"""Hydro reservoirs mapping pipeline for the GloHydroRes database (static map export).

Main entry points
-----------------
- ``load_hydro_reservoirs``: read + clean the CSV for a target country list.
- ``summarize_hydro_sites``: aggregate key metrics (site counts, capacity, head).
- ``build_hydro_map``: static PDF map styled by plant type.
- CLI/``__main__``: convenience runner that targets ``output_workflow/hydro_reservoirs`` with user-editable params.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import re

try:  # Optional background map support.
    import geopandas as gpd
except ImportError:  # pragma: no cover - geopandas is optional.
    gpd = None

from difflib import SequenceMatcher

BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "dataset" / "GloHydroRes_vs1.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output_workflow" / "hydro_reservoirs"
DEFAULT_COUNTRY_SHP = (
    BASE_DIR / "dataset" / "maps" / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"
)

# --------------------------------------------------------------------------- #
# Local utilities (duplicated to avoid extra dependencies)
# --------------------------------------------------------------------------- #


def require_file(path, hint=None):
    """Validate that a required external file exists; raise with a helpful message if missing."""
    path = Path(path)
    if not path.exists():
        extra = f" {hint}" if hint else " Place the file in the expected directory."
        raise FileNotFoundError(f"File not found: {path}.{extra}")
    return path


def resolve_country_name(input_name, available_names, threshold=0.75, verbose=True, allow_missing=False):
    """Resolve a user-provided country name against available names with fuzzy matching."""
    available = [name for name in available_names if isinstance(name, str)]
    normalized = {name.strip().lower(): name for name in available}
    candidate = input_name.strip().lower()

    if candidate in normalized:
        match = normalized[candidate]
        if verbose:
            print(f"[country-resolver] Exact match: '{input_name}' -> '{match}'")
        return match

    best, best_score = None, 0
    for name in available:
        score = SequenceMatcher(None, candidate, name.strip().lower()).ratio()
        if score > best_score:
            best, best_score = name, score

    if best_score >= threshold:
        if verbose:
            pct = f"{best_score*100:.1f}%"
            print(f"[country-resolver] Using closest match ({pct}): '{input_name}' -> '{best}'")
        return best

    if allow_missing:
        suggestions = ", ".join(available[:5])
        print(
            f"[country-resolver] Missing '{input_name}'. Closest='{best}' ({best_score:.2f}). "
            f"Available examples: {suggestions}. Skipping."
        )
        return None

    suggestions = ", ".join(available[:5])
    raise ValueError(
        f"Could not match country '{input_name}'. Top suggestion: '{best}' "
        f"(score {best_score:.2f}). Available examples: {suggestions}"
    )


def _strip_country(label: str, country: str | None) -> str:
    """Remove country names from the label to keep site names concise."""
    if not label or not country:
        return label
    pattern = re.compile(re.escape(country), flags=re.IGNORECASE)
    cleaned = pattern.sub("", label)
    return cleaned


def _clean_site_label(label: str, country: str | None = None) -> str:
    """Remove generic words and country names from facility labels."""
    if not label:
        return "Unknown"
    cleaned = GENERIC_SITE_LABEL_PATTERN.sub("", label)
    cleaned = _strip_country(cleaned, country)
    cleaned = re.sub(r"[,-]\s*,*", " ", cleaned)  # drop stray punctuation
    cleaned = " ".join(cleaned.split())
    return cleaned or label

DEFAULT_WORLD_MAP_SHAPEFILE = DEFAULT_COUNTRY_SHP

# Simple styling keyed by plant_type codes in the dataset (STO=storage, ROR=run-of-river, PS=pumped storage).
TYPE_COLORS = {"STO": "#1f77b4", "ROR": "#2ca02c", "PS": "#ff7f0e", "Unknown": "#6c757d"}
GENERIC_SITE_LABEL_WORDS = (
    "hydroelectric power plant",
    "hydroelectric plant",
    "hydropower plant",
    "power plant",
    "power station",
    "pumped storage power plant",
    "pumped storage",
    "storage",
    "plant",
    "hpp",
)
GENERIC_SITE_LABEL_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, GENERIC_SITE_LABEL_WORDS)) + r")\b", flags=re.IGNORECASE
)


def _normalize_countries(countries: Iterable[str], available: Sequence[str], verbose: bool) -> list[str]:
    """Resolve requested country names against the dataset list using fuzzy matching."""
    resolved = []
    for country in countries:
        match = resolve_country_name(country, available, verbose=verbose, allow_missing=False)
        resolved.append(match)
    return resolved


def load_hydro_reservoirs(
    dataset_path: Path | str | None = None,
    countries: Iterable[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read and clean the GloHydroRes CSV.

    Columns retained: ID, country, name, capacity_mw, plant_lat, plant_lon, plant_type, river, head_m, dam_height_m,
    res_area_km2, res_vol_km3, year. Extra columns are preserved for transparency.
    """
    path = Path(dataset_path or DEFAULT_DATASET)
    if not path.is_absolute():
        # First try relative to the script directory.
        candidate = SCRIPT_DIR / path
        # If a bare filename was provided, also try the dataset/ subfolder.
        if not candidate.exists():
            candidate_dataset = BASE_DIR / "dataset" / path.name
            if candidate_dataset.exists():
                candidate = candidate_dataset
        path = candidate
    path = require_file(path, hint="Place GloHydroRes_vs1.csv under pre-analysis/dataset/")
    if verbose:
        print(f"[hydro] Loading dataset from {path}")

    df = pd.read_csv(path)

    # Normalize core columns and enforce numeric dtypes where appropriate.
    for col in ("capacity_mw", "plant_lat", "plant_lon", "head_m", "dam_height_m", "res_area_km2", "res_vol_km3", "year"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    df["plant_type"] = df.get("plant_type", pd.Series(dtype=str)).fillna("Unknown").astype(str).str.upper()
    df["name"] = df.get("name", pd.Series(dtype=str)).fillna("Unknown").astype(str)
    df["country"] = df.get("country", pd.Series(dtype=str)).fillna("Unknown").astype(str)

    if countries:
        available = sorted(df["country"].dropna().unique())
        targets = _normalize_countries(countries, available, verbose=verbose)
        df = df[df["country"].isin(targets)].reset_index(drop=True)
        if verbose:
            print(f"[hydro] Filtered to {len(df)} sites across {len(targets)} country selection(s)")

    # Drop records missing coordinates since they cannot be mapped.
    mapped = df.dropna(subset=["plant_lat", "plant_lon"])
    if verbose and len(mapped) < len(df):
        missing = len(df) - len(mapped)
        print(f"[hydro] Dropped {missing} site(s) without coordinates for mapping")
    return mapped


def summarize_hydro_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate summary statistics per country."""
    if df.empty:
        return pd.DataFrame(columns=["country", "site_count", "total_capacity_mw", "median_capacity_mw", "avg_head_m", "avg_dam_height_m"])

    summary = (
        df.groupby("country")
        .agg(
            site_count=("ID", "count"),
            total_capacity_mw=("capacity_mw", "sum"),
            median_capacity_mw=("capacity_mw", "median"),
            avg_head_m=("head_m", "mean"),
            avg_dam_height_m=("dam_height_m", "mean"),
        )
        .reset_index()
        .sort_values("total_capacity_mw", ascending=False)
    )
    return summary


def plant_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of sites per plant_type and country."""
    if df.empty:
        return pd.DataFrame(columns=["country"])
    pivot = (
        df.pivot_table(index="country", columns="plant_type", values="ID", aggfunc="count", fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return pivot


def build_hydro_map(
    df: pd.DataFrame,
    output_path: Path | str,
    zoom_start: int = 6,  # retained for parity; not used in static map but kept for CLI compatibility
) -> Path:
    """Render a static PDF map with markers sized by capacity."""
    if df.empty:
        raise ValueError("No hydro sites available to plot.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw background map when geopandas and shapefile are available (same style as generators_pipeline).
    if gpd is not None and DEFAULT_WORLD_MAP_SHAPEFILE.exists():
        try:
            world = gpd.read_file(DEFAULT_WORLD_MAP_SHAPEFILE)
            world.plot(ax=ax, color="#f5f5f0", edgecolor="#cacaca", linewidth=0.4)
        except Exception:
            pass  # Continue with plain scatter if map fails to load.

    # Marker size scales with sqrt(capacity) for readability.
    capacities = df["capacity_mw"].fillna(0).clip(lower=0)
    sizes = 25 + (capacities**0.5) * 3

    for plant_type, sub in df.groupby("plant_type"):
        ax.scatter(
            sub["plant_lon"],
            sub["plant_lat"],
            s=sizes.loc[sub.index],
            label=plant_type,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            color=TYPE_COLORS.get(plant_type, TYPE_COLORS["Unknown"]),
        )

    # Label top 5 sites by capacity directly on the map.
    top_sites = df.sort_values("capacity_mw", ascending=False).head(5)
    for _, row in top_sites.iterrows():
        if pd.isna(row.get("plant_lat")) or pd.isna(row.get("plant_lon")):
            continue
        name = _clean_site_label(row.get("name", "Unknown"), country=row.get("country"))
        capacity = row.get("capacity_mw")
        volume = row.get("res_vol_km3")
        cap_label = f"{capacity:.0f} MW" if pd.notna(capacity) else "N/A"
        vol_label = f", {volume:.2f} km³" if pd.notna(volume) else ""
        ax.annotate(
            f"{name} ({cap_label}{vol_label})",
            xy=(row["plant_lon"], row["plant_lat"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
            color="#333333",
        )

    # Center plot on sites with padding and hide ticks/labels (mirrors generators_pipeline static map).
    lon_min, lon_max = float(df["plant_lon"].min()), float(df["plant_lon"].max())
    lat_min, lat_max = float(df["plant_lat"].min()), float(df["plant_lat"].max())
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    lon_padding = max(abs(lon_range) * 0.2, 0.5)
    lat_padding = max(abs(lat_range) * 0.2, 0.5)
    ax.set_xlim(lon_min - lon_padding, lon_max + lon_padding)
    ax.set_ylim(lat_min - lat_padding, lat_max + lat_padding)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax.set_title("Hydro reservoirs (GloHydroRes selection)")
    ax.legend(title="Plant type", frameon=True)

    plt.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _load_config(config_path: Path | str | None):
    """Load YAML config when provided; return empty dict when absent."""
    if not config_path:
        return {}
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _config_block(cfg: dict) -> dict:
    """Extract the hydro_reservoirs block from the open-data config."""
    return cfg.get("hydro_reservoirs", {})


def run_hydro_pipeline(
    dataset_path: Path | str | None,
    countries: Iterable[str],
    output_dir: Path | str,
    zoom_start: int = 6,
    verbose: bool = True,
) -> dict[str, Path]:
    """End-to-end pipeline: load -> summarize -> map -> export CSVs."""
    df = load_hydro_reservoirs(dataset_path=dataset_path, countries=countries, verbose=verbose)

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_csv = out_dir / "hydro_sites_cleaned.csv"
    df.to_csv(cleaned_csv, index=False)

    summary = summarize_hydro_sites(df)
    summary_csv = out_dir / "hydro_sites_summary.csv"
    summary.to_csv(summary_csv, index=False)

    breakdown = plant_type_breakdown(df)
    breakdown_csv = out_dir / "hydro_sites_type_breakdown.csv"
    breakdown.to_csv(breakdown_csv, index=False)

    map_path = build_hydro_map(df, out_dir / "hydro_sites_map.pdf", zoom_start=zoom_start)

    return {
        "cleaned_csv": cleaned_csv,
        "summary_csv": summary_csv,
        "breakdown_csv": breakdown_csv,
        # Keep legacy key name for compatibility while emitting a PDF.
        "map_png": map_path,
        "map_pdf": map_path,
    }



if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # User-editable parameters (tweak in-place from your IDE)
    # ------------------------------------------------------------------ #
    config_path = BASE_DIR / "config" / "open_data_config.yaml"  # Set to None to skip config lookup.
    use_config = True  # Flip to False to ignore config file entirely.
    countries = []  # e.g., ["Bosnia and Herzegovina", "Croatia"]. Falls back to config if empty.
    dataset_path = DEFAULT_DATASET  # Override with a custom CSV path if needed.
    output_dir = DEFAULT_OUTPUT_DIR  # Relative paths are resolved under pre-analysis/.
    zoom_start = 6  # Retained for API compatibility; not used for static map scaling.
    verbose = True

    cfg = _load_config(config_path) if (use_config and config_path) else {}
    hydro_cfg = _config_block(cfg)

    countries = countries or hydro_cfg.get("countries") or []
    if not countries:
        raise ValueError("No countries provided. Set `countries` above or in config.hydro_reservoirs.countries.")

    dataset_path = dataset_path or hydro_cfg.get("dataset") or DEFAULT_DATASET
    output_dir = output_dir or hydro_cfg.get("output_dir") or DEFAULT_OUTPUT_DIR
    zoom_start = zoom_start or hydro_cfg.get("zoom_start", 6)

    outputs = run_hydro_pipeline(dataset_path, countries, output_dir, zoom_start=zoom_start, verbose=verbose)
    if verbose:
        print("[hydro] Finished. Outputs:")
        for label, path in outputs.items():
            print(f"  - {label}: {path}")
