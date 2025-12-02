#!/usr/bin/env python3
"""Render an automatic Markdown report from workflow outputs."""

from __future__ import annotations

import argparse
import calendar
import os
import shutil
import shlex
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from jinja2 import Template

# Force a headless matplotlib backend so Snakemake worker threads don't try to spawn GUI windows.
os.environ["MPLBACKEND"] = "Agg"
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    matplotlib = None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = BASE_DIR / "report.md.j2"
DEFAULT_OUTPUT = BASE_DIR / "output_workflow" / "report.md"
DEFAULT_CONFIG = BASE_DIR / "config" / "open_data_config.yaml"
DISABLE_FLOAT_TEX = BASE_DIR / "disable_float.tex"


def _vprint(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[report][verbose] {message}", file=sys.stderr)


def _abspath_for_display(path: Path) -> str:
    """Return an absolute path string for clearer debugging."""
    try:
        return path.resolve().as_posix()
    except OSError:
        return path.as_posix()


def _slug(text: str) -> str:
    """Return a filesystem-friendly slug."""
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def _resolve_relative(base: Path, maybe_path: Path | str) -> Path:
    path = Path(maybe_path)
    return path if path.is_absolute() else (base / path)


CATEGORY_OUTPUT_SUBDIRS = {
    "load": "load",
    "vre": "vre",
    "supply": "supply",
    "socioeconomic": "socioeconomic",
}


def _resolve_category_output_dir(output_root: Path, relative: Optional[str], category: str) -> Path:
    """Resolve a category-specific output directory much like the workflow does."""
    subdir = CATEGORY_OUTPUT_SUBDIRS.get(category)
    if subdir is None:
        raise ValueError(f"Unknown output category: {category}")
    base = output_root / subdir
    if not relative:
        return base
    rel_path = Path(relative)
    if rel_path.is_absolute():
        return rel_path
    return base / rel_path


def _relpath_for_display(path: Path) -> str:
    """Return a BASE_DIR-relative path string when possible for concise display."""
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _human_join(items: Sequence[str]) -> str:
    """Join a list of strings with commas and a final 'and' for readability."""
    items = [item for item in items if isinstance(item, str)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _latest_mtime(paths: Iterable[Path]) -> Optional[datetime]:
    """Return the most recent modification timestamp (UTC) among the provided paths."""
    timestamps = []
    for path in paths:
        try:
            if path and path.exists():
                timestamps.append(path.stat().st_mtime)
        except OSError:
            continue
    if not timestamps:
        return None
    return datetime.fromtimestamp(max(timestamps), tz=timezone.utc)


def _resolution_with_date(resolution: str, paths: Sequence[Path]) -> str:
    """Return resolution label without appending extraction dates."""
    return resolution


def _ensure_sequence(value) -> Sequence[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            print(f"Warning: could not parse config {config_path}: {exc}", file=sys.stderr)
            return {}


def default_report_output(config_path: Path, output_dir_override: Optional[Path]) -> Path:
    """Pick a sensible default report path under the workflow output root."""
    cfg = load_config(config_path)
    out_root = find_output_dir(cfg, output_dir_override)
    return out_root / "report.md"


def find_output_dir(config: Dict, override: Optional[Path]) -> Path:
    """Pick an output directory; fall back to defaults when missing."""
    if override:
        return override.resolve()

    root = _resolve_relative(BASE_DIR, config.get("output_workflow_dir", "output_workflow"))
    candidates = [root, root / "output"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def readable_country(slug: str, slug_map: Dict[str, str]) -> str:
    return slug_map.get(slug, slug.replace("_", " ").title())


def collect_load_profiles(load_dir: Path, slug_map: Dict[str, str]) -> Tuple[List[Dict], List[Path], List[Path], List[Path]]:
    stats: List[Dict] = []
    figures: List[Path] = []
    heatmaps: List[Path] = []
    boxplots: List[Path] = []

    for csv_path in sorted(load_dir.glob("load_profile_*.csv")):
        df = pd.read_csv(csv_path)
        country_slug = csv_path.stem.replace("load_profile_", "")
        country_name = readable_country(country_slug, slug_map)
        if "load_mw" not in df.columns:
            continue

        stats.append(
            {
                "Country": country_name,
                "Average Load (MW)": df["load_mw"].mean(),
                "Peak Load (MW)": df["load_mw"].max(),
                "Minimum Load (MW)": df["load_mw"].min(),
            }
        )

        pdf_path = csv_path.with_suffix(".pdf")
        if pdf_path.exists():
            figures.append(pdf_path)

        for heatmap_path in sorted(load_dir.glob(f"heatmap_load_{country_slug}.*")):
            heatmaps.append(heatmap_path)

        for boxplot_path in sorted(load_dir.glob(f"boxplot_load_{country_slug}.*")):
            boxplots.append(boxplot_path)

    return stats, figures, heatmaps, boxplots


def collect_rninja_profiles(vre_dir: Path) -> Tuple[List[Dict], Dict[str, List[Dict[str, Optional[Path]]]], List[Path]]:
    """Summarise Renewables Ninja capacity factors and gather per-zone figures."""
    summaries: List[Dict] = []
    zone_order: Dict[str, List[str]] = {"solar": [], "wind": []}
    zone_slug_map: Dict[str, Dict[str, str]] = {"solar": {}, "wind": {}}
    boxplots: List[Path] = []

    def _tech_key_from_label(label: str) -> Optional[str]:
        lower = label.lower()
        if "pv" in lower or "solar" in lower:
            return "solar"
        if "wind" in lower:
            return "wind"
        return None

    def _register_zone(tech_key: str, zone_label: str) -> None:
        slug = _slug(str(zone_label))
        if not slug:
            return
        mapping = zone_slug_map.setdefault(tech_key, {})
        if slug not in mapping:
            mapping[slug] = zone_label
            zone_order.setdefault(tech_key, []).append(slug)

    def _collect_csvs(patterns: Sequence[str]) -> List[Path]:
        paths: List[Path] = []
        for pattern in patterns:
            paths.extend(sorted(vre_dir.glob(pattern)))
        seen: set[Path] = set()
        unique_paths: List[Path] = []
        for path in paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        return unique_paths

    csv_files = _collect_csvs(["rninja_data_*.csv", "vre_rninja_*.csv"])
    for csv_path in csv_files:
        tech = "solar" if "solar" in csv_path.stem.lower() or "_pv_" in csv_path.stem.lower() else "wind"
        df = pd.read_csv(csv_path)
        # Exclude calendar/index columns so only capacity-factor series (e.g., yearly columns) are averaged
        meta_cols = {"zone", "season", "month", "day", "hour", "timestamp", "timestamp_utc"}
        value_cols = [c for c in df.columns if c not in meta_cols]
        for zone, group in df.groupby("zone"):
            for col in value_cols:
                summaries.append(
                    {
                        "zone": zone,
                        "tech": tech,
                        "period": col,
                        "mean_capacity_factor": float(group[col].mean()),
                    }
                )
            if isinstance(zone, str) and zone.strip():
                _register_zone(tech, zone)

    per_zone_paths: Dict[str, Dict[str, Dict[str, Optional[Path]]]] = {"solar": {}, "wind": {}}

    def _match_zone_slug(part: str, candidates: Sequence[str]) -> Optional[str]:
        best: Optional[str] = None
        for slug in candidates:
            if part == slug or part.startswith(f"{slug}_"):
                if best is None or len(slug) > len(best):
                    best = slug
        return best

    def _parse_filename(
        stem: str, prefix: str, suffix: str
    ) -> Tuple[Optional[str], Optional[str]]:
        if not stem.startswith(prefix) or not stem.endswith(suffix):
            return None, None
        body = stem[len(prefix) : -len(suffix)]
        tech_label, _, zone_part = body.partition("_")
        if not tech_label or not zone_part:
            return None, None
        return tech_label, zone_part

    for heatmap in sorted(vre_dir.glob("heatmap_*_rninja.*")):
        tech_label, slug_part = _parse_filename(heatmap.stem, "heatmap_", "_rninja")
        if not tech_label or not slug_part:
            continue
        tech_key = _tech_key_from_label(tech_label)
        if tech_key is None:
            continue
        slug_candidates = list(zone_slug_map.get(tech_key, {}).keys())
        zone_slug = _match_zone_slug(slug_part, slug_candidates)
        if not zone_slug:
            continue
        per_zone_paths.setdefault(tech_key, {}).setdefault(zone_slug, {"heatmap": None, "boxplot": None})[
            "heatmap"
        ] = heatmap

    for boxplot in sorted(vre_dir.glob("boxplot_*_rninja.*")):
        tech_label, slug_part = _parse_filename(boxplot.stem, "boxplot_", "_rninja")
        if not tech_label or not slug_part:
            continue
        tech_key = _tech_key_from_label(tech_label)
        if tech_key is None:
            continue
        slug_candidates = list(zone_slug_map.get(tech_key, {}).keys())
        zone_slug = _match_zone_slug(slug_part, slug_candidates)
        if not zone_slug:
            continue
        per_zone_paths.setdefault(tech_key, {}).setdefault(zone_slug, {"heatmap": None, "boxplot": None})[
            "boxplot"
        ] = boxplot
        boxplots.append(boxplot)

    structured: Dict[str, List[Dict[str, Optional[Path]]]] = {}
    for tech_key in ("solar", "wind"):
        entries: List[Dict[str, Optional[Path]]] = []
        seen: set[str] = set()
        order = zone_order.get(tech_key, [])
        candidate_map = zone_slug_map.get(tech_key, {})
        for slug in order:
            data = per_zone_paths.get(tech_key, {}).get(slug)
            if not data or not (data.get("heatmap") or data.get("boxplot")):
                continue
            entries.append(
                {
                    "country": candidate_map.get(slug, slug.replace("_", " ").title()),
                    "heatmap": data.get("heatmap"),
                    "boxplot": data.get("boxplot"),
                }
            )
            seen.add(slug)
        for slug, data in per_zone_paths.get(tech_key, {}).items():
            if slug in seen:
                continue
            if not (data.get("heatmap") or data.get("boxplot")):
                continue
            entries.append(
                {
                    "country": candidate_map.get(slug, slug.replace("_", " ").title()),
                    "heatmap": data.get("heatmap"),
                    "boxplot": data.get("boxplot"),
                }
            )
        if entries:
            structured[tech_key] = entries

    return summaries, structured, boxplots


def plot_gap_project_locations(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Quick scatter plot for GAP-selected projects (lat/lon, scaled by capacity)."""
    if df.empty or not {"Latitude", "Longitude"}.issubset(df.columns):
        return None

    try:
        import matplotlib

        backend = getattr(matplotlib, "get_backend", lambda: "")()
        if str(backend).lower() != "agg":
            matplotlib.use("Agg", force=True)  # headless backend for non-main-thread calls
        import matplotlib.pyplot as plt
    except Exception:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    df_local = df.copy()
    df_local["capacity"] = pd.to_numeric(df_local["Capacity (MW)"], errors="coerce").fillna(0)
    df_local["size"] = 40 + 6 * df_local["capacity"].pow(0.5)
    color_map = {"solar": "#f4a261", "wind": "#2a9d8f"}

    for tech, group in df_local.groupby("Type"):
        color = color_map.get(str(tech).lower(), "#577590")
        ax.scatter(
            group["Longitude"],
            group["Latitude"],
            s=group["size"],
            color=color,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.35,
            label=str(tech).title(),
        )
        for _, row in group.iterrows():
            label = row.get("City") or row.get("Plant / Project name") or ""
            if isinstance(label, str) and label.strip():
                ax.annotate(
                    label.strip(),
                    (row["Longitude"], row["Latitude"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    alpha=0.8,
                )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GIP-2025 selected projects")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    if len(df_local["Type"].unique()) > 1:
        ax.legend(title="Technology")

    fig.tight_layout()
    plot_path = output_dir / "gap_project_locations.pdf"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    return plot_path


def collect_gap_projects(
    supply_dir: Path, slug_map: Dict[str, str]
) -> Tuple[Optional[str], Optional[Path], Optional[Path]]:
    gap_files = sorted(supply_dir.glob("most_relevant_projects_*.csv"))
    if not gap_files:
        return None, None, None

    gap_path = gap_files[0]
    df = pd.read_csv(gap_path)

    if not {"Country", "Type", "Capacity (MW)", "Plant / Project name"}.issubset(df.columns):
        return None, gap_path, None

    df["Country"] = df["Country"].apply(lambda x: readable_country(_slug(str(x)), slug_map) if isinstance(x, str) else x)

    def _representative_location(group: pd.DataFrame) -> str:
        for col in ("City", "Plant / Project name"):
            for val in group.get(col, []):
                if isinstance(val, str) and val.strip():
                    return val.strip()
        for _, row in group.iterrows():
            lat, lon = row.get("Latitude"), row.get("Longitude")
            if pd.notna(lat) and pd.notna(lon):
                return f"{lat:.2f}, {lon:.2f}"
        return ""

    summary_rows: List[Dict] = []
    for (country, tech), group in df.groupby(["Country", "Type"]):
        summary_rows.append(
            {
                "Country": country,
                "Technology": tech.title() if isinstance(tech, str) else tech,
                "Total capacity (MW)": group["Capacity (MW)"].sum(),
                "Example location": _representative_location(group) or "n/a",
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["Country", "Technology"])
    summary_md = summary.to_markdown(index=False, floatfmt=".0f")

    plot_path = plot_gap_project_locations(df, supply_dir)

    return summary_md, gap_path, plot_path


def collect_generation_map(
    supply_dir: Path, config: Dict
) -> Tuple[Optional[Path], Optional[Path], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    gen_cfg = config.get("generation_map", {})
    map_candidates = []
    if gen_cfg.get("map_filename"):
        map_candidates.append(supply_dir / gen_cfg["map_filename"])
    map_candidates.append(supply_dir / "generation_map.html")
    map_candidates.extend(sorted(supply_dir.glob("generation_map*.html")))

    map_path = None
    map_all_path = None
    for candidate in map_candidates:
        if not candidate.exists():
            continue
        stem = candidate.stem.lower()
        is_all = stem.endswith("_all")
        if is_all and map_all_path is None:
            map_all_path = candidate
        if not is_all and map_path is None:
            map_path = candidate
        if map_path and map_all_path:
            break
    if map_path is None:
        map_path = map_all_path

    summary_path = None
    data_path = None
    if gen_cfg.get("summary_filename"):
        candidate = supply_dir / gen_cfg["summary_filename"]
        if candidate.exists():
            summary_path = candidate
    if summary_path is None:
        fallback = supply_dir / "generation_sites_summary.csv"
        if fallback.exists():
            summary_path = fallback

    summary_df = None
    data_df = None
    if gen_cfg.get("data_filename"):
        candidate = supply_dir / gen_cfg["data_filename"]
        if candidate.exists():
            data_path = candidate
    if data_path is None:
        fallback = supply_dir / "generation_sites.csv"
        if fallback.exists():
            data_path = fallback

    if summary_path and summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path)
            if summary_df.empty:
                summary_df = None
        except Exception as exc:  # pragma: no cover - defensive logging only
            print(f"Warning: could not read generation map summary {summary_path}: {exc}", file=sys.stderr)

    if data_path and data_path.exists():
        try:
            data_df = pd.read_csv(data_path)
            if data_df.empty:
                data_df = None
        except Exception as exc:  # pragma: no cover - defensive logging only
            print(f"Warning: could not read generation site data {data_path}: {exc}", file=sys.stderr)

    return map_path, map_all_path, summary_df, data_df


def collect_hydro_reservoirs(
    output_root: Path, config: Dict
) -> Tuple[Optional[Path], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Path], Optional[Path]]:
    """Locate hydropower map + summary tables from the hydro_reservoirs workflow."""
    hydro_cfg = config.get("hydro_reservoirs", {})
    if not hydro_cfg.get("enabled", False):
        return None, None, None

    search_dirs = []
    # Preferred: supply/hydro_reservoirs (or configured subdir).
    try:
        hydro_dir = _resolve_category_output_dir(output_root, hydro_cfg.get("output_dir"), "supply")
        search_dirs.append(hydro_dir)
    except Exception:
        pass
    # Fallbacks: standalone outputs or directly under pre-analysis.
    search_dirs.extend(
        [
            BASE_DIR / "output_standalone" / "hydro_reservoirs",
            BASE_DIR / "hydro_reservoirs",
        ]
    )

    map_path = None
    summary_df = None
    data_df = None
    summary_path = None
    data_path = None

    for folder in search_dirs:
        if not folder.exists():
            continue
        candidate_map = folder / "hydro_sites_map.pdf"
        if map_path is None and not candidate_map.exists():
            candidate_map = folder / "hydro_sites_map.png"
        candidate_summary = folder / "hydro_sites_summary.csv"
        candidate_data = folder / "hydro_sites_cleaned.csv"
        if map_path is None and candidate_map.exists():
            map_path = candidate_map
        if summary_df is None and candidate_summary.exists():
            try:
                df = pd.read_csv(candidate_summary)
                if not df.empty:
                    summary_df = df
                    summary_path = candidate_summary
            except Exception as exc:
                print(f"Warning: could not read hydro summary {candidate_summary}: {exc}", file=sys.stderr)
        if data_df is None and candidate_data.exists():
            try:
                df = pd.read_csv(candidate_data)
                if not df.empty:
                    data_df = df
                    data_path = candidate_data
            except Exception as exc:
                print(f"Warning: could not read hydro data {candidate_data}: {exc}", file=sys.stderr)
        if map_path and summary_df is not None and data_df is not None:
            break

    return map_path, summary_df, data_df, summary_path, data_path


def _format_owid_latest(latest: pd.DataFrame) -> str:
    """Prettify the latest-year OWID energy table."""
    if latest is None or latest.empty:
        return ""
    df = latest.copy()
    rename = {
        "country": "Country",
        "year": "Year",
        "population_millions": "Population (M)",
        "gdp_billions_usd": "GDP (bn USD)",
        "electricity_demand_twh": "Electricity demand (TWh)",
        "electricity_demand_per_capita_kwh": "Demand per capita (kWh)",
        "energy_per_capita_kwh": "Energy per capita (kWh)",
    }
    df = df.rename(columns=rename)
    numeric_cols = [col for col in rename.values() if col in df.columns and col not in ("Country", "Year")]

    def _fmt(value: object, decimals: int = 1) -> str:
        try:
            if pd.isna(value):
                return "-"
            return f"{float(value):.{decimals}f}"
        except Exception:
            return "-"

    for col in numeric_cols:
        df[col] = df[col].apply(lambda v: _fmt(v, decimals=1))

    return _wrap_table(df.to_markdown(index=False))


def collect_owid_energy(output_root: Path, config: Dict) -> Dict[str, object]:
    """Gather OWID energy figures + summary tables from workflow outputs."""
    cfg = config.get("owid_energy", {}) or {}
    if not cfg.get("enabled", False):
        return {"figures": {}, "summary": "", "data_files": [], "outdir": None}

    outdir = _resolve_category_output_dir(output_root, cfg.get("output_dir", "owid_energy"), "socioeconomic")
    basename = cfg.get("output_basename", "owid_energy")
    figures = {
        "population": outdir / f"{basename}_population.pdf",
        "gdp": outdir / f"{basename}_gdp.pdf",
        "electricity": outdir / f"{basename}_electricity.pdf",
        "electricity_per_capita": outdir / f"{basename}_electricity_per_capita.pdf",
    }
    summary_md = ""
    data_files: List[Path] = []
    latest_path = outdir / f"{basename}_latest.csv"
    summary_path = outdir / f"{basename}_summary.csv"

    table_df: Optional[pd.DataFrame] = None
    for candidate in (latest_path, summary_path):
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                if not df.empty:
                    table_df = df
                    data_files.append(candidate)
                    break
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Warning: could not read OWID summary {candidate}: {exc}", file=sys.stderr)
    if table_df is not None:
        summary_md = _format_owid_latest(table_df)

    return {"figures": figures, "summary": summary_md, "data_files": data_files, "outdir": outdir}


def collect_climate_overview(output_root: Path, config: Dict) -> Dict[str, object]:
    cfg = config.get("climate_overview", {})
    if not cfg.get("enabled", False):
        return {"summary": "", "figures": {}, "period": "", "data_files": [], "countries": ""}

    outdir = _resolve_relative(output_root, cfg.get("output_dir", "climate"))
    summary_md = ""
    summary_path = outdir / "climate_summary.csv"
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
            # Build one table per variable; drop Period and include a caption with units.
            required = {"Country", "Variable", "Mean", "Min", "Max", "Units"}
            if required.issubset(df.columns):
                df = df[list(required)]  # drop Period and keep only needed stats
                tables = []
                for var, group in df.groupby("Variable"):
                    unit = group["Units"].iloc[0] if "Units" in group.columns else ""
                    cols = ["Max", "Mean", "Min"]  # desired order
                    table = group[["Country", *cols]].set_index("Country")
                    table_md = table.to_markdown(index=True, floatfmt=".1f")
                    caption = f"_{var} ({unit})_"
                    tables.append(f"{table_md}\n\n{caption}")
                summary_md = "\n\n".join(tables)
            else:
                summary_md = df.to_markdown(index=False, floatfmt=".1f")
        except Exception as exc:  # pragma: no cover - defensive logging only
            print(f"Warning: could not read climate summary {summary_path}: {exc}", file=sys.stderr)

    figures = {
        "spatial": sorted(outdir.glob("spatial_mean_*.pdf")),
        "monthly": sorted(outdir.glob("monthly_mean_*.pdf")),
        "heatmap_precipitation": sorted(outdir.glob("monthly_precipitation_heatmap.*")),
        "heatmap_temperature": sorted(outdir.glob("monthly_temperature_heatmap.*")),
        "scatter": sorted(outdir.glob("scatter_annual_spatial_means_*.pdf")),
    }
    data_files = sorted(outdir.glob("*.nc"))
    if not data_files:
        data_files = sorted(outdir.glob("*.zip"))

    start_year = cfg.get("start_year")
    end_year = cfg.get("end_year")
    if start_year and end_year:
        period = f"{start_year}–{end_year}"
    else:
        period = str(start_year or end_year or "")

    countries_cfg = cfg.get("countries")
    if countries_cfg:
        countries = _ensure_sequence(countries_cfg)
    else:
        labels = cfg.get("label_map") or {}
        countries = [labels.get(iso, iso) for iso in _ensure_sequence(cfg.get("iso_a2"))]
    countries_inline = _human_join([c for c in countries if c])

    return {
        "summary": summary_md,
        "figures": figures,
        "period": period,
        "data_files": data_files,
        "outdir": outdir,
        "countries": countries_inline,
    }


def find_rep_day_figures(base_dir: Path, output_dir: Path) -> List[Path]:
    """Collect representative-day diagnostic figures from common output locations."""
    candidate_dirs = [
        base_dir / "prepare-data" / "representative_days" / "output",
        base_dir / "representative_days" / "output",
        output_dir / "representative_days",
        output_dir,
    ]
    figs: List[Path] = []
    seen: set[Path] = set()
    for rep_dir in candidate_dirs:
        try:
            rep_dir = rep_dir.resolve()
        except OSError:
            continue
        if rep_dir in seen or not rep_dir.exists():
            continue
        seen.add(rep_dir)
        for ext in ("*.pdf", "*.png"):
            figs.extend(sorted(rep_dir.glob(ext)))
    return figs


def load_representative_seasons(output_dir: Path, config: Dict) -> Tuple[str, Optional[Path], List[Path], Dict[int, int]]:
    """Load season map CSV and diagnostics for representative seasons."""
    rep_season_cfg = config.get("representative_seasons", {}) or {}
    outdir_cfg = rep_season_cfg.get("output_dir") or "representative_seasons"
    candidate_dirs = []
    outdir_path = Path(outdir_cfg)
    if outdir_path.is_absolute():
        candidate_dirs.append(outdir_path)
    else:
        candidate_dirs.append(output_dir / outdir_path)
    candidate_dirs.append(output_dir / "representative_seasons")
    candidate_dirs.append(BASE_DIR / "prepare-data" / "representative_days" / "representative_seasons")

    map_path: Optional[Path] = None
    figures: List[Path] = []
    seen_dirs: set[Path] = set()
    for directory in candidate_dirs:
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved in seen_dirs or not resolved.exists():
            continue
        seen_dirs.add(resolved)
        candidate_map = resolved / "seasons_map.csv"
        if candidate_map.exists():
            map_path = candidate_map
        for fname in (
            "monthly_features_heatmap.pdf",
            "season_means_heatmap.pdf",
            "monthly_features_heatmap.png",
            "season_means_heatmap.png",
        ):
            fpath = resolved / fname
            if fpath.exists() and fpath not in figures:
                figures.append(fpath)

    seasons_map: Dict[int, int] = {}
    table_md = ""
    if map_path and map_path.exists():
        try:
            df = pd.read_csv(map_path)
            seasons_map = {int(row["month"]): int(row["season"]) for _, row in df.iterrows()}
            table_md = df.to_markdown(index=False)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: could not read seasons map {map_path}: {exc}", file=sys.stderr)

    return table_md, map_path, figures, seasons_map


def _percent_no_decimals(value) -> str:
    """Format a fractional value as a percentage with no decimal places."""
    if isinstance(value, str) and value.strip() == "-":
        return "-"
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value) * 100:.0f}%"
    except Exception:
        return "-"


def load_representative_days_summary(output_dir: Path, config: Dict) -> Tuple[str, str]:
    """Load and prettify the representative_days_summary.csv table."""
    rep_cfg = config.get("representative_days", {}) or {}
    rep_subdir = rep_cfg.get("epm_output_dir") or rep_cfg.get("output_dir") or "representative_days"
    summary_dir_cfg = rep_cfg.get("summary_dir")
    summary_dir = (
        _resolve_relative(output_dir, summary_dir_cfg)
        if summary_dir_cfg
        else output_dir / "representative_days"
    )
    candidates = [
        summary_dir / "representative_days_summary.csv",
        output_dir / "representative_days_summary.csv",
        output_dir / "epm_export" / rep_subdir / "representative_days_summary.csv",
        output_dir / rep_subdir / "representative_days_summary.csv",
        BASE_DIR / "prepare-data" / "representative_days" / "output" / "representative_days_summary.csv",
    ]

    summary_path: Optional[Path] = None
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            summary_path = resolved
            break

    if not summary_path:
        return "", ""

    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return "", _relpath_for_display(summary_path)
    if df.empty:
        return "", _relpath_for_display(summary_path)

    friendly = df.copy()
    rename = {"season": "Season", "day": "Day", "category": "Category", "weight_pct": "Weight"}
    friendly = friendly.rename(columns=rename)

    def _clean_col(col: str) -> str:
        text = str(col)
        if text in rename.values():
            return text
        if text.endswith("_avg_cf"):
            base = text[: -len("_avg_cf")]
            return f"{base.replace('_', ' ')} Avg CF (%)"
        return text.replace("_", " ")

    friendly.columns = [_clean_col(c) for c in friendly.columns]

    # Keep season labels for grouping/separators; drop day for brevity.
    if "Day" in friendly.columns:
        friendly = friendly.drop(columns=["Day"])

    # Rename category label
    if "Category" in friendly.columns:
        friendly["Category"] = (
            friendly["Category"]
            .apply(lambda v: v if pd.isna(v) else str(v))
            .str.replace("_", " ", regex=False)
            .replace({"representative": "Repr. Days"})
        )

    def _pct(value: object, decimals: int = 2) -> str:
        """Format numeric values as percentages; pass through non-numeric placeholders."""
        try:
            if pd.isna(value):
                return "-"
            return f"{float(value) * 100:.{decimals}f}%"
        except Exception:
            return "-"

    # Format weights as percentages with no decimals
    if "Weight" in friendly.columns:
        friendly["Weight"] = friendly["Weight"].apply(lambda v: _pct(v, decimals=0))

    # Convert all Avg CF (%) columns to percentage strings without decimals
    cf_cols = [c for c in friendly.columns if c.endswith("Avg CF (%)")]
    for col in cf_cols:
        friendly[col] = friendly[col].apply(lambda v: _pct(v, decimals=0))

    # Rename Avg CF columns to use tech-ISO2 headers (drop repeated "Avg CF" text)
    try:
        import pycountry
    except Exception:
        pycountry = None

    def _iso2_from_name(name: str) -> str:
        """Best-effort ISO2 lookup; fallback to original name on failure."""
        if not pycountry:
            return name
        try:
            country = pycountry.countries.lookup(name)
            return country.alpha_2
        except Exception:
            return name

    rename_cols = {}
    for col in cf_cols:
        if not col.endswith("Avg CF (%)"):
            continue
        base = col[:-len(" Avg CF (%)")] if col.endswith(" Avg CF (%)") else col
        parts = base.split(" ", 1)
        if len(parts) == 2:
            tech, country = parts
            iso = _iso2_from_name(country)
            rename_cols[col] = f"{tech}-{iso}"
    if rename_cols:
        friendly = friendly.rename(columns=rename_cols)

    # Insert horizontal separators between seasons and before benchmarks for readability.
    season_labels = list(friendly["Season"]) if "Season" in friendly.columns else []
    season_breaks = []
    for idx in range(1, len(season_labels)):
        prev, curr = season_labels[idx - 1], season_labels[idx]
        if pd.isna(prev) or pd.isna(curr):
            continue
        if str(prev) == "-" or str(curr) == "-":
            continue
        if prev != curr:
            season_breaks.append(idx)

    benchmark_idx: Optional[int] = None
    if "Category" in friendly.columns:
        benchmark_mask = friendly["Category"].astype(str).str.contains("benchmark", case=False, na=False)
        benchmark_hits = [i for i, flag in enumerate(benchmark_mask) if flag]
        if benchmark_hits:
            benchmark_idx = benchmark_hits[0]

    def _insert_row_separators(table_md: str, n_cols: int, break_rows: List[int]) -> str:
        """Inject markdown rows made of '---' to visually separate row groups."""
        if not table_md or n_cols <= 0 or not break_rows:
            return table_md
        lines = table_md.splitlines()
        if len(lines) <= 2:
            return table_md
        separator = "|" + "|".join([" --- "] * n_cols) + "|"
        offset = 0
        for row_idx in sorted(set(break_rows)):
            insert_at = 2 + row_idx + offset
            if insert_at <= len(lines):
                lines.insert(insert_at, separator)
                offset += 1
        return "\n".join(lines)

    break_rows: List[int] = season_breaks.copy()
    if benchmark_idx is not None:
        break_rows.append(benchmark_idx)

    summary_md = friendly.to_markdown(index=False)
    summary_md = _insert_row_separators(summary_md, len(friendly.columns), break_rows)
    summary_md = _wrap_table(summary_md)
    # Add a brief caption explaining benchmarks/averages.
    caption = "_Capacity-factor columns are percent values per tech/ISO2; weights reflect representative-day optimisation._"
    return f"{summary_md}\n\n{caption}", _relpath_for_display(summary_path)


FIGURE_PLACEHOLDER_SIGNATURE = b"[PLACEHOLDER]"


def _is_valid_figure(path: Path) -> bool:
    """Return True when a figure file exists and has content to avoid placeholders."""
    try:
        if not path.exists() or not path.is_file():
            return False
        stat = path.stat()
        if stat.st_size == 0:
            return False
        if stat.st_size >= len(FIGURE_PLACEHOLDER_SIGNATURE):
            with path.open("rb") as fh:
                signature = fh.read(len(FIGURE_PLACEHOLDER_SIGNATURE))
            if signature == FIGURE_PLACEHOLDER_SIGNATURE:
                return False
        return True
    except OSError:
        return False


def format_figures(fig_paths: Sequence[Path], label: str, add_link: bool = True) -> str:
    valid_paths = [path for path in fig_paths if _is_valid_figure(path)]
    if not valid_paths:
        return ""

    supported_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    lines = []
    for path in valid_paths:
        suffix = path.suffix.lower()
        if suffix not in supported_suffixes:
            continue
        rel = path.relative_to(BASE_DIR)
        lines.append(
            "::: {.nonfloat .center}\n"
            f"![{label}]({rel.as_posix()})\n"
            ":::"
        )

    if not lines:
        return ""
    return "\n\n".join(lines)


def _wrap_table(table_md: str) -> str:
    """Wrap markdown tables in a scrollable div for better rendering."""
    if not table_md:
        return ""
    return f'<div style="overflow-x: auto;">\n\n{table_md}\n\n</div>'


def format_hydro_summary(summary_df: Optional[pd.DataFrame]) -> str:
    """Render per-country hydro summary markdown."""
    if summary_df is None or summary_df.empty:
        return ""
    df = summary_df.copy()
    rename = {
        "country": "Country",
        "site_count": "Sites",
        "total_capacity_mw": "Total capacity (MW)",
        "median_capacity_mw": "Median capacity (MW)",
        "avg_head_m": "Average head (m)",
        "avg_dam_height_m": "Average dam height (m)",
    }
    df = df.rename(columns=rename)
    return _wrap_table(df.to_markdown(index=False, floatfmt=".1f"))


def format_hydro_appendix(data_df: Optional[pd.DataFrame]) -> str:
    """Render per-country hydro plant tables for the appendix."""
    if data_df is None or data_df.empty:
        return ""
    parts: List[str] = []
    cols = ["name", "plant_type", "capacity_mw", "river", "res_vol_km3"]
    rename = {
        "name": "Name",
        "plant_type": "Type",
        "capacity_mw": "Capacity (MW)",
        "river": "River",
        "res_vol_km3": "Reservoir volume (km³)",
    }
    for country, group in data_df.groupby("country"):
        df = group.copy()
        df = df[cols]
        df = df.rename(columns=rename)
        table = _wrap_table(df.to_markdown(index=False, floatfmt=".2f"))
        parts.append(f"**{country}**\n\n{table}")
    return "\n\n".join(parts)


def _extract_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Best-effort datetime extraction; fallback to synthetic hourly index."""
    for col in ("datetime", "timestamp"):
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().any():
                return pd.DatetimeIndex(dt)
    return pd.date_range("2015-01-01", periods=len(df), freq="h")


def build_load_aggregate_plots(load_dir: Path, slug_map: Dict[str, str]) -> Dict[str, Path]:
    """Create monthly and daily average load plots across countries."""
    try:
        import matplotlib

        backend = getattr(matplotlib, "get_backend", lambda: "")()
        if str(backend).lower() != "agg":
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    figures: Dict[str, Path] = {}
    load_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(load_dir.glob("load_profile_*.csv"))
    if not csv_paths:
        return figures

    monthly = {}
    daily = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "load_mw" not in df.columns:
            continue
        dt_index = _extract_datetime_index(df)
        df = df.assign(month=dt_index.month, hour=dt_index.hour)
        country_slug = csv_path.stem.replace("load_profile_", "")
        country_name = readable_country(country_slug, slug_map)
        monthly[country_name] = df.groupby("month")["load_mw"].mean()
        daily[country_name] = df.groupby("hour")["load_mw"].mean()

    if monthly:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for country, series in monthly.items():
            series.sort_index().plot(ax=ax, marker="o", label=country)
        ax.set_xlabel("Month")
        ax.set_ylabel("Average load (MW)")
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = load_dir / "load_avg_month.pdf"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figures["avg_month"] = path

    if daily:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for country, series in daily.items():
            series.sort_index().plot(ax=ax, marker="o", label=country)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Average load (MW)")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path = load_dir / "load_avg_day.pdf"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figures["avg_day"] = path

    return figures


def format_generation_summary(df: Optional[pd.DataFrame]) -> str:
    """Create a readable markdown table for generation-map summaries."""
    if df is None or df.empty:
        return ""

    friendly = df.copy()

    status_col = next((col for col in ("status", "Status", "Category") if col in friendly.columns), None)
    if status_col:
        friendly["_status_category"] = friendly[status_col].apply(_categorize_status)
        friendly = friendly[friendly["_status_category"] == "Operating"]
        friendly = friendly.drop(columns=["_status_category", status_col], errors="ignore")

    if friendly.empty:
        return ""

    friendly = friendly.drop(columns=[c for c in ["site_count", "avg_capacity_mw", "status", "Category"] if c in friendly.columns], errors="ignore")

    rename = {
        "country": "Country",
        "technology": "Technology",
        "capacity_mw": "Total capacity (MW)",
        "total_capacity_mw": "Total capacity (MW)",
    }
    friendly = friendly.rename(columns=rename)

    if "Total capacity (MW)" in friendly.columns:
        friendly["Total capacity (MW)"] = pd.to_numeric(friendly["Total capacity (MW)"], errors="coerce").round(0)

    ordered_cols = [c for c in ["Country", "Technology", "Total capacity (MW)"] if c in friendly.columns]
    tail = [c for c in friendly.columns if c not in ordered_cols]
    friendly = friendly[ordered_cols + tail]

    return friendly.to_markdown(index=False, floatfmt=".0f")


def format_generation_summary_per_country(df: Optional[pd.DataFrame]) -> str:
    """Create per-country capacity tables with totals, plus an overall total table."""
    if df is None or df.empty:
        return ""

    lines = []
    country_col = "country" if "country" in df.columns else "Country"
    for country, group in df.groupby(country_col):
        friendly = group.copy()
        friendly = friendly.rename(columns={"technology": "Technology", "status": "Status"})
        friendly = friendly.drop(columns=[c for c in ["avg_capacity_mw", "site_count", country_col] if c in friendly.columns], errors="ignore")
        friendly = friendly.rename(columns={"capacity_mw": "Total capacity (MW)", "total_capacity_mw": "Total capacity (MW)"})
        if "Total capacity (MW)" in friendly.columns:
            friendly["Total capacity (MW)"] = friendly["Total capacity (MW)"].round(0)
        ordered_cols = [c for c in ["Technology", "Status", "Total capacity (MW)"] if c in friendly.columns]
        tail = [c for c in friendly.columns if c not in ordered_cols]
        friendly = friendly[ordered_cols + tail]
        if "Total capacity (MW)" in friendly.columns:
            total_row = {"Technology": "Total", "Total capacity (MW)": friendly["Total capacity (MW)"].sum()}
            if "Status" in friendly.columns:
                total_row["Status"] = ""
            friendly = pd.concat([friendly, pd.DataFrame([total_row])], ignore_index=True)
        title = country if isinstance(country, str) else str(country)
        lines.append(f"#### {title}\n" + _wrap_table(friendly.to_markdown(index=False, floatfmt=".0f")))

    overall = df.copy()
    overall = overall.rename(columns={"technology": "Technology", "capacity_mw": "Total capacity (MW)", "total_capacity_mw": "Total capacity (MW)"})
    if "Total capacity (MW)" in overall.columns:
        overall["Total capacity (MW)"] = overall["Total capacity (MW)"].round(0)
    overall_summary = overall.groupby("Technology")["Total capacity (MW)"].sum().reset_index().sort_values("Technology")
    overall_total = pd.DataFrame([{"Technology": "Total", "Total capacity (MW)": overall_summary["Total capacity (MW)"].sum()}])
    overall_table = pd.concat([overall_summary, overall_total], ignore_index=True)
    lines.append("#### All countries\n" + _wrap_table(overall_table.to_markdown(index=False, floatfmt=".0f")))

    return "\n\n".join(lines)


def _categorize_status(status: str) -> str:
    """Group raw statuses into three robust buckets."""
    text = str(status or "").strip().lower()
    normalized = " ".join(text.replace("_", " ").replace("-", " ").split())

    if normalized in {"", "nan", "none", "na", "n/a"}:
        return "Announced / pre-construction"

    def _has(tokens):
        return any(tok in normalized for tok in tokens)

    if _has(["shelv", "cancel", "retir", "decomm", "mothball", "abandon", "suspend", "halt"]):
        return "Shelved / cancelled"
    if _has(["operat", "running", "commission", "in service", "active", "existing"]):
        return "Operating"
    if _has(["construct", "announ", "plan", "pre construction", "preconstruction", "proposal", "feasib", "develop", "permit", "license", "licence", "committed", "unknown"]):
        return "Announced / pre-construction"

    # Default to the middle bucket so unexpected labels still show up.
    return "Announced / pre-construction" if not normalized else "Other"


def _short_name(row: pd.Series, max_len: int = 48) -> str:
    """Pick a concise plant name from available columns."""
    for col in [
        "name",
        "Plant / Project name (other)",
        "Plant / Project name (local)",
        "Plant Name",
        "Plant / Project name",
    ]:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            candidate = " ".join(val.strip().split())
            return candidate if len(candidate) <= max_len else candidate[: max_len - 3].rstrip() + "..."
    return "(unnamed)"


def format_generation_sites_by_status(df: Optional[pd.DataFrame]) -> str:
    """Create plant-level tables per country grouped into three status categories."""
    if df is None or df.empty:
        return ""

    working = df.copy()
    working = working.rename(columns={"country": "Country", "technology": "Technology", "status": "Status"})
    if "Country" not in working:
        working["Country"] = ""
    if "Technology" not in working:
        working["Technology"] = "Unknown"
    if "Status" not in working:
        working["Status"] = "Unknown"

    working["Plant"] = working.apply(_short_name, axis=1)
    working["name"] = working.get("name", "")
    working["Generator"] = working["name"].apply(lambda val: " ".join(str(val).split()) if isinstance(val, str) else "")
    working["Generator"] = working["Generator"].where(working["Generator"] != "", working["Plant"])
    working["Capacity (MW)"] = pd.to_numeric(working.get("capacity_mw"), errors="coerce")
    working["Category"] = working["Status"].apply(_categorize_status)

    category_order = ["Operating", "Announced / pre-construction", "Shelved / cancelled"]
    lines: List[str] = []

    for country, group in working.groupby("Country"):
        lines.append(f"#### {country}")
        for category in category_order:
            subset = group[group["Category"] == category].copy()
            if subset.empty:
                continue
            subset = subset.sort_values(["Technology", "Capacity (MW)"], ascending=[True, False])
            table_cols = ["Technology", "Generator", "Status", "Capacity (MW)"]
            subset_table = subset[table_cols]
            lines.append(f"**{category}**\n" + _wrap_table(subset_table.to_markdown(index=False, floatfmt=".0f")))

        other = group[~group["Category"].isin(category_order)]
        if not other.empty:
            other = other.sort_values(["Technology", "Capacity (MW)"] , ascending=[True, False])
            lines.append(
                "**Other**\n"
                + _wrap_table(other[["Technology", "Generator", "Status", "Capacity (MW)"]].to_markdown(index=False, floatfmt=".0f"))
            )

    return "\n\n".join(lines)


def summarize_parameters(config: Dict, seasons_override: Optional[Dict[int, int]] = None) -> Dict[str, str]:
    """Return human-readable key parameters and season mapping."""
    lines: List[str] = []
    rn_cfg = config.get("rninja", {})
    if rn_cfg:
        rn_start = rn_cfg.get("start_year")
        rn_end = rn_cfg.get("end_year")
        if rn_end is not None:
            rn_end -= 1  # exclusive upper bound in config
        rn_period = f"{rn_start}–{rn_end}" if rn_start and rn_end else rn_start or rn_end or ""
        lines.append(
            f"- Renewables Ninja: dataset `{rn_cfg.get('dataset', 'n/a')}`, years {rn_period}, "
            f"height {rn_cfg.get('height', 'n/a')} m, tilt {rn_cfg.get('tilt', 'n/a')}°, "
            f"azimuth {rn_cfg.get('azim', 'n/a')}°, tracking {rn_cfg.get('tracking', 'n/a')}, "
            f"system loss {rn_cfg.get('system_loss', 'n/a')*100 if isinstance(rn_cfg.get('system_loss'), (int, float)) else rn_cfg.get('system_loss')}%, "
            f"turbine `{rn_cfg.get('turbine', 'n/a')}`."
        )

    rep_cfg = config.get("representative_days", {})
    seasons_map = seasons_override or rep_cfg.get("seasons_map", {})
    season_buckets: Dict[str, List[str]] = {}
    for month, bucket in seasons_map.items():
        label = f"Season {bucket}"
        season_buckets.setdefault(label, []).append(calendar.month_abbr[int(month)])
    seasons_text = "; ".join(f"{season}: {', '.join(sorted(months))}" for season, months in sorted(season_buckets.items()))
    if rep_cfg:
        lines.append(
            f"- Representative days: {rep_cfg.get('n_representative_days', 'n/a')} days from {rep_cfg.get('n_clusters', 'n/a')} clusters; "
            f"{rep_cfg.get('n_bins', 'n/a')} bins in optimisation."
        )

    return {"bullets": "\n".join(lines), "seasons": seasons_text}


def build_repro_checklist(
    workflow_path: Path,
    config_path: Path,
    output_dir: Path,
    workflow_command: str,
    report_command: str,
) -> str:
    """Create a reproducibility checklist block that can live in the appendix."""
    return "\n".join(
        [
            "- This project is built on the open-source **EPM** stack: [GitHub](https://github.com/ESMAP-World-Bank-Group/EPM) · [Docs](https://esmap-world-bank-group.github.io/EPM/home.html).",
            "- First clone EPM and follow the docs to set up the environment; then run from this repo (paths below are relative to the EPM folder, e.g., `EPM_WestBalkans`).",
            f"- Workflow: `{_relpath_for_display(workflow_path)}` using config `{_relpath_for_display(config_path)}`; outputs stored in `{_relpath_for_display(output_dir)}`.",
            "- Re-run full open-data workflow (regenerates all inputs):",
            f"  `{workflow_command}`",
            "- Regenerate this report only:",
            f"  `{report_command}`",
            "- Archive the rendered report, the config YAML, and the `output_workflow` directory together; timestamps in the tables reflect extraction dates of the raw open datasets.",
        ]
    )


def build_appendix_sections(
    repro_checklist: str,
    rninja_summary: List[Dict],
    gen_status_tables: str,
    boxplots: List[Path],
    gap_plot: Optional[Path] = None,
    generation_map_all_path: Optional[Path] = None,
    generation_map_all_static_figs: str = "",
    hydro_appendix_body: str = "",
    workflow_parameters: str = "",
    season_mapping: str = "",
    extra_sections: Sequence[Dict[str, str]] = (),
    rninja_period_label: str = "",
    rep_days_body: str = "",
    rep_days_fig: str = "",
    socio_map_fig: str = "",
    load_heatmaps: Sequence[Path] = (),
    load_boxplots: Sequence[Path] = (),
) -> List[Dict[str, str]]:
    """Assemble appendix sections with headings and content (numbered A, B, C, ...)."""
    sections: List[Dict[str, str]] = []
    letter = ord("A")

    def add_section(title: str, body: str):
        nonlocal letter
        if not body:
            return
        sections.append({"title": f"Appendix {chr(letter)}: {title}", "body": body})
        letter += 1

    # Order mirrors the main narrative: Climate → Load → VRE → Generation → Hydro → Socio → Rep days → Repro.
    # A. Climate (extras)
    if extra_sections:
        climate_body = "\n\n".join(section.get("body", "") for section in extra_sections if section.get("body"))
        add_section("Climate", climate_body)

    # B. Load diagnostics
    load_parts: List[str] = []
    load_heatmap_fig = format_figures(load_heatmaps, "Load heatmap", add_link=False) if load_heatmaps else ""
    load_boxplot_fig = format_figures(load_boxplots, "Load distribution", add_link=False) if load_boxplots else ""
    for fig in (load_heatmap_fig, load_boxplot_fig):
        if fig:
            load_parts.append(fig)
    add_section("Load diagnostics", "\n\n".join(load_parts))

    # C. VRE (boxplots or other diagnostics)
    if boxplots:
        add_section("VRE diagnostics", format_figures(boxplots, "Capacity-factor distribution", add_link=False))

    # D. Generation (assets + map)
    gen_body_parts: List[str] = []
    if gen_status_tables:
        gen_body_parts.append(gen_status_tables)
    map_lines: List[str] = []
    if generation_map_all_path:
        rel = _relpath_for_display(generation_map_all_path)
        map_lines.append(f"Interactive generation map (all statuses): [{generation_map_all_path.name}]({rel})")
    if generation_map_all_static_figs:
        map_lines.append(generation_map_all_static_figs)
    if map_lines:
        gen_body_parts.append("\n\n".join(map_lines))
    add_section("Generation assets and map", "\n\n".join(part for part in gen_body_parts if part))

    # E. Hydropower assets
    add_section("Hydropower assets", hydro_appendix_body)

    # F. Socio-economic maps
    add_section("Socio-economic density maps", socio_map_fig)

    # G. Representative days
    rep_day_parts: List[str] = []
    if rep_days_body:
        rep_day_parts.append(rep_days_body)
    if rep_days_fig:
        rep_day_parts.append(rep_days_fig)
    add_section("Representative days", "\n\n".join(rep_day_parts))

    # H. Reproducibility + workflow parameters
    repro_parts: List[str] = []
    if workflow_parameters:
        body = workflow_parameters
        if season_mapping:
            body = f"{body}\n\n- Season grouping: {season_mapping}"
        repro_parts.append(body)
    if repro_checklist:
        repro_parts.append(repro_checklist)
    add_section("Reproducibility and workflow", "\n\n".join(repro_parts))

    return sections


def render_report(
    template_path: Path,
    output_path: Path,
    config_path: Path,
    output_dir_override: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    _vprint(verbose, f"Template: {_abspath_for_display(template_path)}")
    _vprint(verbose, f"Config: {_abspath_for_display(config_path)}")
    _vprint(verbose, f"Markdown output: {_abspath_for_display(output_path)}")
    if output_dir_override:
        _vprint(verbose, f"Output dir override: {_abspath_for_display(output_dir_override)}")

    config = load_config(config_path)
    _vprint(verbose, f"Loaded config keys: {sorted(config.keys())}" if config else "Config empty or missing.")
    slug_map = {_slug(name): name for name in config.get("gap", {}).get("countries", [])}

    output_dir = find_output_dir(config, output_dir_override)
    _vprint(verbose, f"Resolved workflow output dir: {_abspath_for_display(output_dir)}")

    load_cfg = config.get("load_profile", {})
    rninja_cfg = config.get("rninja", {})
    genmap_cfg = config.get("generation_map", {})
    socio_cfg = config.get("socioeconomic_maps", {})
    owid_cfg = config.get("owid_energy", {})
    load_dir = _resolve_category_output_dir(output_dir, load_cfg.get("output_dir"), "load")
    vre_dir = _resolve_category_output_dir(output_dir, rninja_cfg.get("output_dir"), "vre")
    supply_dir = _resolve_category_output_dir(output_dir, genmap_cfg.get("output_dir"), "supply")
    socio_dir = _resolve_category_output_dir(output_dir, socio_cfg.get("output_dir"), "socioeconomic")
    owid_dir = _resolve_category_output_dir(output_dir, owid_cfg.get("output_dir"), "socioeconomic")
    _vprint(verbose, f"Load outputs: {_abspath_for_display(load_dir)}")
    _vprint(verbose, f"VRE outputs: {_abspath_for_display(vre_dir)}")
    _vprint(verbose, f"Supply outputs: {_abspath_for_display(supply_dir)}")
    _vprint(verbose, f"Socio-economic maps: {_abspath_for_display(socio_dir)}")
    _vprint(verbose, f"OWID energy outputs: {_abspath_for_display(owid_dir)}")

    climate = collect_climate_overview(output_dir, config)
    load_stats, load_figs, load_heatmaps, load_boxplots = collect_load_profiles(load_dir, slug_map)
    load_agg_figs = build_load_aggregate_plots(load_dir, slug_map)
    rninja_summary, rninja_zone_figures, boxplots = collect_rninja_profiles(vre_dir)
    rninja_avg_cf_table = ""
    rninja_tech_display = {"solar": "Solar PV", "wind": "Wind"}
    rninja_zone_sections: List[Dict[str, object]] = []
    for tech_key in ("solar", "wind"):
        entries = rninja_zone_figures.get(tech_key, [])
        if not entries:
            continue
        label = rninja_tech_display.get(tech_key, tech_key.title())
        zone_rows: List[Dict[str, str]] = []
        for entry in entries:
            heatmap_md = (
                format_figures(
                    [entry["heatmap"]],
                    f"Capacity-factor heatmap — {label} — {entry['country']}",
                    add_link=False,
                )
                if entry.get("heatmap")
                else ""
            )
            boxplot_md = (
                format_figures(
                    [entry["boxplot"]],
                    f"Capacity-factor distribution — {label} — {entry['country']}",
                    add_link=False,
                )
                if entry.get("boxplot")
                else ""
            )
            if not heatmap_md and not boxplot_md:
                continue
            zone_rows.append(
                {
                    "country": entry["country"],
                    "heatmap": heatmap_md,
                    "boxplot": boxplot_md,
                }
            )
        if zone_rows:
            rninja_zone_sections.append({"tech": tech_key, "label": label, "zones": zone_rows})
    gap_summary, gap_path, gap_plot = collect_gap_projects(supply_dir, slug_map)
    gen_map_path, gen_map_all_path, gen_summary_df, gen_sites_df = collect_generation_map(supply_dir, config)
    (
        hydro_map_path,
        hydro_summary_df,
        hydro_data_df,
        hydro_summary_path,
        hydro_data_path,
    ) = collect_hydro_reservoirs(output_dir, config)
    owid_energy = collect_owid_energy(output_dir, config)
    static_pdfs = sorted(supply_dir.glob("generation_map_static_*.pdf"))
    static_all_files = [path for path in static_pdfs if path.stem.lower().endswith("_all")]
    static_main_files = [path for path in static_pdfs if path not in static_all_files]
    static_map_files = [*static_main_files, *static_all_files]
    static_map_figs = format_figures(static_main_files, "Generation map static export", add_link=False)
    static_map_all_figs = format_figures(
        static_all_files,
        "Generation map static export (all statuses)",
        add_link=False,
    )
    def _load_socio_status(folder: Path) -> Dict[str, str]:
        status_path = folder / "socioeconomic_status.json"
        if not status_path.exists():
            return {}
        try:
            entries = json.loads(status_path.read_text(encoding="utf-8"))
            return {entry.get("basename"): entry.get("message", "") for entry in entries if isinstance(entry, dict)}
        except Exception as exc:
            print(f"Warning: could not read socio status {status_path}: {exc}", file=sys.stderr)
            return {}

    socio_status = _load_socio_status(socio_dir)

    socio_entries: List[Tuple[str, Path, str]] = []
    socio_warnings: List[str] = []
    if socio_cfg.get("enabled", False):
        socio_datasets = socio_cfg.get("datasets") or []
        default_basename = socio_cfg.get("output_basename")
        for entry in socio_datasets:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label") or entry.get("title_prefix") or entry.get("key") or entry.get("name") or "Socio-economic map"
            basename = entry.get("output_basename") or default_basename or f"{_slug(str(entry.get('key') or label))}_map"
            socio_entries.append((str(label), socio_dir / f"{basename}.pdf", basename))
            if basename in socio_status and socio_status[basename]:
                socio_warnings.append(f"- {label}: {socio_status[basename]}")
    socio_files = [path for _, path, _ in socio_entries]
    socio_map_fig = "\n\n".join(
        format_figures([path], label, add_link=False)
        for label, path, _ in socio_entries
        if path
    )
    if socio_warnings:
        warning_lines = "\n".join(socio_warnings)
        notice = f"_Note: socio-economic maps encountered issues:_\n{warning_lines}"
        socio_map_fig = f"{socio_map_fig}\n\n{notice}" if socio_map_fig else notice
    owid_population_fig = format_figures(
        [owid_energy["figures"].get("population")] if owid_energy.get("figures") else [],
        "Population (OWID)",
        add_link=False,
    )
    owid_gdp_fig = format_figures(
        [owid_energy["figures"].get("gdp")] if owid_energy.get("figures") else [],
        "GDP (OWID)",
        add_link=False,
    )
    owid_electricity_fig = format_figures(
        [owid_energy["figures"].get("electricity")] if owid_energy.get("figures") else [],
        "Electricity demand (OWID)",
        add_link=False,
    )
    owid_electricity_pc_fig = format_figures(
        [owid_energy["figures"].get("electricity_per_capita")] if owid_energy.get("figures") else [],
        "Per-capita electricity demand & total energy (OWID)",
        add_link=False,
    )
    owid_summary_table = owid_energy.get("summary", "")
    hydro_map_fig = format_figures([hydro_map_path], "Hydropower map", add_link=False) if hydro_map_path else ""
    hydro_summary_table = format_hydro_summary(hydro_summary_df)
    hydro_appendix_tables = format_hydro_appendix(hydro_data_df)
    rep_figs = find_rep_day_figures(BASE_DIR, output_dir)
    rep_days_fig_md = format_figures(rep_figs, "Representative days")
    rep_summary_table, rep_summary_path = load_representative_days_summary(output_dir, config)
    rep_seasons_table, rep_seasons_map_path, rep_seasons_figs, rep_seasons_map = load_representative_seasons(
        output_dir, config
    )
    rep_days_appendix_body = ""
    if rep_summary_table:
        rep_days_appendix_body = rep_summary_table
        if rep_summary_path:
            rep_days_appendix_body = f"{rep_days_appendix_body}\n\n_Source: {rep_summary_path}_"

    period_candidates = [entry["period"] for entry in rninja_summary if "period" in entry]
    period_numbers = []
    for value in period_candidates:
        try:
            period_numbers.append(int(str(value)))
        except (TypeError, ValueError):
            continue

    rn_cfg = config.get("rninja", {})
    rn_start = rn_cfg.get("start_year")
    rn_end = rn_cfg.get("end_year")
    if rn_end is not None:
        rn_end = rn_end - 1  # end_year is exclusive in the workflow
    if rn_start is None and period_numbers:
        rn_start = min(period_numbers)
    if rn_end is None and period_numbers:
        rn_end = max(period_numbers)

    params = summarize_parameters(config, seasons_override=rep_seasons_map or None)
    season_map_desc = params.get("seasons", "")
    parameter_summary = params.get("bullets", "")
    rep_cfg = config.get("representative_days", {}) or {}
    rep_days_count = rep_cfg.get("n_representative_days") or ""
    rep_days_clusters = rep_cfg.get("n_clusters") or ""
    rep_days_bins = rep_cfg.get("n_bins") or ""

    countries_cfg = config.get("gap", {}).get("countries")
    countries = _ensure_sequence(countries_cfg) if countries_cfg else []
    if not countries:
        # Fallback to climate countries if GAP countries are absent.
        clim_countries = climate.get("countries")
        if clim_countries:
            countries = _ensure_sequence(clim_countries)
    countries_text = "\n".join(f"- {name}" for name in countries) if countries else ""
    countries_inline = _human_join(countries)

    load_summary_table = ""
    if load_stats:
        df_load = pd.DataFrame(load_stats)
        load_summary_table = df_load.to_markdown(index=False, floatfmt=".0f")

    gen_summary_country = format_generation_summary_per_country(gen_summary_df)
    gen_sites_by_status = format_generation_sites_by_status(gen_sites_df)
    gen_appendix_tables = gen_sites_by_status or gen_summary_country

    load_csvs = sorted(load_dir.glob("load_profile_*.csv"))
    rninja_csvs = sorted(vre_dir.glob("rninja_data_*.csv"))
    if not rninja_csvs:
        rninja_csvs = sorted(vre_dir.glob("vre_rninja_*.csv"))
    if not rninja_csvs:
        rninja_csvs = sorted(output_dir.glob("rninja_data_*.csv"))
    if not rninja_csvs:
        rninja_csvs = sorted(output_dir.glob("vre_rninja_*.csv"))
    gen_files = [p for p in (gen_map_path, gen_map_all_path) if p]
    gen_files.extend([supply_dir / "generation_sites_summary.csv", supply_dir / "generation_sites.csv"])
    gen_files.extend(static_map_files)
    hydro_files = [p for p in (hydro_map_path, hydro_summary_path, hydro_data_path) if p]
    socio_files = socio_files or []

    rn_period_label = ""
    if rn_start and rn_end:
        rn_period_label = f"Jan–Dec {rn_start}" if rn_start == rn_end else f"Jan–Dec {rn_start} – {rn_end}"
    elif rn_start:
        rn_period_label = f"Jan–Dec {rn_start}"
    elif rn_end:
        rn_period_label = f"Jan–Dec {rn_end}"

    if rninja_summary:
        df_rn = pd.DataFrame(rninja_summary)
        tech_labels = {"solar": "Solar PV", "wind": "Wind"}
        df_rn_wide = (
            df_rn.groupby(["zone", "tech"])["mean_capacity_factor"]
            .mean()
            .unstack("tech")
            .rename(columns=tech_labels)
            .reset_index()
        )
        column_order = ["zone"] + [label for label in tech_labels.values() if label in df_rn_wide.columns]
        df_rn_wide = df_rn_wide[column_order]
        df_rn_wide = df_rn_wide.rename(columns={"zone": "Country"})
        for col in df_rn_wide.columns:
            if col == "Country":
                continue
            df_rn_wide[col] = df_rn_wide[col].apply(lambda v: "-" if pd.isna(v) else f"{float(v)*100:.0f}%")
        rninja_period_text = f"Average capacity factors ({rn_period_label})" if rn_period_label else "Average capacity factors"
        rninja_avg_cf_table = f"**{rninja_period_text}**\n" + _wrap_table(df_rn_wide.to_markdown(index=False))

    workflow_path = BASE_DIR / "Snakefile"
    report_script = BASE_DIR / "generate_report.py"
    workflow_path_rel = _relpath_for_display(workflow_path)
    report_script_rel = _relpath_for_display(report_script)
    config_label = _relpath_for_display(config_path)
    output_dir_label = _relpath_for_display(output_dir)
    workflow_command = f"snakemake -s {shlex.quote(workflow_path_rel)} --cores 4"
    report_command = (
        f"python {shlex.quote(report_script_rel)}"
        f" --config {shlex.quote(_relpath_for_display(config_path))}"
        f" --output-dir {shlex.quote(_relpath_for_display(output_dir))}"
        f" --output {shlex.quote(_relpath_for_display(output_path))}"
    )
    repro_checklist_md = build_repro_checklist(workflow_path, config_path, output_dir, workflow_command, report_command)

    data_overview_rows = [
        {
            "Dataset": "Socio-economic trends (OWID energy)",
            "Resolution": _resolution_with_date("Annual", owid_energy.get("data_files", [])),
            "Source": "[Our World in Data — Energy](https://ourworldindata.org/energy)",
        },
        {
            "Dataset": "Socio-economic density rasters",
            "Resolution": _resolution_with_date("Gridded (1 km)", socio_files),
            "Source": "User-provided rasters configured under socioeconomic_maps (e.g., GDP 2005 PPP, population 2020).",
        },
        {
            "Dataset": "Climate (ERA5-Land monthly)",
            "Resolution": _resolution_with_date(f"Monthly ({climate['period'] or 'N/A'})", climate["data_files"]),
            "Source": "[ERA5-Land monthly means](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means)",
        },
        {
            "Dataset": "Load profiles",
            "Resolution": _resolution_with_date("Hourly", load_csvs),
            "Source": "[Open data for electricity modelling (Toktarova 2019)](https://www.sciencedirect.com/science/article/abs/pii/S0142061518336196)",
        },
        {
            "Dataset": "Renewables Ninja capacity factors",
            "Resolution": _resolution_with_date(f"Hourly ({rn_period_label or 'N/A'})", rninja_csvs),
            "Source": "[Renewables.ninja](https://www.renewables.ninja/)",
        },
        {
            "Dataset": "Generation assets",
            "Resolution": _resolution_with_date("Plant-level", gen_files),
            "Source": "[Global Integrated Power (GIP) 2025](https://datasets.wri.org/dataset/globalpowerplantdatabase)",
        },
        {
            "Dataset": "Hydropower reservoirs",
            "Resolution": _resolution_with_date("Plant-level", hydro_files),
            "Source": "[Global hydropower infrastructure (Sci Data 2025)](https://www.nature.com/articles/s41597-025-04975-0#Sec5) ([Zenodo](https://zenodo.org/records/14526360))",
        },
    ]
    data_overview_table = pd.DataFrame(data_overview_rows).to_markdown(index=False)

    climate_spatial_fig = format_figures(climate.get("figures", {}).get("spatial", []), "Spatial mean climate", add_link=False)
    climate_monthly_fig = format_figures(climate.get("figures", {}).get("monthly", []), "Monthly climate averages", add_link=False)
    climate_precip_heatmap = format_figures(
        climate.get("figures", {}).get("heatmap_precipitation", []),
        "Monthly precipitation heatmap",
        add_link=False,
    )
    climate_temperature_heatmap = format_figures(
        climate.get("figures", {}).get("heatmap_temperature", []),
        "Monthly temperature heatmap",
        add_link=False,
    )
    climate_scatter_fig = format_figures(climate.get("figures", {}).get("scatter", []), "Temperature vs precipitation", add_link=False)

    climate_appendix_sections: List[Dict[str, str]] = []
    climate_appendix_lines: List[str] = []
    if climate_monthly_fig:
        climate_appendix_lines.append("#### Monthly averages by country\n" + climate_monthly_fig)
    if climate_spatial_fig:
        climate_appendix_lines.append("#### Spatial diagnostics\n" + climate_spatial_fig)
    if climate_scatter_fig:
        climate_appendix_lines.append("#### Temperature vs precipitation diagnostics\n" + climate_scatter_fig)
    if climate_appendix_lines:
        climate_appendix_sections.append(
            {"title": "Climate diagnostics", "body": "\n\n".join(climate_appendix_lines)}
        )

    appendix_sections = build_appendix_sections(
        repro_checklist_md,
        rninja_summary,
        gen_appendix_tables,
        boxplots,
        gap_plot,
        generation_map_all_path=gen_map_all_path,
        generation_map_all_static_figs=static_map_all_figs,
        hydro_appendix_body=hydro_appendix_tables,
        workflow_parameters=parameter_summary,
        season_mapping=season_map_desc,
        extra_sections=climate_appendix_sections,
        rninja_period_label=rn_period_label,
        rep_days_body=rep_days_appendix_body,
        socio_map_fig=socio_map_fig,
        load_heatmaps=load_heatmaps,
        load_boxplots=load_boxplots,
    )

    report_scope = countries_inline or "Energy System Modelling"

    context = {
        "date": str(date.today()),
        "report_scope": report_scope,
        "countries_inline": countries_inline,
        "countries_text": countries_text,
        "rn_countries_inline": countries_inline,
        "climate_summary": climate.get("summary", ""),
        "climate_spatial_fig": climate_spatial_fig,
        "climate_monthly_fig": climate_monthly_fig,
        "climate_precip_heatmap": climate_precip_heatmap,
        "climate_temperature_heatmap": climate_temperature_heatmap,
        "climate_scatter_fig": climate_scatter_fig,
        "climate_countries": climate.get("countries", ""),
        "climate_period": climate.get("period", ""),
        "owid_population_fig": owid_population_fig,
        "owid_gdp_fig": owid_gdp_fig,
        "owid_electricity_fig": owid_electricity_fig,
        "owid_electricity_pc_fig": owid_electricity_pc_fig,
        "owid_energy_summary": owid_summary_table,
        "load_profile_fig": format_figures(load_figs, "Load profile", add_link=False),
        "load_month_fig": format_figures([load_agg_figs["avg_month"]], "Average load per month", add_link=False) if "avg_month" in load_agg_figs else "",
        "load_day_fig": format_figures([load_agg_figs["avg_day"]], "Average load per day", add_link=False) if "avg_day" in load_agg_figs else "",
        "load_heatmap_fig": format_figures(load_heatmaps, "Load heatmap", add_link=False),
        "load_boxplot_fig": format_figures(load_boxplots, "Load distribution", add_link=False),
        "load_profile_summary": load_summary_table,
        "vre_location_plot": format_figures([gap_plot], "GIP project picks", add_link=False) if gap_plot else "",
        "vre_location_map": (
            gap_summary
            if gap_summary and gap_path
            else ""
        ),
        "generation_map_text": (
            f"- Interactive generation map: [{gen_map_path.name}]({gen_map_path.relative_to(BASE_DIR).as_posix()})"
            if gen_map_path
            else ""
        ),
        "generation_map_static_figs": static_map_figs,
        "generation_map_summary": gen_summary_country,
        "generation_map_summary_appendix": gen_appendix_tables,
        "hydro_map_fig": hydro_map_fig,
        "hydro_summary_table": hydro_summary_table,
        "hydro_appendix_table": hydro_appendix_tables,
        "socio_map_fig": socio_map_fig,
        "rn_period_start": rn_start or "",
        "rn_period_end": rn_end or "",
        "rn_period_months": "Jan–Dec",
        "rn_period_label": rn_period_label,
        "rninja_avg_cf_table": rninja_avg_cf_table,
        "rninja_zone_sections": rninja_zone_sections,
        "rep_days_fig": rep_days_fig_md,
        "rep_days_summary_table": rep_summary_table,
        "rep_days_summary_path": rep_summary_path,
        "rep_days_count": rep_days_count,
        "rep_days_clusters": rep_days_clusters,
        "rep_days_bins": rep_days_bins,
        "rep_seasons_table": rep_seasons_table,
        "rep_seasons_fig": format_figures(
            rep_seasons_figs,
            "Capacity factors by representative seasons",
            add_link=False,
        ),
        "rep_seasons_map_path": rep_seasons_map_path,
        "rep_season_map": season_map_desc,
        "appendix_sections": appendix_sections,
        "data_overview": data_overview_table,
        "workflow_path": _relpath_for_display(workflow_path),
        "config_path_display": config_label,
        "output_dir_display": output_dir_label,
        "workflow_command": workflow_command,
        "report_command": report_command,
        "repro_checklist": repro_checklist_md,
    }

    template = Template(template_path.read_text())
    rendered = template.render(**context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render pre-analysis/report.md from a Jinja template.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="Path to the Jinja2 template.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config used by the workflow.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the rendered Markdown report (defaults to <output_workflow_dir>/report.md).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the workflow output directory (defaults to config output_workflow_dir).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information about resolved input/output paths.",
    )
    return parser.parse_args()


def export_report_variants(
    markdown_path: Path,
    pdf_path: Optional[Path] = None,
    docx_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Export a rendered Markdown report to PDF and DOCX via Pandoc."""

    def _run_pandoc(target: Path, args: Sequence[str], label: str) -> Optional[Path]:
        if not shutil.which("pandoc"):
            print(f"Warning: pandoc not found; skipping {label} export.", file=sys.stderr)
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Use BASE_DIR as the working directory so relative figure paths resolve.
            subprocess.run(args, check=True, cwd=BASE_DIR)
        except subprocess.CalledProcessError as exc:
            print(f"Warning: failed to export {label}: {exc}", file=sys.stderr)
            return None
        return target.resolve()

    outputs: Dict[str, Path] = {}
    pdf_target = pdf_path or markdown_path.with_suffix(".pdf")
    docx_target = docx_path or markdown_path.with_suffix(".docx")

    if pdf_target:
        pdf_args = [
            "pandoc",
            "--from=markdown",
            "--to=pdf",
            "-V",
            "geometry:margin=1.5cm",
            "-V",
            "float-placement=H",
            "-H",
            str(DISABLE_FLOAT_TEX),
            "--output",
            str(pdf_target),
            str(markdown_path),
        ]
        pdf_output = _run_pandoc(pdf_target, pdf_args, "PDF")
        if pdf_output:
            outputs["pdf"] = pdf_output

    if docx_target:
        docx_args = [
            "pandoc",
            "--from=gfm",
            "--to=docx",
            "--output",
            str(docx_target),
            str(markdown_path),
        ]
        docx_output = _run_pandoc(docx_target, docx_args, "DOCX")
        if docx_output:
            outputs["docx"] = docx_output

    return outputs


if __name__ == "__main__":
    
    # User-editable defaults for IDE runs (no CLI flags needed)
    USER_TEMPLATE: Path = DEFAULT_TEMPLATE
    USER_CONFIG: Path = DEFAULT_CONFIG
    USER_OUTPUT: Optional[Path] = None  # Set to a path to override report.md
    USER_OUTPUT_DIR_OVERRIDE: Optional[Path] = None  # Set to override workflow output root
    USER_VERBOSE: bool = True
    
    template_path = USER_TEMPLATE
    config_path = USER_CONFIG
    output_dir_override = USER_OUTPUT_DIR_OVERRIDE
    output_path = USER_OUTPUT or default_report_output(config_path, output_dir_override)

    _vprint(USER_VERBOSE, f"Final Markdown target: {_abspath_for_display(output_path)}")
    render_report(template_path, output_path, config_path, output_dir_override, verbose=USER_VERBOSE)

    exports = export_report_variants(output_path)
    display_md = _relpath_for_display(output_path.resolve())
    messages = [f"Markdown: {display_md}"]
    if "pdf" in exports:
        messages.append(f"PDF: {_relpath_for_display(exports['pdf'])}")
    if "docx" in exports:
        messages.append(f"DOCX: {_relpath_for_display(exports['docx'])}")
    print("[report] " + "; ".join(messages))
