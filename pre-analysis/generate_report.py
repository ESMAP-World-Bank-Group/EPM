#!/usr/bin/env python3
"""Render an automatic Markdown report from workflow outputs."""

from __future__ import annotations

import argparse
import calendar
import os
import shlex
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
    """Append extraction date to a resolution label when file timestamps exist."""
    ts = _latest_mtime(paths)
    if not ts:
        return resolution
    return f"{resolution} (extracted {ts.date().isoformat()})"


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

    for csv_path in sorted(vre_dir.glob("rninja_data_*.csv")):
        tech = "solar" if "solar" in csv_path.stem.lower() or "_pv_" in csv_path.stem.lower() else "wind"
        df = pd.read_csv(csv_path)
        value_cols = [c for c in df.columns if c not in {"zone", "season", "day", "hour"}]
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
            if not data or (data.get("heatmap") is None and data.get("boxplot") is None):
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
            if data.get("heatmap") is None and data.get("boxplot") is None:
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
    plot_path = output_dir / "gap_project_locations.png"
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


def collect_climate_overview(output_root: Path, config: Dict) -> Dict[str, object]:
    cfg = config.get("climate_overview", {})
    if not cfg.get("enabled", False):
        return {"summary": "", "figures": {}, "period": "", "data_files": [], "countries": ""}

    outdir = _resolve_relative(output_root, cfg.get("output_dir", "climate"))
    summary_md = ""
    summary_path = outdir / "climate_summary.csv"
    if summary_path.exists():
        try:
            summary_md = pd.read_csv(summary_path).to_markdown(index=False, floatfmt=".1f")
        except Exception as exc:  # pragma: no cover - defensive logging only
            print(f"Warning: could not read climate summary {summary_path}: {exc}", file=sys.stderr)

    figures = {
        "spatial": sorted(outdir.glob("spatial_mean_*.pdf")),
        "monthly": sorted(outdir.glob("monthly_mean_*.pdf")),
        "heatmap": sorted(outdir.glob("monthly_precipitation_heatmap.*")),
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


def find_rep_day_figures(base_dir: Path) -> List[Path]:
    rep_dir = base_dir / "prepare-data" / "representative_days" / "output"
    if not rep_dir.exists():
        return []
    figs: List[Path] = []
    for ext in ("*.pdf", "*.png"):
        figs.extend(sorted(rep_dir.glob(ext)))
    return figs


def _is_valid_figure(path: Path) -> bool:
    """Return True when a figure file exists and has content to avoid empty placeholders."""
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
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
        lines.append(f"![{label}]({rel.as_posix()})")

    if not lines:
        return ""
    return "\n\n".join(lines)


def _wrap_table(table_md: str) -> str:
    """Wrap markdown tables in a scrollable div for better rendering."""
    if not table_md:
        return ""
    return f'<div style="overflow-x: auto;">\n\n{table_md}\n\n</div>'


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
        path = load_dir / "load_avg_month.png"
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
        path = load_dir / "load_avg_day.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figures["avg_day"] = path

    return figures


def format_generation_summary(df: Optional[pd.DataFrame]) -> str:
    """Create a readable markdown table for generation-map summaries."""
    if df is None or df.empty:
        return ""

    friendly = df.copy()
    friendly = friendly.drop(columns=[c for c in ["site_count", "avg_capacity_mw"] if c in friendly.columns])

    rename = {
        "country": "Country",
        "technology": "Technology",
        "status": "Status",
        "capacity_mw": "Total capacity (MW)",
        "total_capacity_mw": "Total capacity (MW)",
    }
    friendly = friendly.rename(columns=rename)

    ordered_cols = [c for c in ["Country", "Technology", "Status", "Total capacity (MW)"] if c in friendly.columns]
    tail = [c for c in friendly.columns if c not in ordered_cols]
    friendly = friendly[ordered_cols + tail]

    return friendly.to_markdown(index=False, floatfmt=".0f")


def format_generation_summary_per_country(df: Optional[pd.DataFrame]) -> str:
    """Create one table per country without the country column to save space."""
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
        title = country if isinstance(country, str) else str(country)
        lines.append(f"#### {title}\n" + _wrap_table(friendly.to_markdown(index=False, floatfmt=".0f")))
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
            table_cols = ["Technology", "Plant", "Status", "Capacity (MW)"]
            subset_table = subset[table_cols]
            lines.append(f"**{category}**\n" + _wrap_table(subset_table.to_markdown(index=False, floatfmt=".0f")))

        other = group[~group["Category"].isin(category_order)]
        if not other.empty:
            other = other.sort_values(["Technology", "Capacity (MW)"] , ascending=[True, False])
            lines.append(
                "**Other**\n"
                + _wrap_table(other[["Technology", "Plant", "Status", "Capacity (MW)"]].to_markdown(index=False, floatfmt=".0f"))
            )

    return "\n\n".join(lines)


def summarize_parameters(config: Dict) -> Dict[str, str]:
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
    seasons_map = rep_cfg.get("seasons_map", {})
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
            f"- Workflow: `{_relpath_for_display(workflow_path)}` using config `{_relpath_for_display(config_path)}`; outputs stored in `{_relpath_for_display(output_dir)}`.",
            "- Re-run full open-data workflow (regenerates all inputs):  ",
            f"  `{workflow_command}`",
            "- Regenerate this report only:  ",
            f"  `{report_command}`",
            "- Archive the rendered report, the config YAML, and the `output_workflow` directory together; timestamps in the tables reflect extraction dates of the raw open datasets.",
        ]
    )


def build_appendix_sections(
    repro_checklist: str,
    rninja_summary: List[Dict],
    gen_summary_by_country: str,
    boxplots: List[Path],
    gap_plot: Optional[Path] = None,
    generation_map_all_path: Optional[Path] = None,
    generation_map_all_static_figs: str = "",
    workflow_parameters: str = "",
    season_mapping: str = "",
) -> List[Dict[str, str]]:
    """Assemble appendix sections with headings and content."""
    sections: List[Dict[str, str]] = []
    letter = ord("A")

    def _title(suffix: str) -> str:
        nonlocal letter
        title = f"Appendix {chr(letter)}: {suffix}"
        letter += 1
        return title

    if workflow_parameters:
        body = workflow_parameters
        if season_mapping:
            body = f"{body}\n\n- Season grouping: {season_mapping}"
        sections.append({"title": _title("Key workflow parameters"), "body": body})

    if repro_checklist:
        sections.append({"title": _title("Reproducibility and audit trail"), "body": repro_checklist})

    diagnostics: List[str] = []
    if gap_plot:
        diagnostics.append("#### GIP project locations figure\n" + format_figures([gap_plot], "GIP project picks", add_link=False))

    if rninja_summary:
        df_rn = pd.DataFrame(rninja_summary)
        diagnostics.append(
            "#### Renewables Ninja average capacity factors\n"
            + _wrap_table(df_rn.sort_values(["tech", "zone", "period"]).to_markdown(index=False, floatfmt=".3f"))
        )

    if diagnostics:
        sections.append({"title": _title("VRE selections and diagnostics"), "body": "\n\n".join(diagnostics)})

    if gen_summary_by_country:
        sections.append({"title": _title("Generation assets by country"), "body": gen_summary_by_country})

    if boxplots:
        sections.append(
            {"title": _title("Additional figures"), "body": format_figures(boxplots, "Capacity-factor distribution", add_link=False)}
        )

    map_all_lines: List[str] = []
    if generation_map_all_path:
        rel = generation_map_all_path.relative_to(BASE_DIR).as_posix()
        map_all_lines.append(
            f"- Interactive generation map (all statuses): [{generation_map_all_path.name}]({rel})"
        )
    if generation_map_all_static_figs:
        map_all_lines.append(
            "- Static exports (all statuses):\n" + generation_map_all_static_figs
        )
    if map_all_lines:
        sections.append(
            {"title": _title("Generation map (all statuses)"), "body": "\n\n".join(map_all_lines)}
        )

    return sections


def render_report(
    template_path: Path,
    output_path: Path,
    config_path: Path,
    output_dir_override: Optional[Path] = None,
) -> None:
    config = load_config(config_path)
    slug_map = {_slug(name): name for name in config.get("gap", {}).get("countries", [])}

    output_dir = find_output_dir(config, output_dir_override)
    load_cfg = config.get("load_profile", {})
    rninja_cfg = config.get("rninja", {})
    genmap_cfg = config.get("generation_map", {})
    load_dir = _resolve_category_output_dir(output_dir, load_cfg.get("output_dir"), "load")
    vre_dir = _resolve_category_output_dir(output_dir, rninja_cfg.get("output_dir"), "vre")
    supply_dir = _resolve_category_output_dir(output_dir, genmap_cfg.get("output_dir"), "supply")
    climate = collect_climate_overview(output_dir, config)
    load_stats, load_figs, load_heatmaps, load_boxplots = collect_load_profiles(load_dir, slug_map)
    load_agg_figs = build_load_aggregate_plots(load_dir, slug_map)
    rninja_summary, rninja_zone_figures, boxplots = collect_rninja_profiles(vre_dir)
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
                    f"{label} heatmap ({entry['country']})",
                    add_link=False,
                )
                if entry.get("heatmap")
                else ""
            )
            boxplot_md = (
                format_figures(
                    [entry["boxplot"]],
                    f"{label} boxplot ({entry['country']})",
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
    rep_figs = find_rep_day_figures(BASE_DIR)

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

    params = summarize_parameters(config)
    season_map_desc = params.get("seasons", "")
    parameter_summary = params.get("bullets", "")

    countries = config.get("gap", {}).get("countries", [])
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

    rn_period_label = ""
    if rn_start and rn_end:
        rn_period_label = f"Jan–Dec {rn_start}" if rn_start == rn_end else f"Jan–Dec {rn_start} – {rn_end}"
    elif rn_start:
        rn_period_label = f"Jan–Dec {rn_start}"
    elif rn_end:
        rn_period_label = f"Jan–Dec {rn_end}"

    workflow_path = BASE_DIR / "Snakefile"
    report_script = BASE_DIR / "generate_report.py"
    config_label = _relpath_for_display(config_path)
    output_dir_label = _relpath_for_display(output_dir)
    workflow_command = f"snakemake -s {shlex.quote(str(workflow_path))} --cores 4"
    report_command = (
        f"python {shlex.quote(str(report_script))}"
        f" --config {shlex.quote(str(config_path))}"
        f" --output-dir {shlex.quote(str(output_dir))}"
        f" --output {shlex.quote(str(output_path))}"
    )
    repro_checklist_md = build_repro_checklist(workflow_path, config_path, output_dir, workflow_command, report_command)

    data_overview_rows = [
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
    ]
    data_overview_table = pd.DataFrame(data_overview_rows).to_markdown(index=False)

    appendix_sections = build_appendix_sections(
        repro_checklist_md,
        rninja_summary,
        gen_appendix_tables,
        boxplots,
        gap_plot,
        generation_map_all_path=gen_map_all_path,
        generation_map_all_static_figs=static_map_all_figs,
        workflow_parameters=parameter_summary,
        season_mapping=season_map_desc,
    )

    report_scope = countries_inline or "Energy System Modelling"

    context = {
        "date": str(date.today()),
        "report_scope": report_scope,
        "countries_text": countries_text,
        "rn_countries_inline": countries_inline,
        "climate_summary": climate.get("summary", ""),
        "climate_spatial_fig": format_figures(climate.get("figures", {}).get("spatial", []), "Spatial mean climate", add_link=False),
        "climate_monthly_fig": format_figures(climate.get("figures", {}).get("monthly", []), "Monthly climate averages", add_link=False),
        "climate_heatmap": format_figures(climate.get("figures", {}).get("heatmap", []), "Monthly precipitation heatmap", add_link=False),
        "climate_scatter_fig": format_figures(climate.get("figures", {}).get("scatter", []), "Temperature vs precipitation", add_link=False),
        "climate_countries": climate.get("countries", ""),
        "climate_period": climate.get("period", ""),
        "load_profile_fig": format_figures(load_figs, "Load profile", add_link=False),
        "load_month_fig": format_figures([load_agg_figs["avg_month"]], "Average load per month", add_link=False) if "avg_month" in load_agg_figs else "",
        "load_day_fig": format_figures([load_agg_figs["avg_day"]], "Average load per day", add_link=False) if "avg_day" in load_agg_figs else "",
        "load_heatmap_fig": format_figures(load_heatmaps, "Load heatmap", add_link=False),
        "load_boxplot_fig": format_figures(load_boxplots, "Load distribution", add_link=False),
        "load_profile_summary": load_summary_table,
        "vre_location_plot": format_figures([gap_plot], "GIP project picks", add_link=False) if gap_plot else "",
        "vre_location_map": (
            gap_summary + f"\n\nSource: [{gap_path.name}]({gap_path.relative_to(BASE_DIR).as_posix()})"
            if gap_summary and gap_path
            else ""
        ),
        "generation_map_text": (
            f"- Interactive generation map: [{gen_map_path.name}]({gen_map_path.relative_to(BASE_DIR).as_posix()})"
            if gen_map_path
            else ""
        ),
        "generation_map_static_figs": static_map_figs,
        "generation_map_summary": format_generation_summary(gen_summary_df),
        "generation_map_summary_appendix": gen_appendix_tables,
        "rn_period_start": rn_start or "",
        "rn_period_end": rn_end or "",
        "rn_period_months": "Jan–Dec",
        "rn_period_label": rn_period_label,
        "rninja_zone_sections": rninja_zone_sections,
        "rep_days_fig": format_figures(rep_figs, "Representative days"),
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output or default_report_output(args.config, args.output_dir)
    render_report(args.template, output_path, args.config, args.output_dir)
    try:
        display_path = output_path.resolve().relative_to(BASE_DIR)
    except ValueError:
        display_path = output_path
    print(f"Wrote report to {display_path}")
