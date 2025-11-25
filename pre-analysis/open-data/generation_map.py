from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import unicodedata
import warnings
from typing import Dict, Iterable, Optional

import folium
import matplotlib
matplotlib.use("Agg")  # Headless backend for static map exports.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - geopandas optional for static map background.
    gpd = None

from utils_renewables import require_file, resolve_country_name

# Column aliases to normalize GAP headers into a compact, consistent schema.
COLUMN_ALIASES: Dict[str, str] = {
    "Plant / Project name": "name",
    "Plant Name": "name",
    "Name": "name",
    "Technology": "technology",
    "Type": "technology",
    "Fuel": "technology",
    "Capacity (MW)": "capacity_mw",
    "Capacity_MW": "capacity_mw",
    "CapacityMW": "capacity_mw",
    "Status": "status",
    "Country/area": "country",
    "Country": "country",
    "Latitude": "latitude",
    "Longitude": "longitude",
}

# Default styles used on the interactive map.
DEFAULT_STATUS_COLORS: Dict[str, str] = {
    "Operating": "green",
    "Under Construction": "orange",
    "Planned": "blue",
    "Announced": "purple",
    "Mothballed": "gray",
    "Cancelled": "red",
    "Retired": "black",
}

DEFAULT_TECH_ICONS: Dict[str, str] = {
    "Hydro": "tint",
    "Solar": "sun",
    "Wind": "wind",
    "Thermal": "fire",
    "Gas": "gas-pump",
    "Coal": "industry",
    "Oil": "oil-can",
    "Nuclear": "atom",
    "Geothermal": "temperature-high",
    "Biomass": "leaf",
}

DEFAULT_TECH_MARKERS = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">", "h", "8"]


def _collapse_duplicate_columns(df: pd.DataFrame, column_names: Iterable[str]) -> pd.DataFrame:
    """Collapse duplicated columns by taking the first non-null value."""
    for name in column_names:
        mask = [col == name for col in df.columns]
        if sum(mask) > 1:
            # Combine duplicate columns, keeping the first non-null entry across them.
            collapsed = df.loc[:, mask].bfill(axis=1).iloc[:, 0]

            # Drop all duplicate-name columns then re-insert the collapsed Series at
            # the position of the first occurrence to preserve column order.
            first_idx = mask.index(True)
            keep_cols = [col for col, is_dupe in zip(df.columns, mask) if not is_dupe]
            df = df.loc[:, keep_cols]
            df.insert(first_idx, name, collapsed)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common GAP columns and coerce expected dtypes."""
    rename = {col: COLUMN_ALIASES[col] for col in df.columns if col in COLUMN_ALIASES}
    df = df.rename(columns=rename)

    # Consolidate duplicated normalized columns (e.g., Technology/Type/Fuel -> technology).
    df = _collapse_duplicate_columns(df, set(COLUMN_ALIASES.values()))

    for col in ("name", "technology", "status", "country"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    for col in ("capacity_mw", "latitude", "longitude"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _match_key(value: str, options: Iterable[str]) -> Optional[str]:
    """Return the first option that appears in value (case-insensitive)."""
    value_lower = str(value).lower()
    for candidate in options:
        if candidate.lower() in value_lower:
            return candidate
    return None


def load_generation_sites(
    xlsx_path,
    countries: Iterable[str],
    sheet_name: str = "Power facilities",
    verbose: bool = False,
) -> pd.DataFrame:
    """Load and standardize generation sites from the Global Atlas Power Excel."""
    path = require_file(
        xlsx_path,
        hint="Place Global-Integrated-Power-April-2025.xlsx under pre-analysis/open-data/dataset/ or point the config to the correct location.",
    )
    df_raw = pd.read_excel(path, sheet_name=sheet_name, header=0, index_col=None)
    df = _normalize_columns(df_raw)
    if verbose:
        print(f"[generation-map] Loaded {len(df_raw)} raw rows from {path} ({sheet_name}).")

    available = df["country"].dropna().unique()
    resolved_map: Dict[str, str] = {}
    for country in countries or []:
        match = resolve_country_name(country, available, verbose=verbose, allow_missing=True)
        if match:
            resolved_map[country] = match
        else:
            print(f"[generation-map] No GAP rows matched '{country}'; skipping.")

    if resolved_map:
        df = df[df["country"].isin(resolved_map.values())].copy()
        reverse_map = {v: k for k, v in resolved_map.items()}
        df["country"] = df["country"].map(reverse_map).fillna(df["country"])

    df = df.dropna(subset=["latitude", "longitude"])
    df = df.drop_duplicates(subset=["country", "name", "technology", "latitude", "longitude"])
    df["status"] = df["status"].replace("", "Unknown")
    df["technology"] = df["technology"].replace("", "Unknown")
    if verbose:
        print(f"[generation-map] {len(df)} rows remain after country filter + dropping missing coords.")

    return df.reset_index(drop=True)


def summarize_generation_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize capacity by country/technology/status (drop raw site counts)."""
    if df.empty:
        return pd.DataFrame(columns=["country", "technology", "status", "total_capacity_mw", "avg_capacity_mw"])

    # Defensive: ensure grouping columns are unique even if upstream inputs carried duplicates.
    df = _collapse_duplicate_columns(df, ["country", "technology", "status", "name", "capacity_mw"])

    summary = (
        df.groupby(["country", "technology", "status"], dropna=False)
        .agg(total_capacity_mw=("capacity_mw", "sum"), avg_capacity_mw=("capacity_mw", "mean"))
        .reset_index()
        .sort_values(["country", "technology", "status"])
    )
    return summary


def _build_popup(row: pd.Series) -> str:
    """HTML popup content for a single site."""
    fields = [
        ("Name", row.get("name", "")),
        ("Technology", row.get("technology", "")),
        ("Capacity (MW)", row.get("capacity_mw", "")),
        ("Status", row.get("status", "")),
        ("Country", row.get("country", "")),
    ]
    lines = [f"<b>{label}:</b> {val}" for label, val in fields if pd.notna(val) and val != ""]
    return "<br>".join(lines)


def _legacy_cluster_js(tech_icons: Dict[str, str]) -> str:
    """Return the JS used by the notebook to style clusters by capacity/tech."""
    tech_mapping = json.dumps(tech_icons)
    return f"""
function(cluster) {{
    var markers = cluster.getAllChildMarkers();
    var totalCapacity = 0;
    var techCounts = {{}};
    var dominantTech = '';
    var maxCount = 0;

    markers.forEach(function(marker) {{
        if (marker.options.capacity) {{
            totalCapacity += marker.options.capacity;
        }}
        if (marker.options.technology) {{
            var tech = marker.options.technology;
            if (!techCounts[tech]) {{
                techCounts[tech] = 0;
            }}
            techCounts[tech]++;
            if (techCounts[tech] > maxCount) {{
                maxCount = techCounts[tech];
                dominantTech = tech;
            }}
        }}
    }});

    var size = Math.sqrt(totalCapacity) * 1.5;
    if (size < 20) size = 20;

    var techIcon = 'bolt';
    var techMapping = {tech_mapping};
    for (var tech in techMapping) {{
        if (dominantTech && dominantTech.toLowerCase().includes(tech.toLowerCase())) {{
            techIcon = techMapping[tech];
            break;
        }}
    }}

    return L.divIcon({{
        html: '<div style="background-color: #3388ff; color: white; border-radius: 50%; text-align: center; width: ' + size + 'px; height: ' + size + 'px; line-height: ' + size + 'px; font-size: ' + (size/2) + 'px;"><i class="fa fa-' + techIcon + '"></i></div>',
        className: 'marker-cluster',
        iconSize: L.point(size, size)
    }});
}}
"""


def create_generation_map(
    df: pd.DataFrame,
    map_path: Path,
    status_colors: Optional[Dict[str, str]] = None,
    tech_icons: Optional[Dict[str, str]] = None,
    tiles: str = "cartodbpositron",
    verbose: bool = False,
) -> None:
    """Notebook-style map: capacity-scaled DivIcons plus custom cluster bubbles."""
    status_colors = status_colors or DEFAULT_STATUS_COLORS
    tech_icons = tech_icons or DEFAULT_TECH_ICONS
    df = df.copy()

    for col in ("name", "technology", "status", "country"):
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)

    if "capacity_mw" not in df.columns:
        df["capacity_mw"] = np.nan
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")

    for col in ("latitude", "longitude"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_coords = df["latitude"].isna() | df["longitude"].isna()
    invalid_coords = invalid_coords | (~np.isfinite(df["latitude"])) | (~np.isfinite(df["longitude"]))
    if invalid_coords.any():
        if verbose:
            print(f"[generation-map] Dropping {invalid_coords.sum()} rows without valid coordinates.")
        df = df[~invalid_coords]
    if verbose:
        print(f"[generation-map] Rendering legacy map with {len(df)} sites after cleaning.")

    map_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        if verbose:
            print("[generation-map] No sites passed to legacy map rendering; writing placeholder HTML.")
        map_path.write_text("<html><body><p>No generation sites available.</p></body></html>", encoding="utf-8")
        return

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    power_map = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles=tiles)

    all_layer = folium.FeatureGroup(name="All Power Plants")
    status_layers = {status: folium.FeatureGroup(name=f"Status: {status}") for status in status_colors}
    tech_layers = {tech: folium.FeatureGroup(name=f"Technology: {tech}") for tech in tech_icons}

    cluster_js = _legacy_cluster_js(tech_icons)
    main_cluster = MarkerCluster(icon_create_function=cluster_js).add_to(all_layer)
    status_clusters = {status: MarkerCluster(icon_create_function=cluster_js).add_to(layer) for status, layer in status_layers.items()}
    tech_clusters = {tech: MarkerCluster(icon_create_function=cluster_js).add_to(layer) for tech, layer in tech_layers.items()}

    all_layer.add_to(power_map)
    for layer in status_layers.values():
        layer.add_to(power_map)
    for layer in tech_layers.values():
        layer.add_to(power_map)

    def _status_color(value: str) -> str:
        key = _match_key(value, status_colors.keys())
        return status_colors.get(key, "cadetblue")

    def _tech_icon(value: str) -> str:
        key = _match_key(value, tech_icons.keys())
        return tech_icons.get(key, "bolt")

    def _scale_capacity(capacity) -> float:
        if pd.isna(capacity) or capacity <= 0:
            return 5.0
        return float(np.sqrt(capacity) * 1.5)

    skip_cols = {"name", "technology", "status", "country", "capacity_mw", "latitude", "longitude"}

    for _, row in df.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        capacity = row.get("capacity_mw")
        technology = row.get("technology", "Unknown")
        status = row.get("status", "Unknown")

        popup_lines = [
            ("Name", row.get("name", "Unknown")),
            ("Technology", technology),
            ("Capacity (MW)", capacity if pd.notna(capacity) else ""),
            ("Status", status),
            ("Country", row.get("country", "")),
        ]
        for col, val in row.items():
            if col in skip_cols or pd.isna(val) or val == "":
                continue
            popup_lines.append((col, val))
        popup_html = "<br>".join(f"<b>{label}:</b> {val}" for label, val in popup_lines if val not in ("", None))

        icon_size = _scale_capacity(capacity)
        icon_html = f"""
        <div style="
            background-color: {_status_color(status)};
            color: white;
            border-radius: 50%;
            text-align: center;
            width: {icon_size*2}px;
            height: {icon_size*2}px;
            line-height: {icon_size*2}px;
            font-size: {max(icon_size, 8)}px;
        ">
            <i class="fa fa-{_tech_icon(technology)}"></i>
        </div>
        """

        marker = folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                icon_size=(icon_size * 2, icon_size * 2),
                icon_anchor=(icon_size, icon_size),
                html=icon_html,
            ),
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{row.get('name', 'Unknown')} - {technology} - {capacity or 'N/A'} MW",
        )
        marker.options["capacity"] = float(capacity) if pd.notna(capacity) else 0
        marker.options["technology"] = str(technology)
        marker.add_to(main_cluster)

        status_key = _match_key(status, status_layers.keys())
        if status_key and status_key in status_clusters:
            marker_copy = folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    icon_size=(icon_size * 2, icon_size * 2),
                    icon_anchor=(icon_size, icon_size),
                    html=icon_html,
                ),
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{row.get('name', 'Unknown')} - {technology} - {capacity or 'N/A'} MW",
            )
            marker_copy.add_to(status_clusters[status_key])

        tech_key = _match_key(technology, tech_layers.keys())
        if tech_key and tech_key in tech_clusters:
            marker_copy = folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    icon_size=(icon_size * 2, icon_size * 2),
                    icon_anchor=(icon_size, icon_size),
                    html=icon_html,
                ),
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{row.get('name', 'Unknown')} - {technology} - {capacity or 'N/A'} MW",
            )
            marker_copy.add_to(tech_clusters[tech_key])

    bounds = [
        [df["latitude"].min(), df["longitude"].min()],
        [df["latitude"].max(), df["longitude"].max()],
    ]
    power_map.fit_bounds(bounds, padding=(10, 10))

    legend_html = [
        '<div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: auto; '
        'border:2px solid grey; z-index:9999; font-size:14px; background-color:white; padding: 10px; '
        'border-radius: 6px;">',
        '<p style="margin-top: 0; margin-bottom: 5px;"><b>Status</b></p>',
    ]
    for status, color in status_colors.items():
        legend_html.append(
            f'<div style="display: flex; align-items: center; margin-bottom: 3px;">'
            f'<div style="background-color:{color}; width:15px; height:15px; margin-right:5px; border-radius:50%;"></div>'
            f'<span>{status}</span></div>'
        )
    legend_html.extend(
        [
            '<p style="margin-top: 10px; margin-bottom: 5px;"><b>Size</b></p>',
            "<div>Marker size is proportional to Capacity (MW)</div>",
            "<div>Icon background color represents Status</div>",
            "<div>Tooltip shows Name, Technology, and Capacity</div>",
            "</div>",
        ]
    )
    power_map.get_root().html.add_child(folium.Element("\n".join(legend_html)))
    folium.LayerControl().add_to(power_map)
    power_map.save(map_path)
    if verbose:
        print(f"[generation-map] Saved legacy-style map to {map_path}")
        all_counts = len(getattr(main_cluster, "_children", {}))
        print(f"[generation-map] Marker counts – all: {all_counts}, "
              f"status: { {k: len(getattr(v, '_children', {})) for k, v in status_clusters.items()} }, "
              f"tech: { {k: len(getattr(v, '_children', {})) for k, v in tech_clusters.items()} }")



def create_static_generation_map(
    df: pd.DataFrame,
    output_dir: Path,
    basename: str = "generation_map_static",
    status_colors: Optional[Dict[str, str]] = None,
    tech_markers: Optional[Iterable[str]] = None,
    figsize=(11, 7),
    dpi: int = 200,
    verbose: bool = False,
) -> Dict[str, Path]:
    """Render a matplotlib version of the generation map for PNG/PDF export."""
    status_colors = status_colors or DEFAULT_STATUS_COLORS
    markers = list(tech_markers or DEFAULT_TECH_MARKERS) or DEFAULT_TECH_MARKERS

    df_plot = df.copy()
    for col in ("name", "technology", "status", "country"):
        if col not in df_plot.columns:
            df_plot[col] = "Unknown"
        df_plot[col] = df_plot[col].fillna("Unknown").astype(str)

    for coord in ("latitude", "longitude"):
        df_plot[coord] = pd.to_numeric(df_plot.get(coord), errors="coerce")
    df_plot = df_plot.dropna(subset=("latitude", "longitude"))

    if df_plot.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No generation sites to plot.", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if gpd is not None:
            try:
                world = gpd.read_file("dataset/maps/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
                world.plot(ax=ax, color="#f5f5f0", edgecolor="#cacaca", linewidth=0.4)
            except Exception as exc:  # pragma: no cover - geopandas failures should not break exports.
                if verbose:
                    print(f"[generation-map] Could not render world background: {exc}")

        status_order = list(dict.fromkeys(df_plot["status"]))
        tech_order = []
        tech_iter = itertools.cycle(markers)
        tech_marker_map: Dict[str, str] = {}
        for tech in df_plot["technology"]:
            if tech not in tech_marker_map:
                tech_marker_map[tech] = next(tech_iter)
                tech_order.append(tech)

        def _status_color_static(value: str) -> str:
            key = _match_key(value, status_colors.keys())
            return status_colors.get(key, "#6c6c6c")

        capacities = pd.to_numeric(df_plot["capacity_mw"], errors="coerce").fillna(0)
        df_plot["marker_size"] = 30 + np.sqrt(np.maximum(capacities, 0)) * 6

        for status in status_order:
            group_status = df_plot[df_plot["status"] == status]
            color = _status_color_static(status)
            for tech in group_status["technology"].unique():
                subset = group_status[group_status["technology"] == tech]
                marker = tech_marker_map.get(tech, markers[0])
                ax.scatter(
                    subset["longitude"],
                    subset["latitude"],
                    s=subset["marker_size"],
                    marker=marker,
                    color=color,
                    edgecolors="black",
                    linewidths=0.6,
                    alpha=0.8,
                )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Generation sites – status (color) & technology (marker)")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

        status_handles = [
            Patch(facecolor=_status_color_static(status), edgecolor="black", label=status, alpha=0.7)
            for status in status_order
        ]
        tech_handles = [
            Line2D(
                [0],
                [0],
                marker=tech_marker_map[tech],
                color="#444444",
                linestyle="",
                markerfacecolor="#777777",
                markeredgecolor="black",
                markersize=8,
                label=tech,
            )
            for tech in tech_order
        ]

        if status_handles:
            status_legend = ax.legend(handles=status_handles, title="Status", loc="lower left", frameon=True)
            ax.add_artist(status_legend)
        if tech_handles:
            ax.legend(
                handles=tech_handles,
                title="Technology",
                loc="lower right",
                frameon=True,
                ncol=min(3, max(1, len(tech_handles))),
            )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path, dpi=dpi)
    plt.close(fig)

    if verbose:
        print(f"[generation-map] Saved static map to {png_path} and {pdf_path}")

    return {"png": png_path, "pdf": pdf_path}


DEFAULT_STATUS_TO_CODE: Dict[str, int] = {
    "operating": 1,
    "existing": 1,
    "under construction": 2,
    "construction": 2,
    "committed": 2,
    "planned": 3,
    "announced": 3,
    "unknown": 3,
}

DEFAULT_TECH_FUEL: Dict[str, Dict[str, str]] = {
    "solar": {"tech": "PV", "fuel": "Solar"},
    "wind": {"tech": "OnshoreWind", "fuel": "Wind"},
    "hydro": {"tech": "Hydro", "fuel": "Water"},
    "hydropower": {"tech": "Hydro", "fuel": "Water"},
    "gas": {"tech": "CCGT", "fuel": "Gas"},
    "coal": {"tech": "ST", "fuel": "Coal"},
    "oil": {"tech": "OCGT", "fuel": "Oil"},
    "biomass": {"tech": "BiomassPlant", "fuel": "Biomass"},
    "geothermal": {"tech": "Geothermal", "fuel": "Geothermal"},
}


def _slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in ascii_text)
    compact = "_".join(filter(None, cleaned.split("_")))
    return compact.lower().strip("_")


def _countries_slug(countries: Iterable[str]) -> str:
    slugs = [_slugify(country) for country in countries]
    slugs = [slug for slug in slugs if slug]
    if not slugs:
        return "all-countries"
    # Preserve the provided order while removing duplicates.
    unique_slugs = list(dict.fromkeys(slugs))
    return "-".join(unique_slugs)


def export_pgen_data_input(
    df: pd.DataFrame,
    output_path: Path,
    status_map: Optional[Dict[str, int]] = None,
    tech_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Path:
    """Export GAP-style generation rows to a minimal pGenDataInput CSV."""
    status_map = status_map or DEFAULT_STATUS_TO_CODE
    tech_map = tech_map or DEFAULT_TECH_FUEL

    def _status_code(text: str) -> int:
        key = _match_key(text or "", status_map.keys()) or "unknown"
        return status_map.get(key, 3)

    def _tech_fuel(text: str) -> Dict[str, str]:
        key = _match_key(text or "", tech_map.keys())
        if key and key in tech_map:
            return tech_map[key]
        cleaned = str(text or "Unknown").strip()
        return {"tech": cleaned.replace(" ", ""), "fuel": cleaned}

    columns = [
        "gen",
        "zone",
        "tech",
        "fuel",
        "Status",
        "StYr",
        "RetrYr",
        "Capacity",
        "DescreteCap",
        "fuel2",
        "HeatRate2",
        "BuildLimitperYear",
        "Life",
        "MinLimitShare",
        "HeatRate",
        "RampUpRate",
        "RampDnRate",
        "OverLoadFactor",
        "ResLimShare",
        "Capex",
        "FOMperMW",
        "VOM",
        "ReserveCost",
        "UnitSize",
    ]

    records = []
    gen_counter: Dict[str, int] = {}

    for idx, row in df.iterrows():
        zone = str(row.get("country", "")).strip() or "unknown_zone"
        name = str(row.get("name", "")).strip() or f"site_{idx+1}"
        base_gen = _slugify(name) or f"site_{idx+1}"
        gen_counter[base_gen] = gen_counter.get(base_gen, 0) + 1
        gen_name = base_gen if gen_counter[base_gen] == 1 else f"{base_gen}_{gen_counter[base_gen]}"

        tech_fuel = _tech_fuel(row.get("technology", ""))

        record = {
            "gen": gen_name,
            "zone": zone,
            "tech": tech_fuel["tech"],
            "fuel": tech_fuel["fuel"],
            "Status": _status_code(row.get("status", "")),
            "StYr": "",
            "RetrYr": "",
            "Capacity": row.get("capacity_mw"),
            "DescreteCap": "",
            "fuel2": "",
            "HeatRate2": "",
            "BuildLimitperYear": "",
            "Life": "",
            "MinLimitShare": "",
            "HeatRate": "",
            "RampUpRate": "",
            "RampDnRate": "",
            "OverLoadFactor": "",
            "ResLimShare": "",
            "Capex": "",
            "FOMperMW": "",
            "VOM": "",
            "ReserveCost": "",
            "UnitSize": "",
        }
        records.append(record)

    export_df = pd.DataFrame(records, columns=columns)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)
    return output_path


def build_generation_map(
    xlsx_path,
    countries: Iterable[str],
    sheet_name: str = "Power facilities",
    output_dir="output",
    map_filename: Optional[str] = None,
    data_filename: Optional[str] = None,
    summary_filename: Optional[str] = None,
    pgen_filename: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Path]:
    """Main entry point: clean GAP data, export CSVs, and render the legacy map."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = list(countries)
    country_slug = _countries_slug(countries)

    map_filename = map_filename or f"generation_map_{country_slug}.html"
    data_filename = data_filename or f"generation_sites_{country_slug}.csv"
    summary_filename = summary_filename or f"generation_sites_summary_{country_slug}.csv"

    df_sites = load_generation_sites(xlsx_path, countries, sheet_name=sheet_name, verbose=verbose)
    summary = summarize_generation_sites(df_sites)

    data_path = output_dir / data_filename
    summary_path = output_dir / summary_filename
    map_path = output_dir / map_filename
    pgen_path = output_dir / pgen_filename if pgen_filename else None

    df_sites.to_csv(data_path, index=False)
    summary.to_csv(summary_path, index=False)
    if verbose:
        print(
            f"[generation-map] Loaded {len(df_sites)} sites "
            f"(after filtering on countries and dropping missing coordinates)."
        )
        print(f"[generation-map] Saving cleaned sites to {data_path}")
        print(f"[generation-map] Saving summary to {summary_path}")

    create_generation_map(df_sites, map_path, verbose=verbose)

    static_paths = create_static_generation_map(
        df_sites,
        output_dir,
        basename=f"generation_map_static_{country_slug}",
        verbose=verbose,
    )

    outputs = {
        "map": map_path,
        "data": data_path,
        "summary": summary_path,
        "static_png": static_paths["png"],
        "static_pdf": static_paths["pdf"],
    }
    if pgen_path:
        export_pgen_data_input(df_sites, pgen_path)
        outputs["pgen"] = pgen_path
    return outputs


def _find_gap_excel(source: Path) -> Path:
    """Resolve the GAP Excel from a folder (or pass through an explicit file)."""
    if source.is_file():
        return source

    candidates = sorted(source.glob("Global-Integrated-Power-*.xlsx")) or sorted(source.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(
            f"No Excel files found under {source}. "
            "Drop Global-Integrated-Power-April-2025.xlsx in that folder or pass --excel explicitly."
        )
    return candidates[0]


def main(default_verbose: bool = False) -> None:
    script_dir = Path(__file__).resolve().parent
    default_source = script_dir / "dataset"

    parser = argparse.ArgumentParser(description="Quick-start helper for GAP-based generation maps.")
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=default_source,
        help="Folder containing the GAP Excel (defaults to ./dataset) or the Excel file itself.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write map + CSVs (defaults to ./output when using the bundled dataset, or <source>/output otherwise).",
    )
    parser.add_argument(
        "--pgen-filename",
        type=str,
        default=None,
        help="Optional filename for exporting a minimal pGenDataInput CSV alongside the map.",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["Albania"],
        help="Countries to include (defaults to Croatia). Use --all-countries to include everything.",
    )
    parser.add_argument(
        "--all-countries",
        action="store_true",
        help="Ignore country filtering and include all rows with coordinates.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=default_verbose,
        help="Print debug details about filtering and map rendering.",
    )
    args = parser.parse_args()

    try:
        excel_path = _find_gap_excel(args.source)
    except FileNotFoundError as exc:
        print(exc)
        return

    if args.output_dir:
        output_dir = args.output_dir
    elif args.source.resolve() == default_source:
        output_dir = script_dir / "output"
    else:
        output_dir = excel_path.parent / "output"
    countries = [] if args.all_countries else args.countries

    try:
        outputs = build_generation_map(
            excel_path,
            countries,
            sheet_name="Power facilities",
            output_dir=output_dir,
            pgen_filename=args.pgen_filename,
            verbose=args.verbose,
        )
        print(f"[generation-map] Wrote map + CSVs to {outputs['map'].parent}")
        for label, path in outputs.items():
            print(f"  - {label}: {path}")
    except FileNotFoundError as exc:
        print(exc)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="Unknown extension is not supported and will be removed",
        category=UserWarning,
        module=r"openpyxl\.worksheet\._read_only",
    )
    main(default_verbose=True)
