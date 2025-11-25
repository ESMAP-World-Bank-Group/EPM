from __future__ import annotations

import argparse
from pathlib import Path
import unicodedata
import warnings
from typing import Dict, Iterable, Optional

import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster

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


def _marker_icon(
    technology: str,
    status: str,
    status_colors: Dict[str, str],
    tech_icons: Dict[str, str],
) -> folium.Icon:
    status_key = _match_key(status, status_colors.keys())
    tech_key = _match_key(technology, tech_icons.keys())
    color = status_colors.get(status_key) or "blue"
    icon = tech_icons.get(tech_key) or "info-sign"
    try:
        return folium.Icon(color=color, icon=icon, prefix="fa")
    except Exception:
        return folium.Icon(color="blue", icon="info-sign", prefix="fa")


def create_generation_map(
    df: pd.DataFrame,
    map_path: Path,
    status_colors: Optional[Dict[str, str]] = None,
    tech_icons: Optional[Dict[str, str]] = None,
    tiles: str = "cartodbpositron",
    verbose: bool = False,
) -> None:
    """Render an interactive folium map for the given sites."""
    status_colors = status_colors or DEFAULT_STATUS_COLORS
    tech_icons = tech_icons or DEFAULT_TECH_ICONS
    df = df.copy()

    # Normalize key string fields to avoid missing labels downstream.
    for col in ("name", "technology", "status", "country"):
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)

    for col in ("capacity_mw",):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("latitude", "longitude"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that cannot be plotted because coordinates are invalid.
    invalid_coords = df["latitude"].isna() | df["longitude"].isna()
    invalid_coords = invalid_coords | (~np.isfinite(df["latitude"])) | (~np.isfinite(df["longitude"]))
    if invalid_coords.any():
        if verbose:
            print(f"[generation-map] Dropping {invalid_coords.sum()} rows without valid coordinates.")
        df = df[~invalid_coords]

    map_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        if verbose:
            print("[generation-map] No sites passed to map rendering; writing placeholder HTML.")
        map_path.write_text("<html><body><p>No generation sites available.</p></body></html>", encoding="utf-8")
        return

    if verbose:
        print(
            f"[generation-map] Rendering map for {len(df)} sites across {df['country'].nunique()} countries."
        )
        print(
            "[generation-map] Coordinate coverage: "
            f"lat [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}], "
            f"lon [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]"
        )

    # Center the initial view on the average coordinates of all sites.
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=tiles)

    # Base layer that always contains the full set of sites.
    all_layer = folium.FeatureGroup(name="All generation sites")
    all_cluster = MarkerCluster(name="All sites")
    all_cluster.add_to(all_layer)
    all_layer.add_to(fmap)

    # Build toggle-able sublayers per status.
    status_layers: Dict[str, MarkerCluster] = {}
    for status in status_colors:
        fg = folium.FeatureGroup(name=f"Status: {status}")
        cluster = MarkerCluster(name=f"Status: {status}")
        cluster.add_to(fg)
        fg.add_to(fmap)
        status_layers[status] = cluster

    # Build toggle-able sublayers per technology.
    tech_layers: Dict[str, MarkerCluster] = {}
    for tech in tech_icons:
        fg = folium.FeatureGroup(name=f"Technology: {tech}")
        cluster = MarkerCluster(name=f"Technology: {tech}")
        cluster.add_to(fg)
        fg.add_to(fmap)
        tech_layers[tech] = cluster

    markers_added = 0
    status_hit: Dict[str, int] = {k: 0 for k in status_colors}
    tech_hit: Dict[str, int] = {k: 0 for k in tech_icons}

    # Populate all layers with markers, falling back to circles if needed.
    for _, row in df.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])

        tooltip = f"{row.get('name', 'Unknown')} – {row.get('technology', '')}"
        if pd.notna(row.get("capacity_mw")):
            tooltip += f" ({row['capacity_mw']:.1f} MW)"

        popup = folium.Popup(_build_popup(row), max_width=320)
        icon = _marker_icon(row.get("technology", ""), row.get("status", ""), status_colors, tech_icons)

        def _add_marker(target_cluster: MarkerCluster):
            try:
                marker = folium.Marker(
                    location=[lat, lon],
                    icon=icon,
                    tooltip=tooltip,
                    popup=popup,
                )
                marker.add_to(target_cluster)
                return True
            except Exception as exc:
                if verbose:
                    print(
                        f"[generation-map] Marker failed for {row.get('name','?')} ({lat},{lon}): {exc}; "
                        "falling back to CircleMarker."
                    )
                try:
                    circle = folium.CircleMarker(
                        location=[lat, lon], radius=5, color="blue", fill=True, tooltip=tooltip, popup=popup
                    )
                    circle.add_to(target_cluster)
                    return True
                except Exception as exc2:
                    if verbose:
                        print(f"[generation-map] CircleMarker also failed for {row.get('name','?')}: {exc2}")
                    return False

        if _add_marker(all_cluster):
            markers_added += 1

        status_key = _match_key(row.get("status", ""), status_colors.keys())
        if status_key and status_key in status_layers:
            if _add_marker(status_layers[status_key]):
                status_hit[status_key] = status_hit.get(status_key, 0) + 1

        tech_key = _match_key(row.get("technology", ""), tech_icons.keys())
        if tech_key and tech_key in tech_layers:
            if _add_marker(tech_layers[tech_key]):
                tech_hit[tech_key] = tech_hit.get(tech_key, 0) + 1

    if verbose:
        active_status = [k for k, v in status_hit.items() if v]
        active_tech = [k for k, v in tech_hit.items() if v]
        print(
            f"[generation-map] Added {markers_added} markers. "
            f"Status layers populated: {active_status or 'none'}. "
            f"Tech layers populated: {active_tech or 'none'}."
        )

    # Ensure the map view fits all points, even for single-country runs.
    bounds = [
        [df["latitude"].min(), df["longitude"].min()],
        [df["latitude"].max(), df["longitude"].max()],
    ]
    fmap.fit_bounds(bounds, padding=(10, 10))

    # Enable layer toggles and write out the HTML map.
    folium.LayerControl(collapsed=False).add_to(fmap)

    if verbose:
        # Introspect folium objects for a quick sanity check on what was rendered.
        cluster_counts: Dict[str, int] = {}
        cluster_counts["All sites"] = len(getattr(all_cluster, "_children", {}))
        for key, cluster in status_layers.items():
            cluster_counts[f"Status: {key}"] = len(getattr(cluster, "_children", {}))
        for key, cluster in tech_layers.items():
            cluster_counts[f"Technology: {key}"] = len(getattr(cluster, "_children", {}))

        layer_names = [
            getattr(child, "layer_name", type(child).__name__)
            for child in fmap._children.values()
            if hasattr(child, "layer_name")
        ]

        print(f"[generation-map] Folium layer names: {layer_names}")
        print(f"[generation-map] Cluster marker counts: {cluster_counts}")

    fmap.save(map_path)
    if verbose:
        print(f"[generation-map] Saved interactive map to {map_path}")


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


def export_pgen_data_input(
    df: pd.DataFrame,
    output_path: Path,
    status_map: Optional[Dict[str, int]] = None,
    tech_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Path:
    """Export GAP-style generation rows to a minimal pGenDataInput CSV."""
    status_map = status_map or DEFAULT_STATUS_TO_CODE
    tech_map = tech_map or DEFAULT_TECH_FUEL

    def _slug(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(text))
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in ascii_text)
        compact = "_".join(filter(None, cleaned.split("_")))
        return compact.lower().strip("_")

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
        base_gen = _slug(name) or f"site_{idx+1}"
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
    map_filename: str = "generation_map.html",
    data_filename: str = "generation_sites.csv",
    summary_filename: str = "generation_sites_summary.csv",
    pgen_filename: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Path]:
    """Main entry point: clean GAP data, export CSVs, and render the map."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    outputs = {"map": map_path, "data": data_path, "summary": summary_path}
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
