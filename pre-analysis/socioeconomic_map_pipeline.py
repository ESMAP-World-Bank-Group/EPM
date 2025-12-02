"""Quick helper to render a static socio-economic raster map (GDP, population, etc.)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Headless backend for static map exports.
import matplotlib.pyplot as plt
import numpy as np

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - geopandas is optional but recommended.
    gpd = None

try:
    import rasterio
    from rasterio.plot import plotting_extent
    from rasterio.windows import from_bounds as window_from_bounds
except ImportError as exc:  # pragma: no cover - runtime guard for missing dependency.
    raise ImportError(
        "rasterio is required to render socio-economic raster maps. "
        "Install it via `pip install rasterio` or conda-forge (`conda install -c conda-forge rasterio`)."
    ) from exc

from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
from matplotlib import patheffects

from load_pipeline import require_file, resolve_country_name

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_RASTER = DATASET_DIR / "GDP2005_1km.tif"
DEFAULT_WORLD_MAP_SHAPEFILE = (
    DATASET_DIR / "maps" / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"
)
DEFAULT_CITIES_SHAPEFILE = (
    DATASET_DIR / "maps" / "ne_110m_populated_places" / "ne_110m_populated_places.shp"
)
LOG_PREFIX = "[socio-map]"
STATUS_FILENAME = "socioeconomic_status.json"


def _write_warning_placeholder(output_path: Path, title: str, reason: str, dpi: int = 220) -> Path:
    """Generate a lightweight placeholder PDF with a warning message."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=14, fontweight="bold", wrap=True)
    ax.text(0.5, 0.4, reason, ha="center", va="center", fontsize=11, wrap=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _update_status(output_dir: Path, *, basename: str, dataset_label: str, message: Optional[str]) -> None:
    """Record or clear a status message for a socio-economic map."""
    status_path = output_dir / STATUS_FILENAME
    records = []
    if status_path.exists():
        try:
            records = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            records = []
    records = [rec for rec in records if rec.get("basename") != basename]
    if message:
        records.append({"basename": basename, "label": dataset_label, "message": message})
    if records:
        status_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    else:
        status_path.unlink(missing_ok=True)


def _filter_countries(world: gpd.GeoDataFrame, countries: Iterable[str], verbose: bool = False) -> gpd.GeoDataFrame:
    """Return only the rows matching the requested countries using fuzzy resolution."""
    if not countries:
        return world

    available = world["ADMIN"].dropna().tolist()
    matches = []
    for country in countries:
        match = resolve_country_name(country, available, allow_missing=True, verbose=verbose)
        if match:
            matches.append(match)
        elif verbose:
            print(f"{LOG_PREFIX} Skipping missing country: {country}")
    if not matches:
        return world
    filtered = world[world["ADMIN"].isin(matches)].copy()
    return filtered if not filtered.empty else world


def render_socioeconomic_map(
    raster_path,
    countries: Iterable[str],
    *,
    dataset_label: str = "Dataset",
    shapefile: Optional[Path] = None,
    city_shapefile: Optional[Path] = None,
    output_dir: Path | str = "output",
    output_basename: str = "socioeconomic_map",
    percentile_clip: Tuple[float, float] = (2, 98),
    cmap: str = "YlGnBu",
    use_log_scale: bool = True,
    dpi: int = 250,
    title_prefix: Optional[str] = None,
    title: Optional[str] = None,
    scale_label: Optional[str] = None,
    verbose: bool = False,
) -> Path:
    """Render a static raster map for socio-economic layers and return the PDF path."""
    if isinstance(countries, str):
        countries = [countries]

    tag = f"{LOG_PREFIX}[{output_basename}]"

    raster_hint = f"Place {Path(raster_path).name} under pre-analysis/dataset/ or update the config path."
    raster_path = require_file(raster_path, hint=raster_hint)
    if verbose:
        print(f"{tag} Reading raster: {raster_path}")
    shapefile = shapefile or DEFAULT_WORLD_MAP_SHAPEFILE
    shapefile = require_file(shapefile, hint="Download Natural Earth admin boundaries under dataset/maps/.")
    if verbose:
        print(f"{tag} Using country shapefile: {shapefile}")
    city_shapefile = city_shapefile or DEFAULT_CITIES_SHAPEFILE
    city_shapefile = require_file(city_shapefile, hint="Place ne_110m_populated_places under dataset/maps/.")
    if verbose:
        print(f"{tag} Using city shapefile: {city_shapefile}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if gpd is None:
        raise ImportError("geopandas is required to overlay country borders; install via `pip install geopandas`.")

    world_raw = gpd.read_file(shapefile)
    if verbose:
        print(f"{tag} World shapefile CRS: {world_raw.crs}")
    world_filtered = _filter_countries(world_raw, countries, verbose=verbose)
    if world_filtered.empty:
        world_filtered = world_raw
        if verbose:
            print(f"{tag} No country filter applied; using full world outlines.")
    else:
        if verbose:
            matched = ", ".join(sorted(world_filtered["ADMIN"].unique()))
            print(f"{tag} Matched countries: {matched}")

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if verbose:
            crs_label = raster_crs.to_string() if raster_crs else "None"
            print(f"{tag} Raster CRS (GeoTIFF): {crs_label}")
        # Clip to the country bounds to speed up plotting when possible.
        data = src.read(1, masked=True)
        transform = src.transform
        if verbose:
            print(f"{tag} Full raster shape: {data.shape}")
        if not world_filtered.empty and raster_crs:
            world_proj = world_filtered.to_crs(raster_crs)
            bounds = world_proj.total_bounds  # xmin, ymin, xmax, ymax
            if verbose:
                print(f"{tag} Clipping raster to bounds {bounds}")
            window = window_from_bounds(*bounds, transform=src.transform)
            data = src.read(1, window=window, masked=True)
            transform = src.window_transform(window)
            if verbose:
                print(f"{tag} Window shape: {data.shape}")
        extent = plotting_extent(data, transform)
        if verbose:
            print(f"{tag} Plot extent: {extent}")

    # Clean and normalize data for plotting.
    data = np.ma.masked_invalid(data)
    data = np.ma.masked_less_equal(data, 0)
    if verbose:
        positives = int((data > 0).sum())
        valid_vals = data.compressed()
        vmin_dbg = valid_vals.min() if valid_vals.size else None
        vmax_dbg = valid_vals.max() if valid_vals.size else None
        print(f"{tag} Raster stats (after mask>0): positives={positives}, min={vmin_dbg}, max={vmax_dbg}")
    valid = data.compressed()
    if valid.size == 0:
        message = f"{dataset_label} raster has no positive values in the selected extent."
        print(f"{tag} Warning: {message}")
        output_path = output_dir / f"{output_basename}.pdf"
        _update_status(output_dir, basename=output_basename, dataset_label=dataset_label, message=message)
        return _write_warning_placeholder(output_path, dataset_label, message, dpi=dpi)
    vmin, vmax = np.percentile(valid, percentile_clip)
    if vmax <= 0:
        message = f"{dataset_label} raster normalization failed; check input data."
        print(f"{tag} Warning: {message}")
        output_path = output_dir / f"{output_basename}.pdf"
        _update_status(output_dir, basename=output_basename, dataset_label=dataset_label, message=message)
        return _write_warning_placeholder(output_path, dataset_label, message, dpi=dpi)
    norm = LogNorm(vmin=max(vmin, 1e-6), vmax=vmax) if use_log_scale else Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    img = ax.imshow(
        data,
        extent=extent,
        cmap=cmap,
        norm=norm,
        origin="upper",
    )

    world_plot = world_filtered.to_crs(raster_crs) if raster_crs else world_filtered.to_crs(epsg=4326)
    if verbose:
        print(f"{tag} World outlines CRS for plotting: {world_plot.crs}")
    if world_plot.empty and verbose:
        print(f"{tag} Warning: world outlines empty after reprojection; skipping borders.")
    else:
        world_plot.boundary.plot(ax=ax, color="#2c3e50", linewidth=0.5)

    # Overlay major cities from the bundled Natural Earth dataset (filtered to the viewport).
    try:
        if world_plot.empty:
            raise ValueError("world outline empty; cannot place cities.")

        cities = gpd.read_file(city_shapefile)
        if verbose:
            print(f"{tag} Cities shapefile CRS: {cities.crs}")
        cities = cities.to_crs(world_plot.crs)
        if verbose:
            print(f"{tag} Cities CRS after reprojection: {cities.crs}")
        bbox = world_plot.total_bounds
        if np.any(np.isnan(bbox)):
            raise ValueError("invalid world bounds for city clipping.")
        cities = cities.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        if verbose:
            print(f"{tag} City candidates in view: {len(cities)}")
        if not cities.empty:
            cities.plot(ax=ax, markersize=10, color="#d84315", alpha=0.85, edgecolor="white", linewidth=0.4)
            label_fields = ("NAME", "NAMEASCII", "name", "Name", "NAMEALT", "GEO_NAME")
            pop_fields = ("POP_MAX", "POP_MIN", "POP_EST")
            for _, row in cities.iterrows():
                label = next(
                    (
                        row.get(field)
                        for field in label_fields
                        if field in row and isinstance(row.get(field), str) and row.get(field).strip()
                    ),
                    None,
                ) or "City"
                pop_val = next(
                    (
                        row.get(field)
                        for field in pop_fields
                        if field in row and row.get(field) not in (None, "")
                    ),
                    None,
                )
                pop_text = f" ({int(pop_val):,})" if pop_val is not None else ""
                geom = row.geometry
                ax.annotate(
                    f"{label}{pop_text}",
                    xy=(geom.x, geom.y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color="#e7eef7",
                    weight="bold",
                    path_effects=[patheffects.withStroke(linewidth=1.2, foreground="#2f3f50")],
                )
    except Exception as exc:
        if verbose:
            print(f"{LOG_PREFIX} Skipping city overlay: {exc}")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    title_suffix = ", ".join(countries) if countries else "World"
    resolved_title = title or f"{title_prefix or dataset_label} – {title_suffix}"
    ax.set_title(resolved_title, fontsize=12)

    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    resolved_scale_label = scale_label or (f"{dataset_label} (log scale)" if use_log_scale else dataset_label)
    cbar.set_label(resolved_scale_label)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(round(v)):,}" if v >= 1 else f"{v:g}"))

    output_path = output_dir / f"{output_basename}.pdf"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    _update_status(output_dir, basename=output_basename, dataset_label=dataset_label, message=None)
    if verbose:
        print(f"{tag} Saved static map to {output_path}")
    return output_path


if __name__ == "__main__":
    # Quick IDE-friendly configuration for ad hoc runs (no CLI args needed).
    selected_countries = ["Spain"]
    verbose = True
    use_log_scale = True
    output_root = SCRIPT_DIR / "output_standalone"

    default_maps = [
        {
            "key": "gdp",
            "raster": DEFAULT_RASTER,
            "basename": "gdp_map",
            "dataset_label": "GDP (2005 PPP $)",
            "title_prefix": "GDP density (2005, 1 km)",
            "cmap": "YlGnBu",
        },
        {
            "key": "population",
            "raster": DATASET_DIR / "pop" / "SSP2" / "SSP2_2020.tif",
            "basename": "population_map",
            "dataset_label": "Population (2020, persons)",
            "title_prefix": "Population density (2020, 1 km)",
            "cmap": "OrRd",
        },
    ]

    for config in default_maps:
        output = render_socioeconomic_map(
            raster_path=config["raster"],
            countries=selected_countries,
            shapefile=DEFAULT_WORLD_MAP_SHAPEFILE,
            city_shapefile=DEFAULT_CITIES_SHAPEFILE,
            output_dir=output_root / config["basename"],
            output_basename=config["basename"],
            dataset_label=config["dataset_label"],
            title_prefix=config["title_prefix"],
            cmap=config["cmap"],
            use_log_scale=use_log_scale,
            verbose=verbose,
        )
        print(f"{LOG_PREFIX} {config['key']} map saved to: {output}")
