"""Lightweight ERA5-Land climate overview utilities extracted from the notebook."""

from __future__ import annotations

import calendar
import glob
import os
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr

os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

try:  # Optional dependency; downloads are skipped when unavailable.
    import cdsapi  # type: ignore
except Exception:  # pragma: no cover - best-effort import
    cdsapi = None

# The ERA5 GRIB files require the `cfgrib` backend (install via `pip install cfgrib`)

VARIABLE_DISPLAY_NAMES = {
    "t2m": "2m Temperature",
    "2m_temperature": "2m Temperature",
    "tp": "Total Precipitation",
    "total_precipitation": "Total Precipitation",
}


def _get_var_display_name(var_name: str, custom: Optional[str] = None) -> str:
    if custom:
        return custom
    return VARIABLE_DISPLAY_NAMES.get(var_name, var_name.replace("_", " ").title())


def _get_var_units(dataset_collection, var: str) -> str:
    if isinstance(dataset_collection, dict):
        datasets = dataset_collection.values()
    else:
        datasets = [dataset_collection]
    for ds in datasets:
        if isinstance(ds, xr.Dataset) and var in ds:
            return ds[var].attrs.get("units", "")
    return ""


def get_bbox(ISO_A2: str) -> Tuple[float, float, float, float]:
    try:
        from cartopy.io import shapereader
    except Exception as exc:  # pragma: no cover
        raise ImportError("cartopy is required to resolve country bounding boxes.") from exc

    shp = shapereader.Reader(
        shapereader.natural_earth(
            resolution="10m", category="cultural", name="admin_0_countries"
        )
    )
    try:
        record = next(filter(lambda c: c.attributes["ISO_A2"] == ISO_A2, shp.records()))
    except StopIteration as exc:
        raise ValueError(f"No Natural Earth record found for ISO code {ISO_A2}") from exc
    x_west, y_south, x_east, y_north = record.geometry.bounds
    return x_west, y_south, x_east, y_north


def read_grib_file(grib_path: Path, step_type: Optional[str] = None, max_retries: int = 3) -> xr.Dataset:
    def remove_cfgrib_index_files(grib_path: Path) -> None:
        idx_files = glob.glob(f"{grib_path}.*.idx")
        for idx in idx_files:
            try:
                os.remove(idx)
            except Exception as exc:
                print(f"Warning: failed to remove idx file {idx}: {exc}")

    remove_cfgrib_index_files(grib_path)
    print(f"Opening GRIB file: {grib_path}")
    dataset: Optional[xr.Dataset] = None
    for attempt in range(max_retries):
        try:
            if step_type:
                ds_avgid = xr.open_dataset(
                    grib_path,
                    engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"stepType": "avgid"}},
                    decode_timedelta=True,
                )
                ds_avgas = xr.open_dataset(
                    grib_path,
                    engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"stepType": "avgas"}},
                    decode_timedelta=True,
                )
                dataset = xr.merge([ds_avgid, ds_avgas], compat="override")
            else:
                dataset = xr.open_dataset(grib_path, engine="cfgrib", decode_timedelta=True)
            break
        except Exception as exc:
            print(f"[Attempt {attempt+1}] Failed to open {grib_path}: {exc}")
            time.sleep(1)
    if dataset is None:
        raise RuntimeError(f"Unable to read GRIB file {grib_path}")
    return dataset


def extract_data(zip_path: Path, step_type: Optional[str] = None, extract_to: Path = Path("era5_extracted_files")) -> xr.Dataset:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    extract_to.mkdir(parents=True, exist_ok=True)
    zip_name = zip_path.stem
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    dataset: Optional[xr.Dataset] = None
    for root, _, files in os.walk(extract_to):
        for file in files:
            if file.lower().endswith((".grib", ".grb")):
                grib_path = Path(root) / file
                new_grib_name = f"{zip_name}.grib"
                new_grib_path = Path(root) / new_grib_name
                if grib_path != new_grib_path:
                    grib_path.rename(new_grib_path)
                    grib_path = new_grib_path
                dataset = read_grib_file(grib_path, step_type=step_type)
    if dataset is None:
        raise RuntimeError(f"No GRIB files were extracted from {zip_path}")
    return dataset


def convert_dataset_units(ds: xr.Dataset) -> xr.Dataset:
    conversions = {
        "t2m": {"factor": 1, "offset": -273.15, "new_unit": "degC"},
        "tp": {"factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "ro": {"factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "sro": {"factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "pev": {"factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "e": {"factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "sd": {"factor": 1000, "offset": 0, "new_unit": "mm"},
    }
    ds_converted = ds.copy()
    for var, conv in conversions.items():
        if var in ds_converted.data_vars:
            ds_converted[var] = ds_converted[var] * conv["factor"] + conv["offset"]
            ds_converted[var].attrs["units"] = conv["new_unit"]
    return ds_converted


def plot_monthly_mean(
    dataset: Dict[str, xr.Dataset],
    var: str,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    folder: Optional[Path] = None,
    display: bool = False,
    label_map: Optional[Dict[str, str]] = None,
    var_display_name: Optional[str] = None,
) -> None:
    n = len(dataset)
    label_map = label_map or {}
    var_label = _get_var_display_name(var, var_display_name)
    fig, axs = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True, sharey=True)
    if n == 1:
        axs = [axs]
    for ax, (iso, ds) in zip(axs, dataset.items()):
        da = ds[var]
        spatial_mean = da.mean(dim=[lat_name, lon_name])
        months_per_year = spatial_mean.groupby("time.year").count()
        complete_years = months_per_year.where(months_per_year == 12, drop=True)["year"].values
        spatial_mean_complete = spatial_mean.where(spatial_mean["time.year"].isin(complete_years), drop=True)
        by_year = spatial_mean_complete.groupby("time.year")
        annual_totals = by_year.sum()
        max_year = annual_totals["year"].values[annual_totals.argmax().item()]
        min_year = annual_totals["year"].values[annual_totals.argmin().item()]
        monthly_mean = spatial_mean.groupby("time.month").mean()
        max_year_monthly = spatial_mean.where(spatial_mean["time.year"] == max_year, drop=True).groupby(
            "time.month"
        ).mean()
        min_year_monthly = spatial_mean.where(spatial_mean["time.year"] == min_year, drop=True).groupby(
            "time.month"
        ).mean()
        ax.plot(
            [pd.Timestamp(f"2025-{m:02d}-01").strftime("%b") for m in monthly_mean["month"].values],
            monthly_mean,
            label="Mean (all years)",
            marker="o",
        )
        ax.plot(
            [pd.Timestamp(f"2025-{m:02d}-01").strftime("%b") for m in max_year_monthly["month"].values],
            max_year_monthly,
            label=f"Max Year ({max_year})",
            linestyle="--",
            marker="^",
        )
        ax.plot(
            [pd.Timestamp(f"2025-{m:02d}-01").strftime("%b") for m in min_year_monthly["month"].values],
            min_year_monthly,
            label=f"Min Year ({min_year})",
            linestyle=":",
            marker="v",
        )
        country_label = label_map.get(iso, iso)
        units = ds[var].attrs.get("units", "")
        ax.set_title(f"{country_label} - Monthly Average of {var_label}")
        ax.set_ylabel(f"{var_label} ({units})" if units else var_label)
        ax.grid(True)
        ax.legend()
    axs[-1].set_xlabel("Month")
    plt.tight_layout()
    if folder is not None:
        plt.savefig(folder / f"monthly_mean_{var}.pdf")
    if display:
        plt.show()
    plt.close(fig)


def scatter_annual_spatial_means(
    data: Dict[str, xr.Dataset],
    var_x: str = "t2m",
    var_y: str = "tp",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    folder: Optional[Path] = None,
    display: bool = False,
    label_map: Optional[Dict[str, str]] = None,
    var_x_display: Optional[str] = None,
    var_y_display: Optional[str] = None,
) -> None:
    markers = ["o", "s", "^", "D", "v", "<", ">", "P", "*", "X"]
    colors = plt.cm.tab10.colors
    label_map = label_map or {}
    fig = plt.figure(figsize=(10, 6))
    for i, (iso, ds) in enumerate(data.items()):
        x_spatial = ds[var_x].mean(dim=[lat_name, lon_name])
        y_spatial = ds[var_y].mean(dim=[lat_name, lon_name])
        x_annual = x_spatial.groupby("time.year").mean()
        y_annual = y_spatial.groupby("time.year").mean()
        plt.scatter(
            x_annual.values,
            y_annual.values,
            label=label_map.get(iso, iso),
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            edgecolor="black",
        )
    x_label = _get_var_display_name(var_x, var_x_display)
    y_label = _get_var_display_name(var_y, var_y_display)
    x_units = _get_var_units(data, var_x)
    y_units = _get_var_units(data, var_y)
    plt.xlabel(f"{x_label} ({x_units})" if x_units else x_label)
    plt.ylabel(f"{y_label} ({y_units})" if y_units else y_label)
    plt.title(f"Annual Spatial Mean: {x_label} vs {y_label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if folder:
        fig.savefig(folder / f"scatter_annual_spatial_means_{var_x}_{var_y}.pdf")
    if display:
        plt.show()
    plt.close(fig)


def plot_spatial_mean_timeseries_all_iso(
    dataset: Dict[str, xr.Dataset],
    var: str = "tp",
    agg: Optional[str] = None,
    folder: Optional[Path] = None,
    display: bool = False,
    label_map: Optional[Dict[str, str]] = None,
    var_display_name: Optional[str] = None,
) -> None:
    lat_name, lon_name = "latitude", "longitude"
    label_map = label_map or {}
    var_label = _get_var_display_name(var, var_display_name)
    units = _get_var_units(dataset, var)
    fig = plt.figure(figsize=(12, 6))
    for iso, ds in dataset.items():
        da = ds[var]
        spatial_mean = da.mean(dim=[lat_name, lon_name])
        if agg == "avg":
            spatial_mean = spatial_mean.groupby("time.year").mean()
            time_axis = spatial_mean["year"].values
        elif agg == "sum":
            spatial_mean = spatial_mean.groupby("time.year").sum()
            time_axis = spatial_mean["year"].values
        else:
            time_axis = ds["time"].values
        plt.plot(time_axis, spatial_mean, label=f"{label_map.get(iso, iso)} - mean")
    plt.title(f"Temporal Evolution of Spatial Mean of {var_label}")
    plt.xlabel("Time")
    plt.ylabel(f"{var_label} ({units})" if units else var_label)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    if folder is not None:
        fig.savefig(folder / f"spatial_mean_{var}_{agg}.pdf")
    if display:
        plt.show()
    plt.close(fig)


def plot_monthly_precipitation_heatmap(
    dataset: Dict[str, xr.Dataset],
    var: str = "tp",
    cmap: str = "cividis",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    path: Optional[Path] = None,
    display: bool = False,
    label_map: Optional[Dict[str, str]] = None,
    var_display_name: Optional[str] = None,
) -> None:
    data_rows = []
    label_map = label_map or {}
    var_label = _get_var_display_name(var, var_display_name)
    units = _get_var_units(dataset, var)
    units_label = f" ({units})" if units else ""
    for iso, ds in dataset.items():
        if var not in ds:
            continue
        da = ds[var]
        spatial_mean = da.mean(dim=[lat_name, lon_name])
        monthly_total = spatial_mean.groupby("time.month").mean(dim="time")
        monthly_series = monthly_total.to_series()
        country_label = label_map.get(iso, iso)
        for month in range(1, 13):
            value = monthly_series.get(month, 0.0)
            data_rows.append({"country": country_label, "month": month, "precip": value})
    df = pd.DataFrame(data_rows)
    heatmap_data = df.pivot(index="country", columns="month", values="precip").fillna(0)
    heatmap_data = heatmap_data.reindex(columns=range(1, 13))
    desired_order = [label_map.get(iso, iso) for iso in dataset.keys()]
    desired_order = [country for country in desired_order if country in heatmap_data.index]
    heatmap_data = heatmap_data.reindex(desired_order)
    fig_height = max(4, 0.5 * len(heatmap_data))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        annot=True,
        fmt=".0f",
        cbar_kws={"label": f"Mean {var_label}{units_label}"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Monthly Mean {var_label} per Country{units_label}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Country")
    ax.set_xticks([i + 0.5 for i in range(12)])
    ax.set_xticklabels([calendar.month_abbr[i + 1] for i in range(12)], rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    if display:
        plt.show()
    plt.close(fig)

ERA5_DATASET = "reanalysis-era5-land-monthly-means"
VARIABLE_CODES = {
    "total_precipitation": "tp",
    "surface_runoff": "sro",
    "runoff": "ro",
    "snow_depth_water_equivalent": "sd",
    "2m_temperature": "t2m",
    "potential_evaporation": "pev",
    "total_evaporation": "e",
}
DISPLAY_NAMES = {
    "t2m": "Temperature",
    "tp": "Precipitation",
}


def _require_utils():
    try:
        from cartopy.io import shapereader  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "climate_overview requires cartopy/shapereader (and related xarray stack) for spatial helpers."
        ) from exc


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _var_label(var: str) -> str:
    _require_utils()
    return _get_var_display_name(var, DISPLAY_NAMES.get(var))


def _build_download_plan(
    iso_codes: Sequence[str],
    variables: Sequence[str],
    start_year: int,
    end_year: int,
    dataset_name: str,
    api_dir: Path,
) -> Dict[str, Path]:
    suffix = "_".join(VARIABLE_CODES.get(v, v) for v in variables)
    plan = {}
    for iso in iso_codes:
        filename = f"{dataset_name}_{iso}_{start_year}_{end_year}_{suffix}.zip"
        plan[iso] = api_dir / filename
    return plan


def _natural_earth_records():
    try:
        from cartopy.io import shapereader
    except Exception:  # pragma: no cover
        return tuple()

    try:
        reader = shapereader.Reader(
            shapereader.natural_earth(
                resolution="10m", category="cultural", name="admin_0_countries"
            )
        )
    except Exception:
        return tuple()

    return tuple(reader.records())


@lru_cache(maxsize=None)
def _natural_earth_name_lookup() -> Dict[str, str]:
    records = _natural_earth_records()
    lookup: Dict[str, str] = {}
    for record in records:
        iso = record.attributes.get("ISO_A2")
        if not iso:
            continue
        iso = iso.upper()
        for key in ("NAME_LONG", "ADMIN", "NAME"):
            name = record.attributes.get(key)
            if isinstance(name, str):
                normalized = name.strip().lower()
                if normalized and normalized not in lookup:
                    lookup[normalized] = iso
    return lookup


def _natural_earth_label_map(iso_codes: Sequence[str]) -> Dict[str, str]:
    records = _natural_earth_records()
    if not records:
        return {}

    targets = {code.upper() for code in iso_codes if isinstance(code, str)}
    labels: Dict[str, str] = {}
    for record in records:
        iso = record.attributes.get("ISO_A2")
        if not iso:
            continue
        iso = iso.upper()
        if iso not in targets or iso in labels:
            continue
        for key in ("NAME_LONG", "ADMIN", "NAME"):
            name = record.attributes.get(key)
            if isinstance(name, str) and name.strip():
                labels[iso] = name.strip()
                break
    return labels


def _countries_to_iso_codes(countries: Sequence[str]) -> List[str]:
    lookup = _natural_earth_name_lookup()
    seen: List[str] = []
    missing: List[str] = []
    for country in countries:
        if not country:
            continue
        candidate = str(country).strip()
        if not candidate:
            continue
        normalized = candidate.lower()
        if len(normalized) == 2 and normalized.isalpha():
            iso = normalized.upper()
        else:
            iso = lookup.get(normalized)
        if iso:
            seen.append(iso)
        else:
            missing.append(candidate)
    if missing:
        raise ValueError(f"Could not resolve ISO codes for: {', '.join(missing)}")
    return seen


def _enrich_label_map(
    iso_codes: Sequence[str], custom_label_map: Optional[Dict[str, str]]
) -> Dict[str, str]:
    label_map = dict(custom_label_map or {})
    existing_keys = {key.upper() for key in label_map if isinstance(key, str)}
    iso_lookup = {code.upper(): code for code in iso_codes if isinstance(code, str)}
    missing = [code for code in iso_lookup.keys() if code not in existing_keys]
    if missing:
        native_labels = _natural_earth_label_map(missing)
        for iso_upper, name in native_labels.items():
            original_iso = iso_lookup.get(iso_upper, iso_upper)
            label_map.setdefault(original_iso, name)
    return label_map


def maybe_download_era5(
    plan: Dict[str, Path],
    iso_bboxes: Dict[str, Tuple[float, float, float, float]],
    variables: Sequence[str],
    start_year: int,
    end_year: int,
    dataset_name: str = ERA5_DATASET,
    download: bool = False,
) -> None:
    """Download missing ERA5-Land files when requested (best-effort)."""
    missing = [iso for iso, path in plan.items() if not path.exists()]
    if not missing:
        return
    if not download:
        raise FileNotFoundError(
            f"Missing ERA5 archives for {', '.join(missing)}; set download=True to fetch them."
        )
    if cdsapi is None:
        raise ImportError("cdsapi is required to download ERA5 data.")

    years = [str(y) for y in range(start_year, end_year + 1)]
    client = cdsapi.Client()

    for iso in missing:
        lon_w, lat_s, lon_e, lat_n = iso_bboxes[iso]
        print(f"[climate] Downloading {iso} to {plan[iso]}...")
        client.retrieve(
            dataset_name,
            {
                "format": "grib",
                "product_type": "monthly_averaged_reanalysis",
                "variable": variables,
                "year": years,
                "month": [f"{m:02d}" for m in range(1, 13)],
                "time": "00:00",
                "area": [lat_s, lon_w, lat_n, lon_e],
            },
            str(plan[iso]),
        )


def load_era5_archives(
    plan: Dict[str, Path],
    extract_dir: Path,
    convert_units_flag: bool = True,
) -> Dict[str, xr.Dataset]:
    _require_utils()
    datasets: Dict[str, xr.Dataset] = {}
    for iso, zip_path in plan.items():
        print(f"[climate] Reading {zip_path.name} for {iso}")
        ds = extract_data(str(zip_path), step_type=True, extract_to=str(extract_dir / iso))
        if convert_units_flag:
            ds = convert_dataset_units(ds)
        datasets[iso] = ds
    return datasets


def summarize_climate(
    datasets: Dict[str, xr.Dataset],
    variables: Sequence[str],
    label_map: Optional[Dict[str, str]] = None,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> pd.DataFrame:
    """Create a compact summary table with overall means per country/variable."""
    rows: List[Dict[str, object]] = []
    for iso, ds in datasets.items():
        country = (label_map or {}).get(iso, iso)
        years = pd.to_datetime(ds.time.values).year
        period = f"{years.min()}–{years.max()}"
        for var in variables:
            var_key = VARIABLE_CODES.get(var, var)
            if var_key not in ds:
                continue
            spatial = ds[var_key].mean(dim=[lat_name, lon_name])
            rows.append(
                {
                    "Country": country,
                    "Variable": _var_label(var_key),
                    "Period": period,
                    "Mean": float(spatial.mean()),
                    "Min": float(spatial.min()),
                    "Max": float(spatial.max()),
                    "Units": ds[var_key].attrs.get("units", ""),
                }
            )
    return pd.DataFrame(rows)


def default_output_paths(output_dir: Path, variables: Sequence[str]) -> List[Path]:
    """Standard figures/files used by the workflow for reporting."""
    outputs: List[Path] = [output_dir / "climate_summary.csv"]
    for var in variables:
        var_key = VARIABLE_CODES.get(var, var)
        outputs.append(output_dir / f"spatial_mean_{var_key}_None.pdf")
        agg = "avg" if var_key == "t2m" else "sum"
        outputs.append(output_dir / f"spatial_mean_{var_key}_{agg}.pdf")
        outputs.append(output_dir / f"monthly_mean_{var_key}.pdf")
    outputs.append(output_dir / "scatter_annual_spatial_means_t2m_tp.pdf")
    outputs.append(output_dir / "monthly_precipitation_heatmap.png")
    return outputs


def run_climate_overview(
    iso_codes: Sequence[str],
    start_year: int,
    end_year: int,
    variables: Sequence[str],
    api_dir: Path,
    extract_dir: Path,
    output_dir: Path,
    dataset_name: str = ERA5_DATASET,
    download: bool = False,
    generate_plots: bool = True,
    save_netcdf: bool = True,
    label_map: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Main entry point to generate climate datasets + figures."""
    _require_utils()
    api_dir = _ensure_dir(Path(api_dir))
    extract_dir = _ensure_dir(Path(extract_dir))
    output_dir = _ensure_dir(Path(output_dir))

    plan = _build_download_plan(iso_codes, variables, start_year, end_year, dataset_name, api_dir)
    bboxes = {iso: get_bbox(iso) for iso in iso_codes}
    maybe_download_era5(plan, bboxes, variables, start_year, end_year, dataset_name, download=download)

    datasets = load_era5_archives(plan, extract_dir, convert_units_flag=True)

    summary_df = summarize_climate(datasets, variables, label_map=label_map)
    summary_path = output_dir / "climate_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    outputs = {"summary": summary_path, "figures": []}
    if save_netcdf:
        for iso, ds in datasets.items():
            suffix = "_".join(VARIABLE_CODES.get(v, v) for v in variables)
            nc_path = output_dir / f"{dataset_name}_{iso}_{start_year}_{end_year}_{suffix}.nc"
            ds.to_netcdf(nc_path)
            outputs["figures"].append(nc_path)

    if not generate_plots:
        return outputs

    label_map = label_map or {}
    for var in variables:
        var_key = VARIABLE_CODES.get(var, var)
        display_name = DISPLAY_NAMES.get(var_key)
        if any(var_key in ds for ds in datasets.values()):
            plot_spatial_mean_timeseries_all_iso(
                datasets,
                var=var_key,
                agg=None,
                folder=output_dir,
                display=False,
                label_map=label_map,
                var_display_name=display_name,
            )
        agg = "avg" if var_key == "t2m" else "sum"
        if any(var_key in ds for ds in datasets.values()):
            plot_spatial_mean_timeseries_all_iso(
                datasets,
                var=var_key,
                agg=agg,
                folder=output_dir,
                display=False,
                label_map=label_map,
                var_display_name=display_name,
            )
            plot_monthly_mean(
                datasets,
                var_key,
                folder=output_dir,
                display=False,
                label_map=label_map,
                var_display_name=display_name,
            )

    if {"t2m", "tp"}.issubset({VARIABLE_CODES.get(v, v) for v in variables}):
        scatter_annual_spatial_means(
            datasets,
            var_x="t2m",
            var_y="tp",
            folder=output_dir,
            display=False,
            label_map=label_map,
            var_x_display=DISPLAY_NAMES.get("t2m"),
            var_y_display=DISPLAY_NAMES.get("tp"),
        )

    if "total_precipitation" in variables or "tp" in variables:
        plot_monthly_precipitation_heatmap(
            datasets,
            var="tp",
            path=output_dir / "monthly_precipitation_heatmap.png",
            display=False,
            label_map=label_map,
            var_display_name=DISPLAY_NAMES.get("tp"),
        )

    return outputs


# === Simple IDE-friendly defaults ===
# Tweak these variables before running this script directly from an editor.
COUNTRIES: List[str] = ["Albania"]
START_YEAR: int = 1990
END_YEAR: int = 2000
VARIABLES: List[str] = ["2m_temperature", "total_precipitation"]
OUTPUT_ROOT: Path = HERE / "climate_debug"
API_DIR: Optional[Path] = None
EXTRACT_DIR: Optional[Path] = None
OUTPUT_DIR: Optional[Path] = None
DATASET: str = ERA5_DATASET
DOWNLOAD: bool = True
GENERATE_PLOTS: bool = True
SAVE_NETCDF: bool = True


if __name__ == "__main__":
    iso_codes = _countries_to_iso_codes(COUNTRIES)
    api_dir = API_DIR or (OUTPUT_ROOT / "api")
    extract_dir = EXTRACT_DIR or (OUTPUT_ROOT / "extract")
    output_dir = OUTPUT_DIR or (OUTPUT_ROOT / "output")
    label_map = _enrich_label_map(iso_codes, None)

    print(f"[climate_overview] Preparing {len(iso_codes)} countries: {', '.join(iso_codes)}")
    print(f"[climate_overview] Output root: {OUTPUT_ROOT}")

    run_climate_overview(
        iso_codes=iso_codes,
        start_year=START_YEAR,
        end_year=END_YEAR,
        variables=VARIABLES,
        api_dir=api_dir,
        extract_dir=extract_dir,
        output_dir=output_dir,
        dataset_name=DATASET,
        download=DOWNLOAD,
        generate_plots=GENERATE_PLOTS,
        save_netcdf=SAVE_NETCDF,
        label_map=label_map,
    )
