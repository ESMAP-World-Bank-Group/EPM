"""Lightweight ERA5-Land climate overview utilities extracted from the notebook."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import xarray as xr

os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

try:  # Optional dependency; downloads are skipped when unavailable.
    import cdsapi  # type: ignore
except Exception:  # pragma: no cover - best-effort import
    cdsapi = None

_IMPORT_ERROR: Optional[Exception] = None
try:
    from utils_climatic import (
        convert_dataset_units,
        extract_data,
        get_bbox,
        plot_monthly_mean,
        plot_monthly_precipitation_heatmap,
        plot_spatial_mean_timeseries_all_iso,
        scatter_annual_spatial_means,
        _get_var_display_name,
    )
except Exception as exc:  # pragma: no cover - optional dependency guard
    _IMPORT_ERROR = exc
    convert_dataset_units = extract_data = get_bbox = None  # type: ignore
    plot_monthly_mean = plot_monthly_precipitation_heatmap = None  # type: ignore
    plot_spatial_mean_timeseries_all_iso = scatter_annual_spatial_means = None  # type: ignore
    _get_var_display_name = lambda name, custom=None: custom or name  # type: ignore

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
    "t2m": "2m Temperature",
    "tp": "Total Precipitation",
}


def _require_utils():
    if _IMPORT_ERROR:
        raise ImportError(
            "climate_overview requires utils_climatic dependencies (cartopy/xarray stack)."
        ) from _IMPORT_ERROR


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
