# Open-Data Workflows

This folder contains the exploratory and ingestion notebooks that pull hydrologic and renewable datasets into a common structure before they feed the EPM model. Use the catalog below to pick the workflow that matches the question you are trying to answer.

## Notebook catalog

| Notebook | Focus & what you get | Key inputs (relative to `pre-analysis/open-data`) | Outputs |
|----------|----------------------|---------------------------------------------------|---------|
| `hydro_atlas_comparison.ipynb` | Compares utility-reported capacity factors against the African Hydropower Atlas to validate magnitude and seasonality before adopting Atlas curves. Generates per-plant plots for QA/QC. | Utility CSVs dropped in `input/utility/` plus `input/African_Hydropower_Atlas_v2-0.xlsx`. | PNG/inline plots plus cleaned comparison tables in memory (save manually as needed). |
| `hydro_basins.ipynb` | Visualizes GRDC catchment polygons (`stationbasins.geojson`) and inspects metadata such as drainage areas, pour points, and quality flags so you can trace basins tied to plants. | Sub-folders under `data_grdc_hydro_capp/input/**/stationbasins.geojson`. | Map layers rendered in-notebook and aggregated GeoDataFrames for export if needed. |
| `hydro_capacity_factors.ipynb` | Work-in-progress pipeline to merge the African Hydropower Atlas with the Global Hydropower Tracker for a consolidated hydropower capacity catalog. | `input/African_Hydropower_Atlas_v2-0.xlsx`; `input/Global-Hydropower-Tracker-*.xlsx`. | Draft merged tables (WIP—inspect notebook before relying on outputs). |
| `hydro_inflow.ipynb` | End-to-end GRDC workflow: load NetCDF station data, intersect with HydroRIVERS and the Global Hydropower Tracker, and export cleaned inflow/runoff datasets plus exploratory maps. | `input/grdc_input/**/GRDC-Monthly.nc`; `input/river_input/HydroRIVERS_v10_af_shp`; `input/Global-Hydropower-Tracker-April-2025.xlsx`. | Processed CSVs/GeoPackages in `output/`, interactive Folium maps, diagnostic plots. |

### Directory tips
- `input/` holds the raw GRDC NetCDF, HydroRIVERS shapefiles, Atlas workbooks, and any utility CSVs you manually download.
- `output/` is git-ignored so you can iterate freely before copying vetted tables onward.
- `in_progress/` notebooks should document any assumptions here before promoting them into the main workflow.

## Key datasets referenced by these notebooks

### Global Runoff Data Centre (GRDC)
The GRDC hosts the most complete collection of quality-assured river discharge observations worldwide. Data are downloaded manually from the [GRDC Data PORTAL](https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser#dataDownload/StationCatalogue) as station-level time series (no API or gridded option). Review the [FAQ](https://grdc.bafg.de/help/faq/) to choose stations and formats.

### VegDischarge
VegET-based gridded runoff is routed through the mizuRoute model, producing discharge estimates for 63k African river reaches. Five routing algorithms are available—IRF, KWT, KW, MC, DW—and prior work suggests KWT or KW gives the most reliable hydrology for our purposes.

### ERA5-Land monthly averages
- Period: **Jan 1950 to ~2–3 months before present**
- Resolution: 0.1° (~9 km)
- Dataset: `monthly_averaged_reanalysis` via the CDS API

Key variables and units:

| Variable | Unit | Notes |
|----------|------|-------|
| `2m_temperature` | Kelvin | Subtract 273.15 for °C. |
| `total_precipitation` | m/day | Multiply by 1000 for mm/day, then by days-per-month for totals. |
| `surface_runoff` | m/day | Same conversion as precipitation. |
| `snow_depth_water_equivalent` | m | Instantaneous depth. |
| `potential_evaporation` | m/day | Multiply by 1000 for mm/day. |
| `total_evaporation` | m/day | Multiply by 1000 for mm/day. |

Accumulated variables are stored as daily means; multiply by the number of days per month to recover totals. Instantaneous variables already represent monthly means.

To read GRIB files, filter by `stepType` to separate instantaneous (`avgid`) from accumulated (`avgas`) fields:

```python
import xarray as xr

ds_instant = xr.open_dataset("data.grib", engine="cfgrib",
                             backend_kwargs={"filter_by_keys": {"stepType": "avgid"}})
ds_accum = xr.open_dataset("data.grib", engine="cfgrib",
                           backend_kwargs={"filter_by_keys": {"stepType": "avgas"}})
```

### GRUN
GRUN provides a global reconstruction of monthly runoff for 1902–2014 on a 0.5° grid (mm/day), distributed as NetCDFv4. The model is trained with GSIM streamflow and meteorological forcings from GSWP3. Download from [figshare](https://figshare.com/articles/dataset/GRUN_Global_Runoff_Reconstruction/9228176) and see the [ESSD publication](https://essd.copernicus.org/articles/11/1655/2019/) for methodology.
