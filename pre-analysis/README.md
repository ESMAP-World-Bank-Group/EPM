# Open-Data Workflows

This folder contains the exploratory and ingestion notebooks that pull hydrologic and renewable datasets into a common structure before they feed the EPM model. Use the catalog below to pick the workflow that matches the question you are trying to answer.

## Running the Snakemake workflow
- Prep inputs/config: place `dataset/Global-Integrated-Power-April-2025.xlsx` (and IRENA CSVs under `irena.input_dir`) in the expected folders; copy `config/api_tokens.example.ini` → `config/api_tokens.ini` and add a `renewables_ninja` token (or export `API_TOKEN_RENEWABLES_NINJA`); tweak countries/years in `config/open_data_config.yaml` as needed.
- Fix the conda env path so Snakemake can find it: the Snakefile points to `envs/renewables.yaml` but the file is `renewables.yml`. Either create the expected path (`mkdir -p envs && ln -s ../renewables.yml envs/renewables.yaml`) or edit the Snakefile to reference `renewables.yml`.
- Create the env once (or let Snakemake do it after the path fix): `conda env create -f renewables.yml -n epm-open-data`.
- CBC solver is included in the env (`coin-or-cbc`) so PuLP never falls back to Gurobi; make sure `cbc` is on your PATH (the conda env activation does this) to avoid license warnings.
- Run from `pre-analysis/open-data`: `snakemake --snakefile Snakefile --cores 1 --use-conda --conda-frontend mamba` (drop `--use-conda` only if you already activated `epm-open-data`).
- Outputs land in `output/`: GAP filtered CSV + Renewables Ninja CSVs + IRENA CSVs. VRE outputs now follow the unified naming/shape `vre_<source>_<label?>_<tech>.csv` with columns `zone,season,day,hour,<year columns>`.
- Optional: to compute representative days via the prepare-data pipeline, fill `representative_days` in `config/open_data_config.yaml` (set `enabled: true`, point `input_files` to your hourly CSVs, adjust seasons/map and counts). Snakemake will then emit `repr_days.csv`, `pHours.csv`, and `pVREProfile.csv` under the configured `output_dir` (plus `pDemandProfile.csv` when a load series is provided).
- Socio-economic static maps (GDP, population): enable the `socioeconomic_maps` block in `config/open_data_config.yaml` to render PDFs under `output_workflow/socioeconomic/` from the configured rasters using Natural Earth country outlines. For a quick ad-hoc run from the IDE, execute `python socioeconomic_map_pipeline.py` after adjusting the dataset list and `selected_countries` at the bottom of the file.

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

## API tokens
- Copy `config/api_tokens.example.ini` to `config/api_tokens.ini` and fill in your keys under the `[api_tokens]` section (git-ignored).
- Renewables Ninja looks for `renewables_ninja`; you can add other APIs (e.g., `enstoe`) as new keys.
- You can also set environment variables instead of a file: `API_TOKEN_RENEWABLES_NINJA=<token>`. To point to a non-default config file, set `API_TOKENS_PATH=/path/to/api_tokens.ini`.
- Token config is stored at the project root under `config/api_tokens.ini` so all open-data utilities can share it.

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
