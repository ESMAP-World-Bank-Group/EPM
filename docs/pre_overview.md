# Preanalysis Folder Overview

The `pre-analysis/` workspace separates exploratory data ingestion from model-ready processing. 
- Use **`prepare-data/`** to turn those datasets into Electricity Planning Model (EPM) inputs such as `pAvailability`, `pVREgenProfile`, and demand profiles.
- Use **`open-data/`** to download, QA, and harmonize external datasets. 

---

## Objective

Produce clean, versioned inputs for EPM by:
- reshaping and validating those datasets against the current perimeter (prepare-data stage)
- exporting consistent CSVs that match the structure in `epm/input/data_capp`
- curating third-party climate, hydro, renewable, and generation data (open-data stage)

---

## Workspace Layout

| Path | Role | Highlights |
|------|------|------------|
| `pre-analysis/open-data/` | Exploratory notebooks that ingest APIs, shapefiles, and atlas workbooks, plus QA tools (plots, Folium maps). | Renewable Ninja & IRENA harvesters, GRDC inflow prep, hydro basin QA, hydro atlas comparisons. |
| `pre-analysis/prepare-data/` | Deterministic workflows that reshape curated datasets into EPM-ready CSVs and diagnostics. | Climatic overview, load profile builders, representative days, hydro availability, supply-demand balance checks. |

---

## prepare-data workflows

| Notebook / Module | Purpose | Key outputs |
|-------------------|---------|-------------|
| `climatic_overview.ipynb` | Profiles ERA5-Land temperature/precipitation to define seasons, wet/dry periods, and candidate representative years for each zone. | Climate diagnostics in `output/` plus summary CSVs used downstream. |
| `load_profile.ipynb` | Builds hourly demand profiles by fusing monthly means with hourly shapes and sanity checks. | Hourly `load_profile.csv` saved under `output/` for direct use in EPM. |
| `load_profile_treatment.ipynb` | Cleans historical load measurements (outlier removal, missing-data infill) before feeding the builder. | Treated historical series in `output/load_profile_treated.csv`. |
| `load_plot.ipynb` | Generates forecast and QA plots for stakeholder review (peak vs average, growth trends). | PNG/HTML dashboards in `output/plots/`. |
| `representative_days/representative_days.ipynb` | Clusters climate and load time series to produce reduced time slices. | `pHours.csv`, `load/pDemandProfile.csv`, `supply/pVREProfile.csv`. |
| `supply_demand_balance.ipynb` | Checks that the supply fleet plus renewables meet the treated demand under each scenario; flags deficits before GAMS runs. | Balance tables/plots in `output/` plus optional CSV deltas. |
| `hydro_availability.ipynb` | Converts monthly hydro shapes into reservoir `pAvailabilityCustom.csv` and ROR `pVREgenProfile.csv`, validating against `pHours`. | Final hydro CSVs under `output/`. |
| `hydro_representative_years.ipynb` | Experimental picker for representative hydropower years; use to sample dry/baseline/wet seasons before exporting availability tables. | Candidate `pAvailability_*.csv` files (review manually). |
| `utils_climatic.py` | Shared helpers for ERA5 extraction, aggregation, and plotting. | Imported across notebooks; no standalone output. |
| `legacy_to_new_format/` | Migration scripts that convert historic SPLAT/EPM spreadsheets into the current column naming. | Intermediate CSVs stored locally before copying to `epm/input`. |

**Inputs & outputs**: Every subfolder follows the same rule—drop raw/intermediate assets into `input/`, and keep notebook-produced artifacts inside `output/` until you promote them into `epm/input`.

---

## open-data notebooks

| Notebook | Focus & what you get | Typical outputs |
|----------|----------------------|-----------------|
| `get_renewables_irena_data.ipynb` | Downloads IRENA wind/solar profiles using SPLAT naming, producing hourly capacity-factor tables per zone-season. | CSV grids plus QA plots under `output/`. |
| `get_renewable_ninja_data.ipynb` | Calls the Renewable Ninja API using coordinates from the generation catalog; writes harmonized solar/wind profiles. | Hourly CF CSVs (`zone,season,day,hour,<year>`). |
| `get_renewables_coordinate.ipynb` | Builds the coordinate list (lat/lon) from generation assets so Renewable Ninja pulls the right plants. | Coordinate CSV consumed by the Ninja notebook. |
| `get_generation_maps.ipynb` | Visualizes generation databases on interactive maps to verify coverage and technology tagging. | HTML/PNG maps in `output/maps/`. |
| `hydro_atlas_comparison.ipynb` | Compares utility capacity factors with the African Hydropower Atlas before adopting Atlas curves. | QA plots plus comparison tables (save manually as needed). |
| `hydro_basins.ipynb` | Inspects GRDC catchments and HydroRIVERS shapefiles to link plants with upstream basins. | GeoDataFrames/maps stored under `output/`. |
| `hydro_capacity_factors.ipynb` | (WIP) Merges African Hydropower Atlas profiles with Global Hydropower Tracker metadata for a consolidated catalog. | Draft merged tables in `output/`. |
| `hydro_inflow.ipynb` | Processes GRDC NetCDF station data, intersects HydroRIVERS, and exports inflow/runoff diagnostics. | Cleaned CSVs/GeoPackages plus Folium maps. |

Use these notebooks when you need to refresh the underlying open datasets. Once the exploratory outputs look correct, feed them into the deterministic routines inside `prepare-data/`.

---

## Shared input/output conventions

- **`input/`**: raw downloads, API responses, shapefiles, and any intermediate CSVs that need to be versioned.
- **`output/`**: notebook artifacts (plots, QA tables, temporary CSVs). Copy only the vetted deliverables into `epm/input/data_capp` or `epm/input/data_<region>` to keep git noise low.

Maintaining this separation preserves reproducibility, makes it clear which datasets entered the model, and accelerates updates when new countries or data vintages are added.

---
