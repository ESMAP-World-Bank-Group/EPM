# Data Preparation

The `pre-analysis/` workspace sits upstream of EPM — it turns raw external datasets into clean, model-ready CSV inputs. It is split into two stages:

| Stage | Folder | Role |
|---|---|---|
| **Open data** | `pre-analysis/open-data/` | Download, QA, and harmonize external datasets (APIs, shapefiles, atlases) |
| **Prepare data** | `pre-analysis/prepare-data/` | Reshape curated datasets into EPM-ready CSVs (demand profiles, hydro availability, VRE profiles, representative days) |

**Convention:** drop raw and intermediate files in `input/`, keep notebook outputs in `output/`. Only copy vetted deliverables into `epm/input/data_<region>/` to keep the model folder clean.

---

## prepare-data workflows

| Notebook | Purpose | Key outputs |
|---|---|---|
| `climatic_overview.ipynb` | Profiles ERA5-Land climate to define seasons, wet/dry periods, and representative years per zone | Climate diagnostics and summary CSVs |
| `load_profile.ipynb` | Builds hourly demand profiles from monthly means and hourly shapes | `load_profile.csv` |
| `load_profile_treatment.ipynb` | Cleans historical load data (outlier removal, gap filling) | `load_profile_treated.csv` |
| `load_plot.ipynb` | QA plots for demand forecasts (peak vs average, growth trends) | PNG/HTML dashboards |
| `representative_days.ipynb` | Clusters climate and load time series into reduced time slices | `pHours.csv` · `pDemandProfile.csv` · `pVREProfile.csv` |
| `supply_demand_balance.ipynb` | Checks that supply meets demand before running GAMS; flags deficits | Balance tables and plots |
| `hydro_availability.ipynb` | Converts monthly hydro shapes into reservoir availability and ROR profiles | `pAvailabilityCustom.csv` · `pVREgenProfile.csv` |
| `hydro_representative_years.ipynb` | Selects representative hydropower years (dry/baseline/wet) | Candidate `pAvailability_*.csv` (review manually) |
| `utils_climatic.py` | Shared helpers for ERA5 extraction, aggregation, and plotting | — |
| `legacy_to_new_format/` | Migrates legacy SPLAT/EPM spreadsheets to the current column format | Intermediate CSVs |

---

## open-data notebooks

| Notebook | Focus | Outputs |
|---|---|---|
| `get_renewables_irena_data.ipynb` | IRENA wind/solar capacity-factor profiles by zone and season | Hourly CF CSVs |
| `get_renewable_ninja_data.ipynb` | Renewable Ninja API — solar/wind profiles from plant coordinates | Hourly CF CSVs |
| `get_renewables_coordinate.ipynb` | Builds coordinate list (lat/lon) from generation catalog for Renewable Ninja | Coordinate CSV |
| `get_generation_maps.ipynb` | Interactive maps to verify generation database coverage and technology tagging | HTML/PNG maps |
| `hydro_atlas_comparison.ipynb` | Compares utility capacity factors with the African Hydropower Atlas | QA plots and comparison tables |
| `hydro_basins.ipynb` | Links plants to upstream GRDC catchments via HydroRIVERS shapefiles | GeoDataFrames and maps |
| `hydro_capacity_factors.ipynb` | *(WIP)* Merges African Hydropower Atlas with Global Hydropower Tracker | Draft merged tables |
| `hydro_inflow.ipynb` | Processes GRDC NetCDF inflow data and exports runoff diagnostics | Cleaned CSVs and Folium maps |

Once exploratory outputs look correct, feed them into the deterministic routines in `prepare-data/`.
