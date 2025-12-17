# Prepare-Data Workflows

Use this folder after the raw hydrologic and renewable datasets have been downloaded. The notebooks here reshape, validate, and export EPM-ready CSV inputs such as `pAvailability`, `pVREgenProfile`, and other model tables.

## Notebook catalog

| Notebook                                          | Focus & what you get                                                                                                                                                                               | Key inputs (relative to `pre-analysis/prepare-data`)                                                                                                              | Outputs                                                                                                                       |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `generate_hydro_capacity_inputs.ipynb`            | Converts Hydropower Atlas profiles into GAMS-ready seasonal inputs. Builds reservoir `pAvailability` and run-of-river `pVREgenProfile` tables across baseline/dry/wet scenarios.                   | `input/African_Hydropower_Atlas_v2-0.xlsx`; `../../epm/input/data_capp/supply/pGenDataInput_clean.csv`; `../../epm/input/data_capp/pHours.csv`.                   | Scenario-specific `output/pAvailability_<scenario>.csv` and `output/pVREgenProfile_<scenario>.csv`.                           |
| `hydro_availability.ipynb`                        | Converts monthly hydro profiles into the `pAvailabilityCustom.csv` and hourly ROR `pVREgenProfile.csv` templates expected by the EPM model. Includes validation against `pHours.csv`.              | `input/hydro_profile_*.csv` (default `hydro_profile_dry.csv`) with `gen/zone/tech/month` columns; `../../epm/input/data_capp/pHours.csv`.                         | `output/pAvailabilityCustom.csv`; `output/pVREgenProfile.csv`.                                                                |
| `hydro_representative_years.ipynb`                | Prototype workflow for selecting representative hydropower years (with reservoir management rules) and exporting EPM-ready `pAvailability` CSVs. Inspect carefully before relying on results.      | Cleaned hydro generation profiles and installed-capacity metadata assembled upstream.                                                                             | CSVs written to `output/` once the logic is finalized (currently experimental).                                               |
| `representative_days/representative_days.ipynb`   | Clusters hourly load, VRE, and hydro traces into a small set of representative seasons/days, then produces the weights and time-slices needed by the dispatch model (plus supporting GAMS checks). | Place harmonized hourly profiles for each technology in `representative_days/input/` and update `representative_days/repdays_pipeline.py` if you add new metrics. | Weighted day tables and QA figures in `representative_days/output/`; GAMS data dumps saved under `representative_days/gams/`. |
| `legacy_to_new_format/legacy_to_new_format.ipynb` | Maps legacy `pre-analysis` outputs into the new `prepare-data` folder structure so older studies can leverage the refreshed pipelines without re-running raw data pulls.                           | Drop archived CSVs into `legacy_to_new_format/input/` (matching the legacy naming), then point the notebook to the desired scenario.                              | Normalized CSVs exported to `legacy_to_new_format/output/`, ready to be copied into `epm/input`.                              |

### Directory tips

- Place raw-to-intermediate inputs under `input/` so notebooks remain reproducible.
- Outputs are git-ignored; copy only vetted CSVs into `epm/input`.
- Keep notes about any exploratory steps inside this README before promoting them to the main workflow.

## Working with ERA5-Land variables

Several notebooks rely on ERA5-Land `monthly_averaged_reanalysis` data (Jan 1950–present, 0.1° resolution). Variables such as `total_precipitation`, `surface_runoff`, `potential_evaporation`, and `total_evaporation` are stored as daily means in meters of water equivalent. Multiply by 1000 to convert to mm/day, then by the number of days in the target month for totals. Instantaneous variables like `2m_temperature` and `snow_depth_water_equivalent` are already averaged and require no accumulation.

When reading ERA5-Land GRIB files, filter by `stepType` so instantaneous (`avgid`) fields are opened separately from accumulated (`avgas`) fields. Attempting to read a mixed file without filtering causes `cfgrib` errors.

## Reference datasets

- **VegDischarge**: Routed VegET runoff (63k African river reaches) is available via the mizuRoute model. Choose KWT or KW routing for the most reliable fits.
- **GRUN**: Monthly runoff reconstruction for 1902–2014 on a 0.5° grid (mm/day), provided as NetCDFv4 and trained on GSIM + GSWP3 forcings. Download from [figshare](https://figshare.com/articles/dataset/GRUN_Global_Runoff_Reconstruction/9228176) and see the [ESSD paper](https://essd.copernicus.org/articles/11/1655/2019/).
