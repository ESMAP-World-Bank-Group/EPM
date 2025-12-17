# Data Preparation Workflow Documentation

Use this guide to run the **pre-analysis** notebooks in the right order and generate EPM-ready inputs. The primary workflows now live inside `pre-analysis/prepare-data/`, while `pre-analysis/open-data/` remains available whenever you need to refresh the underlying renewable or hydro datasets.

---

## Summary Table

| Step | Notebook(s) | Key inputs | Outputs / Notes |
|------|-------------|------------|-----------------|
| 1 | `zcmap.csv` | Country/zone perimeter | Defines the modeling scope consumed by every notebook. |
| 2 | `pre-analysis/prepare-data/climatic_overview.ipynb` | `zcmap.csv`, ERA5-Land data | Season definitions, precipitation/temperature plots, CSV summaries. |
| 3 | `pre-analysis/open-data/get_renewables_irena_data.ipynb` or `get_renewable_ninja_data.ipynb` | SPLAT names or coordinates, API keys | Fresh wind/solar capacity factors when you need to refresh raw inputs before re-running prepare-data notebooks. |
| 4 | `pre-analysis/prepare-data/load_profile_treatment.ipynb` → `load_profile.ipynb` | Historical load measurements, monthly means | Cleaned load history plus synthetic/forecast hourly load profiles. |
| 4b | `pre-analysis/prepare-data/load_plot.ipynb` | Outputs from step 4 | Shareable demand plots for QA/stakeholder review. |
| 5 | `pre-analysis/prepare-data/representative_days/representative_days.ipynb` | Climate outputs (step 2), load profiles (step 4), renewables | `pHours.csv`, `load/pDemandProfile.csv`, `supply/pVREProfile.csv`. |
| 6 | `pre-analysis/prepare-data/hydro_availability.ipynb` (+ `hydro_representative_years.ipynb` if sampling years) | Hydropower profiles/Atlas tables, `pHours.csv`, `pGenDataInput_clean.csv` | `pAvailabilityCustom.csv`, ROR `pVREgenProfile.csv`, optional scenario variants. |
| 7 | `pre-analysis/prepare-data/supply_demand_balance.ipynb` | Demand, renewables, hydro availability, generation fleet | Balance dashboards ensuring supply meets demand before launching GAMS runs. |

> **Tip:** open-data notebooks such as `hydro_inflow.ipynb`, `hydro_basins.ipynb`, and `get_generation_maps.ipynb` are your toolkit for refreshing the raw datasets that feed the prepare-data workflows above.

---

## 1. Define Perimeter Countries/Zones

- Populate `zcmap.csv` in the repo root with the countries or zones that match your study.
- Use consistent SPLAT/EPM names because these IDs drive joins in both `prepare-data` and `open-data`.

---

## 2. Run the Climatic Overview (`pre-analysis/prepare-data/climatic_overview.ipynb`)

- Objective: quantify precipitation and temperature regimes so you can justify representative seasons or years.
- Inputs: zone list from `zcmap.csv`, ERA5-Land downloads placed in `pre-analysis/prepare-data/input/`.
- Outputs: plots (available under `pre-analysis/prepare-data/output/`) and summary CSVs that downstream notebooks read.

![Temperature & Precipitation](dwld/pre-analysis/scatter_annual_spatial_means_t2m_tp.png)
![Monthly Precipitation](dwld/pre-analysis/monthly_precipitation_heatmap.png)

---

## 3. Refresh Renewable Resource Data (open-data stage)

When base-year renewable profiles are outdated, switch to `pre-analysis/open-data/`:

- **IRENA route** — `get_renewables_irena_data.ipynb`
  - Inputs: list of SPLAT zones plus the IRENA workbook.
  - Output format: `zone, season, day, hour, <climatic_year>` hourly capacity factors.
- **Renewable Ninja route** — `get_renewable_ninja_data.ipynb`
  - Requires coordinates from `get_renewables_coordinate.ipynb`.
  - Useful for scenario-specific solar/wind traces.

Save the resulting CSVs under `pre-analysis/prepare-data/input/` before moving on. You can also use `get_generation_maps.ipynb` for quick QA plots of the generation fleet.

---

## 4. Build Demand Profiles (`load_profile_treatment.ipynb`, `load_profile.ipynb`, `load_plot.ipynb`)

1. **Treat historical data** — run `load_profile_treatment.ipynb` to clean utility load logs (remove spikes, fill gaps).  
2. **Generate the hourly profile** — run `load_profile.ipynb` to blend treated history with monthly targets or growth assumptions.  
3. **Plot for QA** — use `load_plot.ipynb` to export PNG/HTML charts for stakeholder review.

Deliverables: smoothed historical load, modeled hourly demand, and plots placed in `pre-analysis/prepare-data/output/`.

---

## 5. Generate Representative Days (`pre-analysis/prepare-data/representative_days/representative_days.ipynb`)

- Inputs: load profiles from step 4, renewable capacity-factor tables (step 3), climate summaries (step 2).
- Process: cluster the full-year time series into a manageable subset of days while preserving seasonal statistics.
- Outputs ready for EPM:
  - `epm/input/data_capp/pHours.csv`
  - `epm/input/data_capp/load/pDemandProfile.csv`
  - `epm/input/data_capp/supply/pVREProfile.csv`

---

## 6. Hydropower Preparation (`hydro_availability.ipynb` + helpers)

- `hydro_availability.ipynb` ingests monthly hydro profiles or Atlas curves and exports:
  - `pAvailabilityCustom.csv` for reservoirs.
  - Run-of-river `pVREgenProfile.csv`.
- `hydro_representative_years.ipynb` (optional) samples wet/baseline/dry years before feeding them into `hydro_availability.ipynb`.
- Need new inflow data or basin checks? Use the open-data notebooks (`hydro_inflow.ipynb`, `hydro_basins.ipynb`, `hydro_atlas_comparison.ipynb`) to regenerate the raw profiles, then drop the outputs back into `prepare-data/input/`.

---

## 7. Supply vs Demand Balance (`pre-analysis/prepare-data/supply_demand_balance.ipynb`)

- Objective: confirm that the cleaned generation fleet, renewable additions, and hydro schedules cover the demand from step 4.
- Inputs: `pGenDataInput_clean.csv`, demand/renewable outputs, hydro availability, `pHours`.
- Outputs: deficit tables, stacked supply-demand plots, and sanity checks prior to running `epm/main.gms`.

---

## Notes

- Keep naming conventions consistent across demand, renewable, and hydro files (zone, technology, scenario).
- Store raw downloads in `pre-analysis/open-data/input/` or `pre-analysis/prepare-data/input/` and only copy vetted CSVs into `epm/input`.
- Whenever you introduce a new data vintage, rerun the balance notebook (step 7) before executing the GAMS model.
