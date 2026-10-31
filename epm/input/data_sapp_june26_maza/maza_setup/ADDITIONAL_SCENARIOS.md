# MAZA scenario extension (22 October 2026)

Use `scenarios_maza_extended.csv` with the existing full-year `config_maza.csv`. It preserves every prior scenario and adds five matched MAZA / no-MAZA pairs. MAZA is 400 MW and MOMA remains at the 200 MW baseline in the new pairs; only the named assumption changes.

| Scenario | MAZA run | No-MAZA comparator | Change from baseline |
|---|---|---|---|
| Mpatamanga shutdown | `MpatShutdown` | `MpatShutdown_NoMAZA` | Remove `Mpatamanga_Opt_HY_MWI` (358 MW) from the generator input for all model years. |
| No Growth Change | `MalawiNoGrowthChange23` | `MalawiNoGrowthChange23_NoMAZA` | Hold the Malawi 2027 energy and peak forecast fixed as the anchor, then grow both 2.3% each year through 2060. |
| Conservative Growth | `MalawiConservativeHalfGrowth` | `MalawiConservativeHalfGrowth_NoMAZA` | Take half of each year's baseline projected GWh and MW increase above the 2027 level. |
| No Mpatamanga + No Growth Change | `MpatShutdown_MalawiNoGrowthChange23` | `MpatShutdown_MalawiNoGrowthChange23_NoMAZA` | Remove Mpatamanga and use the 2.3% Malawi demand path. |
| No Mpatamanga + Conservative Growth | `MpatShutdown_MalawiConservativeHalfGrowth` | `MpatShutdown_MalawiConservativeHalfGrowth_NoMAZA` | Remove Mpatamanga and use half of projected Malawi demand growth. |

The baseline 2027 Malawi values are 4,095 GWh and 627 MW. The 2060 values are **58,103.6 GWh / 8,892.6 MW** in the current forecast, **31,099.3 GWh / 4,759.8 MW** under Conservative Growth, and **8,672.6 GWh / 1,327.9 MW** under No Growth Change. The configured baseline demand file previously ended in 2056; its 2057–2060 values for each zone and demand type use that row's 2055–2056 growth rate. Both Malawi alternatives use this extended baseline and keep all other zones identical. Every other country and every Malawi year through 2027 is unchanged. The 2.3% is Malawi's 10-year historical growth in **served energy**, reported by the Ministry of Energy demand-forecast study (section 2.1): https://www.energy.gov.mw/docs/uploads/Demand-forecast-final-report-v1f-9.pdf . Served energy is constrained by supply and connections; No Growth Change is an intentionally severe low-demand stress test, not a forecast of latent demand or electrification.

From the `epm` directory, run the five new scenario pairs (10 EPM cases) with:

```bash
python epm.py --folder_input data_sapp_june26 --config config_maza.csv \
  --scenarios scenarios_maza_extended.csv \
  --selected_scenarios MpatShutdown MpatShutdown_NoMAZA \
  MalawiNoGrowthChange23 MalawiNoGrowthChange23_NoMAZA \
  MalawiConservativeHalfGrowth MalawiConservativeHalfGrowth_NoMAZA \
  MpatShutdown_MalawiNoGrowthChange23 MpatShutdown_MalawiNoGrowthChange23_NoMAZA \
  MpatShutdown_MalawiConservativeHalfGrowth MpatShutdown_MalawiConservativeHalfGrowth_NoMAZA \
  --cpu 1 --reduced_output --no_plot_dispatch --simulation_label MAZA_additional_20261022
```

For a run through 2060, use a planning-year file that includes years through 2060; the local `y.csv` currently ends in 2050. Use the same cloud environment and GAMS setup as the prior full run. The new scenario pairs are listed in `maza_efa/new_scenario_pairs.csv` for later EFA ingestion. The current EFA report and workbook reflect only the six previously completed scenario pairs; the new EPM cases have been configured but not solved.
