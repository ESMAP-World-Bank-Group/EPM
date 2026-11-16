# MAZA scenarios on data_sapp_nuc_rsa_moz

This package provides 11 matched MAZA / no-MAZA comparisons (22 EPM runs). EPM creates the MAZA baseline run from the config; the other 21 run names are in scenarios_maza_extended.csv.

## Run configuration

Use config_maza.csv for the new dataset's configured 2025–2056 planning years. Demand files also contain 2057–2060 columns. config_maza_2060.csv selects maza_setup/y_maza_2060.csv for a proposed 2060 run, but other year-dependent files in the new dataset have not all been extended to 2060; check their treatment before solving that longer horizon.

From the epm directory, run all cases with this command (or split into smaller batches):

    python epm.py --folder_input data_sapp_nuc_rsa_moz --config config_maza.csv \
      --scenarios scenarios_maza_extended.csv --selected_scenarios \
      baseline NoMAZA \
      MOMA50Until2035 MOMA50Until2035_NoMAZA \
      MMRestricted50 MMRestricted50_NoMAZA \
      MpatDelay2035 MpatDelay2035_NoMAZA \
      MpatDelay2035_MOMA50Until2035 MpatDelay2035_MOMA50Until2035_NoMAZA \
      MpatDelay2035_MMRestricted50 MpatDelay2035_MMRestricted50_NoMAZA \
      MpatShutdown MpatShutdown_NoMAZA \
      MalawiNoGrowthChange23 MalawiNoGrowthChange23_NoMAZA \
      MalawiConservativeHalfGrowth MalawiConservativeHalfGrowth_NoMAZA \
      MpatShutdown_MalawiNoGrowthChange23 MpatShutdown_MalawiNoGrowthChange23_NoMAZA \
      MpatShutdown_MalawiConservativeHalfGrowth MpatShutdown_MalawiConservativeHalfGrowth_NoMAZA \
      --cpu 1 --reduced_output --no_plot_dispatch --simulation_label MAZA_nuc_rsa_moz_2056

## Scenario definitions

| MAZA run | No-MAZA comparator | Difference from baseline |
|---|---|---|
| baseline | NoMAZA | MAZA 400 MW, MOMA 200 MW |
| MOMA50Until2035 | MOMA50Until2035_NoMAZA | MOMA 50 MW through 2034, 200 MW from 2035 |
| MMRestricted50 | MMRestricted50_NoMAZA | MOMA 50 MW throughout |
| MpatDelay2035 | MpatDelay2035_NoMAZA | Mpatamanga starts in 2035 |
| MpatDelay2035_MOMA50Until2035 | MpatDelay2035_MOMA50Until2035_NoMAZA | Mpatamanga delay plus MOMA delay |
| MpatDelay2035_MMRestricted50 | MpatDelay2035_MMRestricted50_NoMAZA | Mpatamanga delay plus MOMA restriction |
| MpatShutdown | MpatShutdown_NoMAZA | No Mpatamanga |
| MalawiNoGrowthChange23 | MalawiNoGrowthChange23_NoMAZA | Malawi energy and peak grow 2.3% annually from 2027 |
| MalawiConservativeHalfGrowth | MalawiConservativeHalfGrowth_NoMAZA | Half of each year's new-baseline demand increase above 2027 |
| MpatShutdown_MalawiNoGrowthChange23 | MpatShutdown_MalawiNoGrowthChange23_NoMAZA | No Mpatamanga plus 2.3% Malawi growth |
| MpatShutdown_MalawiConservativeHalfGrowth | MpatShutdown_MalawiConservativeHalfGrowth_NoMAZA | No Mpatamanga plus half-growth Malawi demand |

## Baseline decisions to verify

- The new config.csv selects the corrected v5 demand forecast, v5 nonnuclear South Africa generator file, and November 2026 transmission file. The MAZA configs preserve those selections except for scenario overrides and the agreed MOMA correction.
- The selected November transmission file has MOMA as two 300 MW lines (600 MW). This study previously specified one 200 MW MOMA interconnector, so the MAZA baseline transmission override changes it to 200 MW. The untouched new config.csv run is not the comparable MAZA baseline; rerun baseline using config_maza.csv.
- The selected v5 generator file gives Mpatamanga_Opt_HY_MWI a 2033 start and 2043 retirement. The delay file changes only its start to 2035, preserving 2043. Confirm the 2043 retirement is intended before relying on results.
- The corrected v5 demand forecast ends in 2056. For 2057–2060, the MAZA baseline demand file extrapolates each zone/type using its 2055–2056 growth factor. Values through 2056 are copied unchanged.
- Malawi 2060 energy demand is 8,317.3 GWh baseline, 5,765.4 GWh under 2.3% growth, and 5,519.8 GWh under half-growth. In this new dataset, the 2.3% path is higher than the half-growth path; the labels are retained, but their severity order has changed.
- The 50-to-200 MW MOMA case uses modeled 50 MW and 150 MW stages with proportional capital costs. These stages sum to the 200 MW study assumption.

The input files and scenario references were checked. EPM has not been solved with this rebuilt package.

