# Server runs of the v5 scenario grid

Four batches, one per roadmap step (3a to 3d), launched by hand on the server from the `epm`
folder after the repository and the data are refreshed. The scenario names are the columns of
`epm/input/data_blacksea/scenarios.csv` (built by `tools/scenarios_build.py`); the model type and
the cplex option file come from that file, so no `--modeltype` flag is needed. The `--scenarios`
flag must stay bare and before `-S` (without it epm.py reads no scenario file at all).

Input set v5 (2026-11-02): v4 without the 0.2 cap on external trade, and the Bulgarian batteries
at 4.5 GW and 14 GWh. Only the Int columns change; the Ext inputs are byte for byte those of v4.
The v4 run of batch 3a (grid5y_core_20261101, cap 0.2) is kept as the cap sensitivity, not merged.

Before every batch:

```
git pull
dvc pull -r r2
cd epm
```

The log should print `Scenarios file: scenarios.csv`. A scenario reported as "no feasible solution
found" with `CPLEX Error 1001` in its `_main.log` ran out of memory: relaunch it alone with `--cpu 1`.

## 3a: core cases, five-year grid (5 scenarios, Int)

```
python epm.py --folder_input data_blacksea --scenarios -S Int_Base_5y Int_BSSC_5y Int_GECO_5y Int_Onland_5y Int_FreeExp_5y --cpu 3 --simulation_label grid5y_core_20261102 --output_zip
```

## 3b: Ext scope, five-year grid (12 scenarios: 3 core + 9 EU price)

```
python epm.py --folder_input data_blacksea --scenarios -S Ext_Base_5y Ext_BSSC_5y Ext_GECO_5y Ext_Base_EUVeryLow_5y Ext_BSSC_EUVeryLow_5y Ext_GECO_EUVeryLow_5y Ext_Base_EUCrisis_5y Ext_BSSC_EUCrisis_5y Ext_GECO_EUCrisis_5y Ext_Base_EUCBAM_5y Ext_BSSC_EUCBAM_5y Ext_GECO_EUCBAM_5y --cpu 3 --simulation_label grid5y_ext_20261102 --output_zip
```

## 3c: Int sensitivities, five-year grid (24 scenarios)

```
python epm.py --folder_input data_blacksea --scenarios -S Int_Base_ETSLo_5y Int_BSSC_ETSLo_5y Int_GECO_ETSLo_5y Int_Base_ETSHi_5y Int_BSSC_ETSHi_5y Int_GECO_ETSHi_5y Int_Base_GasLo_5y Int_BSSC_GasLo_5y Int_GECO_GasLo_5y Int_Base_GasHi_5y Int_BSSC_GasHi_5y Int_GECO_GasHi_5y Int_Base_DemLo_5y Int_BSSC_DemLo_5y Int_GECO_DemLo_5y Int_Base_DemHi_5y Int_BSSC_DemHi_5y Int_GECO_DemHi_5y Int_Base_AzGas10_5y Int_BSSC_AzGas10_5y Int_GECO_AzGas10_5y Int_Base_ResHi_5y Int_BSSC_ResHi_5y Int_GECO_ResHi_5y --cpu 3 --simulation_label grid5y_sens_20261102 --output_zip
```

## 3d: core cases, two-year grid (8 scenarios: 5 Int + 3 Ext)

Nine model years instead of five: start at `--cpu 2` and fall back to `--cpu 1` on a CPLEX Error 1001.

```
python epm.py --folder_input data_blacksea --scenarios -S Int_Base_2y Int_BSSC_2y Int_GECO_2y Int_Onland_2y Int_FreeExp_2y Ext_Base_2y Ext_BSSC_2y Ext_GECO_2y --cpu 2 --simulation_label grid2y_core_20261102 --output_zip
```

## After a batch

Copy the zip to the laptop and merge it into the explorer run of its grid (step 4a), one run folder
per grid, never two grids in one folder:

```
python tools/merge_runs.py --target epm/output_view/grid5y_<date> <zip ...>
python tools/merge_runs.py --target epm/output_view/grid2y_<date> <zip ...>
python tools/scenario_register.py
```

The register then flips the merged scenarios from "to run" to "done"; `--r2` after the upload
flips them to "published".
