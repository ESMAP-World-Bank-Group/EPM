# Runs of the v6 scenario grid

Input version v6 (2026-11-21): edits 1 to 8 of the round 9 plan. The scenario names are the columns of
`epm/input/data_blacksea/scenarios_v6.csv`, built by
`tools/scenarios_build.py --grid internal_analysis/scenarios/scenario_grid_v6.yaml`. The v5 grid, its
scenario file and its runs are untouched (`server_runs.md`, `scenario_register.csv`).

The v6 scenarios keep the v5 names. Three things keep the two versions apart:

- the run labels and the explorer runs of v6 start with `v6`;
- `scenario_register_v6.csv` reads its results only from the explorer runs named `v6_*`
  (keys `register` and `result_runs` of the grid file) and shows the v5 result of the same name as twin;
- `scenario_register.csv` (v5) ignores every run whose name starts with a version tag.

## Before a batch

```
git pull
dvc pull -r r2
dvc status
```

`epm/input/data_blacksea.dvc` must read `f6367a3a887b0e81f5da434db10b494e.dir` (428 files). No builder
that writes into the input folder and no `dvc add` while a GAMS run is alive.

## Batch C5: the 17 central cases, five-year grid, laptop

One epm.py call per scenario, one after the other, `--cpu 1`. About 20 minutes per Int scenario and
12 per Ext scenario with the machine awake; peak memory about 6 GB.

```
python tools/run_sequential.py --scenarios scenarios_v6.csv --prefix v6c5 --date 20261121 -S Int_Base_5y Ext_Base_5y Int_BSSC_5y Int_GECO_5y Int_Onland_5y Ext_BSSC_5y Ext_GECO_5y Ext_Onland_5y Int_EWTC_5y Int_TRIPP_5y Int_TCasp_5y Int_BSTN_5y Int_ArTur_5y Int_ArAz_5y Int_AllProjects_5y Int_FreeExp_5y Ext_FreeExp_5y
```

The laptop must stay on mains power with the lid open. It uses modern standby: on battery or with the
lid closed Windows suspends the solve whatever the script asks, and the run only resumes when the
machine wakes. On 2026-11-21 the first scenario took three hours for that reason (two standby
periods of more than an hour inside the barrier, visible as two iterations of about 4,600 and 4,900
seconds in `<scenario>_main.log`); the result is unaffected.

Each scenario comes back as `epm/output/v6c5_<scenario>_20261121.zip`. The command can be relaunched:
a scenario whose zip is complete is skipped. `status.csv` and the logs sit in
`epm/output/_sequential/v6c5_20261121`. The call that one line stands for:

```
python epm.py --folder_input data_blacksea --scenarios scenarios_v6.csv -S <scenario> --cpu 1 --simulation_label v6c5_<scenario>_20261121 --output_zip --no_plot_dispatch
```

The run of 2026-11-21 was launched with the first version of this script, kept outside the repository,
with the same epm.py call and the same labels.

Reproducibility check: the four test runs made on the same inputs gave Int_Base_5y 316.44,
Ext_Base_5y 227.57, Int_AllProjects_5y 307.30 and Ext_FreeExp_5y 222.82 BUSD.

## Batch C2: the eight core cases, two-year grid

Nine model years instead of five. Int_Base_2y goes first and alone, as the measure of time and memory.

```
python tools/run_sequential.py --scenarios scenarios_v6.csv --prefix v6c2 -S Int_Base_2y
python tools/run_sequential.py --scenarios scenarios_v6.csv --prefix v6c2 -S Int_Base_2y Ext_Base_2y Int_BSSC_2y Int_GECO_2y Int_Onland_2y Ext_BSSC_2y Ext_GECO_2y Ext_Onland_2y
```

Dates that sit off the five-year grid move back on the two-year grid: EWTC 2034, first TRIPP segment
2028, Kozloduy 8 in 2038.

## After a batch

One explorer run per grid, never two grids in one folder. The merge tool does not expand `*` itself:
from PowerShell, list the zips one by one. A merge can be repeated as the zips arrive.

```
python tools/merge_runs.py --target epm/output_view/v6_central5y_20261121 epm/output/v6c5_*_20261121.zip
python tools/merge_runs.py --target epm/output_view/v6_core2y_<date> epm/output/v6c2_*_<date>.zip
python tools/scenario_register.py --grid internal_analysis/scenarios/scenario_grid_v6.yaml
```

The register flips the merged scenarios from "to run" to "done"; `--r2` after the upload
(`tools/upload_output_view_to_r2.py`) flips them to "published".

## Reading the results

A committed project line carries no cost in the objective: the difference of system cost between a
project row and the Base is a gross gain, to set against the capex of
`epm/input/data_blacksea/capex_interconnectors_expost_v6.csv`. FreeExp is the exception: its candidate
lines carry their capex in the objective, so its difference with the Base is already net of the lines
it builds.

The capacity built on a candidate line toward an external zone is only in `epmresults.gdx`
(`pAdditionalTransmissionCapacityExt`, `pNewTransmissionCapacityExt`). `NewTransmissionCapacity` in
`pTransmissionMerged.csv` is the increment of each model year, not the cumulated capacity.

## Interim assumptions carried by these runs

Flags F1 to F12 of the round 9 plan are open: the runs use the interim values and are to be rerun on
the scenarios a flag touches once the country teams answer.
