# Runs of the v6 scenario grid

Input version v6 (2026-11-21): edits 1 to 8 of the round 9 plan. Input version v6.1 (2026-11-22):
the same files with twelve corrections and twelve more scenario columns. On 2026-11-23 the first lot of
sensitivity files joined the folder (eight new files, 26 more scenario columns): no file of v6.1 changed apart
from the scenario file, so the runs of 2026-11-22 stay valid. The scenario names are the columns of
`epm/input/data_blacksea/scenarios_v6.csv`, built by
`tools/scenarios_build.py --grid internal_analysis/scenarios/scenario_grid_v6.yaml`. The v5 grid, its
scenario file and its runs are untouched (`server_runs.md`, `scenario_register.csv`).

The v6 scenarios keep the v5 names. Three things keep the two versions apart:

- the run labels and the explorer runs of v6 start with `v6`, those of v6.1 with `v61`;
- `scenario_register_v6.csv` reads its results only from the explorer runs named `v61_*`
  (keys `register` and `result_runs` of the grid file) and shows as twin the most recent earlier
  result of the same name held in `epm/output_view`: the v5 run since 2026-11-23, when the run of
  2026-11-21 left the explorer (see "After a batch");
- `scenario_register.csv` (v5) ignores every run whose name starts with a version tag.

## Before a batch

```
git pull
dvc pull -r r2
dvc status
```

`epm/input/data_blacksea.dvc` must read `5f6099489e82de80ff3e7629e1718d54.dir` (442 files, input
version v6.1 with the sensitivity files of 2026-11-23). Batch V61 ran on
`f6bc00ed996cba4c3a1b37a884da19bc.dir` (434 files, input version v6.1 of 2026-11-22): the same 434
files, the scenario file with 78 columns. The runs of 2026-11-21 below were made on
`f6367a3a887b0e81f5da434db10b494e.dir` (428 files, input version v6). No builder that writes into the
input folder and no `dvc add` while a GAMS run is alive.

## Input version v6.1: corrections of 22 November 2026

Twelve corrections are applied in place in the v6 files (same file names), through
`pre-analysis/pipelines/build_robg_v6.py` (edit 9) and the table `v6_corrections_20261122.csv` kept
with the inputs. The state of 2026-11-21 stays available at git `345f095c` with the DVC version
`f6367a3a`. Every result of 2026-11-21 below is the "before corrections" reference.

| Code | Zone | Correction |
|---|---|---|
| A1 | Georgia, Armenia | no line in service on the pair (150 MW before); the Base keeps the committed 350 MW line from 2032 |
| A2 | Armenia | modelling assumption: the existing nuclear plant leaves the model one year earlier, in 2035 |
| A3 | Armenia | central gas price path revised; the path of 2026-11-21 becomes a sensitivity |
| A4 | Armenia | distributed solar in service raised to the level connected in March 2026 |
| A5 | Armenia | capex of the new nuclear unit |
| A6 | Armenia | size and heat rate of one gas unit |
| R1 | Romania | Cernavoda 1 after refurbishment at 650 MW (707 before) |
| R2 | Romania | batteries: 2,000 MW and 4,000 MWh from 2028 (two committed rows) |
| Z1 | Azerbaijan | three units in operation become existing, four solar units under construction become committed |
| TR1 | Türkiye | yearly build limit of onshore wind x 0.4 (2,180 MW a year) |
| G1 | Georgia | yearly build limit of wind 140 MW (256 before) |
| GA | two-year grid | wind candidates open in 2026, so the 2030 ceiling is the same on both grids |

Four new files and twelve new scenario columns (78 in all), on the five-year grid only:

- `Int_NoCTN_5y`: the Base without the committed Georgia to Armenia line;
- `ArmeniaGasEU` (the gas price path of 2026-11-21) and `ArmeniaNoNewNuclear`, each on Base, ArTur,
  ArAz and TRIPP;
- `EUNuclearDelay` (the six committed reactors of Romania, Bulgaria and Hungary one model step
  later) on Base, BSSC and GECO.

Checks before the runs: the four builders in `--check` mode (0 difference), the read-only acceptance
script of the corrections (140 passed, 0 failed) and a cell by cell comparison of the folder with a
dry-run copy. Known limits: Türkiye wind in 2030 has the same ceiling on both grids (28.8 GW with the
existing fleet), but 43 other candidates of the Int files (38 of the Ext files) with a start year
between 2027 and 2030 still have a lower 2030 ceiling on the two-year grid.

## Batch V61: the rerun on the corrected inputs, server

Labels `v61c5_<scenario>_20261122` (five-year grid) and `v61c2_<scenario>_20261122` (two-year grid),
never the labels of 2026-11-21. From the `epm` folder of the server, inside `tmux`, after the refresh
of "Before a batch". Int_Base_5y goes first and is the test of the corrected inputs (316.44 BUSD
before corrections). The first call runs lots 1 to 3 (45 five-year scenarios, about 7 hours), the
second lot 4 (the eight core cases on the two-year grid, about 16 hours).

```
python ../tools/run_sequential.py --scenarios scenarios_v6.csv --prefix v61c5 --date 20261122 -S Int_Base_5y Ext_Base_5y Int_BSSC_5y Int_GECO_5y Int_Onland_5y Ext_BSSC_5y Ext_GECO_5y Ext_Onland_5y Int_EWTC_5y Int_TRIPP_5y Int_TCasp_5y Int_BSTN_5y Int_ArTur_5y Int_ArAz_5y Int_AllProjects_5y Int_FreeExp_5y Ext_FreeExp_5y Int_NoCTN_5y Int_Base_ArmeniaGasEU_5y Int_ArTur_ArmeniaGasEU_5y Int_ArAz_ArmeniaGasEU_5y Int_TRIPP_ArmeniaGasEU_5y Int_Base_ArmeniaNoNewNuclear_5y Int_ArTur_ArmeniaNoNewNuclear_5y Int_ArAz_ArmeniaNoNewNuclear_5y Int_TRIPP_ArmeniaNoNewNuclear_5y Int_Base_EUNuclearDelay_5y Int_BSSC_EUNuclearDelay_5y Int_GECO_EUNuclearDelay_5y Ext_Base_EUVeryLow_5y Ext_BSSC_EUVeryLow_5y Ext_GECO_EUVeryLow_5y Ext_Onland_EUVeryLow_5y Ext_Base_EUCrisis_5y Ext_BSSC_EUCrisis_5y Ext_GECO_EUCrisis_5y Ext_Onland_EUCrisis_5y Ext_Base_EUCBAM_5y Ext_BSSC_EUCBAM_5y Ext_GECO_EUCBAM_5y Ext_Onland_EUCBAM_5y Ext_Base_EUVeryLowCBAM_5y Ext_BSSC_EUVeryLowCBAM_5y Ext_GECO_EUVeryLowCBAM_5y Ext_Onland_EUVeryLowCBAM_5y
python ../tools/run_sequential.py --scenarios scenarios_v6.csv --prefix v61c2 --date 20261122 -S Int_Base_2y Ext_Base_2y Int_BSSC_2y Int_GECO_2y Int_Onland_2y Ext_BSSC_2y Ext_GECO_2y Ext_Onland_2y
```

| Lot | Scenarios | Content |
|---|---|---|
| 1 | 18 | the 17 central cases of 2026-11-21 and Int_NoCTN_5y |
| 2 | 11 | ArmeniaGasEU and ArmeniaNoNewNuclear on four cases, EUNuclearDelay on three |
| 3 | 16 | the four EU price cases of the Ext scope on Base, BSSC, GECO and Onland |
| 4 | 8 | the core cases on the two-year grid, Int_Base_2y first |

Both calls can be relaunched: a scenario whose zip is complete is skipped, and a scenario that fails
does not stop the others. The zips `epm/output/v61c5_*_20261122.zip` and `v61c2_*_20261122.zip` are
then copied to `epm/output/` on the laptop and merged into the explorer runs
`v61_central5y_20261122` (lot 1), `v61_sens5y_20261122` (lots 2 and 3: every sensitivity has its own
Base row, the two central Bases stay in the central run) and `v61_core2y_20261122` (lot 4). The
register reads the runs named `v61_*`.

To read in the runs without the new Armenian nuclear unit: unserved energy and unmet reserve in
Armenia after 2035, when the existing plant has left.

Outcome (2026-11-23): 53 of 53 scenarios solved to optimality on the server. The zips were copied to
`epm/output/` on the laptop with an md5 check, merged and published the same day: 18 scenarios in
`v61_central5y_20261122`, 27 in `v61_sens5y_20261122`, 8 in `v61_core2y_20261122`. The merged runs
give back the values read in the zips. Objective of the central scenarios against the run of
2026-11-21, which they replace in the explorer:

| Scenario | v6.1, BUSD | v6 of 21 Nov, BUSD | Difference |
|---|---|---|---|
| Int_Base_5y | 322.84 | 316.44 | +6.40 |
| Int_BSSC_5y | 321.22 | 314.74 | +6.48 |
| Int_GECO_5y | 318.72 | 312.23 | +6.49 |
| Int_Onland_5y | 319.41 | 313.07 | +6.34 |
| Int_EWTC_5y | 321.30 | 314.85 | +6.45 |
| Int_TRIPP_5y | 322.07 | 315.73 | +6.34 |
| Int_TCasp_5y | 322.50 | 316.09 | +6.41 |
| Int_BSTN_5y | 322.52 | 316.14 | +6.38 |
| Int_ArTur_5y | 322.45 | 316.08 | +6.37 |
| Int_ArAz_5y | 322.81 | 316.40 | +6.41 |
| Int_AllProjects_5y | 313.78 | 307.30 | +6.48 |
| Int_FreeExp_5y | 319.10 | 312.76 | +6.34 |
| Int_NoCTN_5y | 322.92 | n/a | n/a |
| Ext_Base_5y | 233.25 | 227.57 | +5.68 |
| Ext_BSSC_5y | 231.33 | 225.58 | +5.75 |
| Ext_GECO_5y | 227.65 | 221.91 | +5.74 |
| Ext_Onland_5y | 229.49 | 223.77 | +5.72 |
| Ext_FreeExp_5y | 228.62 | 222.82 | +5.80 |

## Batch SA: first lot of sensitivities, server

Labels `v61s5_<scenario>_20261123`, five-year grid, 26 scenarios (18 Int, 8 Ext), about 4 hours. The
files come from `pre-analysis/pipelines/build_sens_v6.py` (`--check` mode, 0 difference) and, for the
two EU price cases, from `tools/build_ext_inputs_v6.py`. The first seven scenarios of the call read
every new file once: they are the test of the lot. From the `epm` folder of the server, inside `tmux`,
after the refresh of "Before a batch".

```
python ../tools/run_sequential.py --scenarios scenarios_v6.csv --prefix v61s5 --date 20261123 -S Int_Base_ArmeniaGasContract_5y Int_Base_ArmeniaDemandLow_5y Int_GECO_CaucasusREHubs_5y Int_Base_EUCarbonLow_5y Int_Base_TurkiyeCarbonPrice_5y Ext_Base_EULow_5y Ext_Base_EUHigh_5y Int_ArTur_ArmeniaGasContract_5y Int_ArAz_ArmeniaGasContract_5y Int_TRIPP_ArmeniaGasContract_5y Int_ArTur_ArmeniaDemandLow_5y Int_ArAz_ArmeniaDemandLow_5y Int_TRIPP_ArmeniaDemandLow_5y Int_FreeExp_CaucasusREHubs_5y Int_BSSC_EUCarbonLow_5y Int_GECO_EUCarbonLow_5y Int_Onland_EUCarbonLow_5y Int_BSSC_TurkiyeCarbonPrice_5y Int_GECO_TurkiyeCarbonPrice_5y Int_Onland_TurkiyeCarbonPrice_5y Ext_BSSC_EULow_5y Ext_GECO_EULow_5y Ext_Onland_EULow_5y Ext_BSSC_EUHigh_5y Ext_GECO_EUHigh_5y Ext_Onland_EUHigh_5y
```

| Sensitivity | Cases | What moves against the central |
|---|---|---|
| ArmeniaGasContract | Base, ArTur, ArAz, TRIPP | Armenian gas on the border contract path, without the margin of correction A3 |
| ArmeniaDemandLow | Base, ArTur, ArAz, TRIPP | Armenian peak and energy on the forecast in use before the country team update |
| CaucasusREHubs | GECO, FreeExp | the five solar and wind candidates of the two GEC hubs become 14 committed steps (2032, 2036, 2040); the steps replace the candidates |
| EUCarbonLow | Base, BSSC, GECO, Onland | CO2 of Romania, Bulgaria and Hungary on the TYNDP 2024 path (132 USD/t in 2030, 172 in 2040, against 106 and 322); prices of the nine neighbours priced on TYNDP 2026 scaled by the cost of gas with CO2 (factor 1.10 in 2030, 0.80 in 2036, 0.70 in 2040) |
| TurkiyeCarbonPrice | Base, BSSC, GECO, Onland | from 2030 the central CO2 price on Turkish gas, imported coal, lignite and fuel oil |
| EULow, EUHigh | Ext Base, BSSC, GECO, Onland | the low and the high EU price case |

To keep in mind when reading. EUCarbonLow is above the central from 2026 to 2031 and below it from
2032: it is a lower path for the long run only. Its border price rule is a rule of this study: it
keeps the hourly shape and scales the level, hours set by renewables included. CaucasusREHubs forces
the hub capacity in, with its capex in the objective: its difference with the central reads as the
cost or the gain of the full build, not as a project gain. TurkiyeCarbonPrice has no price path in
Turkish law: an upper bound. The zips are merged into the explorer run `v61_sensA5y_20261123`; the
central cases they compare to stay in `v61_central5y_20261122`.

Outcome (2026-11-24): 26 of 26 scenarios solved to optimality on the server. The zips were copied to
`epm/output/` on the laptop with an md5 check, merged into `v61_sensA5y_20261123` and, after a review
of the results against the central twins, published on 2026-11-24. The merged run gives back the
values read in the zips. Each run differs from its central twin by the intended input files only,
and unmet demand stays below 1.5 GWh per scenario over the five model years. The register counts
79 scenarios published and 25 to run. Gain of each case, Base minus case, against the Base of the
same sensitivity (CaucasusREHubs has no Base of its own and is compared with the central Base):

| Sensitivity | Base, BUSD | Case | Gain, BUSD | Central gain, BUSD |
|---|---|---|---|---|
| ArmeniaGasContract | 322.55 | ArTur | 0.51 | 0.39 |
| ArmeniaGasContract | 322.55 | ArAz | 0.02 | 0.04 |
| ArmeniaGasContract | 322.55 | TRIPP | 0.83 | 0.78 |
| ArmeniaDemandLow | 322.18 | ArTur | 0.44 | 0.39 |
| ArmeniaDemandLow | 322.18 | ArAz | 0.05 | 0.04 |
| ArmeniaDemandLow | 322.18 | TRIPP | 0.80 | 0.78 |
| EUCarbonLow | 325.67 | BSSC | 1.11 | 1.62 |
| EUCarbonLow | 325.67 | GECO | 2.70 | 4.13 |
| EUCarbonLow | 325.67 | Onland | 3.11 | 3.43 |
| TurkiyeCarbonPrice | 489.19 | BSSC | 1.70 | 1.62 |
| TurkiyeCarbonPrice | 489.19 | GECO | 4.19 | 4.13 |
| TurkiyeCarbonPrice | 489.19 | Onland | 8.50 | 3.43 |
| EULow, Ext | 233.34 | BSSC | 1.28 | 1.92 |
| EULow, Ext | 233.34 | GECO | 3.32 | 5.60 |
| EULow, Ext | 233.34 | Onland | 2.99 | 3.76 |
| EUHigh, Ext | 232.86 | BSSC | 2.89 | 1.92 |
| EUHigh, Ext | 232.86 | GECO | 8.96 | 5.60 |
| EUHigh, Ext | 232.86 | Onland | 5.45 | 3.76 |
| CaucasusREHubs | 322.84, central | GECO | 3.28 | 4.13 |
| CaucasusREHubs | 322.84, central | FreeExp | 2.09 | 3.75 |

Reading, from the review of 2026-11-24. The EU price cases are in order (very low, low, central,
high, crisis) in border prices, exports and gains. Onland keeps a floor near 3.0 BUSD because the
line also carries imports of cheap EU power into Türkiye. The two Armenian sensitivities are
coherent: with the contract gas price Armenia turns into a net exporter, and ArTur and TRIPP gain
more. Four sensitivities need a label when they are quoted:

- EUCarbonLow is the TYNDP 2024 carbon path, not a low case in 2030 (132 USD/t against 106). Its Base
  is 2.83 BUSD above the central Base. The project gains read as a lower carbon path because the
  projects start in 2032.
- TurkiyeCarbonPrice is a stress test. Its Base is 166 BUSD above the central Base, of which about
  128 is the carbon payment itself (emissions times price, hand estimate) and about 38 a resource
  cost. Turkish coal stays at 48 TWh a year, its floor: the fleet is kept by the no retirement rule
  and holds a minimum generation share of 0.25. The Onland gain (8.50 against 3.43) includes avoided
  carbon payments in Türkiye and emissions moved to the Caucasus (about 2 Mt a year, hand estimate):
  it reads as an upper bound. BSSC and GECO do not move.
- ArmeniaDemandLow is the forecast in use before the country team update. Its energy is lower in
  every year and its 2030 peak slightly higher: it is not a pure low case.
- CaucasusREHubs forces 11.3 GW into the two hubs by 2040, against 5.3 GW chosen by the model in the
  central GECO case. The forced build costs 0.85 BUSD in GECO and 1.66 BUSD in FreeExp against the
  free choice. Under FreeExp the model then builds the cable from the Georgian hub to Romania
  (1.29 GW in 2040), which the central FreeExp case does not build.

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

Outcome on the laptop (2026-11-21 and 22): 14 of the 17 scenarios solved to optimality and are merged
in `epm/output_view/v6_central5y_20261121`. Int_EWTC_5y, Int_AllProjects_5y and Int_FreeExp_5y ended
on `CPLEX Error 1001: Out of memory`, which epm.py prints as "no feasible solution found". The cause
is the machine, not the model: an Int scenario needs about 15 GB of commit memory (GAMS 5.8 GB, the
CPLEX link and the loaded model), and the other applications and sessions open that day left 7 to
10 GB out of 51. A retry with 10 GB left failed the same way after 28 seconds of barrier. Before a
local batch, check that at least 18 GB of commit memory is free (Task Manager, Performance, Memory,
"Committed"); if not, close applications or use the server.

## The three missing five-year scenarios: server

Same labels as the laptop batch, so the three zips merge into the same explorer run. From the
repository root on the server, after the refresh of "Before a batch" (from the `epm` folder, call
`python ../tools/run_sequential.py` with the same arguments):

```
python tools/run_sequential.py --scenarios scenarios_v6.csv --prefix v6c5 --date 20261121 -S Int_EWTC_5y Int_AllProjects_5y Int_FreeExp_5y
```

The three files `epm/output/v6c5_<scenario>_20261121.zip` are then copied to `epm/output/` on the
laptop.

Done on 2026-11-22, 10 minutes per scenario on the server (peak memory 5.9 GB): Int_EWTC_5y 314.845,
Int_AllProjects_5y 307.296 (the test run gave 307.30) and Int_FreeExp_5y 312.756 BUSD. The 17
scenarios are merged in `epm/output_view/v6_central5y_20261121`. The status file and the logs of
these three runs stay on the server, in `epm/output/_sequential/v6c5_20261121`.

## Batch C2: the eight core cases, two-year grid, server

Not launched: replaced on 2026-11-22 by lot 4 of batch V61, on the corrected inputs. The paragraph
is kept for the method.

Nine model years instead of five, so more memory than the laptop can give: this batch runs on the
server, one scenario at a time (`--cpu 1` is set by the script). Int_Base_2y goes first, as the
measure of time and memory; the script then carries on with the seven others. Expect about two hours
per Int scenario.

```
python tools/run_sequential.py --scenarios scenarios_v6.csv --prefix v6c2 --date 20261122 -S Int_Base_2y Ext_Base_2y Int_BSSC_2y Int_GECO_2y Int_Onland_2y Ext_BSSC_2y Ext_GECO_2y Ext_Onland_2y
```

Launch it inside `tmux` or `screen` so that it survives the end of the session. Every five minutes
the script prints the minutes elapsed and the last line of the GAMS log of the running scenario
(`--progress 0` for none). The command can be relaunched after an interruption: a scenario whose zip
is complete is skipped. A scenario that fails does not stop the others;
`epm/output/_sequential/v6c2_20261122/status.csv` gives one row per attempt. The eight files
`epm/output/v6c2_<scenario>_20261122.zip` are then copied to `epm/output/` on the laptop.

Dates that sit off the five-year grid move back on the two-year grid: EWTC 2034, first TRIPP segment
2028, Kozloduy 8 in 2038.

## After a batch

One explorer run per grid, never two grids in one folder. The merge tool does not expand `*` itself:
from PowerShell, list the zips one by one. A merge can be repeated as the zips arrive. The label
`v61c5` feeds two explorer runs: a name made of a scope and a case goes to the central run, a name
that also carries a sensitivity goes to the sensitivity run. From Git Bash:

```
ALL=$(ls epm/output/v61c5_*_20261122.zip)
CENTRAL=$(echo "$ALL" | grep -E '/v61c5_(Int|Ext)_[A-Za-z]+_5y_20261122\.zip$')
SENS=$(echo "$ALL" | grep -v -E '/v61c5_(Int|Ext)_[A-Za-z]+_5y_20261122\.zip$')
python tools/merge_runs.py --target epm/output_view/v61_central5y_20261122 $CENTRAL
python tools/merge_runs.py --target epm/output_view/v61_sens5y_20261122 $SENS
python tools/merge_runs.py --target epm/output_view/v61_core2y_20261122 epm/output/v61c2_*_20261122.zip
python tools/merge_runs.py --target epm/output_view/v61_sensA5y_20261123 epm/output/v61s5_*_20261123.zip
python tools/scenario_register.py --grid internal_analysis/scenarios/scenario_grid_v6.yaml
```

The register flips the merged scenarios from "to run" to "done"; `--r2` after the upload
(`tools/upload_output_view_to_r2.py --only "<run>/*"`) flips them to "published". On 2026-11-23 the
53 scenarios of batch V61 are published in the three runs `v61_*_20261122`. On 2026-11-24 the 26
scenarios of batch SA are published in `v61_sensA5y_20261123`.

The run of 2026-11-21, `v6_central5y_20261121`, was removed from `epm/output_view` on 2026-11-23,
replaced by `v61_central5y_20261122`. The upload rewrites the list of runs the explorer shows from
the folders present, so the run left the explorer with that upload. The upload tool never deletes:
the csv files of the removed run stay on the store until they are deleted there. Its 17 zips
`epm/output/v6c5_*_20261121.zip` are kept and rebuild it:

```
python tools/merge_runs.py --target epm/output_view/v6_central5y_20261121 epm/output/v6c5_*_20261121.zip
```

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
