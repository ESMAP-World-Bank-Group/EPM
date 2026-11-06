# Run EPM from Python - Advanced Features

The Python interface supports advanced features via command-line options.

## A. Run Multiple Scenarios

To run additional scenarios beyond the baseline, use the `--scenarios` argument. By default, invoking the flag without a value loads `scenarios.csv` from your `folder_input`:

```sh
python epm.py --folder_input my_data --scenarios
```

If your scenarios file uses a different name or sits in another location relative to `folder_input`, pass it explicitly:

```sh
python epm.py --folder_input my_data --scenarios my_scenarios.csv
```

Each row overrides specific input files defined in config.csv.

- Files not listed in scenarios.csv default to those in config.csv.
- This works in combination with Monte Carlo simulations, sensitivity analyses, or policy scenarios.

You can limit execution to a selection of scenarios using the `--selected_scenarios` (short form `-S`) argument. For example:

```sh
python epm.py --folder_input my_data --scenarios my_scenarios.csv -S baseline Retrade
```

This will only run the baseline and Retrade scenarios, even if more are defined in scenarios.csv.

This is useful if:

- You want to rerun only a specific scenario
- You are debugging or testing one configuration
- You want to prioritize certain scenarios during batch runs

## B. Sensitivity Analysis

The EPM model includes a built-in sensitivity analysis feature that allows users to test the robustness of results by varying key input parameters. This is particularly useful for exploring uncertainty or conducting scenario-based assessments.

---

### 1. How to Enable Sensitivity Analysis

To activate sensitivity runs, follow these steps:

1. **Create a `sensitivity.csv` file**  
   This file must be located in your input folder (same level as `scenarios.csv` or `config.csv`) and indicate which parameters you want to vary.

2. **Add the `--sensitivity` flag**  
   When running the model using the CLI or the Python API, pass the sensitivity flag as an argument.

---

### 2. Example `sensitivity.csv`

```csv
parameter,sensitivity
pSettings,TRUE
y,TRUE
pDemandForecast,FALSE
pDemandProfile,FALSE
pAvailabilityDefault,TRUE
pCapexTrajectoriesDefault,TRUE
pFuelPrice,TRUE
ResLimShare,TRUE
BuildLimitperYear,TRUE
pVREProfile,TRUE
```

Each row defines a parameter for which a sensitivity analysis will be triggered. A value of `1` means "active", and `0` or omission means "inactive".

#### 3. Run with Sensitivity

**From the Command Line:**

```bash
python epm.py --input_folder input/your_data_folder --sensitivity
```

### Tested Sensitivity Parameters and Variations

| Parameter                                 | Description                                                                | Variations / Tested Values                          |
| ----------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------- |
| `pSettings`                               | Model setting parameters (all tested within the same `pSettings.csv` file) | See detailed parameters below                       |
| ├─ `VoLL`                                 | Value of Lost Load                                                         | 250                                                 |
| ├─ `fApplyPlanningReserveConstraint`      | Planning reserve constraints                                               | 0                                                   |
| ├─ `sVREForecastErrorPct`                 | VRE forecast error                                                         | 0, 0.3                                              |
| ├─ `fApplyCountrySpinReserveConstraint`   | Country-level spinning reserve enforcement                                 | 0                                                   |
| ├─ `fApplySystemSpinReserveConstraint`    | System-level spinning reserve enforcement                                  | 0                                                   |
| ├─ `CostSurplus`                          | Cost of surplus generation                                                 | 1, 5                                                |
| ├─ `CostCurtail`                          | Cost of curtailment                                                        | 1, 5                                                |
| ├─ `fEnableInternalExchange`              | Enable internal exchanges                                                  | 0, 1                                                |
| ├─ `fEnableExternalExchange`              | Enable exchanges with external zones                                       | 0, 1                                                |
| ├─ `fAllowTransferExpansion`              | Allow expansion of transfer limits                                         | 0, 1                                                |
| └─ `sIntercoReserveContributionPct`       | Contribution of interconnection reserves                                   | 0, 0.5                                              |
| `y`                                       | Year definitions                                                           | Full continuous years range; first & last year only |
| `pDemandForecast`                         | Total demand forecast scaling                                              | -25%, -10%, +10%, +25%                              |
| `pDemandProfile`                          | Demand profile shape                                                       | Flat profile (equal hourly shares, 1/24)            |
| `pAvailabilityDefault`                    | Thermal plant availability (default values)                                | 0.3, 0.7                                            |
| `pCapexTrajectoriesDefault`               | Capital expenditure trajectories                                           | Flat trajectory (all values = 1)                    |
| `pFuelPrice`                              | Fuel price scaling                                                         | -20%, +20%                                          |
| `ResLimShare` (in `pGenDataInputDefault`) | Contribution to reserves                                                   | -50%, -100%                                         |
| `BuildLimitperYear`                       | Build limit removal for candidate plants                                   | Set build limit = plant capacity (removal)          |
| `pVREProfile`                             | Variable renewable energy generation scaling                               | -20%, +20%                                          |

- Modified input files are saved in a `sensitivity` folder next to the baseline input files.
- New scenarios are created with updated paths to the modified input files.
- Sensitivity scenarios use the `~` (tilde) separator in their names (e.g., `baseline~NoBiomass`, `baseline~pDemandForecast_015`).
- Assessment scenarios use the `@` (at) separator (e.g., `baseline@rehabilitation`).
- Combined scenarios are supported: `baseline~NoBiomass@rehabilitation` (sensitivity first, then assessment).
- Only parameters enabled in the `sensitivity` dict are processed.
- This approach supports automated scenario generation for robustness checks.

## C. Single-Country Runs with Border Prices

For detailed country-level analysis, you can run a two-step workflow:

1. **Regional run** with `--focus_country` to extract border prices
2. **Single-country run** using those prices as fixed external inputs

### Step 1: Run Regional Model with Focus Country

```bash
python epm.py --folder_input data_test --focus_country DRC
```

This runs the full regional model and generates single-country input files in `folder_input`:

```text
input/data_test/
├── config_drc.csv                    # Modified config for single-country run
└── single_country_drc/
    ├── zcmap_drc.csv                 # Only DRC zones
    ├── zext_drc.csv                  # Neighbors as external zones
    ├── pTradePrice_drc.csv           # Border prices from pHourlyPrice
    ├── pExtTransferLimit_drc.csv     # Cross-border capacities
    └── pTransferLimit_drc.csv        # Internal transmission only
```

### Step 2: Run Single-Country Model

```bash
python epm.py --folder_input data_test --config config_drc.csv
```

The single-country model treats neighboring countries as external zones with fixed prices from the regional run.

### How It Works

- **Neighbors identified** from both `pTransferLimit.csv` (existing) and `pNewTransmission.csv` (candidate lines)
- **Border prices** extracted from `pHourlyPrice` output, pivoted to `pTradePrice` format
- **Multi-zone countries** supported (e.g., DRC includes both `DRC` and `DRC_South` zones)
- **Price interpolation** handled automatically by input treatment for missing years

### CLI Reference

| Argument          | Description                                                      |
|-------------------|------------------------------------------------------------------|
| `--focus_country` | Country code to generate single-country inputs for (e.g., `DRC`) |

---

## D. Monte Carlo analysis

The Monte Carlo analysis tests how robust each scenario's investment plan is to uncertainties: fuel prices, demand, hydro, random outages of thermal plants and interconnections, and the weather year for solar and wind. The approach follows adequacy assessments such as ENTSO-E's Mid-term Adequacy Forecast, within the limits of EPM's representative days (see the note on LOLE below).

How it works:

1. Each selected scenario (the *parent*) is solved as usual, optimising investment and dispatch.
2. For each sample, a Monte Carlo run re-solves the dispatch of each parent with its investment plan fixed. Builds and retirements of generation and storage and the new transmission lines are fixed to the parent's solution. The sampled uncertainties are applied to the inputs.
3. The results are summarised as distributions across samples: adequacy (energy not served, loss of load), costs, and what drives the risk.

What is fixed in the Monte Carlo runs (`main.gms`, block active when `LOADSOLPATH` is set):

- The **decisions** of the parent are fixed: `vBuild`, `vRetire`, `vBuildStor`, `vRetireStor` and the cumulative new transmission lines `vNewTransmissionLine`, rounded to 6 decimals. Capacities (`vCap`, `vCapStor`) are left free and follow from the capacity equations, so they come out identical to the parent's.
- Only decisions that were free in the parent run are fixed. With `PA.HoldFixed = 1`, variables already fixed by their bounds (committed plants and lines, existing units) are not written to `PA_p.gdx`; they keep the bounds set from the inputs, which are the same as in the parent run.
- Not fixed: H2 assets and CSP thermal storage, which are re-optimised in each sample when they are used.

### 1. Define the uncertainties

Create a CSV file with one row per uncertain input:

```csv
feature,type,lowerbound,upperbound,zones,fuels,techs,probability,lines,count,seasons,file,season_months
hydro,Uniform,-0.3,0,Congo;DRC_South,,,,,,,,
fossilfuel,Uniform,0,0.3,,,,,,,,,
demand,Uniform,-0.1,0.1,Gabon,,,,,,,,
thermal_outage,Uniform,0.3,1,,,,0.2,,,1,,
interco_outage,Uniform,0.5,1,,,,,,1,1,,
climate_year,Uniform,0,0,,,,,,,,supply/vre_monthly_cf.csv,Q1:1-6;Q2:7-12
```

Common columns:

- `feature`: the uncertain input (see below)
- `zones` (optional): model zones (or countries) concerned, separated by `;`. Names are checked against `zcmap`. Empty means all zones of the model.

**Continuous features** draw one value per sample from a [`chaospy`](https://pypi.org/project/chaospy/) distribution. `type` is the distribution (`Uniform`, `Normal`, ...) and `lowerbound`/`upperbound` are its two arguments, as relative changes (`-0.3` = −30%).

| Feature | Effect |
|---|---|
| `fossilfuel` | Scales the `pFuelPrice` of fossil fuels: Coal, Lignite, Gas, LNG, HFO, LFO, Diesel, Oil and their variants (`Gas_Moz`, `Coal_Hwan`, ...). Optional `fuels` column to choose the fuels. Zones are mapped to their country through `zcmap`. |
| `fuel` | Same for every fuel of `pFuelPrice` |
| `demand` | Scales the demand forecast (peak and energy) |
| `hydro` | Scales the availability of ROR and ReservoirHydro plants (`pAvailabilityDefault`, `pAvailability`), capped at 1 |

**Random outages** are drawn independently for each sample. `lowerbound`/`upperbound` give the range of the share of capacity lost during an outage (0–1).

| Feature | Columns | Effect |
|---|---|---|
| `thermal_outage` | `techs` (default ST, CCGT, OCGT, ICE, Nuclear), `probability` (default 0.1), `seasons` (default 1) | Each plant has an outage with the given probability. Its availability is reduced by a share drawn in the range, during `seasons` randomly chosen season(s), in every year. Written to `pAvailability`. |
| `interco_outage` | `lines` (`ZoneA-ZoneB;...`, default all corridors), `count` (default 1), `seasons` (default 1) | `count` corridors lose a share of their transfer capacity, existing and new lines, in both directions, during random season(s). Written to `pTransferOutage`. |

An outage applies to the drawn season(s) in **every model year** of the run. Only plants and corridors of the model's zones (`zcmap`) are drawn, even if the input files list other zones. `lines` names a corridor in either order (`DRC-DRC_South` or `DRC_South-DRC`). The reserve contribution of interconnections (planning reserve) is not reduced by `interco_outage`.

Thermal outages are **re-centred**: the input availability already includes average outages, so every plant in scope is first scaled by `1 / (1 − expected loss)`, where expected loss = probability × share of seasons hit × mean depth. Averaged over many samples, the availability then equals the input availability. Samples spread around it instead of being systematically pessimistic.

**Weather years (`climate_year`)**: each sample uses a historical weather year for solar and wind in all zones at once, which keeps the correlation between zones and technologies. For each zone, technology and season, `pVREProfile` and `pVREgenProfile` are multiplied by the ratio between that year's mean capacity factor and the average over all years, capped at 1. The daily shape of the profiles is kept. PV and CSP follow the solar series; onshore and offshore wind follow the wind series.

- `file`: the climate-year dataset (`vre_monthly_cf.csv`, see below), relative to `folder_input`, to the `epm` folder, or absolute
- `lowerbound`/`upperbound`: first and last weather year to draw from (0 = all years in the file)
- `season_months`: months of each model season, e.g. `Q1:1-3;Q2:4-6;Q3:7-9;Q4:10-12` or `Q1:12,1,2;...`. If empty, the months are split evenly over the seasons in calendar order, with a warning. Use the same months as the season mapping used to build the representative days of your input folder (`seasons_map` in the representative-days pipeline); otherwise the factors are computed on the wrong months.

The years are shuffled once and used in turn, so every weather year is used before any repeats: use at least as many samples as weather years (30 for the default dataset) for every year to appear. Zones without climate data keep their profiles (with a warning). In `pVREgenProfile`, only plants whose technology in `pGenDataInput` is PV, CSP or wind are scaled.

The outage and weather-year draws use a fixed seed (`MONTECARLO_SEED` in `preprocessing.py`). A sample therefore has the same draws in every parent scenario, so parents are compared on the same conditions, and adding samples does not change the earlier ones.

### 2. Build the climate-year dataset (once, reusable)

**A ready-made dataset** covers 1994–2023 for the SAPP zones (South_Africa, Lesotho, Eswatini, Mozambique_South, Mozambique_Central, Mozambique_North, Mozambique, Mozal, Zimbabwe_East, Zimbabwe_West, Botswana, Namibia, Zambia, Angola_North, Angola_Central, Angola_South, Angola_East, Angola, Tanzania, DRC, DRC_South, Malawi) and the data_test zones (Congo, CAR, Gabon): `pre-analysis/climate_years/data/era5_1994_2023_vre_monthly_cf.csv`, with its `era5_1994_2023_metadata.json`. Copy it into your input folder (for example `supply/vre_monthly_cf.csv`) or point the `file` column to it. Zone names must match the model's.

To build or extend it, `pre-analysis/climate_years/fetch_climate_years.py` downloads ERA5 hourly point series (dataset `reanalysis-era5-single-levels-timeseries`) from the Copernicus Climate Data Store, one site per zone. It converts them to PV and wind capacity factors and stores:

- `hourly/<zone>.csv.gz`: hourly capacity factors (UTC), reusable, for example for representative days per weather year
- `vre_monthly_cf.csv`: zone, tech (PV / Wind), year, month, cf. This is the input of `climate_year`.
- `metadata.json`: source, years, sites and conversion parameters

```sh
cd pre-analysis/climate_years
python fetch_climate_years.py --sites zone_sites_sapp.csv --start-year 1994 --end-year 2023 --output-dir output/sapp
```

It needs a CDS account, the API key in `~/.cdsapirc`, and the ERA5 licence accepted on the dataset page. A site takes about 40 seconds for 30 years. Downloads are cached per site in `raw/`, so an interrupted run can be resumed; `--zones "ZoneA;ZoneB"` limits the run to some zones. The `output/` folder is ignored by git: copy the monthly file to `data/` to keep it. The sites file (`zone, lat, lon`) uses approximate zone centroids. Replace them with the main solar and wind resource areas for better results. The PV and wind conversions are deliberately simple: they capture year-to-year variations, which is what the Monte Carlo uses, and do not replace a resource assessment.

### 3. Configuration

Add `solvemode` to your configuration file (`config.csv`):

```csv
"Solve mode: 2 = solve, 1 = solve and save PA_p.gdx (required for Monte Carlo), 0 = reuse PA_p.gdx",solvemode,1
```

The parent scenarios must be solved with `solvemode = 1` so that they save `PA_p.gdx`, which the Monte Carlo runs load to fix the investment plan. The Monte Carlo runs are solved with `solvemode = 2` and a short report (no hourly outputs).

> `chaospy` is an optional dependency used only for the continuous features. If it is missing, EPM installs it automatically (`pip install --only-binary=numpoly chaospy`). If this fails, for example without internet access, install it manually.

### 4. Run

```sh
python epm.py --folder_input data_test --scenarios -S baseline HighDemand --montecarlo --montecarlo_samples 30 --uncertainties mc_uncertainties.csv --no_plot_dispatch
```

The uncertainties file can be given relative to `folder_input` or to the `epm` folder. Each parent and sample gives a run named `<scenario>_MC001`, `<scenario>_MC002`, ... The modified inputs are written in `montecarlo/` folders next to the original files (ignored by git and overwritten by each run).

- With `--simple`, the samples are built after the reduction, so parents and samples use the same years.
- Number of samples: P90/P95 values and exceedance curves need many samples; use 20–30 at least, and at least the number of weather years with `climate_year`. Each sample is one dispatch run per parent scenario; `--parallel` runs them side by side.
- Postprocessing only: `python epm.py --folder_input <folder> --postprocess output/<run> --montecarlo` rebuilds the charts of an existing run.

### 5. Outputs

Results folder:

- `montecarlo_scenarios.csv`: for each Monte Carlo run, its parent, its sample, the sampled values, the weather year and the number of outages. This file links each run to the scenario whose investment plan it uses, so scenario names may contain `_`.
- `montecarlo_events.csv`: every outage drawn (run, type, plant or corridor, zone, season, depth)
- `samples_montecarlo.csv`, `input_scenarios_montecarlo.csv`: sampled values and input files of each run

The standard charts are produced for the parent scenarios. The Monte Carlo charts and their tables are in the `montecarlo/` subfolder of the graphs folder:

| Output | Content |
|---|---|
| `NPV_distribution.png` | System NPV across samples (dots), P10–P90 band, P50, and the deterministic parent run |
| `EENS_system.png`, `LOLE_system.png` | System energy not served (GWh) and loss of load (h/year) by year, with the reliability standard (3 h) |
| `EENS_exceedance.png` | Share of samples at or above each level of energy not served, last year |
| `LOLE_zones.png` | Mean loss of load by zone and year |
| `Cost_change.png` | Mean change of each cost component compared with the parent run, with P10–P90 |
| `EENS_drivers.png` | Energy not served against each sampled input, weather year and outage count |
| `montecarlo_summary.csv` | By parent: NPV mean/P10/P90, and by year EENS mean/P90, LOLE mean/P95, share of samples above the standard |
| `montecarlo_adequacy_zones.csv` | Same adequacy statistics by zone and year |
| `montecarlo_samples.csv` | One row per run: sampled values, NPV, EENS, LOLE |
| `montecarlo_outage_impact.csv` | For each plant or corridor, mean energy not served in samples where it failed compared with samples where it did not |

The adequacy indicators come from the report tables `pAdequacy` (by zone: LOLE in h, EENS in GWh, EENS as % of demand, peak unmet demand in MW) and `pAdequacySystem` (system LOLE = hours with unmet demand in any zone, EENS). They are written in every run, Monte Carlo or not, in the standard, short and extensive reports. An hour counts as loss of load when unmet demand exceeds 0.01 MW.

> **LOLE with representative days:** each modelled hour stands for many hours of the year (its `pHours` weight), so LOLE moves in steps of these weights and is less precise than in an 8,760-hour adequacy model. EENS is less affected. Use more representative days for adequacy studies, and read LOLE as an indicator rather than a measurement.

### 6. Transfer capacity outages outside the Monte Carlo

`pTransferOutage` is also an ordinary optional input, usable in any run to test a corridor outage deterministically. It is the share (0–1) of a corridor's transfer capacity, existing and new lines, that is unavailable in a season and year:

```csv
From,To,q,2025,2030,2040
DRC,DRC_South,Q1,0.9,0.9,0.9
DRC_South,DRC,Q1,0.9,0.9,0.9
```

Add it to `config.csv` (or to a scenario in `scenarios.csv`) as `pTransferOutage,trade/pTransferOutage.csv`. Give both directions and every model year (values are not interpolated). Without the file, EPM reads an empty template (`resources/optional_empty/pTransferOutage.csv`) and nothing changes. The constraint is `vFlow ≤ (pTransferLimit + new lines × CapacityPerLine) × (1 − pTransferOutage)` in `eTransferCapacityLimit` (`base.gms`).
