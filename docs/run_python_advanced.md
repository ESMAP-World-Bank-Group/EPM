# Run EPM from Python - Advanced Features

The Python interface supports advanced features via command-line options.

## A. Run Multiple Scenarios

To run additional scenarios beyond the baseline, use the `--scenarios` argument with a `scenarios.csv file`:

1. Create a `scenarios.csv` file in your `input` folder.
2. Run EPM using:

```sh
python epm.py --folder_input my_data --scenarios input/scenarios.csv
```

Each row overrides specific input files defined in config.csv.

- Files not listed in scenarios.csv default to those in config.csv.
- This works in combination with Monte Carlo simulations, sensitivity analyses, or policy scenarios.

You can limit execution to a selection of scenarios using the `--selected_scenarios` argument. For example:

```sh
python epm.py --folder_input my_data --scenarios input/my_scenarios.csv  --selected_scenarios baseline Retrade
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
| ├─ `zonal_spinning_reserve_constraints`   | Zonal spinning reserve constraints                                         | 0                                                   |
| ├─ `CostSurplus`                          | Cost of surplus generation                                                 | 1, 5                                                |
| ├─ `CostCurtail`                          | Cost of curtailment                                                        | 1, 5                                                |
| ├─ `interconMode`                         | Interconnection mode                                                       | 0, 1                                                |
| ├─ `fCountIntercoForReserves`             | Include interconnection reserves                                           | 0, 1                                                |
| └─ `sIntercoReserveContributionPct`       | Contribution of interconnection reserves                                   | 0, 0.5                                              |
| `y`                                       | Year definitions                                                           | Full continuous years range; first & last year only |
| `pDemandForecast`                         | Total demand forecast scaling                                              | -25%, -10%, +10%, +25%                              |
| `pDemandProfile`                          | Demand profile shape                                                       | Flat profile (equal hourly shares, 1/24)            |
| `pAvailabilityDefault`                    | Thermal plant availability (default values)                                | 0.3, 0.7                                            |
| `pCapexTrajectoriesDefault`               | Capital expenditure trajectories                                           | Flat trajectory (all values = 1)                    |
| `pFuelPrice`                              | Fuel price scaling                                                         | -20%, +20%                                          |
| `ResLimShare` (in `pGenDataExcelDefault`) | Contribution to reserves                                                   | -50%, -100%                                         |
| `BuildLimitperYear`                       | Build limit removal for candidate plants                                   | Set build limit = plant capacity (removal)          |
| `pVREProfile`                             | Variable renewable energy generation scaling                               | -20%, +20%                                          |

- Modified input files are saved in a `sensitivity` folder next to the baseline input files.
- New scenarios are created with updated paths to the modified input files.
- Only parameters enabled in the `sensitivity` dict are processed.
- This approach supports automated scenario generation for robustness checks.

## C. Monte-Carlo analysis (ongoing development)

EPM allows you to run Monte Carlo simulations to test how uncertainties (like fuel prices or demand) affect your results. This feature currently works only via the Python interface and is still under development.

What the code does is:

- You define uncertain parameters and their ranges.
- The model creates several versions (samples) of each scenario based on these uncertainties.
- For each scenario:
  1. It runs the model with your default settings and optimizes investment pathways (classical EPM approach).
  2. Then, it runs Monte Carlo simulations where these investment pathways are fixed and only dispatch is optimized.
- Graphs are automatically generated to show the results and how they vary due to uncertainty.

How to run this feature:

1. Define uncertainty ranges
   Create a CSV file specifying the uncertain parameters. This file should include the following columns:

- `feature`: the name of the uncertain input (e.g., fossilfuel, demand)

- `type`: the type of probability distribution (e.g., Uniform, Normal)

- `lowerbound`: the lower limit of the distribution

- `upperbound`: the upper limit of the distribution

- `zones` (optional): List of zones where the uncertainty applies, separated by semicolons (e.g., `Zambia;Zimbabwe`).
  If left empty, the uncertainty applies to all zones.

Currently, the code supports uniform distributions (i.e., sampling uniformly between lower and upper bounds). Support for additional distributions (e.g., normal, beta) will be added in future versions.
Uncertainty sampling is powered by the [`chaospy` package](https://pypi.org/project/chaospy/), so only distributions available in `chaospy` can be used.

Each row in your uncertainty definition file must correspond to a supported feature. Currently implemented features include:

- `fossilfuel`: scales fuel price trajectories for all fossil fuel types (Coal, HFO, LNG, Gas, Diesel) uniformly by a percentage
- `demand`: scales the entire demand forecast (peak & energy) uniformly by a percentage across zones specified
- `hydro`: scales hydro trajectories uniformly by a percentage across zones specified
  Example file: [mc_uncertainties.csv example](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_test_region/mc_uncertainties.csv).

2. Specify in your command-line:

```sh
python epm.py --folder_input my_data --config input/my_data/my_config.csv --scenarios input/my_data/my_scenarios.csv --selected scenarios baseline Scenario1 Scenario2  --montecarlo --montecarlo_samples 20 --uncertainties input/data_test_region/your_uncertainty_file.csv --no_plot_dispatch
```

This command will:

- Load the uncertainties defined in your file (`--uncertainties input/data_test_region/your_uncertainty_file.csv`)
- Generate 20 samples from the joint probability distribution (`--montecarlo_samples 20`)
- Run the model for each selected scenario
- Run Monte Carlo dispatch simulations for each sample

**Important:** Set `solvemode = 1` in your configuration to obtain the full outputs when running the default scenarios (in `PA_p.gdx` file). This saves detailed results used to fix investment decisions before the Monte Carlo step.
