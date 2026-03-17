# Run from Python

Running EPM through Python is the recommended approach. `epm.py` reads your configuration, launches GAMS for each scenario, and runs postprocessing automatically.

> **Prerequisites:** Python environment set up. If not done yet, see [Installation](run_installation.md).

---

## Quick start

Open a terminal **from inside your EPM repository folder** and run:

```sh
conda activate epm_env
python epm.py --folder_input data_test --config data_test/config.csv
```

- `--folder_input`: your input data folder (inside `epm/input/`)
- `--config`: the configuration file that tells EPM which CSV files to use

Results are written to `output/simulations_run_<timestamp>/`.

---

## Common workflows

=== "Basic run"

    Run EPM with your own input folder:

    ```sh
    python epm.py --folder_input my_country --config my_country/config.csv
    ```

    Use `--simple` to relax integer constraints (faster, good for first tests):

    ```sh
    python epm.py --folder_input my_country --config my_country/config.csv --simple
    ```

=== "Multi-scenario"

    Run all scenarios defined in a `scenarios.csv` file, using 4 CPU cores in parallel:

    ```sh
    python epm.py --folder_input my_country --config my_country/config.csv --scenarios --cpu 4
    ```

    Run only a subset of scenarios:

    ```sh
    python epm.py --folder_input my_country --config my_country/config.csv --scenarios -S baseline HighDemand
    ```

=== "Monte Carlo"

    Run 50 Monte Carlo samples using an uncertainties definition file:

    ```sh
    python epm.py --folder_input my_country --config my_country/config.csv \
      --montecarlo --montecarlo_samples 50 \
      --uncertainties my_country/uncertainties.csv \
      --cpu 8 --reduced_output
    ```

    !!! note "Windows prerequisite"
        Monte Carlo requires `chaospy`. On Windows, install [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first, then `pip install chaospy==4.3.18`.

=== "Postprocess only"

    Re-run postprocessing on an existing results folder without re-solving the model:

    ```sh
    python epm.py --postprocess simulations_run_2025-05-01_12-00-00
    ```

=== "Assessment"

    Assess the value of a specific project by running a counterfactual (model without that project):

    ```sh
    python epm.py --folder_input my_country --config my_country/config.csv \
      --generator_assessment SolarProject
    ```

    Assess the value of an interconnection:

    ```sh
    python epm.py --folder_input my_country --config my_country/config.csv \
      --interco_assessment Angola-Zambia
    ```

---

## All options

The tables below list every argument you can pass to `python epm.py`. Arguments are optional unless stated otherwise (the model uses default values for anything not specified).

### Core inputs

| Argument | Default | Description |
|---|---|---|
| `--folder_input` | `data_test` | Input data folder (inside `epm/input/`) |
| `--config` | `config.csv` | Configuration file path, relative to `folder_input` |
| `--modeltype` | `None` | Override solver type: `MIP` or `RMIP` |

### Scenarios

| Argument | Default | Description |
|---|---|---|
| `--scenarios` | `None` | Scenario file. Use without value to load `scenarios.csv`, or provide a custom path |
| `-S` / `--selected_scenarios` | `None` | Run only a subset of scenarios. Example: `-S baseline HighDemand` |
| `--simple` | `None` | Relax integer constraints. Use without arguments for default `['DiscreteCap', 'y']` |

### Performance

| Argument | Default | Description |
|---|---|---|
| `--cpu` | `1` | Number of CPU cores for parallel scenario runs |
| `--reduced_output` | `False` | Generate smaller output files (recommended for large Monte Carlo runs) |
| `--output_zip` | `False` | Zip the output folder after processing |

### Monte Carlo & sensitivity

| Argument | Default | Description |
|---|---|---|
| `--montecarlo` | `False` | Enable Monte Carlo simulation |
| `--montecarlo_samples` | `10` | Number of random samples |
| `--uncertainties` | `None` | CSV file defining uncertain parameters and their distributions |
| `--sensitivity` | `False` | Enable built-in sensitivity analysis |

### Assessment

| Argument | Default | Description |
|---|---|---|
| `--generator_assessment` | `None` | Exclude named generators to assess their counterfactual value |
| `--project_assessment` | `None` | Create individual and combined assessment scenarios per project |
| `--interco_assessment` | `None` | Assess the value of a named interconnection |

### Regional & country filtering

| Argument | Default | Description |
|---|---|---|
| `--country` | `None` | Filter zones to include only those belonging to this country |
| `--focus_country` | `None` | After a regional run, generate single-country inputs for this country |

### Postprocessing & output

| Argument | Default | Description |
|---|---|---|
| `--postprocess` | `None` | Run postprocessing only on an existing results folder |
| `--generate_figures` | `False` | Generate figures during postprocessing |
| `--no_plot_dispatch` | | Disable dispatch plots (saves time on large runs) |
| `--plot_selected_scenarios` | `all` | Restrict plots to a subset of scenarios |
| `--graphs_folder` | `img` | Folder where postprocessing plots are saved |
| `--reduce_definition_csv` | `False` | Generate reduced yearly CSV files for Tableau |
| `--simulation_label` | timestamp | Custom name for the results folder |

### Debugging

| Argument | Default | Description |
|---|---|---|
| `--debug` | `False` | Enable verbose DEBUG mode in GAMS |
| `--trace` | `False` | Enable TRACE mode in GAMS |

---

## Related

- [Input Setup](../input/input_setup.md): how `config.csv`, `pSettings`, and scenarios work together
- [Input Catalog](../input/input_detailed.md): full reference of all input parameters
- [Debugging](run_debugging.md): what to do when the model does not converge
