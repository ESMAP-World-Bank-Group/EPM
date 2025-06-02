# Run EPM with Python - Summary

First, navigate to the `epm` directory. 

Example: Specify the input folder, enable sensitivity analysis, and use 4 CPU cores for parallel execution.
```bash
python epm.py --folder_input data_sapp --sensitivity --cpu 4
```

Other command-line options allow for further customization of the simulation run, such as specifying scenarios, enabling Monte Carlo analysis, and more.

### General Configuration

| Argument         | Type   | Default             | Description                                                   | Example Usage                    |
|------------------|--------|---------------------|---------------------------------------------------------------|----------------------------------|
| `--config`       | string | `input/config.csv`  | Path to the global configuration CSV file                     | `--config input/my_config.csv`   |
| `--folder_input` | string | `data_test`         | Folder containing input data files                            | `--folder_input data_sapp`       |

---

### Execution Control

| Argument               | Type        | Default                         | Description                                                   | Example Usage                    |
|------------------------|-------------|----------------------------------|---------------------------------------------------------------|----------------------------------|
| `--simple`             | list[str]   | None                             | List of simplified parameters                                 | `--simple DiscreteCap y`         |
| `--cpu`                | integer     | 1                                | Number of CPU cores to use                                    | `--cpu 4`                        |
| `--no_run_multiprocess`| flag        | False (runs in parallel)         | Disable multiprocessing                                       | `--no_run_multiprocess`         |

---

### Postprocessing Options

| Argument                  | Type      | Default     | Description                                                      | Example Usage                               |
|---------------------------|-----------|-------------|------------------------------------------------------------------|---------------------------------------------|
| `--postprocess`           | string    | None        | Only run postprocessing on specified output folder               | `--postprocess simulations_run_2024-10-01`  |
| `--plot_selected_scenarios`| list[str]| `"all"`     | Select scenarios to plot                                         | `--plot_selected_scenarios baseline`        |
| `--no_plot_dispatch`      | flag      | False       | Disable automatic dispatch plots                                 | `--no_plot_dispatch`                        |
| `--graphs_folder`         | string    | `img`       | Folder to save postprocessing graphs                             | `--graphs_folder figures`                   |
| `--reduced_output`        | flag      | False       | Enable reduced output mode                                       | `--reduced_output`                          |

---

### Scenario & Sensitivity Analysis

| Argument              | Type       | Default | Description                                                       | Example Usage                              |
|-----------------------|------------|---------|-------------------------------------------------------------------|--------------------------------------------|
| `--scenarios`         | string     | None    | CSV file defining multiple scenarios                              | `--scenarios input/scenarios.csv`          |
| `--selected_scenarios`| list[str]  | None    | List of scenario names to run                                     | `--selected_scenarios baseline HighDemand` |
| `--sensitivity`       | flag       | False   | Enable sensitivity analysis                                       | `--sensitivity`                            |
| `--project_assessment`| list[str]  | None    | Project(s) to exclude in counterfactual analysis                  | `--project_assessment SolarProject`        |

---

### Monte Carlo Settings

| Argument              | Type     | Default | Description                                                       | Example Usage                              |
|-----------------------|----------|---------|-------------------------------------------------------------------|--------------------------------------------|
| `--montecarlo`        | flag     | False   | Enable Monte Carlo uncertainty analysis                           | `--montecarlo`                             |
| `--montecarlo_samples`| integer  | 10      | Number of samples to generate                                     | `--montecarlo_samples 20`                  |
| `--uncertainties`     | string   | None    | CSV file defining uncertainty parameters                          | `--uncertainties input/uncertainties.csv`  |