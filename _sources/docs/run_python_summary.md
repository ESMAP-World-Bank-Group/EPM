# Run EPM with Python - Summary

Example: Specify the input folder, enable sensitivity analysis, and use 4 CPU cores for parallel execution.
```bash
python run_epm.py --folder_input data_sapp --sensitivity --cpu 4
```

Other command-line options allow for further customization of the simulation run, such as specifying scenarios, enabling Monte Carlo analysis, and more.

| Argument                   | Type       | Default         | Description                                                                                      | Example Usage                                         |
|----------------------------|------------|-----------------|--------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `--config`                 | string     | `input/config.csv` | Path to the global configuration CSV file that defines baseline inputs                          | `--config input/my_config.csv`                        |
| `--folder_input`           | string     | `data_test`     | Folder containing input data files                                                              | `--folder_input data_sapp`                            |
|----------------------------|------------|-----------------|--------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `--simple`                 | list[str]  | None            | List of simplified parameters to enable simplified modeling modes                             | `--simple DiscreteCap y`                              |
| `--cpu`                   | integer    | 1               | Number of CPU cores to use for parallel execution                                              | `--cpu 4`  
| `--no_run_multiprocess`    | flag       | False (run multiprocessing by default) | Disable running simulations in parallel (forces serial execution)                          | `--no_run_multiprocess`                              |
| `--postprocess`            | string     | None            | Only run postprocessing on a specified output folder                                          | `--postprocess simulations_run_2024-10-01_15-00-00` |
| `--plot_selected_scenarios`| list[str]  | `"all"`         | Select scenarios to plot in postprocessing                                                    | `--plot_selected_scenarios baseline`                  |
| `--no_plot_dispatch`       | flag       | False (plot dispatch by default)      | Disable automatic dispatch plots in postprocessing                                           | `--no_plot_dispatch`                                 |
| `--graphs_folder`          | string     | `img`           | Folder name where postprocessing graphs will be saved                                        | `--graphs_folder figures`                            |
|----------------------------|------------|-----------------|--------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `--scenarios`              | string     | None            | Path to a CSV file defining multiple scenarios                                                  | `--scenarios input/scenarios.csv`                     |
| `--selected_scenarios`     | list[str]  | None            | List of scenario names to run from the scenario file                                           | `--selected_scenarios baseline HighDemand`           |
| `--sensitivity`            | flag       | False           | Enable sensitivity analysis (modifies certain input parameters automatically)                   | `--sensitivity`                                       |
| `--project_assessment`     | list[str]  | None            | Name(s) of project(s) to exclude for counterfactual analysis                                  | `--project_assessment SolarProject`  
| `--montecarlo`             | flag       | False           | Enable Monte Carlo uncertainty analysis                                                        | `--montecarlo`                                        |
| `--montecarlo_samples`     | integer    | 10              | Number of samples to generate for Monte Carlo runs                                             | `--montecarlo_samples 20`                             |
| `--uncertainties`          | string     | None            | CSV file defining uncertainty parameters for Monte Carlo analysis                               | `--uncertainties input/uncertainties.csv`            |
| `--reduced_output`         | flag       | False           | Enable reduced output mode (less memory-intensive postprocessing)                              | `--reduced_output`                                    |
---