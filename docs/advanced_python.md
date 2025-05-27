# Advanced Use Python API


Use `run_epm.py` or a notebook to launch the code. To run the code from `run_epm.py`, ensure that the code specified after `if __name__ == '__main__':` does what you want (see the following section on `run_epm.py`) . Then, run the following command in your terminal: `python run_epm.py`. Alternatively, use your favorite IDE (Visual Studio Code, Pycharm) to run the code. 

If you want to run the code from a notebook, add the following lines (example):
```python 
from run_epm import launch_epm_multi_scenarios

# Select the options you want for the run 
config = "input/config.csv"
folder_input = "data_folder"
scenarios = "input/scenarios.csv"
sensitivity = False
reduced_output = False
selected_scenarios = ['baseline']
plot_all = False

# Run 
folder, result = launch_epm_multi_scenarios(config=config,
                                            folder_input=folder_input,
                                            scenarios_specification=scenarios,
                                            sensitivity=sensitivity,
                                            selected_scenarios=selected_scenarios,
                                            cpu=1)
```
---

## Description of `run_epm.py`

The `run_epm.py` script is the main Python execution script. It manages scenario execution, interacts with GAMS, and handles post-processing. It allows users to run simulations locally or on GAMS Engine, execute multiple scenarios in parallel, and process outputs.

The script contains several functions, including:
- **`launch_epm(scenario, scenario_name, ...)`** : Runs a single scenario using GAMS, handles input files, and executes GAMS commands.
- **`launch_epm_multiprocess(df, scenario_name, path_gams, ...)`** : Calls `launch_epm()` with multiprocessing to handle parallel scenario execution.
- **`launch_epm_multi_scenarios(config, scenarios_specification, ...)`** : Manages the execution of multiple scenarios, organizes inputs, and launches simulations.
- **`perform_sensitivity(sensitivity, s)`** : Modifies input parameters for sensitivity analysis.
- **`perform_assessment(project_assessment, s)`** : Creates counterfactual scenarios without specific projects.
- **`postprocess_output(folder, reduced_output, plot_all, folder='postprocessing')`** : Processes model outputs, generates visualizations, and aggregates results.

---

### Adapting the script through the `main()` function

The `main()` function serves as the entry point for the script. It handles:
- Parsing command-line arguments to configure the execution.
- Managing input and output directories.
- Executing `launch_epm_multi_scenarios()` to run the specified scenarios.
- Calling `postprocess_output()` to generate and analyze results.

To add a new command-line argument, follow these steps:

1. Modify the argument parser
```python
    parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="Specify a custom output directory (default: output/)"
    )
```

2. Use this new argument in the script
```output_dir = args.output_dir
    if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```

### `run_epm.py` functions
#### `launch_epm(scenario, scenario_name, ...)`
Runs a single scenario using GAMS.

##### Parameters
- **`scenario`** (pd.DataFrame) : A DataFrame containing input parameters for the scenario.
- **`scenario_name`** (str, optional) : The name of the scenario. If not provided, it is generated based on the timestamp.
- **`path_main_file`** (str) : Path to the main GAMS script (`main.gms`).
- **`path_base_file`** (str) : Path to the base GAMS file.
- **`path_report_file`** (str) : Path to the GAMS script for generating reports.
- **`path_reader_file`** (str) : Path to the GAMS script for reading inputs.
- **`path_verification_file`** (str) : Path to the GAMS script for input verification.
- **`path_treatment_file`** (str) : Path to the GAMS script for input preprocessing.
- **`path_demand_file`** (str) : Path to the GAMS script for demand generation.
- **`path_cplex_file`** (str) : Path to the CPLEX solver options file.
- **`folder_input`** (str, optional) : Path to the input folder.
- **`path_engine_file`** (str, optional) : Path to the GAMS Engine file for remote execution.
- **`prefix`** (str, optional) : Prefix for output folder names.

##### Functionality
- Creates a dedicated folder for the scenario.
- Copies necessary files and constructs the GAMS execution command.
- Runs the scenario either locally or via GAMS Engine.

---

#### `launch_epm_multiprocess(df, scenario_name, path_gams, ...)`
Runs `launch_epm()` in a multiprocessing environment.

##### Parameters
- **`df`** (pd.DataFrame) : Input data for the scenario.
- **`scenario_name`** (str) : Name of the scenario.
- **`path_gams`** (dict) : Dictionary containing paths to required GAMS files.
- **`folder_input`** (str, optional) : Path to the input folder.
- **`path_engine_file`** (str, optional) : Path to the GAMS Engine file.

##### Functionality
- Calls `launch_epm()` for each scenario in parallel.
- Distributes execution across multiple CPU cores.

---

#### `launch_epm_multi_scenarios(config, scenarios_specification, ...)`
Manages the execution of multiple scenarios.

##### Parameters
- **`config`** (str, default: `'config.csv'`) : Path to the global configuration file.
- **`scenarios_specification`** (str, default: `'scenarios.csv'`) : Path to the scenario definitions file.
- **`selected_scenarios`** (list, optional) : List of scenarios to run.
- **`cpu`** (int, default: `1`) : Number of CPU cores to use for parallel execution.
- **`path_gams`** (dict, optional) : Dictionary containing paths to required GAMS files.
- **`sensitivity`** (dict, optional) : Specifies sensitivity parameters.
- **`path_engine_file`** (str, optional) : Path to the GAMS Engine file for remote execution.
- **`folder_input`** (str, optional) : Path to the input folder.
- **`project_assessment`** (str, optional) : Specifies a project to exclude for counterfactual analysis.
- **`simple`** (list, optional) : Simplified model settings.

##### Functionality
- Loads input configurations and scenario definitions.
- Normalizes input paths and processes sensitivity options.
- Creates output directories and runs simulations in parallel using `launch_epm_multiprocess()`.
