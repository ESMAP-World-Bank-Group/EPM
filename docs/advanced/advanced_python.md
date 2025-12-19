# Advanced Use of the Python API

This section provides a guide for advanced users who want to customize EPM runs, integrate the model into larger workflows, or automate scenario analysis using the Python API.

---

## Running the EPM Model Using Python

You can launch the model using:

- The `run_epm.py` script
- A Jupyter notebook

### Running via `run_epm.py`

Edit the code inside the `if __name__ == '__main__':` block to define your run. Then, run the script with:

```sh
python run_epm.py
```

Alternatively, use an IDE like Visual Studio Code or PyCharm to launch it.

---

### Running via a Jupyter Notebook

To run the model from a notebook, use the following pattern:

```python
from run_epm import launch_epm_multi_scenarios

config = "input/config.csv"
folder_input = "data_folder"
scenarios = "input/scenarios.csv"
selected_scenarios = ['baseline']
sensitivity = False
plot_all = False

folder, result = launch_epm_multi_scenarios(
    config=config,
    folder_input=folder_input,
    scenarios_specification=scenarios,
    sensitivity=sensitivity,
    selected_scenarios=selected_scenarios,
    cpu=1
)
```

---

## Overview of `run_epm.py`

`run_epm.py` is the main driver script for launching EPM simulations. It supports:

- Local and remote execution
- Parallel scenario processing
- Post-processing and plotting

Key components include:

- `launch_epm()` – Runs a single scenario
- `launch_epm_multiprocess()` – Runs multiple scenarios in parallel
- `launch_epm_multi_scenarios()` – Manages multi-scenario execution
- `perform_sensitivity()` – Performs sensitivity analysis
- `perform_assessment()` – Excludes specified projects for counterfactuals
- `postprocess_output()` – Summarizes results and generates plots

---

## Customizing the Script

### Modifying the `main()` Function

The `main()` function:

- Parses command-line arguments
- Sets up input/output folders
- Calls `launch_epm_multi_scenarios()`
- Calls `postprocess_output()` to process results

### Adding a New Command-Line Argument

1. Add the argument:

```python
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="Specify a custom output directory (default: output/)"
)
```

2. Use it:

```python
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```

---

## Function Reference

### `launch_epm(scenario, scenario_name, ...)`

Runs a single scenario.

**Key arguments:**

- `scenario` (DataFrame): Input values
- `scenario_name` (str): Name of the scenario
- `folder_input` (str): Input folder path
- `path_main_file`, `path_base_file`, etc.: GAMS file paths
- `path_engine_file` (optional): For GAMS Engine execution
- `prefix` (optional): For output folder names

**What it does:**

- Creates a dedicated output folder
- Copies required files
- Builds and runs the GAMS command

---

### `launch_epm_multiprocess(df, scenario_name, path_gams, ...)`

Executes multiple scenarios using Python multiprocessing.

**Key arguments:**

- `df`: Scenario input DataFrame
- `scenario_name`: Name of the run
- `path_gams`: Dict of GAMS script paths
- `folder_input`, `path_engine_file`: As above

**What it does:**

- Launches `launch_epm()` in parallel across CPU cores

---

### `launch_epm_multi_scenarios(config, scenarios_specification, ...)`

Handles full simulation workflow.

**Key arguments:**

- `config` (str): Path to `config.csv`
- `scenarios_specification` (str): Path to `scenarios.csv`
- `selected_scenarios` (list): Scenarios to run
- `cpu` (int): Number of cores to use
- `path_gams` (dict): GAMS file paths
- `folder_input`: Input folder path
- `sensitivity`, `generator_assessment`, `simple`: Optional flags

**What it does:**

- Reads scenario definitions
- Prepares inputs
- Calls multiprocessing function to execute scenarios
- Organizes outputs

---

### `perform_sensitivity(sensitivity, s)`

Modifies input data to explore uncertainty.

### `perform_assessment(generator_assessment, s)`

Creates counterfactuals by removing specified projects.

### `postprocess_output(folder, reduced_output, plot_all, folder='postprocessing')`

Handles results summarization and plotting.

---

Use this API to flexibly scale EPM for large batches of runs, scenario exploration, or integration into analytical workflows.
