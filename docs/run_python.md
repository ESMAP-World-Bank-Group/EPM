# Running EPM from Python

Using the Python interface enables advanced features such as scenario creation, sensitivity analysis, and Monte Carlo simulations.

You don't need to know Python—just follow the steps below.

You must have Python installed. See the [prerequisites](https://esmap-world-bank-group.github.io/EPM/docs/run_prerequisites.html) for setup instructions.

---

## 1. Create a Python Environment

A Python environment ensures that all required libraries for EPM are available and isolated from other projects.

Follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the folder where you cloned EPM:
   ```sh
   cd EPM
   ```
3. Create a new environment named `epm_env`:
   ```sh
   conda create -n epm_env python=3.10
   ```
4. Activate the environment:
   ```sh
   conda activate epm_env
   ```
5. Install all required libraries:
   ```sh
   pip install -r requirements.txt
   ```

**Important**:
- Before creating the environment, you should have GAMS installed on your computer, with a recent version (ideally >= 48).
- For Windows users: You might see an error when installing some packages (like chaospy, scipy, etc.). This is because Windows needs extra tools to compile them.
    1. Go to [VSCode build tools](https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/)
    2. Download and install Build Tools for Visual Studio.
    3. During installation, check the option: `C++ build tools`
    4. After installation, close and reopen the terminal (Anaconda Prompt), activate your environment, and run the installation command again.

---

## 2. Run the Model (Basic Test)

Once the environment is set up, you can test the model:

1. Navigate to the `epm` code directory:
   ```sh
   cd epm
   ```
2. Run the model:
   ```sh
   python epm.py
   ```

This runs the model using the default input folder and configuration.

---

## 3. Input Data

Input data are defined in a folder `folder_input` and controlled via a `config.csv` file, which specifies what CSV files to use for each parameter in the model.

- Example input structure is provided in the GitHub `main` branch under the `input` folder.
- See the [input documentation](https://esmap-world-bank-group.github.io/EPM/docs/input_overview.html) for full details.

By default, the model uses the `data_test` input folder and `input/config.csv` file, but command line arguments allow you to specify different folders and configurations.

---

## 4. Specifying Arguments for your Run


### `--folder_input`
- **Type:** string  
- **Default:** `data_test`  
- **Purpose:** Folder containing the CSV input data for the simulation.  
- **Notes:** Should contain the input files referenced in the config.


### `--config`
- **Type:** string  
- **Default:** `input/config.csv`  
- **Purpose:** Specifies the configuration file that defines baseline input CSV files and parameters.  
- **Notes:** The config file is required for proper data loading and scenario configuration.

### `--scenarios`
- **Type:** string  
- **Default:** None  
- **Purpose:** Path to a scenario definition file specifying multiple scenarios and their overridden inputs.  
- **Notes:** If omitted, only the baseline scenario will run.

### `--sensitivity`
- **Type:** flag (boolean)  
- **Default:** False  
- **Purpose:** Enables sensitivity analysis which automatically modifies select parameters to evaluate model sensitivity.  
- **Usage:** Include `--sensitivity` to enable.

### `--montecarlo`
- **Type:** flag (boolean)  
- **Default:** False  
- **Purpose:** Enables Monte Carlo simulation by sampling uncertain parameters multiple times.  
- **Usage:** Include `--montecarlo` to enable.

### `--montecarlo_samples`
- **Type:** integer  
- **Default:** 10  
- **Purpose:** Number of random samples/scenarios to generate during Monte Carlo analysis.  
- **Notes:** Larger numbers increase accuracy but require more computation.

### `--uncertainties`
- **Type:** string  
- **Default:** None  
- **Purpose:** CSV file that specifies uncertain parameters and their distributions for Monte Carlo sampling.  
- **Notes:** Required if `--montecarlo` is enabled.

### `--reduced_output`
- **Type:** flag (boolean)  
- **Default:** False  
- **Purpose:** Generates reduced-size outputs to save memory, useful for large Monte Carlo runs.

### `--selected_scenarios`
- **Type:** list of strings  
- **Default:** None  
- **Purpose:** Specifies a subset of scenarios from the scenarios file to run.  
- **Example:** `--selected_scenarios baseline ScenarioA`

### `--cpu`
- **Type:** integer  
- **Default:** 1  
- **Purpose:** Number of CPU cores to use for running scenarios in parallel.

### `--project_assessment`
- **Type:** list of strings  
- **Default:** None  
- **Purpose:** Specifies projects to exclude from the simulation to assess counterfactual scenarios.  
- **Example:** `--project_assessment SolarProject`

### `--simple`
- **Type:** list of strings  
- **Default:** None  
- **Purpose:** Enables simplified model modes by toggling specified parameters.

### `--engine`
- **Type:** string  
- **Default:** None  
- **Purpose:** Path to a GAMS Engine file for running simulations remotely.

### `--postprocess`
- **Type:** string  
- **Default:** None  
- **Purpose:** Runs only postprocessing on an existing output folder, skipping simulation.

### `--plot_selected_scenarios`
- **Type:** list of strings  
- **Default:** `"all"`  
- **Purpose:** Specifies scenarios to plot in postprocessing.

### `--no_run_multiprocess`
- **Type:** flag (boolean)  
- **Default:** False (multiprocessing enabled by default)  
- **Purpose:** Disables parallel processing of scenarios (forces sequential execution).

### `--no_plot_dispatch`
- **Type:** flag (boolean)  
- **Default:** False (dispatch plotting enabled by default)  
- **Purpose:** Disables automatic creation of dispatch plots to save time and memory.

### `--graphs_folder`
- **Type:** string  
- **Default:** `img`  
- **Purpose:** Directory where postprocessing graphs and plots will be saved.

---

## Example Usage

Run EPM with a custom input folder, sensitivity analysis enabled, and multiple CPU cores:

```bash
python run_epm.py --folder_input data_sapp --sensitivity --cpu 4
```

Run postprocessing only on a previous simulation folder:
```bash
python run_epm.py --postprocess simulations_run_2025-05-01_12-00-00
```

Run Monte Carlo analysis with 20 samples and a specified uncertainties file:
```bash
python run_epm.py --montecarlo --montecarlo_samples 20 --uncertainties input/uncertainties.csv
``` 


