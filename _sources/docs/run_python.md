# Run EPM from Python

Using the Python interface enables advanced features such as scenario creation, sensitivity analysis, and Monte Carlo simulations.

You don't need to know Python—just follow the steps below.

You must have Python installed. See the [prerequisites](https://esmap-world-bank-group.github.io/EPM/docs/run_prerequisites.html) for setup instructions.

---

## 1. Create a Python Environment

A Python environment ensures that all required libraries for EPM are available and isolated from other projects.

**Important**:
- Before creating the environment, you should have GAMS installed on your computer, with a recent version (ideally >= 48).

### On Windows

If you plan to run Monte Carlo analysis on **Windows**, first install `Microsoft C++ Build Tools` (needed to compile the optional `chaospy` package):

1. Go to [VSCode build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Download and install Build Tools for Visual Studio.
3. During installation, check the option: `C++ build tools`

Then you can follow these steps:

1. Open a terminal or command prompt (Anaconda Prompt).
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
5. Install the required libraries:
   ```sh
   pip install -r requirements.txt
   ```
6. *(Optional – Monte Carlo only)* Install `chaospy`:
   ```sh
   pip install chaospy==4.3.18
   ```

### On Mac

Follow the same steps as on Windows (minus the Build Tools prerequisite). Install `chaospy` only if you plan to run Monte Carlo analysis.

### On Linux (remote server)

1. Open a terminal or command prompt.
2. Navigate to the folder where you cloned EPM:
   ```sh
   cd EPM
   ```
   
3. Create a new environment named `epm_env`, specifying some Linux compilers.
```sh
conda create -n epm_env python=3.10 numpy cython gcc_linux-64 gxx_linux-64 -c conda-forge
```

4. Activate the environment:
```sh 
conda activate epm_env
export TMPDIR=~/pip_tmp
```

5. Install additional packages from the shared `requirements.txt` file:
```sh 
pip install -r requirements.txt
```

6. *(Optional – Monte Carlo only)* Install `chaospy`:
```sh
pip install chaospy==4.3.18
```

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

**All input data need to be within the `epm/input` folder. The path of the input file are defined based on this reference.**

Input data are defined in a folder `folder_input` and controlled via a `config.csv` file, which specifies what CSV files to use for each parameter in the model.

> Example input structure is provided in the GitHub `main` branch under the `input` folder. See the [input documentation](https://esmap-world-bank-group.github.io/EPM/docs/input_overview.html) for full details.

By default, the model uses the `data_test` input folder and `input/config.csv` file, but command line arguments allow you to specify different folders and configurations.


Input data for EPM is specified through two key components:

1. **Input Folder (`FOLDER_INPUT`)**  
   This folder contains all the necessary `.csv` input files for the model, organized by type (e.g., `data_test_region`). It holds the raw data the model reads.

### `--folder_input`
- **Type:** string  
- **Default:** `data_test`  
- **Purpose:** Folder containing the CSV input data for the simulation.  
- **Notes:** Should contain the input files referenced in the config.

2. **Configuration File (`config.csv`)**  
   This CSV file defines which input files correspond to which parameters in the model for the **baseline scenario**. It includes the parameter `name` and the relative `file` path inside the input folder. An example of a configuration file is available here:  [Example `config.csv`](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_test_region/config.csv)

### `--config`
- **Type:** string  
- **Default:** `input/config.csv`  
- **Purpose:** Specifies the configuration file that defines baseline input CSV files and parameters.  
- **Notes:** The config file is required for proper data loading and scenario configuration.


Important Configuration Options in config.csv

| Option         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `solvemode`    | How the model is solved:<br> `2` = normal (default)<br> `1` = write savepoint `PA_pd.gdx`<br> `0` = generate model only (no solve) |
| `trace`        | Logging verbosity:<br> `0` = minimal (default)<br> `1` = detailed debugging output |
| `reportshort`  | Report size:<br> `0` = full report (default)<br> `1` = compact report for multiple runs (e.g., Monte Carlo) |
| `modeltype`    | Solver type:<br> `MIP` = default<br> `RMIP` = force LP relaxation |


## 4. Specifying Optional Arguments for your Run

Check `Run EPM Advanced Features` for additional explanations on advanced features like sensitivity analysis, Monte Carlo simulations, and scenario management.


### `--simple`
- **Type:** list of strings  
- **Default:** ['DiscreteCap', 'y']  
- **Purpose:** Enables simplified model modes by toggling specified parameters.


### `--cpu`
- **Type:** integer  
- **Default:** 1  
- **Purpose:** Number of CPU cores to use for running scenarios in parallel.

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

### `--no_plot_dispatch`
- **Type:** flag (boolean)  
- **Default:** False (dispatch plotting enabled by default)  
- **Purpose:** Disables automatic creation of dispatch plots to save time and memory.

### `--graphs_folder`
- **Type:** string  
- **Default:** `img`  
- **Purpose:** Directory where postprocessing graphs and plots will be saved.

### `--simulation_label`
- **Type:** string  
- **Default:** timestamp generated name  
- **Purpose:** Replaces the automatic timestamp with a custom name for the results folder.

---

### `--scenarios`
- **Type:** string  
- **Default:** None (flag omitted)  
- **Purpose:** Path to a scenario definition file specifying multiple scenarios and their overridden inputs.  
- **Notes:** Use `--scenarios` with no value to load the default `scenarios.csv`. Provide a custom path (for example `--scenarios alternative.csv`) to select a different file. If the flag is omitted entirely, only the baseline scenario runs.

### `--selected_scenarios`
- **Type:** list of strings  
- **Default:** None  
- **Purpose:** Specifies a subset of scenarios from the scenarios file to run.  
- **Alias:** `-S`  
- **Example:** `-S baseline ScenarioA`

### `--sensitivity`
- **Type:** flag (boolean)  
- **Default:** False  
- **Purpose:** Enables sensitivity analysis which automatically modifies select parameters to evaluate model sensitivity.  
- **Usage:** Include `--sensitivity` to enable.

### `--project_assessment`
- **Type:** list of strings  
- **Default:** None  
- **Purpose:** Specifies projects to exclude from the simulation to assess counterfactual scenarios.  
- **Example:** `--project_assessment SolarProject`

---

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



---

## Example Usage

Run EPM with a custom input folder, sensitivity analysis enabled, and multiple CPU cores:
> It is important to specify the `config.csv` path when running python. Otherwise, a default value will be used, calibrated on the `data_test_region` folder.


```bash
python epm.py --folder_input data_test_region --config input/data_test_region/config.csv --sensitivity --cpu 4
```

Run postprocessing only on a previous simulation folder:
```bash
python epm.py --postprocess simulations_run_2025-05-01_12-00-00
```

Run Monte Carlo analysis with 20 samples and a specified uncertainties file:
```bash
python epm.py --folder_input data_test_region --config input/data_test_region/config.csv --montecarlo --montecarlo_samples 20 --uncertainties input/data_test_region/uncertainties.csv
``` 
