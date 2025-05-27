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

Input data are defined in a folder and controlled via a `config.csv` file, which specifies what CSV files to use for each parameter in the model.

- Example input structure is provided in the GitHub `main` branch under the `input` folder.
- See the [input documentation](https://esmap-world-bank-group.github.io/EPM/docs/input_overview.html) for full details.

---

## 4. Advanced Usage

The Python interface supports advanced features via command-line options.

### A. Run Multiple Scenarios

To run additional scenarios beyond the baseline:

1. Create a `scenarios.csv` file in your input folder.
2. Run EPM using:
   ```sh
   python epm.py --folder_input my_data --scenarios input/my_scenarios.csv
   ```

### B. Sensitivity Analysis

EPM supports sensitivity analysis to assess how changes in key parameters impact results.

- Currently, parameters to vary are hard-coded.
- Example command:
  ```sh
  python epm.py --folder_input my_data --sensitivity
  ```

### C. Monte Carlo Analysis (Experimental)

Monte Carlo allows evaluating uncertainty across input ranges.

1. Create an uncertainty file (`your_uncertainty_file.csv`) with the following columns:
   - `feature`: e.g., `fossilfuel`, `demand`, `hydro`
   - `type`: distribution type (e.g., `Uniform`)
   - `lowerbound`, `upperbound`
   - `zones` (optional): semicolon-separated list of zones, or leave blank for all

2. Example run:
   ```sh
   python epm.py --folder_input my_data \
                 --montecarlo \
                 --montecarlo_samples 20 \
                 --uncertainties input/data/your_uncertainty_file.csv
   ```

3. You can restrict Monte Carlo to selected scenarios:
   ```sh
   --scenarios input/scenarios.csv --selected_scenarios Scenario1 Scenario2
   ```

> Tip: Set `reportshort = 1` in your config to reduce memory use during multiple runs.

---

## 5. Available Command-Line Options

| Argument                  | Description                                                  | Default                         |
|---------------------------|--------------------------------------------------------------|---------------------------------|
| `--config`                | Path to config file                                          | `input/config.csv`              |
| `--folder_input`          | Input folder with model data                                 | `data_gambia`                   |
| `--scenarios`             | Path to scenarios CSV file                                   | *(None)*                        |
| `--selected_scenarios`    | List of scenario names to run                                | All in file                     |
| `--sensitivity`           | Enables sensitivity analysis                                 | `False`                         |
| `--montecarlo`            | Enables Monte Carlo analysis                                 | `False`                         |
| `--montecarlo_samples`    | Number of Monte Carlo samples                                | `10`                            |
| `--uncertainties`         | Path to uncertainty definition CSV                           | *(None)*                        |
| `--postprocess`           | Runs only the postprocessing step                            | *(None)*                        |
| `--no_plot_dispatch`      | Disables automatic plotting of dispatch results              | `True`                          |

---

## 6. Example: Combined Usage

```sh
python epm.py --folder_input input/data_eapp \
              --scenarios input/scenarios.csv \
              --selected_scenarios HighDemand \
              --montecarlo \
              --montecarlo_samples 50 \
              --uncertainties input/uncertainty_file.csv
```

This command runs Monte Carlo simulations on the `HighDemand` scenario using 50 samples.

---

Let me know if you want this section integrated into the full documentation file or need downloadable assets.