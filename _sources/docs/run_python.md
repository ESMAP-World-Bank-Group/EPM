# Running EPM from Python

Using the Python interface enables advanced features such as scenario creation, sensitivity analysis, and Monte Carlo simulations.

You don't need to know Python—just follow the steps below.

You must have Python installed. See the [prerequisites](https://esmap-world-bank-group.github.io/EPM/docs/run_prerequisites.html) for setup instructions.

---

## 1. Create a Python Environment

A Python environment ensures that all required libraries for EPM are available and isolated from other projects.

**Important**:
- Before creating the environment, you should have GAMS installed on your computer, with a recent version (ideally >= 48).

### On Mac

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
   pip install -r requirements_mac_and_windows.txt
   ```

### On Windows

The same steps should be followed. An extra step is however necessary before creating the package since Windows needs extra tools to compile some of the packages (specifically, `chaospy`).

1. Go to [VSCode build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Download and install Build Tools for Visual Studio.
3. During installation, check the option: `C++ build tools`
4. After installation, close and reopen the terminal (Anaconda Prompt), activate your environment, and run the installation command as described in previous section for Mac.

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

5. Install with conda the `chaospy` package, necessary for Monte-Carlo simulations 
```sh 
conda install -c conda-forge chaospy
```

6. Install additional packages from the `requirements.txt` file specific to Linux
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

Currently, parameters to vary are hard-coded. It means you cannot yet specify which parameters to vary directly from the command line. 

To run EPM and enabling sensitivity analysis, use:
```sh
python epm.py --folder_input my_data --sensitivity
```
This will execute EPM and perform sensitivity analysis.

### C. Monte-Carlo analysis (ongoing development)

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
- `hydro`: scales hydro trajectories uniformly by a percentage  across zones specified
Example file: [mc_uncertainties.csv example](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_sapp/mc_uncertainties.csv).

2. Specify in your command-line:
```sh
python epm.py --folder_input my_data --config input/my_data/my_config.csv --scenarios input/my_data/my_scenarios.csv --selected scenarios baseline Scenario1 Scenario2  --montecarlo --montecarlo_samples 20 --uncertainties input/data_sapp/your_uncertainty_file.csv --no_plot_dispatch
```

This command will:
- Load the uncertainties defined in your file (`--uncertainties input/data_sapp/your_uncertainty_file.csv`)
- Generate 20 samples from the joint probability distribution (`--montecarlo_samples 20`)
- Run the model for each selected scenario 
- Run Monte Carlo dispatch simulations for each sample

**Important:** Set `solvemode = 1` in your configuration to obtain the full outputs when running the default scenarios (in `PA_p.gdx` file). This saves detailed results used to fix investment decisions before the Monte Carlo step.



