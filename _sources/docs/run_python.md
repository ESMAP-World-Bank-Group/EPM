# Running EPM from Python

Using the Python API enables additional functionalities such as direct scenario analysis and sensitivity analysis. You don’t need to know Python—just follow these steps !

Prerequisites: You need to have Python installed in your computer. If you don't have it, you can download Anaconda distribution from [here](https://www.anaconda.com/products/distribution).

- Navigate to the EPM directory:```cd EPM```
- Use the terminal to run the command: ```pip install -r requirements.txt```. The goal is to ensure that all required libraries are available (pandas, matplotlib,...). The environment is created using `requirements.txt` file
- Activate the Conda environment to ensure all necessary Python packages are available:
  ```conda activate esmap_env```
- Run the EPM model: ```python epm/epm.py```
- You can also specify additional arguments when launching EPM.


## Input Data
    
In the input folder, create `config.csv` file that contains the `baseline` scenario input specification. This specifies which csv file to use for each parameter under the baseline scenario. You can refer to the epm structure or the input sections to better understand the folder organization. You can look at the GitHub main `input` folder of the main branch for an example.  

If you are interested in running multiple scenarios in addition to the baseline scenario, create `scenarios.csv` file that contains the specification of the scenarios. By default, the model will run the baseline scenario. You can look at the GitHub main `input` folder of the main branch for an example.  


## Available Command-Line Arguments

EPM provides several command-line options to customize your simulation run. Below are the available arguments and their descriptions:

- **`--config`** *(string, default: `input/config.csv`)*  
  Specifies the path to the configuration file. If not provided, the default configuration file located at `input/config.csv` will be used.

- **`--folder_input`** *(string, default: `data_gambia`)*  
  Defines the input folder containing the necessary data files for the simulation. If not specified, it defaults to `data_gambia`.

- **`--scenarios`** *(string, optional)*  
  Allows specifying a scenario file name. If this option is omitted, no scenario file will be used.

- **`--selected_scenarios`** *(list, optional)*  
  Selection of scenarios to run. If this option is omitted, all scenarios will be run.
  - Example usage: --selected_scenarios baseline HighDemand

- **`--sensitivity`** *(flag, default: `False`)*  
  Enables sensitivity analysis when included. If this flag is not set, sensitivity analysis will be disabled.

- **`--montecarlo`** *(flag, default: `False`)*  
  Enables Monte-Carlo analysis when included. If this flag is not set, monte-carlo analysis will be disabled.

- **`--montecarlo_samples`** *(int, default: `10`)*  
  Specifies the number of samples used in the Monte-Carlo run. Number of samples to generate for the Monte Carlo analysis. A higher number improves coverage of the uncertainty space.

- **`--uncertainties`** *(flag, default: `None`)*  
  Allows specifying the uncertainties file name. Specifies which input files are subject to uncertainty. I the option is omitted, no uncertainties file will be used. Required if `montecarlo` is enabled.

- **`--postprocess`** *(flag, default: `None`)*  
  Only runs the postprocessing when included. 

- **`--no_plot_dispatch`** *(flag, default: `True`)*  
  Does not plot specific dispatch plots as automatic outputs when set to True. Speeds up the postprocessing and decreases memory requirements to store graphs.


### Example Command

To run EPM with a specific input folder and enable sensitivity analysis, use:
```sh
python epm.py --folder_input my_data --sensitivity
```
This will execute EPM using the `my_data` folder as input and perform sensitivity analysis.

For advanced users, additional arguments can be combined as needed to customize the simulation workflow.

## Monte-Carlo analysis (ongoing development)

EPM supports Monte Carlo analysis for exploring uncertainty in model inputs. This feature is still under active development, but the current implementation allows running multiple simulations across defined uncertainty ranges.

1. Define uncertainty ranges
Create a CSV file specifying the uncertain parameters. This file should include the following columns:
- `feature`: the name of the uncertain input (e.g., fossilfuel, demand)

- `type`: the type of probability distribution (e.g., Uniform, Normal)

- `lowerbound`: the lower limit of the distribution

- `upperbound`: the upper limit of the distribution

- `zones` (optional): List of zones where the uncertainty applies, separated by semicolons (e.g., `Zambia;Zimbabwe`). 
If left empty, the uncertainty applies to all zones.

Currently, the code supports uniform distributions (i.e., sampling uniformly between lower and upper bounds). Support for additional distributions (e.g., normal, beta) will be added in future versions.
Uncertainty sampling is powered by the `chaospy` package, so only distributions available in `chaospy` can be used.

Each row in your uncertainty definition file must correspond to a supported feature. Currently implemented features include:

- `fossilfuel`: scales fuel price trajectories for all fossil fuel types (Coal, HFO, LNG, Gas, Diesel) uniformly by a percentage
- `demand`: scales the entire demand forecast (peak & energy) uniformly by a percentage across zones specified
- `hydro`: scales hydro trajectories uniformly by a percentage  across zones specified
Example file: [pSettings.csv example](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_sapp/mc_uncertainties.csv).

2. Specify in your command-line:
```sh
python epm.py --folder_input my_data --montecarlo --montecarlo_samples 20 --uncertainties input/data_sapp/your_uncertainty_file.csv
```

This command will:
- Load the uncertainties defined in your file (`--uncertainties input/data_sapp/your_uncertainty_file.csv`)
- Generate 20 samples from the joint probability distribution (`--montecarlo_samples 20`)
- Create one scenario per sample
- Run the EPM model for each scenario

**Tip:** Set `reportshort = 1` in your configuration to reduce memory usage during multiple runs implied by Monte-Carlo analysis.

You can apply the same Monte Carlo simulations to selected scenarios by specifying:
 
```sh 
--scenarios input/data_sapp/your_scenario_file.csv --selected_scenarios Scenario1 Scenario2
```

This allows you for instance to test how specific scenarios (e.g., with or without a new generation or transmission project) perform under uncertainty.

