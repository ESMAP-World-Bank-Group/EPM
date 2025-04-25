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
python epm/epm.py --folder_input my_data --sensitivity
```
This will execute EPM using the `my_data` folder as input and perform sensitivity analysis.

For advanced users, additional arguments can be combined as needed to customize the simulation workflow.
