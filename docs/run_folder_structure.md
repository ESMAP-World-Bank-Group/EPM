# EPM Folder structure

The following structure outlines the key components of the EPM repository:
```plaintext
EPM/
│
├── input/                          # Contains input data files required for the model
│   ├── data_eapp/                  # Input data specific to the EAPP region
│   ├── data_gambia/                # Input data specific to Gambia
│   ├── data_sapp/                  # Input data specific to the SAPP region
│   ├── config.csv                  # Configuration settings for input processing
│
├── postprocessing/                 # Scripts and tools for processing and analyzing model outputs
│   ├── static/                     # Static resources used in post-processing
│   └── utils.py                    # Utility functions for post-processing tasks
│
├── engine_base.gms                 # Defines engine run
├── base.gms                        # Specifies base configurations and parameters
├── cplex.opt                       # Options for the CPLEX solver
├── credentials_engine.json         # Stores credentials for engine access
├── generate_demand.gms             # Script for generating demand data
├── generate_report.gms             # Generates reports from model outputs
├── input_readers.gms               # Handles reading and importing input data
├── input_treatment.gms             # Processes and treats input data
├── input_verification.gms          # Verifies the integrity of input data
├── main.gms                        # Main script orchestrating execution
├── output_verification.gms         # Verifies and validates model outputs
└── run_epm.py                      # Python script to execute the model
```

##### 1. Input folder
The `input/` directory contains all the necessary data and configuration files required to run the model. It is structured into:

###### General configuration files
- **`config.csv`**: Defines global configuration parameters for the model.
- **`scenarios.csv`**: Specifies different scenario configurations that can be run.

###### Data folders
The model input data is organized into `data/` folders. For example, `data_eapp/` contains input data for the Eastern Africa Power Pool and `data_gambia/` contains input data specific to Gambia.

Each of these folders contains CSV files grouped by data type, including:

- **`constraint/`**: Defines various constraints applied in the model.
- **`h2/`**: Includes hydrogen-related data.
- **`load/`**: Contains electricity demand and load profiles.
- **`reserve/`**: Specifies reserve requirements for system reliability.
- **`resources/`**: Contains information on energy resources and availability.
- **`supply/`**: Defines supply-side parameters, including power plants and generation capacity.
- **`trade/`**: Contains parameters related to cross-border electricity trade.

Some files are located directly in the `data/` directory and contain key parameters that are not grouped within a specific subdirectory:

- **`pHours.csv`**: Defines the hourly resolution of the model.
- **`pSettings.csv`**: Contains general model settings and configuration values.
- **`y.csv`**: Defines the time horizon for the simulation.
- **`zcmap.csv`**: Provides zone and country mapping.

A more detailed description of these parameters and their contents is provided in the **Input** section.

---

##### 2. Post-processing folder
The `postprocessing/` directory contains scripts and data files used to process and visualize the model outputs.

- **`utils.py`**: The core Python package for post-processing results. It includes functions for generating various graphs and visualizations. The functions available in this package are detailed in the **Post-Processing** section.
- **`static/`**: Contains CSV files that define characteristics for visualizations, such as:
  - **Color schemes**
  - **Fuel names and mappings**
  - **Other parameters for customizing plots**
  
For more details on the available functions and configuration options, refer to the **Postprocessing** section.

---

##### 3. GAMS model files
These files define and execute the core GAMS model:

- **`main.gms`**: The main file that runs the model.
- **`base.gms`**: Defines the fundamental model structure.
- **`Engine_Base.gms`**: Used when running the model via GAMS Engine.
- **`generate_demand.gms`**: Generates demand scenarios.
- **`generate_report.gms`**: Creates summary reports from simulation results.
- **`input_readers.gms`**: Reads and formats input files.
- **`input_treatment.gms`**: Processes input data before execution.
- **`input_verification.gms`**: Checks input data consistency.
- **`output_verification.gms`**: Validates model outputs.

---

##### 4. Solver configuration
- **`cplex.opt`**: Defines solver settings for CPLEX.

---

##### 5. Execution scripts
- **`run_epm.py`**: Python script to launch EPM from a command line or an external script.

---

##### 6. Output folder
The `output/` directory is not present initially but is generated automatically when simulations are run. Results are stored in subdirectories following this pattern: `output/scenario_name/simulations_run_date_hour`
