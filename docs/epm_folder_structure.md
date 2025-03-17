---

## Directory and File Descriptions

### 1. **Input Directory (`input/`)**
The `input/` directory contains all the necessary data and configuration files required to run the model. It is structured into:

#### **General Configuration Files**
- **`config.csv`**: Defines global configuration parameters for the model.
- **`scenarios.csv`**: Specifies different scenario configurations that can be run.

#### **Data Folders**
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

A more detailed description of these parameters and their contents is provided in the **Input Data** section.

---

### 2. **Post-Processing Directory (`postprocessing/`)**
The `postprocessing/` directory contains scripts and data files used to process and visualize the model outputs.

- **`utils.py`**: The core Python package for post-processing results. It includes functions for generating various graphs and visualizations. The functions available in this package are detailed in the **Post-Processing** section.
- **`static/`**: Contains CSV files that define characteristics for visualizations, such as:
  - **Color schemes**
  - **Fuel names and mappings**
  - **Other parameters for customizing plots**
  
For more details on the available functions and configuration options, refer to the **Post-Processing** section.

---

### 3. **GAMS Model Files**
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

### 4. **Solver Configuration**
- **`cplex.opt`**: Defines solver settings for CPLEX.

---

### 5. **Execution Scripts**
- **`run_epm.py`**: Python script to launch EPM from a command line or an external script.

---

### 6. **Output Directory (`output/`)**
The `output/` directory is not present initially but is generated automatically when simulations are run. Results are stored in subdirectories following this pattern: `output/scenario_name/simulations_run_date_hour`