# EPM Folder Structure

This page describes the folder structure of the EPM repository and the GAMS model directory.

## Repository Structure

The GitHub repository contains the full EPM project including documentation, tools, and the core model:

```plaintext
EPM/                                # Repository root
│
├── docs/                           # Jupyter Book documentation source
├── config/                         # Configuration templates
├── pre-analysis/                   # Data preprocessing pipelines
├── tools/                          # Utility scripts
│
├── epm/                            # Core model directory (see below)
│
├── requirements.txt                # Python dependencies
├── Makefile                        # Documentation build commands
└── README.md                       # Project readme
```

## Model Directory (`epm/`)

The `epm/` folder contains all GAMS model files, Python orchestration scripts, and input/output data. This is the working directory when running the model:

```plaintext
epm/
│
├── input/                          # Contains input data folders
│   ├── data_test/                  # Test dataset
│   │   └── config.csv              # Configuration settings for input processing
│   ├── data_eapp/                  # Input data specific to the EAPP region
│   │   └── config.csv              # Configuration settings for input processing
│   ├── data_gambia/                # Input data specific to Gambia
│   │   └── config.csv              # Configuration settings for input processing
│   └── datapackage.json            # Frictionless Data Package schema for validation
│
├── postprocessing/                 # Scripts and tools for processing model outputs
│   ├── static/                     # Static resources used in post-processing
│   ├── figures_config.json         # Configuration for enabling/disabling figures
│   └── *.py                        # Plotting and analysis modules
│
├── resources/                      # Shared default parameters and headers
│   └── headers/                    # Column definitions and lookup tables
│
├── output/                         # Generated results (created after running)
│
├── main.gms                        # Main GAMS script orchestrating execution
├── base.gms                        # Core model equations and constraints
├── generate_demand.gms             # Script for generating demand data
├── generate_report.gms             # Generates reports from model outputs
├── input_readers.gms               # Handles reading and importing input data
├── hydrogen_module.gms             # Hydrogen module equations
│
├── epm.py                          # Python orchestrator script
├── input_treatment.py              # Python input processing and validation
├── input_verification.py           # Python input data checks
├── output_treatment.py             # Python output processing
├── preprocessing.py                # Python data preprocessing
│
└── cplex.opt                       # CPLEX solver options
```

```{note}
When running EPM from the command line, you execute `python epm.py` from the repository root. The script automatically manages paths to the `epm/` directory.
```

### Input Folder

The `input/` directory contains all the necessary data and configuration files required to run the model. It is structured into:

#### General Configuration Files

- **`config.csv`**: Defines global configuration parameters for the model.
- **`scenarios.csv`**: Specifies different scenario configurations that can be run.

#### Data Folders

The model input data is organized into `data_*/` folders. For example, `data_eapp/` contains input data for the Eastern Africa Power Pool and `data_gambia/` contains input data specific to Gambia.

Each of these folders contains CSV files grouped by data type, including:

- **`constraint/`**: Defines various constraints applied in the model.
- **`h2/`**: Includes hydrogen-related data.
- **`load/`**: Contains electricity demand and load profiles.
- **`reserve/`**: Specifies reserve requirements for system reliability.
- **`supply/`**: Defines supply-side parameters, including power plants and generation capacity.
- **`trade/`**: Contains parameters related to cross-border electricity trade.

Note: Header files and lookup tables (such as `ftfindex.csv`, `pTechData.csv`, `pFuelCarbonContent.csv`, and column header definitions) are now centralized in `epm/resources/headers/` and shared across all input folders.

Some files are located directly in the `data_*/` directory and contain key parameters that are not grouped within a specific subdirectory:

- **`pHours.csv`**: Defines the hourly resolution of the model.
- **`pSettings.csv`**: Contains general model settings and configuration values.
- **`y.csv`**: Defines the time horizon for the simulation.
- **`zcmap.csv`**: Provides zone and country mapping.

A more detailed description of these parameters and their contents is provided in the **Input** section.

---

### Post-processing Folder

The `postprocessing/` directory contains scripts and data files used to process and visualize the model outputs.

- **`figures_config.json`**: Configuration file to enable/disable specific figure categories and individual plots.
- **`static/`**: Contains CSV files that define characteristics for visualizations, such as:
  - **Color schemes**
  - **Fuel names and mappings**
  - **Other parameters for customizing plots**

For more details on the available functions and configuration options, refer to the **Postprocessing** section.

---

### GAMS Model Files

These files define and execute the core GAMS model:

- **`main.gms`**: The main file that runs the model.
- **`base.gms`**: Defines the fundamental model structure.
- **`generate_demand.gms`**: Generates demand scenarios.
- **`generate_report.gms`**: Creates summary reports from simulation results.
- **`input_readers.gms`**: Reads and formats input files.
- **`hydrogen_module.gms`**: Hydrogen module equations.

---

### Solver Configuration

- **`cplex.opt`**: Defines solver settings for CPLEX.

---

### Python Scripts

- **`epm.py`**: Main Python orchestrator to launch EPM from command line.
- **`input_treatment.py`**: Processes and validates input data before execution.
- **`input_verification.py`**: Checks input data consistency.
- **`output_treatment.py`**: Processes model outputs.
- **`preprocessing.py`**: Data preprocessing utilities.

---

### Output Folder

The `output/` directory is not present initially but is generated automatically when simulations are run. Results are stored in subdirectories following this pattern: `output/simulations_run_<timestamp>/`
