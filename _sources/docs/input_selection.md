# Specifying Input Data

**All input data need to be within the `epm/input` folder. The path of the input file are defined based on this reference.**

Input data for EPM is specified through two key components:

1. **Input Folder (`FOLDER_INPUT`)**  
   This folder contains all the necessary `.csv` input files for the model, organized by type (e.g., `data_sapp`). It holds the raw data the model reads.

2. **Configuration File (`config.csv`)**  
   This CSV file defines which input files correspond to which parameters in the model for the **baseline scenario**. It includes the parameter `name` and the relative `file` path inside the input folder.

An example of a configuration file is available here:  
[Example `config.csv`](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_sapp/config.csv)

---

# Running the Model

## Using Python API

When running the model from Python (see *Running EPM from Python*), you must specify:

- The input folder: `--folder_input data_sapp`  
- The configuration file: `--config data_sapp/config.csv`

Example command:
```bash
python run_epm.py --folder_input data_sapp --config data_sapp/config.csv
```

The model will read `config.csv` to locate all parameter files inside the input folder.

Important Configuration Options in config.csv

| Option         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `solvemode`    | How the model is solved:<br> `2` = normal (default)<br> `1` = write savepoint `PA_pd.gdx`<br> `0` = generate model only (no solve) |
| `trace`        | Logging verbosity:<br> `0` = minimal (default)<br> `1` = detailed debugging output |
| `reportshort`  | Report size:<br> `0` = full report (default)<br> `1` = compact report for multiple runs (e.g., Monte Carlo) |
| `modeltype`    | Solver type:<br> `MIP` = default<br> `RMIP` = force LP relaxation |

### Overriding Inputs with scenarios.csv

To run multiple scenarios or variants, use the --scenarios argument with a scenarios.csv file. Each row overrides specific input files defined in config.csv.
- Files not listed in scenarios.csv default to those in config.csv.
- This works in combination with Monte Carlo simulations, sensitivity analyses, or policy scenarios.

## Running from GAMS Studio

You can override inputs by passing command-line parameters:
```sh
--FOLDER_INPUT input/data_sapp --pNewTransmission input/data_sapp/trade/pNewTransmission.csv
```

### Summary Notes
- Defining FOLDER_INPUT is mandatory whether running from Python or GAMS. 
- config.csv controls the baseline input file mappings.
- Scenario-specific overrides happen via scenarios.csv when used.
- Use command-line arguments or modify input_readers.gms to set input files in GAMS Studio.