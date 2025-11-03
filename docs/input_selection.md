# Specifying Input Data

Input data for EPM is specified through `config.csv`, which defines all parameter files used in the **baseline scenario**.
If you're unsure of the file structure, see this [example config.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test_region/config.csv).

The input folder contains all necessary `.csv` files, organized by type (e.g., `data_test_region`). Each file corresponds to a specific parameter in the model.
The `config.csv` file contains the path of the file within the input folder, and the model reads these files to set up the parameters. This allows you to easily switch between different sets of input data without modifying the model code.

Whether running from Python or GAMS Studio, it is **always required** to define the base input folder (`FOLDER_INPUT`) either:

- via command-line argument, or
- by setting it manually in `input_readers.gms`.

This ensures the model correctly points to the appropriate folder of `.csv` files.

---

## Running from Python

As detailed in the section _Running EPM from Python_, the model reads `config.csv` by default.

- **`name`**: name of the parameter (e.g., `pGenDataInputCustom`)
- **`file`**: path to the corresponding `.csv` input file

Only a few additional options are specified directly in `config.csv`:

| Option        | Description                                                                                                                        |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `solvemode`   | How the model is solved:<br> `2` = normal (default)<br> `1` = write savepoint `PA_pd.gdx`<br> `0` = generate model only (no solve) |
| `trace`       | Logging verbosity:<br> `0` = minimal (default)<br> `1` = detailed debugging output                                                 |
| `reportshort` | Report size:<br> `0` = full report (default)<br> `1` = compact report for multiple runs (e.g., Monte Carlo)                        |
| `modeltype`   | Solver formulation:<br> `MIP` = default mixed-integer run<br> `RMIP` = relax integrality for debugging or bounds                   |
| `cplexfile`   | Relative path to the CPLEX options file (for example `cplex/cplex_baseline.opt`) that sets detailed solver controls               |

---

## Overriding Inputs Using `scenarios.csv`

To run multiple variants, use the `--scenarios` argument. Each row in `scenarios.csv` defines a scenario, and only the files listed there will override those in `config.csv`.

- **If a file is not listed**, the model will fall back to the baseline `config.csv`.
- Useful for Monte Carlo, sensitivity analysis, or policy scenarios.

---

### Running from GAMS Studio

When using GAMS Studio, **only one scenario** can be run at a time.

### Option 1: Command-line Arguments (recommended)

You can override files directly in the command line:

```sh
--FOLDER_INPUT input/data_test_region --pNewTransmission input/data_test_region/trade/pNewTransmission.csv
```

This method is flexible but tedious for large sets of files.

### Option 2: Modify `input_readers.gms`

You can hard-code the baseline inputs:

```gams
$if not set pSettings $set pSettings input/%FOLDER_INPUT%/pSettings_baseline.csv
```

> ⚠️ Always make changes in a branch dedicated to your use case—not in the `main` branch.
