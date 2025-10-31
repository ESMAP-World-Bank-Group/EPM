# Output Folder Structure

When launching the EPM model with the Python API, the framework generates a comprehensive set of outputs for analysis and visualization. These include:

- A GDX file (`epmresults.gdx`) with detailed model results
- CSV outputs in the `output_csv/` folder for integration with tools such as Tableau
- Automatically generated plots for dispatch, capacity, and energy mix
- Spreadsheet summaries of key results

All outputs are grouped by scenario (e.g., `baseline`) and stored in a timestamped output folder for traceability.

After running the model, EPM creates a general output folder named:

```
simulations_run_<timestamp>/
```

This folder includes all outputs from the simulation, organized as follows:

```
simulations_run_<timestamp>/
│
├── img/                             # Automatically generated plots
│   ├── 1_capacity/                  # Installed capacity figures
│   ├── 2_cost/                      # Cost breakdown figures
│   ├── 3_energy/                    # Energy balance and generation plots
│   ├── 4_interconnection/           # Interconnection flows and utilization plots
│   ├── 5_dispatch/                  # Hourly dispatch plots
│   └── 6_maps/                      # Spatial visualizations (if applicable)
│
├── baseline/                        # Scenario-specific GAMS outputs and logs
│   ├── main.lst                     # GAMS listing file (modeltype logs and diagnostics)
│
├── epmresults.gdx                   # GDX file with model results
├── input_gdx/                       # Pre-processed input files used by GAMS
├── output_csv/                      # All results exported as CSV (for postprocessing or dashboards)
│
├── cplex.opt                        # modeltype configuration used for the run
├── summary.csv                      # High-level summary of model results
├── summary_detailed.csv             # Extended summary with breakdowns by tech, fuel, zone, etc.
├── simulations_scenarios.csv        # Metadata and status for all scenarios run
```

- This structure is generated **automatically** for each run done with Python.
- When running **multiple scenarios**, results are merged in shared summary and comparison outputs, but each scenario still produces its own dedicated subfolders.
- The `img/` folder only appears if plotting is enabled and includes both per-scenario and cross-scenario visuals.

---

## Ouput Workflows

There are two main workflows for visualizing EPM model results:

- _Tableau_ for interactive exploration and dashboards
- _Python_ for custom plots and analyses, relying on in-house libraries. Check out the Advanced Topics for more details.
