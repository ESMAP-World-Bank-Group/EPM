# Output

To be able to get automatic output from the simulation as well as the good files structure, use the Python API.

## Output Folder Structure

After running the EPM model a general folder output named `simulations_run_<timestamp>` is created in the root directory of your project. This folder contains all the outputs from the simulation run, including plots, raw data, and summary files.


All outputs are then structured by scenario (e.g., `baseline`), with an additiona `img/` folder for plots and figures. 
The `img` only generates 

The structure is as follows:
```
output/
│
├── img/                             # All generated plots and figures
│   ├── baseline/                    # Baseline scenario plots
│   │   ├── dispatch/               # Hourly dispatch plots
│   │   ├── energy/                 # Annual energy mix plots
│   │   ├── capacity/               # New installed capacity timelines
│   │   └── map/                    # Geographic visualizations (if any)
│   └── scenarios_comparison/       # Plots comparing multiple scenarios that have been launched simultaneously
│
├── baseline/                        # GAMS raw outputs and logs
│   ├── main.lst                    # GAMS listing file (detailed solver log)
│
├── epmresults.gdx                   # Main results in GDX format (GAMS)
├── input_gdx/                       # Pre-processed GDX input files
├── output_csv/                      # Model outputs in CSV format (used for Tableau connectivity)
│
├── cplex.opt                        # Solver options used for this run
├── summary.csv                     # Summary of key results (default)
├── summary_detailed.csv            # Extended version of the summary
├── simulations_scenarios.csv       # List of scenarios and status
```

---

### Notes

- This structure is automatically created for each run.
- If you run **multiple scenarios**, each scenario will generate its own `baseline/`-like folder or be included in summary outputs depending on the configuration.