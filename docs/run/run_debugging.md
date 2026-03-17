# Debugging

This page covers what to do when EPM doesn't run, doesn't converge, or produces unrealistic results.

---

## First steps

Start by identifying the symptom:

| Symptom | Where to look |
|---|---|
| Python error before GAMS starts | Terminal output, check input files and config.csv |
| GAMS compilation error | `.lst` file, first `ERROR` line |
| Model infeasible | `PA.gdx` for binding constraints, `.lst` for infeasibility message |
| Model runs but results look wrong | `PA.gdx` variable levels, check input data units |
| Run times out or crashes | `.log` file for memory/time, reduce problem size or use remote server |

Enable verbose logging to get more detail:

```sh
python epm.py --folder_input your_data --config your_data/config.csv --debug
```

Or add `--trace` to log detailed input reading:

```sh
python epm.py --folder_input your_data --config your_data/config.csv --trace
```

---

## Common errors

**Infeasible model**
Usually caused by contradictory constraints — for example a renewable target that cannot be met given available capacity, or a demand that exceeds all possible supply. Check:

- Demand vs. installed + buildable capacity
- Emission caps relative to the fuel mix
- Reserve requirements vs. dispatchable capacity

**Unrealistic results (e.g. enormous unserved energy)**
Usually a data issue. Check:

- Unit consistency (MW vs. GW, MWh vs. GWh)
- Missing or zero capacity values
- Costs set to zero unintentionally (allows infinite dispatch)

**Python crashes before GAMS**
Check `config.csv` — a missing file reference or wrong path is the most common cause. Run with `--trace` to see exactly which input file fails to load.

**GAMS compilation error**
Open the `.lst` file and search for `****` — GAMS marks all errors with four asterisks. The line number points directly to the problem.

---

## GAMS output files

### `PA.gdx` — post-solve values

Generated after a solve. Open in GAMS Studio or with `gdxdump`. Each variable shows:

| Field | Meaning |
|---|---|
| **Level** | Actual solution value |
| **Marginal** | Dual value (shadow price or reduced cost) |
| **Lower / Upper** | Variable bounds |

What to look for:

- **All levels = 0**: the variable may not be connected to any constraint, or has a cost preventing selection
- **Marginal = EPS**: the bound is not active — usually normal
- **Very large levels**: check unit conversions and parameter bounds

### `.log` — execution log

Contains timestamps, file inclusions, model generation time, and solver startup. Useful for diagnosing crashes and timeouts.

### `.lst` — execution listing

Automatically generated. Contains compilation output, equation listings, and solver messages. Search for `****` to find errors quickly.

Tips:

- Set `option limcol = 0; option limrow = 0;` to reduce listing size
- Use the "Variable Listing" and "Equation Listing" sections to inspect values in context

### `.ref` — symbol reference

Shows where every set, parameter, variable, and equation is declared, defined, and referenced. Useful for finding variables that are declared but never used in a constraint.

To generate it in GAMS Studio: add `rf=filename.ref` to the command-line arguments in the Task Bar.

---

## Still stuck?

- Check the [Input Setup](../input/input_setup.md) and [Input Catalog](../input/input_detailed.md) to verify your data format
- Use AI tools or Google with the exact error message from the `.lst` file
- Contact the EPM team
