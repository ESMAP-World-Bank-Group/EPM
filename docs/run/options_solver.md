# Solver Options

EPM solves large-scale mixed-integer programs using **CPLEX** via GAMS. This page covers how to choose a solver type and tune the key parameters.

---

## Solver types

Set via `--modeltype` on the CLI, or the `MODELTYPE` macro in GAMS:

| Type | Description | When to use |
|---|---|---|
| `MIP` | Mixed-integer programming — discrete investment decisions | Default for capacity expansion |
| `RMIP` | Relaxed MIP — integrality constraints dropped | Faster; good for bounds, debugging, or exploratory runs |
| `LP` | Linear programming — no integer variables | Only if the model has no binary/integer decisions |

---

## CPLEX option files

EPM ships ready-made option files in `epm/input/data_test/cplex/`. Point to one via the `cplexfile` row in `config.csv`:

| File | Focus |
|---|---|
| `cplex_baseline.opt` | Balanced defaults — general-purpose |
| `cplex_rmip_fast.opt` | Fast relaxed solves (exploratory RMIP) |
| `cplex_rmip_precision.opt` | Accurate relaxed solves (reproducible RMIP) |
| `cplex_mip_fast.opt` | Fast integer solves (`optcr=0.05`, heuristic-heavy) |
| `cplex_mip_precision.opt` | High-precision integer solves (`optcr=0.01`, deterministic) |
| `cplex_test.opt` | Debug — verbose output, stable barrier settings |

To run a solver comparison across scenarios:

```sh
python epm.py --folder_input data_test --config config.csv --scenarios scenarios_solver.csv
```

---

## Key parameters

Focus on these four first — they have the biggest impact:

| Parameter | Recommended | Notes |
|---|---|---|
| `optcr` | `0.01` | Relative optimality gap. Tighten for production runs, loosen for speed. MIP only. |
| `threads` | `8–12` | More threads = faster, up to a point. Beyond ~16 threads, overhead increases. |
| `lpmethod` | `4` (auto) | Let CPLEX choose between simplex and barrier. Best default for power-system models. |
| `mipemphasis` | `0` (balanced) | `1` = find feasible solutions faster; `2` = prove optimality. Start balanced. |

??? note "All parameters"

    | Parameter | Recommended | Notes |
    |---|---|---|
    | `optcr` | `0.01` | MIP optimality gap |
    | `mipemphasis` | `0` | MIP search strategy |
    | `lpmethod` | `4` | LP algorithm |
    | `threads` | `8–12` | Parallelism |
    | `baralg` | `1` | Barrier + crossover |
    | `barcrossalg` | `1` | Dual simplex crossover |
    | `barcolnz` | `500` | Dense-column threshold for barrier |
    | `barepcomp` | `1e-5` | Barrier complementarity tolerance |
    | `parallelmode` | `1` | Deterministic (reproducible) |
    | `startalg` | `4` | Root-start algorithm (auto) |
    | `memoryemphasis` | `1` | Memory-saving mode |
    | `predual` | `-1` | Primal/dual formulation (auto) |
    | `scaind` | `1` | Automatic matrix scaling |
    | `auxrootthreads` | `0` | Disable helper threads at root (avoids contention on large MIPs) |
    | `heuristics` | `0.05` | Incumbent heuristics effort (MIP only) |
    | `cuts` | `-1` | Cutting planes (auto) |
    | `mipdisplay` | `4–5` | Branch-and-bound verbosity |
    | `solutiontype` | `2` (RMIP), `1` (MIP) | Interior vs basic solution |
    | `names` | `yes` | Keep symbol names for debugging |
    | `iis` | `1` | Extract infeasibility report if model is infeasible |
    | `bardisplay` | `2` | Barrier verbosity |

---

## Stuck or slow?

- **Presolve hangs** (log pauses after "CPLEX Presolve"): set `auxrootthreads = 0`
- **Infeasible model**: enable `iis = 1` to get a diagnostic report — see [Debugging](run_debugging.md)
- **Out of memory**: enable `memoryemphasis = 1`; reduce problem size or move to a remote server

## Resources

- [CPLEX Parameter Reference](https://www.ibm.com/docs/en/icos/20.1.0?topic=reference-cplex-parameters)
- [Internal Tuning Guide (WB)](https://worldbankgroup.sharepoint.com/:b:/t/PowerSystemPlanning-WBGroup/EU2NwUyeOo9CljzcBCJThbsBac_sVZWv7GWmuUWf0XDIyw?e=wLkYhH)
