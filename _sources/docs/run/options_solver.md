# Solver Options for EPM

EPM solves large-scale mixed-integer programs (MIPs). Properly configuring modeltype options can significantly reduce runtime and improve numerical stability.

modeltype settings are defined in a `cplex.opt` file.  
→ See [example `cplex.opt`](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/cplex.opt) for reference.

---

## How the Solver Works (new users)

1. **Build the equations.** EPM translates your inputs into a large linear model. You choose whether it includes integer decisions via the `modeltype` flag (for example `MIP` or `RMIP`).
2. **Presolve trims the problem.** CPLEX removes redundant rows, tightens bounds, and prepares the first LP. If it looks stuck here, the log will linger on “Presolve”; consider adjusting `auxrootthreads` or the data itself.
3. **Root LP is solved.** CPLEX decides automatically between dual-simplex and barrier (interior-point) depending on difficulty. This is where settings like `lpmethod`, `baralg`, and `barcolnz` matter most.
4. **Branch-and-bound (MIP only).** If you run a MIP, the solver explores branches to enforce integrality, guided by `optcr`, `mipemphasis`, `heuristics`, and `cuts`. In RMIP mode there is no branching, so the run finishes here.
5. **Results are reported.** The solver writes solution files, IIS diagnostics (if requested), and logs.

**Decisions you need to make:** (a) pick `modeltype` (MIP for discrete builds, RMIP for relaxed analysis), (b) point `cplexfile` to the preset closest to your needs (speed vs precision), and (c) adjust the headline parameters (`optcr`, `mipemphasis`, `lpmethod`, `threads`) only when you have evidence they are limiting performance.

> **Quick vocabulary**
> - **Barrier / interior-point**: LP algorithm that keeps variables strictly within bounds while driving the objective to optimality; chosen automatically on hard relaxations.
> - **Crossover**: Post-barrier step that converts the interior solution into a simplex basis so branch-and-bound can reuse it efficiently.
> - **Presolve / root node**: Early simplification and the first LP solve before branching. Performance issues here slow down the entire run.
> - **Helper threads**: Auxiliary workers CPLEX uses at the root; disabling them (`auxrootthreads = 0`) can unblock stagnating presolve phases.

## Recommended Solver Parameters

When tuning, focus first on **`optcr`**, **`mipemphasis`**, **`lpmethod`**, and **`threads`**—they have the biggest impact on both runtime and result quality. The remaining parameters help you fine-tune stability and diagnostics once the core settings are in a good place.

- **Begin with the defaults on a small test case.** Run a short scenario using the recommended table first; that will confirm the model is stable before you attempt aggressive tuning on full-scale studies.
- **Barrier controls (`baralg`, `barcolnz`, `barepcomp`) shape LP speed.** Whenever CPLEX switches to the barrier (interior-point) algorithm—whether you are running a relaxed RMIP or a full MIP—these settings determine how quickly those LPs converge. Tweak them only if you see sluggish roots or frequent barrier calls; otherwise the recommended values are a solid default.
- **`solutiontype = 2` (interior solution) is ideal for RMIP/LP analysis but not mandatory for MIPs.** If you need a basic solution for warm starts or detailed tableau-based diagnostics, switch to `solutiontype = 1`; otherwise the interior solution keeps the root solve smooth.
- **Stuck presolve or a hanging root node?** Set `auxrootthreads = 0` to disable helper threads. You will notice this situation in the solver log if the output pauses for a long time right after messages like “CPLEX Presolve” or “Root relaxation ...” without reporting iteration counts; disabling the auxiliary threads usually lets the log resume with normal progress.


| Parameter        | Default (CPLEX) | Recommended Value | Quick Notes                                                                                                                |
| ---------------- | --------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `optcr`          | `0.0001`        | `0.01`            | Relative optimality gap; tighten for production-quality results.                                                           |
| `mipemphasis`    | `0`             | `0 (balanced)`    | Guides MIP search toward speed (`1`) or proof of optimality (`2`); start from balanced.                                    |
| `lpmethod`       | `0`             | `4 (auto)`        | Lets CPLEX pick between dual simplex and barrier; best default for energy planning models.                                 |
| `threads`        | `0 (auto)`      | `8–12`            | Moderate parallelism keeps runtimes down without starving the machine.                                                     |
| `baralg`         | `0`             | `1`               | Uses barrier with crossover to produce a strong basis for MIP nodes.                                                       |
| `barcrossalg`    | `1`             | `1`               | Dual simplex crossover after barrier; typically faster and more robust.                                                    |
| `barcolnz`       | `0 (auto)`      | `500`             | Increase toward 500 to manage dense columns when barrier slows down.                                                       |
| `barepcomp`      | `1e-6`          | `1e-5`            | Complementarity tolerance; lower values boost accuracy, higher values finish faster.                                       |
| `parallelmode`   | `0`             | `1`               | Deterministic parallelism for reproducible results; flip to `0` for exploratory runs.                                      |
| `startalg`       | `0 (auto)`      | `4 (auto)`        | Automatic choice of root-start algorithm; usually dual simplex for energy planning data.                                   |
| `memoryemphasis` | `0`             | `1`               | Activates memory-saving strategies to avoid swapping.                                                                       |
| `predual`        | `-1`            | `-1`              | Lets CPLEX decide between primal/dual formulations; rarely worth forcing manually.                                         |
| `scaind`         | `1`             | `1`               | Automatic scaling to stabilize numerics.                                                                                   |
| `auxrootthreads` | `0 (auto)`      | `0`               | Disables helper threads at the root to reduce contention on large MIPs.                                                    |
| `heuristics`     | `0.05`          | `0.05`            | Light heuristic effort to get good incumbents early without heavy overhead.                                                |
| `cuts`           | `-1 (auto)`     | `-1 (auto)`       | Leave cutting planes on automatic unless diagnosing specific issues.                                                       |
| `mipdisplay`     | `2`             | `4–5`             | Verbosity level for branch-and-bound progress messages.                                                                    |
| `solutiontype`   | `1`             | `2` (RMIP/LP)     | Interior solution for relaxed runs; use `1` on MIPs when you need a basic (tableau) solution.                              |
| `names`          | `auto`          | `yes`             | Keep symbol names for easier debugging and IIS reporting.                                                                  |
| `iis`            | `0`             | `1`               | Enable IIS extraction when infeasibilities occur.                                                                          |
| `bardisplay`     | `0`             | `2`               | Medium verbosity for barrier progress diagnostics.                                                                         |

Top-of-table entries are the most impactful; rows toward the bottom mainly enhance monitoring or debugging.


### `optcr` (recommended 0.01)
- **Default:** `0.0001`
- **What it controls:** Relative optimality gap used to stop branch-and-bound (`(bestBound - incumbent)/|incumbent|`).
- **Precision impact:** Very high — smaller gaps deliver solutions closer to the true optimum.
- **Runtime impact:** High — tightening from 0.05 to 0.01 can multiply solve time; relaxing does the opposite.
- **Applies to:** MIP only; RMIP ignores the gap because it solves a single LP.

### `mipemphasis` (recommended 0 for balanced runs)
- **Default:** `0`
- **What it controls:** Strategic focus of the MIP search (`0` balanced, `1` feasibility, `2` optimality, `3` best bound, `4` hidden feasible).
- **Precision impact:** Medium — emphasis `1` can return decent solutions sooner but may stop with larger gaps.
- **Runtime impact:** Medium — changing emphasis reorders node exploration and can save or cost significant time.
- **Applies to:** MIP only.

### `lpmethod` (recommended 4 for energy planning)
- **Default:** `0` (primal simplex / automatic)
- **What it controls:** Algorithm for LP relaxations (0=Primal simplex, 1=Dual simplex, 2=Network, 3=Barrier, 4=Automatic).
- **Energy-planning tip:** Dual simplex (`1`) excels on sparse models with warm starts, while barrier (`3`) can be faster on tough relaxations; `4` lets CPLEX switch between them automatically and is usually the best choice for power-system planning models.
- **Simple comparison:** Dual simplex walks along the boundary of feasible solutions with quick pivots, whereas barrier glides through the interior using smooth steps; both reach the same optimum but shine on different problem shapes.
- **Precision impact:** Low — mainly affects numerical path, not the optimal solution itself.
- **Runtime impact:** High — the LP engine is called at every node, so a good choice saves minutes or hours.
- **Applies to:** MIP & RMIP (all solves rely on the LP engine).

### `threads` (recommended 8–12)
- **Default:** `0` (automatic thread selection)
- **What it controls:** Maximum number of solver threads.
- **Precision impact:** None — only influences how quickly the solver reaches the same answer.
- **Runtime impact:** High — more threads reduce wall-clock time until oversubscription (>16) adds overhead.
- **Applies to:** MIP & RMIP.

### `baralg` (recommended 1)
- **Default:** `0` (standard barrier)
- **What it controls:** Barrier algorithm variant (0=standard interior-point, 1=barrier followed by crossover).
- **Barrier objective explained:** The barrier method minimizes the LP objective while adding logarithmic penalties that keep solutions strictly inside the feasible region; crossover (option `1`) then converts the interior solution into a basic solution suitable for branch-and-bound.
- **Precision impact:** Medium — crossover produces a usable starting basis for integer phases.
- **Runtime impact:** Medium — the extra crossover work often pays off by reducing effort in later nodes.
- **Applies to:** Primarily MIP (benefits root basis), but also useful for RMIP.

### `barcrossalg` (recommended 1)
- **Default:** `1`
- **What it controls:** Method used during crossover after the barrier step (0=Primal simplex, 1=Dual simplex).
- **Precision impact:** Low — both yield feasible bases, but the dual version maintains numerical stability on sparse models.
- **Runtime impact:** Medium — dual simplex usually delivers faster crossovers for power-system matrices.
- **Applies to:** MIP & RMIP when barrier is used.

### `barcolnz` (default 0, recommended 500)
- **Default:** `0` (automatic dense-column detection)
- **What it controls:** Threshold for treating columns as dense during barrier factorization.
- **Precision impact:** Low — changing the threshold influences numerical conditioning only indirectly.
- **Runtime impact:** Medium to high — leaving the default `0` lets CPLEX decide automatically; setting it near `500` makes dense handling more proactive (lowering it toward 300 treats more columns as dense, raising toward 700 keeps more sparse) and can unlock faster barrier factorizations.
- **Applies to:** MIP & RMIP when barrier runs.

### `barepcomp` (recommended 1e-5)
- **Default:** `1e-6`
- **What it controls:** Complementarity tolerance that stops the barrier iterations.
- **Precision impact:** High — lowering the tolerance (for example to 1e-6) forces the interior-point method closer to the exact optimum; raising it (for example to 1e-4 or 5e-4) accepts looser solutions.
- **Runtime impact:** Medium — tighter tolerances increase iteration counts; loosening them speeds exploratory runs but leaves larger residuals.
- **Applies to:** MIP & RMIP.

### `solutiontype` (recommended 2 for RMIP/LP, 1 for MIP warm starts)
- **Default:** `1`
- **What it controls:** Type of solution returned by the barrier (1=basic, 2=interior point).
- **Precision impact:** Medium — interior solutions are smooth for RMIP/LP reporting; basic solutions expose the tableau structure required by some MIP workflows.
- **Runtime impact:** Medium — keeping `solutiontype = 2` makes relaxed runs faster, while switching to `1` on MIPs can strengthen branch-and-bound warm starts even if the barrier step itself takes a little longer.
- **Applies to:** Use `2` when solving RMIP or LP-only cases; prefer `1` when running MIPs that need basic solutions or detailed tableau diagnostics.

### `parallelmode` (recommended 1 for reproducibility)
- **Default:** `0` (opportunistic)
- **What it controls:** Parallel search strategy (`0` opportunistic for speed, `1` deterministic for repeatability, `2` sequential).
- **Precision impact:** Low — results satisfy the same optimality conditions, but deterministic mode ensures repeatability.
- **Runtime impact:** Medium — opportunistic mode can be a few percent faster on multi-core machines.
- **Applies to:** MIP & RMIP.

### `startalg` (recommended 4)
- **Default:** `0` (automatic)
- **What it controls:** Starting algorithm used at the root (0=automatic primal simplex, 1=automatic dual simplex, 4=automatic).
- **Options explained:** Energy planning models often benefit from dual simplex starts (`1`) when warm-starting from prior runs; barrier starts (`3`) can help ill-conditioned problems; using `4` lets CPLEX choose dynamically based on presolve diagnostics.
- **Precision impact:** Low — influences the initial basis but not the final solution.
- **Runtime impact:** Medium — good root starts cut total solve time.
- **Applies to:** MIP & RMIP.

### `memoryemphasis` (recommended 1)
- **Default:** `0`
- **What it controls:** Memory-saving strategies that limit cache and RAM usage.
- **Precision impact:** None.
- **Runtime impact:** Low to medium — may slow solves slightly but avoids swapping or out-of-memory failures.
- **Applies to:** MIP & RMIP.

### `predual` (recommended -1)
- **Default:** `-1`
- **What it controls:** Whether CPLEX converts the model into its dual form before solving.
- **Precision impact:** Low — automatic choice keeps numerical robustness.
- **Runtime impact:** Medium — letting CPLEX pick typically yields the fastest presolve.
- **Applies to:** MIP & RMIP.

### `scaind` (recommended 1)
- **Default:** `1`
- **What it controls:** Scaling of the constraint matrix (-1 or 0 disables scaling, 1 enables automatic scaling).
- **Precision impact:** Medium — proper scaling stabilizes numerics and reduces rounding issues.
- **Runtime impact:** Medium — better scaling shortens solves and avoids degeneracy problems.
- **Applies to:** MIP & RMIP.

### `auxrootthreads` (recommended 0)
- **Default:** `0` (automatic helper threads)
- **What it controls:** Number of auxiliary threads used during root node processing.
- **Precision impact:** None.
- **Runtime impact:** Medium — turning them off avoids contention on large models and often shortens root solve time.
- **Applies to:** MIP (ignored by RMIP).

### `heuristics` (recommended 0.05)
- **Default:** `0.05`
- **What it controls:** Fraction of effort spent on primal heuristics to find incumbents.
- **Precision impact:** Low — heuristics influence how soon good solutions appear, not their final quality.
- **Runtime impact:** Medium — more heuristics can reduce total runtime by providing better incumbents early.
- **Applies to:** MIP only.

### `cuts` (recommended -1 for automatic)
- **Default:** `-1` (automatic)
- **What it controls:** Strength of generic cutting planes (-1 auto, 0 none, >0 more aggressive).
- **Precision impact:** Low — cuts tighten bounds but do not change feasible points.
- **Runtime impact:** Medium — more cuts add overhead yet shrink the search tree.
- **Applies to:** MIP only.

### `mipdisplay` (recommended 4–5)
- **Default:** `2`
- **What it controls:** Verbosity of branch-and-bound progress reporting.
- **Precision impact:** None.
- **Runtime impact:** Negligible — useful for monitoring without slowing the solver.
- **Applies to:** MIP only.

### `names` (recommended yes)
- **Default:** `auto`
- **What it controls:** Whether CPLEX keeps symbol names in the model it solves.
- **Precision impact:** None.
- **Runtime impact:** Low — storing names costs a little memory but greatly simplifies debugging and IIS analysis.
- **Applies to:** MIP & RMIP.

### `iis` (recommended 1)
- **Default:** `0`
- **What it controls:** Requests an irreducible inconsistent subsystem (IIS) if the model is infeasible.
- **Precision impact:** None.
- **Runtime impact:** Low — only invoked on infeasible models, but invaluable for tracing data issues.
- **Applies to:** MIP & RMIP.

### `bardisplay` (recommended 2)
- **Default:** `0`
- **What it controls:** Verbosity of barrier progress output.
- **Precision impact:** None.
- **Runtime impact:** Negligible — choose higher levels only when diagnosing convergence issues.
- **Applies to:** MIP & RMIP.

---


## Solver Types for EPM

In EPM, the `MODELTYPE` determines the class of optimization problem used during the `Solve` statement. You can set it via a macro like:

```gams
Solve PA using %MODELTYPE% minimizing vNPVcost;
```

Using the Python API, `epm.py`, it's the parameter `--modeltype` that can be used to specify the modeltype (`MIP`or `RMIP`))

| MODELTYPE | Description               | Use Case / Notes                                                                                                                      |
| --------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `LP`      | Linear Programming        | All variables and constraints are linear. Use for fastest solve when there's no need for integer or binary decisions.                 |
| `MIP`     | Mixed Integer Programming | Linear model with some variables constrained to be integer or binary. Use when decisions involve on/off or countable options.         |
| `RMIP`    | Relaxed MIP               | Like MIP, but all integer/binary constraints are relaxed to continuous. Useful for debugging or getting bounds before full MIP solve. |

> - `RMIP` **keeps the full structure of a MIP model** (including integer vars), but relaxes integrality — making it solvable as a continuous problem.
> - `LP` is **purely linear**, and expects all variables to be continuous.
> - Use `RMIP` when you want to check bounds or feasibility of a MIP model without solving the full integer problem.

## Predefined CPLEX Option Files

EPM ships several ready-made option files in `epm/input/data_test/cplex`. Each file aligns with the solver scenarios listed in `scenarios_solver.csv`.

| File | Focus | Notes |
| ---- | ----- | ----- |
| `cplex_baseline.opt` | Balanced defaults | General-purpose settings used by baseline runs. |
| `cplex_rmip_fast.opt` | Fast relaxed solves | Opportunistic parallelism, lighter tolerances for exploratory RMIP runs. |
| `cplex_rmip_precision.opt` | Accurate relaxed solves | Deterministic parallelism, tighter tolerances for reproducible RMIP diagnostics. |
| `cplex_mip_fast.opt` | Fast integer solves | Looser optimality gap (`optcr=0.05`), heuristic-heavy for quick feasible solutions. |
| `cplex_mip_precision.opt` | High-precision integer solves | Tighter gap (`optcr=0.01`), deterministic search, IIS reporting enabled. |
| `cplex_test.opt` | Debug configuration | Verbose output with stable barrier and scaling settings. |

To switch the solver configuration:
- Edit the `cplexfile` row in your `config.csv` so it points to the desired option file (for example `cplex/cplex_mip_precision.opt`).
- Or, for quick experiments, load `scenarios_solver.csv` and pick the column that references the file you want.

Example command (run from the `epm` root) that executes the comparison test:

```bash
python epm.py \
  --folder_input epm/input/data_test \
  --config config.csv \
  --scenarios scenarios_solver.csv
```

## Resources

- [CPLEX User Guide](https://www.ibm.com/docs/en/icos/20.1.0?topic=reference-cplex-parameters)
- [Internal modeltype Tuning Guide (WB)](https://worldbankgroup.sharepoint.com/:b:/t/PowerSystemPlanning-WBGroup/EU2NwUyeOo9CljzcBCJThbsBac_sVZWv7GWmuUWf0XDIyw?e=wLkYhH)
