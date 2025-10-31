# modeltype Options for EPM

EPM solves large-scale mixed-integer programs (MIPs). Properly configuring modeltype options can significantly reduce runtime and improve numerical stability.

modeltype settings are defined in a `cplex.opt` file.  
→ See [example `cplex.opt`](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/cplex.opt) for reference.

---

## Recommended modeltype Parameters

| Parameter        | Recommended Value | Description & Rationale                                                                                                                                                                                                                                                                                 |
| ---------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `threads`        | `8–12`            | Number of modeltype threads. Using more threads than CPU cores (oversubscription) can improve performance through load balancing and OS scheduling. Too many threads (e.g., >16) may increase overhead and slow the model. Test different values.                                                       |
| `lpmethod`       | `4`               | Controls which algorithm is used to solve LPs: <br>`0`: Primal simplex<br>`1`: Dual simplex<br>`2`: Network<br>`3`: Barrier<br>`4`: Automatic (recommended) <br> Let CPLEX choose the best method per instance.                                                                                         |
| `startalg`       | `4`               | Starting algorithm for root LPs (when solving MIPs): same values as `lpmethod`. Default `4` allows CPLEX to choose the best approach dynamically.                                                                                                                                                       |
| `predual`        | `-1`              | Tells CPLEX to solve the primal (`0`), dual (`1`), or to choose automatically (`-1`). Impact is model-specific; automatic is generally safe.                                                                                                                                                            |
| `baralg`         | `1`               | Barrier algorithm type: <br>`0`: Standard barrier <br>`1`: Barrier with crossover (recommended) <br> Produces a basic solution usable by downstream steps.                                                                                                                                              |
| `barcrossalg`    | `1`               | Algorithm used for crossover after barrier: <br>`0`: Primal simplex <br>`1`: Dual simplex (recommended)                                                                                                                                                                                                 |
| `barcolnz`       | `500`             | Sets threshold for treating columns as dense in the barrier method. This can greatly impact performance. Try tuning in steps of ±100.                                                                                                                                                                   |
| `barepcomp`      | `1e-5`            | Complementarity tolerance for barrier method. Lower values improve precision but may slow convergence. `1e-5` offers a good trade-off.                                                                                                                                                                  |
| `solutiontype`   | `2`               | Type of solution to return: <br>`1`: Basic solution <br>`2`: Interior point solution. <br> Use with caution on MIPs—primarily useful for LPs. (`solutiontype = 1` may be slower, but also more useful in post-analysis like Monte Carlo where specific properties for the solution are a nice-to-have). |
| `memoryemphasis` | `1`               | Emphasizes memory-saving strategies, helpful for large models prone to memory issues.                                                                                                                                                                                                                   |
| `bardisplay`     | `2`               | Controls verbosity during barrier solve. Higher values produce more output (good for diagnostics).                                                                                                                                                                                                      |
| `scaind`         | `1`               | Controls constraint matrix scaling: <br>`-1`, `0`: No scaling <br>`1`: Automatic scaling (recommended)                                                                                                                                                                                                  |
| `auxrootthreads` | `0`               | Disables auxiliary tasks in the root node. Reduces time spent in presolve when root node is slow to launch.                                                                                                                                                                                             |

---

## Notes on Performance Tuning

- **Start with default recommendations** and test on a small-to-medium scenario.
- **Barrier method parameters (`baralg`, `barcolnz`, `barepcomp`)** can greatly impact performance, especially for LP relaxations.
- **Interior point solution (`solutiontype = 2`)** is not always suitable for MIPs. Use only when appropriate.
- If presolve seems to hang, try setting:
  ```text
  auxrootthreads = 0
  ```

## modeltype Types for EPM

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

## Resources

- [CPLEX User Guide](https://www.ibm.com/docs/en/icos/20.1.0?topic=reference-cplex-parameters)
- [Internal modeltype Tuning Guide (WB)](https://worldbankgroup.sharepoint.com/:b:/t/PowerSystemPlanning-WBGroup/EU2NwUyeOo9CljzcBCJThbsBac_sVZWv7GWmuUWf0XDIyw?e=wLkYhH)
