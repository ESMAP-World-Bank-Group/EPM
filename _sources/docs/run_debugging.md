# Debugging EPM Models in GAMS

If your run **didn’t converge**, or worse, **converged with unrealistic results**, this guide provides tools and methods to investigate what went wrong.

---

## 1. Understanding the `PA.gdx` Output

The `PA.gdx` file (often auto-generated) contains **post-solve values** for variables and equations. It's critical for diagnosing model behavior.

Each variable includes:

| Field      | Meaning                                                                 |
|------------|-------------------------------------------------------------------------|
| **Level**  | The actual solution value (e.g. power output, demand served, etc.)      |
| **Marginal** | Dual value — often shadow price or reduced cost                      |
| **Lower**  | Lower bound on the variable                                             |
| **Upper**  | Upper bound on the variable                                             |
| **Scale**  | Scaling factor applied (typically 1 unless specified otherwise)         |

### What to Look for

- **Zero levels**: If all levels are zero, check if:
  - The variable was properly connected in constraints.
  - It had a feasible range (`Lower`, `Upper`) that allowed movement.
  - There's a cost or penalty preventing it from being selected.

- **Marginals = EPS**: GAMS uses `EPS` (epsilon) when dual values are **effectively zero**, i.e. the bound is **not active**. This is normal unless you expected that variable to be tightly constrained.

- **Very high or low levels**: Could indicate:
  - Missing unit conversions (e.g. kW vs MW)
  - Mis-specified bounds or parameters
  - A cost that unintentionally encourages unrealistic dispatch

---

## 2. Reference Outputs for Debugging

GAMS provides reference files to trace model structure, usage, and execution flow.

---

### `.log` File — Execution Log

Contains runtime logs and time stamps:

- File inclusions
- Model generation times
- Solver startup and memory use
- Useful for checking if the solver started, crashed, or timed out

---

### `.ref` File — Model Symbol Reference

Enable it within GAMS Studio using command parameters.

This produces `<filename>.ref` which contains:

- All declared **symbols** (sets, parameters, variables, equations)
- Where each symbol was:
  - Declared
  - Defined
  - Assigned
  - Referenced
- Warnings for @**declared but unused** symbols

**Why it's useful**: Quickly shows whether a variable like `vPwrOut(g,f,q,d,t,y)` is declared but never referenced in a constraint — a common source of “zero everywhere” issues.

---

### `.lst` File — Execution Listing

Automatically generated. Includes:

- Compilation diagnostics
- Code listing
- Full equation listings (left-hand side, right-hand side, status)
- Solver messages and summaries

**Tips**:
- Set `option limcol = 0; option limrow = 0;` to limit variable/equation listings.
- Use the **"Variable Listing"** and **"Equation Listing"** sections to examine values in context.

---