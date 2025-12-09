# Dispatch Mode Guide

## Purpose

The Dispatch and Unit Commitment flow runs the full hourly chronology, so it can enforce commitment-level constraints (startup costs, minimum generation shares, ramp limits, minimum up/down windows, storage continuity, etc.) that are otherwise invisible in a representative-day run. This document explains how to turn on the necessary switches, what the new blocks of equations do, and which features are still pending (CSP and hydrogen continuity remain to be added).

## Key features active when `fDispatchMode=1`

1. **Dispatch continuity and ramping**: the model keeps hourly generation (`vPwrOut`) and the slot-aggregated counterpart (`vPwrOutDispatch`) consistent through `eRampContinuity`. Slot-to-slot ramp changes are constrained via `eDispatchRampUpLimit`/`eDispatchRampDownLimit`, while the original hourly `eRampUpLimit`/`eRampDnLimit` stay in place to respect physical ramp rates. Storage SOC transitions (`eStorageHourTransition`, `eStorageDayWrap`, etc.) also operate over the full chronology, so `fDispatchMode` guarantees both generation and storage trajectories are traced continuously. CSP and hydrogen modules still lack the same full-hour continuity enforcement; those will need equivalent equations once their dispatch logic is expanded.

   ```gams
   eRampContinuity(g,q,d,t,AT,y)$((Ramprate(g) and fApplyRampConstraint and fDispatchMode) and mapTS(q,d,t,AT))..
       sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) =e= vPwrOutDispatch(g,q,d,t,AT,y);
   ```

2. **Unit commitment with minimum generation, startup cost, and min up/down**: generators with a `MinGenPoint` see their `isOn/isStartup/isShutdown` binaries linked to both the dispatch output and the objective through `eDispatchMinGenPoint`, `eDispatchMaxGenPoint`, and `eStartupCostConstraint`. The commitment balance equations (`eCommitmentConsistency`, `eCommitmentInitialization`, `eCommitmentSingleTransition`) enforce the transition logic, while the rolling `MinUp...`/`MinDown...` constraints ensure every startup/shutdown is followed by the required persistence window. All of these only run inside dispatch mode, so the flag bundle (`fApplyMinGenerationConstraint` and `fApplyMUDT`) must be set alongside `fDispatchMode`.

   ```gams
   eCommitmentConsistency(g,AT,y)$(ord(AT) > 1 and fDispatchMode)..
       sum((q,d,t)$mapTS(q,d,t,AT), isStartup(g,q,d,t,AT,y) - isShutdown(g,q,d,t,AT,y))
           =e= sum((q,d,t)$mapTS(q,d,t,AT), isOn(g,q,d,t,AT,y))
              - sum((q,d,t)$mapTS(q,d,t,AT-1), isOn(g,q,d,t,AT-1,y));
   ```

   These commitment binaries are tricky because nothing in dispatch mode alone forces `isOn` to follow actual generation unless one of the min-generation constraints is present. We therefore tie positive output to the commitment state via `eDispatchMinGenPoint`/`eDispatchMaxGenPoint`, and startup cost is collected through `eStartupCostConstraint` so flipping `isOn` carries a penalty. The combination of `MinGenPoint(g)`, `fApplyMinGenerationConstraint`, and `fApplyStartupCost` makes the binary meaningful, but it also means you cannot activate the startup-cost reporting or min up/down without enabling the min-generation switch (unless you refactor to add your own big-M linking equation).

   The dispatcher uses the following limits to bind `isOn` with dispatch output. The equality in `eDispatchMinGenPoint` forces any positive generation to raise the commitment flag, while `eDispatchMaxGenPoint` caps output when `isOn` is zero. `eStartupCostConstraint` then counts the observed startups per slot, which feeds `vYearlyStartupCost`.

   ```gams
   eDispatchMinGenPoint(g,q,d,t,y)$((fDispatchMode and fApplyMinGenerationConstraint and MinGenPoint(g) and FD(q,d,t)))..
       sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) =g= pGenData(g,"MinLimitShare")*pGenData(g,"Capacity")
           * sum(AT$mapTS(q,d,t,AT), isOn(g,q,d,t,AT,y));

   eDispatchMaxGenPoint(g,q,d,t,y)$((fDispatchMode and fApplyMinGenerationConstraint and MinGenPoint(g) and FD(q,d,t)))..
       sum(gfmap(g,f), vPwrOut(g,f,q,d,t,y)) =l= pGenData(g,"Capacity")*sum(AT$mapTS(q,d,t,AT), isOn(g,q,d,t,AT,y));
   ```

## Running dispatch mode

1. Set `fDispatchMode=1` and the ancillary switches (`fRampConstraints`, `fApplyRampConstraint`, `fApplyMinGenerationConstraint`, `fApplyStartupCost`, `fApplyMUDT`) inside your `pSettings.csv` or via command-line overrides. `epm/main.gms` pulls these scalars before loading `base.gms`, so every dispatch equation sees the intended flags.
2. Ensure the GDX input contains the hourly sets: `AT`, `pHours`, and the mapping `mapTS(q,d,t,AT)` created by the reader scripts. If you regenerate settings (e.g., toggling `fRampConstraints` or `InitialSOCforBattery`), rerun the Python input treatment so the new scalars persist in `input.gdx`.
3. Launch the usual solver command (`gams epm/main.gms lo=2` or your wrapper). Do not skip the `base.gms` inclusion; it defines the dispatch equations, storage continuity logic, and startup-cost reporting (`vYearlyStartupCost`). The `generate_report` step will now capture startup costs as part of the per-zone variable-cost breakdown.

## Checklist before dispatch runs

- Confirm `fDispatchMode=1` and any of the supporting toggles (`fRampConstraints`, `fApplyRampConstraint`, `fApplyMinGenerationConstraint`, `fApplyStartupCost`, `fApplyMUDT`) that you need are present in the chosen `pSettings` file.
- Validate that `input.gdx` exposes the full chronology (`AT`, `mapTS`, `pHours`) and any storage initialization scalars (`InitialSOCforBattery` / `pStorageInitShare`) the dispatch equations depend on.
- Run the same base script (`gams epm/main.gms` or the Python wrapper) so `base.gms` can generate the dispatch-level variables/histories your analysis relies on.
