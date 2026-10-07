# Dispatch Mode

By default, EPM uses representative days to reduce computation time. Dispatch mode (`fDispatchMode=1`) runs the full hourly chronology instead, enabling commitment-level constraints that are not captured in a representative-day run.

Use dispatch mode when you need to model:

- Startup costs and minimum up/down times
- Ramping constraints over the full time horizon
- Storage continuity across hours and days
- Unit commitment binaries (on/off status per generator per hour)

> For the full mathematical formulation of dispatch equations, see [Dispatch Mode Equations](../model/equations.md#dispatch-mode-equations).

---

## How to activate

Set the following flags in your `pSettings.csv`:

| Flag | Value | Description |
|---|---|---|
| `fDispatchMode` | `1` | **Required.** Activates dispatch mode and full hourly chronology |
| `fRampConstraints` | `1` | Enable ramp rate constraints |
| `fApplyRampConstraint` | `1` | Apply ramp constraints in equations |
| `fApplyMinGenerationConstraint` | `1` | Enforce minimum generation — required to activate unit commitment |
| `fApplyStartupCost` | `1` | Include startup costs in the objective function |
| `fApplyMUDT` | `1` | Enforce minimum up/down time windows |

Only `fDispatchMode` is strictly required. The other flags activate specific features within dispatch mode and can be combined as needed.

---

## Before running

- Make sure `input.gdx` contains the full chronology: sets `AT`, `pHours`, and `mapTS(q,d,t,AT)`
- If you changed any flag in `pSettings.csv`, re-run Python input treatment so the updated scalars are written to `input.gdx`
- Test with a short time horizon first — dispatch mode is significantly more compute-intensive than a representative-day run

---

## Run

Same command as a standard run — GAMS picks up `fDispatchMode` from `pSettings.csv` automatically:

```sh
python epm.py --folder_input your_data --config your_data/config.csv
```
