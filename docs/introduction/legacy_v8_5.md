# Legacy version — EPM v8.5 (Excel)

Before version 9, EPM ran from an **Excel workbook** driving a set of GAMS files. That
generation is archived here for teams that still maintain a v8.5 model.

!!! danger "Archived — not maintained"
    v8.5 receives no fixes, no support and no new features. Use it only to read or re-run an
    existing study. For anything new, use the current version — see
    [Installation](../run/run_installation.md).

---

## Which version should I use?

<div class="grid cards" markdown>

-   **EPM 9 — CSV + Python** *(current)*

    ---

    Inputs as CSV files, orchestrated by Python, solved in GAMS. **Full feature set**:
    scenarios, parallel runs, sensitivity and Monte-Carlo analysis, H2 module, automated
    postprocessing, version control.

    Some familiarity with **Python and GAMS helps** and unlocks customization, but is not
    required for a standard run.

    [→ Get started](../run/run_installation.md)

-   **EPM v8.5 — Excel** *(legacy)*

    ---

    All inputs in one `.xlsb` workbook, launched from GAMS Studio, results written back into
    Excel. **No Python involved** — the whole model is driven from the spreadsheet.

    Frozen feature set, and no way to track changes in git.

    [→ Contents of the pack](#what-is-in-the-pack)

</div>

---

## How v8.5 works

One Excel workbook holds every input. Three GAMS files do the work:

| File | Role |
|---|---|
| `WB_EPM_v8_5_main.gms` | Entry point — reads the workbook, drives the solve |
| `WB_EPM_v8_5_base.gms` | Core equations (equivalent of today's `base.gms`) |
| `WB_EPM_v8_5_Report.gms` | Result extraction back to Excel |

Runs are launched from **GAMS Studio**, not from a command line. A compiled `base.g00` file
speeds up repeated solves — the pack documents how to build it.

---

## What is in the pack

| Folder | Contents |
|---|---|
| `Generic model/` | `WB_EPM_8_5.xlsb` — the blank template — plus the three `.gms` files |
| `Ghana example/` | `WB_EPM_8_5_Ghana_Example.xlsb`, the `.gms` files, `cplex.opt`, `README.txt`, the Ghana decarbonization deck (2022), and `Initial_model/` with a `Ghana_2050_hydro_BaU` variant |
| `Documentation/` | `WB_Electricity_Planning_Model_Documentation_2023.pdf`, `Running EPM/` (how to run EPM, how to build `base.g00`, the Engine looping tool manual) and two training decks |

The v8.5 documentation is **inside the pack** — the 2023 PDF is the reference manual for that
version. Nothing on this site describes v8.5 beyond this page.

---

## Requirements

- **Windows** — the workbook relies on Excel macros
- **Excel** able to open `.xlsb` files, with macros enabled
- **GAMS** with a valid **CPLEX** license, and GAMS Studio to launch runs

!!! warning "Compatibility not guaranteed"
    v8.5 was last touched in 2024 and has **not been tested against recent GAMS releases**.
    It may need adjustment on a current GAMS installation. No support is provided if it
    does not run.

---

## Archived documentation

Two distinct things carry the "v8.5" name — don't confuse them:

| | What it is | Where |
|---|---|---|
| **v8.5 Excel** | The workbook-driven model described on this page. Never on GitHub. | Starter pack (see below) |
| **`v8.5` git tag** | The *transition* era: GAMS files still named `WB_EPM_v8_5_*.gms`, but already wrapped in Python with CSV inputs. Not an ancestor of `main`. | [tree/v8.5/docs](https://github.com/ESMAP-World-Bank-Group/EPM/tree/v8.5/docs) |

The tag preserves documentation pages that were dropped from this site, notably
`old/step_by_step.md` (adapting the `WB_EPM_v8_5_*.gms` files by hand) and
`engine_looping_tool.md` (the Engine looping tool, now superseded by
[Advanced Python Options](../run/run_python_advanced.md)).

---

## Get the pack

<!-- TODO: replace with the public archive URL (Zenodo DOI or GitHub release v8.5-legacy).
     Before publishing: remove `gamslice.txt` from the archive — it is a GAMS license file —
     and confirm the 2023 PDF and the training decks are cleared for external release. -->

The pack is not yet published to a public archive. Contact the EPM team to obtain it.

Earlier releases are archived on [Zenodo](https://zenodo.org/communities/esmap-epm).

---

## Moving to version 9

The current version expects CSV inputs. See
[From Excel to CSV](../input/input_from_excel_to_csv.md) for how to port an existing
workbook.
