# What is EPM?

## Context

The Energy Sector Management Assistance Program (ESMAP) is the World Bank's global energy advisory program. It supports governments and utilities in over 80 countries on power sector planning. The **Electricity Planning Model (EPM)**, a capacity expansion model, is one of the main tools ESMAP uses for this work.

---

## What EPM does

EPM **minimizes the total cost of expanding and operating a power system** over a multi-year horizon, while meeting all technical, economic, and environmental requirements.

It answers questions like:

- How much new capacity is needed, and which technologies?
- What generation mix minimizes costs while meeting renewable targets?
- What is the economic value of regional interconnection?
- How much storage and flexibility does a high-renewables system need?

Concretely, EPM handles:

| Capability | Description |
|---|---|
| **Capacity expansion** | Least-cost generation investment plans across years and zones |
| **Dispatch optimization** | Co-optimizes generation, storage, reserves and cross-border trade, on representative days or a full hourly year |
| **Custom zone modeling** | User-defined zones, from sub-national regions to multi-country representations |
| **Policy testing** | Renewable targets, fuel and import limits, emissions caps and carbon prices |
| **Scenario analysis** | Runs batches of scenarios and sensitivities on uncertain inputs |

---

## History & versions

EPM was first deployed in 2016 and has evolved through several versions. Previous versions are archived on [Zenodo](https://zenodo.org/communities/esmap-epm).

### Which version are you on?

| | **EPM 9** — CSV + Python *(current)* | **EPM v8.5** — Excel *(legacy)* |
|---|---|---|
| Inputs | CSV files, tracked in git | One `.xlsb` workbook |
| Run from | `python epm.py` | GAMS Studio |
| Features | Full set — scenarios, parallel runs, sensitivity & Monte-Carlo, H2, automated postprocessing | Frozen |
| Skills | Python and GAMS knowledge **helps** and unlocks customization, but is not required for a standard run | Excel only |
| Status | Beta — actively developed | Archived, unsupported |
| Where | This documentation | [Legacy version](../run/run_legacy_v8_5.md) |

Porting an existing workbook to the current version: [From Excel to CSV](../input/input_from_excel_to_csv.md).
