# Data Sources — EPM — Black Sea 2026

*Generated 2026-05-30*

---

## Model overview

**Countries**: Turkiye, Armenia  
**Data horizon**: 2024–2053 · step: 1 year


| Category | Item | Parameter | Description | Turkiye | Armenia |
|---|---|---|---|---|---|
| Load | Annual demand forecast | `pDemandForecast` | Historical and projected electricity demand (GWh and MW peak) by year | — | CESI/EPSO (2022) |
| Load | Hourly demand profile | `pDemandProfile` | Typical hourly load curve (8760 h) for a representative year | — | proxy of Turkiye/EastAna |
| Supply | Generator database | `pGenDataInput` | Existing, committed, and candidate plants: name, technology, capacity (MW), COD, CAPEX, O&M, operating constraints | — | CESI/EPSO (2022) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) |
| Supply | Fuel prices | `pFuelPrice` | Gas, coal, diesel, HFO trajectory 2025–2050 ($/GJ) | — | TYNDP / IEA World Energy Outlo… (2022) |
| Supply | Plant availability | `pAvailabilityCustom` | Seasonal capacity factors for thermal, hydro, and other dispatchable units | — | World Nuclear Association (updated annually) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) |
| Supply | Storage assumptions | `pStorageDataInput` | For BESS and PSH: capacity, duration, efficiency, cost assumptions | — | — |
| Supply | VRE and hydro profiles | `pVREProfile` | Hourly capacity factor profiles for solar PV, wind, and run-of-river hydro (normalised 0–1) | — | — |
| Resources | Maximum installable capacity | `pMaxGenerationByFuel` | Maximum new capacity by technology (resource potential and spatial constraints) | — | — |
| Resources | VRE integration assumptions | `pSettings` | VRE curtailment, variability handling, and balancing cost assumptions | — | — |
| Trade | Cross-border transmission | `pTransferLimit` | Existing and planned cross-border interconnectors: capacity (MW), year, routing options | — | — |
| Trade | Transmission losses | `pLossFactorInternal` | Cross-border interconnector losses (% by corridor) | — | — |
| Trade | Trade prices | `pTradePrice` | Import/export prices with temporal variability ($/MWh) — external zones | — | — |
| Reserves | Reserve margin | `pPlanningReserveMargin` | Planning reserve margin (%) and operating reserve assumptions | — | — |
| Other | Carbon pricing | `pCarbonPrice` | Carbon price or emission constraint applied in planning (NDC, ETS membership) | — | — |
| Other | Fuel and import limits | `pMaxFuelLimit` | Caps or floors on fuel use or electricity imports (e.g. gas import quotas) | — | — |

---

<a id="toc"></a>

## Contents

- [Turkiye](#turkiye) — *not yet documented*
- [Armenia](#armenia) — [`pDemandForecast`](#armenia-pdemandforecast) · [`pDemandProfile`](#armenia-pdemandprofile) · [`pAvailabilityCustom`](#armenia-pavailabilitycustom) · [`pGenDataInput`](#armenia-pgendatainput) · [`pFuelPrice`](#armenia-pfuelprice)

---

<a id="turkiye"></a>

## Turkiye

[&#8593; Contents](#toc)

*No data documented yet for this country.*

---

<a id="armenia"></a>

## Armenia

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pDemandForecast`](#armenia-pdemandforecast) | CESI/EPSO (2022) | [MEDIUM] |
| [`pDemandProfile`](#armenia-pdemandprofile) | proxy of Turkiye/EastAna | [LOW] |
| [`pGenDataInput`](#armenia-pgendatainput) | CESI/EPSO (2022) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] |
| [`pFuelPrice`](#armenia-pfuelprice) | TYNDP / IEA World Energy Outlo… (2022) | [MEDIUM] |
| [`pAvailabilityCustom`](#armenia-pavailabilitycustom) | World Nuclear Association (updated annually) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [HIGH] |

<a id="armenia-pdemandforecast"></a>

### `pDemandForecast`

[&#8593; Armenia](#armenia)

**Source**: CESI/EPSO — Armenia Data and Assumptions PLEXOS STUDY (`epso_armenia_plexos_2022`)

**Method**: DIRECT (2030–2050 Base Case) + EXTRAP+INTERP (2024–2029) + EXTRAP (2051–2053)

| Period | Method | Notes |
|--------|--------|-------|
| 2030–2050 | `DIRECT` | CESI Base Case, slide 10 — 5-year milestones |
| 2024 | `EXTRAP` | Peak: 1300 MW (~2022, slide 8) × 1.035² = 1390 MW. Energy: gross demand 2020 ~6385 GWh × 1.0103⁴ = 6650 GWh |
| 2025–2029 | `INTERP` | Linear interpolation between 2024 baseline and 2030 CESI anchor |
| 2031–2049 | `INTERP` | Linear interpolation between successive 5-year CESI anchors |
| 2051–2053 | `EXTRAP` | Extrapolation at 2045–2050 annual rate (+25 MW/yr, +124 GWh/yr) |

> Peak growth rate implied (3.5%/yr) is much higher than energy growth (1.03%/yr), consistent with CESI hypothesis of significant electrification of heating and transport. Historical gross demand 2020 derived from final consumption (slide 8, LOAD FORECAST pptx: 5810 GWh) + T&D losses ~9% (slide 9).

*Confidence: [MEDIUM] · Last updated: 2026-05-29*


<a id="armenia-pdemandprofile"></a>

### `pDemandProfile`

[&#8593; Armenia](#armenia)

**Proxied from**: Turkiye/EastAna  
**Original source**: TEİAŞ — Turkiye hourly load data (likely)

**Method**: PROXY_TurkiyeEastAna

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `PROXY_TurkiyeEastAna` | EastAna (East Anatolia) hourly shape copied verbatim — nearest zone geographically, same model |

> No Armenia SCADA or hourly load data available. Proxied from the Turkiye profile (shared across all TR zones, including EastAna). Key limitation: Armenia's residential sector relies heavily on direct electric heating (unlike Turkiye which has significant gas penetration), implying a sharper winter morning peak and a higher load factor in Q1. Profile should be replaced with GSE/ANRE SCADA data when available.

*Confidence: [LOW] · Last updated: 2026-05-29*


<a id="armenia-pavailabilitycustom"></a>

### `pAvailabilityCustom`

[&#8593; Armenia](#armenia)

**Source**: World Nuclear Association — Reactor Database (`wna_reactor_database`)

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

**Method**: DIRECT (Armenia_ANPP) — EPM generic for all other techs

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Armenia_ANPP: 0.70 flat (Q1–Q4), mean of WNA Energy Availability 2023 (67.3%) and 2024 (70.4%). All other Armenia generators rely on pAvailabilityGeneric defaults (CCGT=0.85, Hydro=0.85, PV/Wind=1.0).
 |

> Nuclear excluded from pAvailabilityGeneric — availability is plant-specific (aging VVER-270 at 0.70 vs new builds at 0.85–0.90). Other countries with nuclear plants should add their own custom entry referencing wna_reactor_database.

*Confidence: [HIGH] · Last updated: 2026-05-30*


<a id="armenia-pgendatainput"></a>

### `pGenDataInput`

[&#8593; Armenia](#armenia)

**Source**: CESI/EPSO — Armenia Data and Assumptions PLEXOS STUDY (`epso_armenia_plexos_2022`)

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

**Method**: DIRECT (capacity, dates from CESI) — tech params from EPM defaults

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Capacity and dates (StYr, RetrYr) from CESI PLEXOS study slides. HeatRate, RampUpRate, RampDnRate, ResLimShare, FOMperMW, VOM left blank — filled automatically at runtime from pGenDataInputGeneric (EPM parameter guide: https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/). Nuclear and ST/Gas defaults added to pGenDataInputGeneric for this deployment.
 |

> 14 generators: 12 Status-1 (existing), 2 Status-2 (committed), 3 Status-3 (candidates). DOUBT — Armenia_Hrazdan_ST: units 1–4 all mothballed per GEM (2024); Capacity set to 0 MW pending confirmation. CESI shows 300–410 MW nameplate but none dispatchable. DOUBT — Armenia_SHC RetrYr=2047: no explicit retirement date found; estimated from post-ADB rehabilitation life extension (loan matures 2029). Commissioned 1960–1962.

*Confidence: [MEDIUM] · Last updated: 2026-05-30*


<a id="armenia-pfuelprice"></a>

### `pFuelPrice`

[&#8593; Armenia](#armenia)

**Source**: TYNDP / IEA World Energy Outlook 2022 — commodity prices (`tyndp_iea_weo_2022`)

**Method**: DIRECT (2030–2050 anchors) + INTERP (intermediates) + EXTRAP (2024, 2051–2053)

| Period | Method | Notes |
|--------|--------|-------|
| 2030–2050 | `DIRECT` | PLEXOS slide 5 anchors — Gas: 4.680/4.838/4.996/5.101/5.206 €/GJ; HFO: 6.16 €/GJ flat |
| 2031–2049 | `INTERP` | Linear interpolation between successive 5-year anchors (slopes: +0.0316 €/GJ/yr on 2030–2040, +0.021 on 2040–2050) |
| 2024 | `EXTRAP` | Gas: 4.680 − 6 × 0.0316 = 4.490 €/GJ (extrapolation arrière au taux 2030–2035); HFO: flat |
| 2025–2029 | `INTERP` | Linear between 2024 and 2030 anchor |
| 2051–2053 | `EXTRAP` | Gas: +0.021 €/GJ/yr (taux 2045–2050); HFO: flat |

> EUR/USD conversion at 1.05 (BCE 2022 annual average). Fuels included: Gas, HFO. Uranium proxied flat from Turkiye (0.97 $/GJ — not covered by TYNDP/IEA WEO). Coal and Lignite excluded — no coal generation in Armenia existing or planned fleet. Gas price reflects TYNDP/IEA market trajectory, not Armenia–Gazprom bilateral contract price (~4.5 $/GJ in 2022); difference is small but methodology diverges from actual cost structure for near-term years.

*Confidence: [MEDIUM] · Last updated: 2026-05-29*


---
