# Data Sources — EPM — Black Sea 2026

*Generated 2026-06-05*

---

## Model overview

**Countries**: Turkiye, Armenia, Georgia, Azerbaijan  
**Data horizon**: 2024–2053 · step: 1 year


| Category | Item | Parameter | Description | Turkiye | Armenia | Georgia | Azerbaijan |
|---|---|---|---|---|---|---|---|
| Load | Annual demand forecast | `pDemandForecast` | Historical and projected electricity demand (GWh and MW peak) by year | — | CESI/EPSO (2022) | ⚠ Georgia Hourly Load Profile wi… (2022) | ⚠ owid_energy_data |
| Load | Hourly demand profile | `pDemandProfile` | Typical hourly load curve (8760 h) for a representative year | — | ⚠ proxy of Turkiye/EastAna | ⚠ Georgia Hourly Load Profile wi… (2022) | ⚠ run_blacksea_data_proxy |
| Supply | Generator database | `pGenDataInput` | Existing, committed, and candidate plants: name, technology, capacity (MW), COD, CAPEX, O&M, operating constraints | — | CESI/EPSO (2022) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | ⚠ SESA/WB Georgia Generation Dat… (2022-07-01) + Georgia Power Sector Data Repository (WB Internal) + WB EPM Georgia v8.5 (2022) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | ⚠ gem_gipt + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) |
| Supply | Fuel prices | `pFuelPrice` | Gas, coal, diesel, HFO trajectory 2025–2050 ($/GJ) | — | TYNDP / IEA World Energy Outlo… (2022) | ⚠ Georgia Fuel Subsidies Databas… (2022) + [TYNDP / IEA World Energy Outlook 2022](https://www.iea.org/reports/world-energy-outlook-2022) | ⚠ imf_energy_subsidies + [TYNDP / IEA World Energy Outlook 2022](https://www.iea.org/reports/world-energy-outlook-2022) |
| Supply | Plant availability | `pAvailabilityCustom` | Seasonal capacity factors for thermal, hydro, and other dispatchable units | — | World Nuclear Association (updated annually) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | ⚠ WB EPM Georgia v8.5 (2022) (2022) + Georgia Hourly Generation Profiles by Technology 2019–2022 | ⚠ WB EPM Georgia v8.5 (2022) (2022) |
| Supply | Storage assumptions | `pStorageDataInput` | For BESS and PSH: capacity, duration, efficiency, cost assumptions | — | — | — | — |
| Supply | VRE and hydro profiles | `pVREProfile` | Hourly capacity factor profiles for solar PV, wind, and run-of-river hydro (normalised 0–1) | — | ⚠ Renewables Ninja (2018–2023) + TEİAŞ | ⚠ WB EPM Georgia 2022 (2022) | — |
| Resources | Maximum installable capacity | `pMaxGenerationByFuel` | Maximum new capacity by technology (resource potential and spatial constraints) | — | — | — | — |
| Resources | VRE integration assumptions | `pSettings` | VRE curtailment, variability handling, and balancing cost assumptions | — | — | — | — |
| Trade | Cross-border transmission | `pTransferLimit` | Existing and planned cross-border interconnectors: capacity (MW), year, routing options | — | — | — | — |
| Trade | Transmission losses | `pLossFactorInternal` | Cross-border interconnector losses (% by corridor) | — | — | — | — |
| Trade | Trade prices | `pTradePrice` | Import/export prices with temporal variability ($/MWh) — external zones | — | — | — | — |
| Reserves | Reserve margin | `pPlanningReserveMargin` | Planning reserve margin (%) and operating reserve assumptions | — | — | — | — |
| Other | Carbon pricing | `pCarbonPrice` | Carbon price or emission constraint applied in planning (NDC, ETS membership) | — | — | — | — |
| Other | Fuel and import limits | `pMaxFuelLimit` | Caps or floors on fuel use or electricity imports (e.g. gas import quotas) | — | — | — | — |

---

<a id="toc"></a>

## Contents

- [Turkiye](#turkiye) — *not yet documented*
- [Armenia](#armenia) — [`pDemandForecast`](#armenia-pdemandforecast) · [`pDemandProfile`](#armenia-pdemandprofile) · [`pVREProfile`](#armenia-pvreprofile) · [`pAvailabilityCustom`](#armenia-pavailabilitycustom) · [`pGenDataInput`](#armenia-pgendatainput) · [`pFuelPrice`](#armenia-pfuelprice)
- [Georgia](#georgia) — [`pGenDataInput`](#georgia-pgendatainput) · [`pDemandForecast`](#georgia-pdemandforecast) · [`pDemandProfile`](#georgia-pdemandprofile) · [`pVREProfile`](#georgia-pvreprofile) · [`pFuelPrice`](#georgia-pfuelprice) · [`pAvailabilityCustom`](#georgia-pavailabilitycustom)
- [Azerbaijan](#azerbaijan) — [`pGenDataInput`](#azerbaijan-pgendatainput) · [`pDemandForecast`](#azerbaijan-pdemandforecast) · [`pDemandProfile`](#azerbaijan-pdemandprofile) · [`pFuelPrice`](#azerbaijan-pfuelprice) · [`pAvailabilityCustom`](#azerbaijan-pavailabilitycustom)

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
| [`pDemandProfile`](#armenia-pdemandprofile) | proxy of Turkiye/EastAna | [LOW] ⚠ |
| [`pGenDataInput`](#armenia-pgendatainput) | CESI/EPSO (2022) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] |
| [`pFuelPrice`](#armenia-pfuelprice) | TYNDP / IEA World Energy Outlo… (2022) | [MEDIUM] |
| [`pAvailabilityCustom`](#armenia-pavailabilitycustom) | World Nuclear Association (updated annually) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [HIGH] |
| [`pVREProfile`](#armenia-pvreprofile) | Renewables Ninja (2018–2023) + TEİAŞ | [LOW] ⚠ |

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

> ⚠ **Needs review**: Obtain GSE/ANRE SCADA hourly load data for Armenia to replace Turkiye/EastAna proxy

**Method**: PROXY_TurkiyeEastAna

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `PROXY_TurkiyeEastAna` | EastAna (East Anatolia) hourly shape copied verbatim — nearest zone geographically, same model |

> No Armenia SCADA or hourly load data available. Proxied from the Turkiye profile (shared across all TR zones, including EastAna). Key limitation: Armenia's residential sector relies heavily on direct electric heating (unlike Turkiye which has significant gas penetration), implying a sharper winter morning peak and a higher load factor in Q1. Profile should be replaced with GSE/ANRE SCADA data when available.

*Confidence: [LOW] · Last updated: 2026-05-29*


<a id="armenia-pvreprofile"></a>

### `pVREProfile`

[&#8593; Armenia](#armenia)

**Source**: Renewables Ninja — PV and Wind capacity factors (`renewables_ninja`)

**Also uses**: TEİAŞ — Turkiye hourly load data (likely) (`teias_hourly_load`)

> ⚠ **Needs review**: Rerun representative days pipeline with all Black Sea countries — d1–d6 currently share same seasonal mean (within-season variability lost). ROR proxied from EastAna.

**Method**: DIRECT seasonal mean (PV, Wind) — PROXY_EastAna (ROR)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | PV + Wind: 6-year mean (2018–2023) from Renewables Ninja at Armenia centroid (44.5°E, 40.2°N). All d1–d6 within a season share the same mean hourly profile.
 |
| 2024–2053 | `PROXY_EastAna` | ROR: EastAna Turkiye zone rows copied verbatim — nearest zone, similar snowmelt hydrology |

> Within-season variability (d1–d6 differentiation) is lost — all daytypes in a season share the same mean profile. This is a known limitation of the current approach. To fix: rerun the full representative days pipeline including Armenia alongside all other Black Sea countries (run_blacksea_data.py already supports this).

*Confidence: [LOW] · Last updated: 2026-05-30*


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

<a id="georgia"></a>

## Georgia

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pDemandForecast`](#georgia-pdemandforecast) | Georgia Hourly Load Profile wi… (2022) | [MEDIUM] ⚠ |
| [`pDemandProfile`](#georgia-pdemandprofile) | Georgia Hourly Load Profile wi… (2022) | [MEDIUM] ⚠ |
| [`pGenDataInput`](#georgia-pgendatainput) | SESA/WB Georgia Generation Dat… (2022-07-01) + Georgia Power Sector Data Repository (WB Internal) + WB EPM Georgia v8.5 (2022) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] ⚠ |
| [`pFuelPrice`](#georgia-pfuelprice) | Georgia Fuel Subsidies Databas… (2022) + [TYNDP / IEA World Energy Outlook 2022](https://www.iea.org/reports/world-energy-outlook-2022) | [LOW] ⚠ |
| [`pAvailabilityCustom`](#georgia-pavailabilitycustom) | WB EPM Georgia v8.5 (2022) (2022) + Georgia Hourly Generation Profiles by Technology 2019–2022 | [MEDIUM] ⚠ |
| [`pVREProfile`](#georgia-pvreprofile) | WB EPM Georgia 2022 (2022) | [MEDIUM] ⚠ |

<a id="georgia-pgendatainput"></a>

### `pGenDataInput`

[&#8593; Georgia](#georgia)

**Source**: SESA/WB Georgia Generation Dataset 2022 (`sesa_georgia_2022`)

**Also uses**: Georgia Power Sector Data Repository (WB Internal) (`ge_power_sector_data_repository`)

**Also uses**: WB EPM Georgia v8.5 (2022) — Technical Parameters (`wb_epm_georgia_v85`)

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

> ⚠ **Needs review**: (1) Committed large hydro (Khudoni 702 MW, Namakhvani 433 MW, Nenskra 280 MW): all politically contested/delayed — StYr estimates uncertain ±3 years. Namakhvani suspended due to protests (2021–2022); may need to downgrade to Status=3. (2) Tbilsresi CCGT (1963): 60+ year old plant, RetrYr=2027 estimated — confirm operational status with CESI/GSE. (3) Kirnati capacity discrepancy: sesa_georgia_2022 shows 27.47 MW, ge_power_sector_data_repository shows 51.22 MW — used sesa_georgia_2022 value. (4) Mtkvari VOM=0.06 $/MWh from wb_epm_georgia_v85 is unusually low — verify. (5) Tbilsresi labeled CCGT in data sources but 1963 vintage — likely old steam turbine. (6) DomesticCoal for Tkibuli: no entry in pFuelPrice for Georgia yet — needs adding.


**Method**: DIRECT (capacity, dates, tech) — old EPM for HeatRate thermal — generic for all other params

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | 113 plants from sesa_georgia_2022 reduced to 46 rows: plants ≥10 MW kept individual; plants <10 MW aggregated into Georgia_AGG_SmallHydro (~224 MW). Capacity: sesa_georgia_2022. StYr: ge_power_sector_data_repository (commissioning year per plant). tech: mapped from Status column (with Reservoir→ReservoirHydro, Seasonal/Small→ROR) cross-checked with ge_power_sector_data_repository type column. HeatRate for Mtkvari (10.3 MMBtu/MWh) and Gardabani CCGT (6.93 MMBtu/MWh) from wb_epm_georgia_v85. All other technical params (VOM, FOM, Capex, RampRate, ResLimShare, Life) left blank → filled at runtime from pGenDataInputGeneric (EPM generic defaults).
 |
| committed | `DIRECT` | 4 committed rows (Status=2): Khudoni 702 MW (StYr=2032), Namakhvani 433 MW (StYr=2030), Nenskra 280 MW (StYr=2029) from List_of_PPAs_May2022 'Construction' stage; Georgia_HydroSHP_Com 549 MW aggregate from PPA construction pipeline.
 |
| candidates | `DIRECT` | 4 candidate rows (Status=3): Wind (300 MW, Capex=1.3 $M/MW), PV (200 MW, Capex=0.8 $M/MW) from RE Pipeline; SmallHydro (300 MW), BESS (200 MW) — engineering estimates.
 |

*Confidence: [MEDIUM] · Last updated: 2026-06-04*


<a id="georgia-pdemandforecast"></a>

### `pDemandForecast`

[&#8593; Georgia](#georgia)

**Source**: Georgia Hourly Load Profile with 3% Annual Growth (2021–2040) (`georgia_demand_load_2022`)

> ⚠ **Needs review**: Peak demand (MW) has no independent cross-validation — only georgia_demand_load_2022 provides peak figures. Energy figures validated against historical balance (error <1% for 2023–2024). Growth rate of 3%/yr is undocumented — confirm with GSE/GNERC official load forecasts. Obtain electrification scenario for post-2030 period (EV, heat pumps) as 3%/yr may underestimate long-term growth.


**Method**: DIRECT (2024–2040 from hourly file) + EXTRAP (2041–2053 at 3%/yr)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2040 | `DIRECT` | Annual peak (MW) = max hourly value per year from Av. 3% Load growth file. Annual energy (GWh) = sum of hourly values / 1000 per year. File covers 2021–2040; 2024 is the first model-relevant year.
 |
| 2041–2053 | `EXTRAP` | Extrapolation at 3%/yr from 2040 base (same growth rate as file assumption). |

*Confidence: [MEDIUM] · Last updated: 2026-06-04*


<a id="georgia-pdemandprofile"></a>

### `pDemandProfile`

[&#8593; Georgia](#georgia)

**Source**: Georgia Hourly Load Profile with 3% Annual Growth (2021–2040) (`georgia_demand_load_2022`)

> ⚠ **Needs review**: Within-season variability (d1–d6 differentiation) is lost — all daytypes in a season share the same seasonal mean profile. Same limitation as Armenia pDemandProfile. To fix: obtain GSE SCADA hourly load data and rerun the representative days pipeline including Georgia.


**Method**: DIRECT seasonal mean from 2025 hourly data, normalized by peak

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Seasonal mean hourly profile extracted from 2025 data in Av. 3% Load growth file (2025 chosen as first model year). Mean computed per (season, hour) → 96 unique hourly values. Normalized by max seasonal-mean value (2105 MW in Q4 evening peak). All d1–d6 daytypes within a season share the same profile (simplified approach).
 |

*Confidence: [MEDIUM] · Last updated: 2026-06-04*


<a id="georgia-pvreprofile"></a>

### `pVREProfile`

[&#8593; Georgia](#georgia)

**Source**: WB EPM Georgia 2022 — VRE Timeseries (Typical Year) (`wb_epm_georgia_timeseries`)

> ⚠ **Needs review**: (1) Single typical year — no multi-year average. (2) Wind profile: Timeseries mean CF ~0.27 vs actual Qartli 2021 CF ~0.46 — Timeseries likely represents a generic Georgian wind site, not Qartli's specific high-wind location. Existing Georgia_Qartli_Wind may be under-dispatched in the model; consider a separate pVREProfile entry or pAvailabilityCustom override for Qartli. (3) PV data origin undocumented — replace with Renewables Ninja multi-year average when running representative days pipeline for Georgia. (4) d1–d6 all share same seasonal mean (within-season variability lost).


**Method**: DIRECT seasonal mean from typical-year hourly CFs, normalized by tech peak

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Three techs (ROR, OnshoreWind, PV) from Timeseries all data.xlsx (sheet RE data). Seasonal mean computed per (season, hour) for 8,760 hourly CF values. Normalized by the maximum seasonal-mean hourly value across all seasons/hours for each tech separately. All d1–d6 daytypes share the same seasonal mean. Seasonal CF characteristics: ROR — Q2 peak (spring snowmelt) cf_mean=0.977, Q4 minimum cf_mean=0.566. Wind — Q2 highest cf_mean=0.902, Q3 lowest cf_mean=0.738. PV — Q3 highest (more sun hours), Q1 lowest.
 |

*Confidence: [MEDIUM] · Last updated: 2026-06-04*


<a id="georgia-pfuelprice"></a>

### `pFuelPrice`

[&#8593; Georgia](#georgia)

**Source**: Georgia Fuel Subsidies Database 2022 (IMF/World Bank methodology) (`georgia_fuel_subsidies_2022`)

**Also uses**: [TYNDP / IEA World Energy Outlook 2022 — commodity prices](https://www.iea.org/reports/world-energy-outlook-2022)

> ⚠ **Needs review**: Gas price HIGHLY uncertain. georgia_fuel_subsidies_2022 retail price (11.99 USD/GJ) is ~3× the estimated wholesale price. South Caucasus proxy (4.5 USD/MMBtu) is based on trade data estimates, not confirmed generator-level tariffs. Confirm actual Gardabani/Mtkvari/GPower fuel cost with CESI or GNERC tariff orders. DomesticCoal: Tkibuli coal quality is sub-bituminous (low calorific value) — price may be expressed per ton not per energy unit in actual contracts.


**Method**: Gas: South Caucasus wholesale proxy + Armenia growth rate. DomesticCoal: DIRECT from fuel-subsidies file.

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `PROXY_Armenia` | Gas: base 4.50 USD/MMBtu in 2024 (South Caucasus wholesale price estimate for Azerbaijan→Georgia gas at ~$150/1000m3 ÷ 35.3 GJ/1000m3 ÷ 0.9478 MMBtu/GJ). Year-on-year growth applied from Armenia trajectory (tyndp_iea_weo_2022): +0.033 USD/MMBtu/yr to 2040, +0.022 USD/MMBtu/yr beyond. Result: 2024=4.50 → 2030=4.70 → 2040=5.03 → 2053=5.32 USD/MMBtu.
 |
| 2024–2053 | `DIRECT` | DomesticCoal: 3.82 USD/MMBtu flat (= 3.62 USD/GJ from georgia_fuel_subsidies_2022 power sector coal price, 2021). Tkibuli domestic coal, minimal price variation.
 |

*Confidence: [LOW] · Last updated: 2026-06-04*


<a id="georgia-pavailabilitycustom"></a>

### `pAvailabilityCustom`

[&#8593; Georgia](#georgia)

**Source**: WB EPM Georgia v8.5 (2022) — Technical Parameters (`wb_epm_georgia_v85`)

**Also uses**: Georgia Hourly Generation Profiles by Technology 2019–2022 (`georgia_generation_profiles_2019_2022`)

> ⚠ **Needs review**: (1) Dzevruli Q3=0.04 and Shaori Q1=0.56 are unusual patterns from old model calibration — verify against actual plant hydrology with CESI. (2) Committed large reservoir (Khudoni/Namakhvani/Nenskra): proxy from Enguri — no plant-specific hydrological data available. (3) ROR aggregate: individual plant availability variability lost (Rioni ~0.70 flat vs Vartsikhe Q3=0.35 in old model). Acceptable for planning model.


**Method**: DIRECT from WB EPM v8.5 GenAvailability (ReservoirHydro) + AGGREGATE 2019-2022 (ROR)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | 7 existing ReservoirHydro plants: quarterly CFs from WB_EPM_v8_5.xlsb GenAvailability sheet (monthly factors m1-m12 averaged to quarters). Plant-specific hydrological calibration from WB EPM Georgia 2022 study: Enguri Q1=0.21/Q2=0.48/Q3=0.52/Q4=0.20; Vardnili Q1=0.30/Q2=0.49/Q3=0.48/Q4=0.28; Khrami-1&2 Q1=0.12/Q2=0.31/Q3=0.27/Q4=0.32; Jinvali Q1=0.28/Q2=0.30/Q3=0.22/Q4=0.18; Dzevruli Q1=0.38/Q2=0.17/Q3=0.04/Q4=0.17 (very low summer — specific hydrology); Shaori Q1=0.56/Q2=0.30/Q3=0.14/Q4=0.44 (peaks in winter — specific regime). 3 committed reservoir plants (Khudoni/Namakhvani/Nenskra): proxy from Enguri (western Georgia large reservoir profile).
 |
| 2024–2053 | `DIRECT_aggregate` | 26 ROR plants (23 individual ≥10 MW + AGG_SmallHydro + committed + candidate): uniform Q1=0.45, Q2=0.81, Q3=0.54, Q4=0.40 from georgia_generation_profiles_2019_2022. 4-year average (2019-2022) of total RoR hourly generation / installed RoR capacity. pVREProfile for ROR set to 1.0 flat — all seasonal variation in pAvailabilityCustom.
 |

*Confidence: [MEDIUM] · Last updated: 2026-06-04*


---

<a id="azerbaijan"></a>

## Azerbaijan

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pDemandForecast`](#azerbaijan-pdemandforecast) | owid_energy_data | [MEDIUM] ⚠ |
| [`pDemandProfile`](#azerbaijan-pdemandprofile) | run_blacksea_data_proxy | [LOW] ⚠ |
| [`pGenDataInput`](#azerbaijan-pgendatainput) | gem_gipt + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] ⚠ |
| [`pFuelPrice`](#azerbaijan-pfuelprice) | imf_energy_subsidies + [TYNDP / IEA World Energy Outlook 2022](https://www.iea.org/reports/world-energy-outlook-2022) | [MEDIUM] ⚠ |
| [`pAvailabilityCustom`](#azerbaijan-pavailabilitycustom) | WB EPM Georgia v8.5 (2022) (2022) | [LOW] ⚠ |

<a id="azerbaijan-pgendatainput"></a>

### `pGenDataInput`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: gem_gipt (`gem_gipt`)

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

> ⚠ **Needs review**: All tech params from generic defaults — no country-specific calibrated model. Committed large CCGT (Mingecevir 1280 MW) status/year to confirm. Old Soviet-era plants (8x300 MW ST Gas, 1990) excluded as pre-2025 retirees.


**Method**: DIRECT from GEM GIPT September 2025 via compute_epm_gendata.py

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | 57 GEM/GIPT plants mapped to 39 EPM rows (plants with RetrYr<2025 excluded). Gas: year<2000 -> ST, year>=2000 -> CCGT. Hydro -> ReservoirHydro. Technical params (HeatRate, VOM, FOM, Capex) from epm_generic_defaults. Committed: Mingecevir CCGT 4x320 MW (StYr=2025).
 |

*Confidence: [MEDIUM] · Last updated: 2026-06-05*


<a id="azerbaijan-pdemandforecast"></a>

### `pDemandForecast`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: owid_energy_data (`owid_energy_data`)

> ⚠ **Needs review**: Peak demand estimated from energy via load factor (0.58) — no independent peak data source. CAGR 1.9%/yr based on 5-year trend; may underestimate if electrification accelerates. Confirm with SOCAR/Ministry energy balance.


**Method**: DIRECT from OWID electricity_demand + CAGR extrapolation + load-factor peak

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT_EXTRAPOLATED` | Anchor: OWID electricity_demand 2025 = 27.17 TWh (net consumption). CAGR = 1.9%/yr computed from OWID 2020-2025 trend. Peak estimated via load_factor=0.58: Peak_MW = (demand_TWh*1e6/8760) / 0.58. 2024: Energy=26666 GWh, Peak=5248 MW. Computed via compute_epm_demand.py --country AZE.
 |

*Confidence: [MEDIUM] · Last updated: 2026-06-05*


<a id="azerbaijan-pdemandprofile"></a>

### `pDemandProfile`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: run_blacksea_data_proxy (`run_blacksea_data_proxy`)

> ⚠ **Needs review**: PROXY — no Azerbaijan-specific hourly load data. Turkey shape used as proxy (similar climate: continental, hot summers, cold winters). TO RECOMPUTE: run full representative-days pipeline for all Black Sea countries (including AZ) with VRE profiles when all country data is available. Command: python run_blacksea_data.py, then rerun compute_epm_demand.py --profile.


**Method**: PROXY Turkey shape from ENTSO-E, scaled to AZ demand, seasonal mean

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `PROXY_Turkey` | Load shape: Turkey ENTSO-E hourly profile scaled to AZ annual energy (from run_blacksea_data.py, AZE_ANNUAL_MWH=29.3 TWh). Seasonal mean per quarter (Q1-Q4), all d1-d6 share same mean profile. Q1_mean=0.737 (winter heating peak), Q3_mean=0.651 (summer). Computed via compute_epm_demand.py --country AZE --profile.
 |

*Confidence: [LOW] · Last updated: 2026-06-05*


<a id="azerbaijan-pfuelprice"></a>

### `pFuelPrice`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: imf_energy_subsidies (`imf_energy_subsidies`)

**Also uses**: [TYNDP / IEA World Energy Outlook 2022 — commodity prices](https://www.iea.org/reports/world-energy-outlook-2022)

> ⚠ **Needs review**: Supply price (4.46 $/MMBtu) is between domestic subsidized retail (~$1.5-2/MMBtu) and export opportunity cost (~$6-8/MMBtu via TANAP). Choice justification: reflects actual generator economics for planning model. If WB study requires economic (opportunity) cost approach, use ~$6.5/MMBtu. IMF data covers 2015-2025; forward trajectory based on Armenia CAGR.


**Method**: DIRECT supply price from IMF subsidy database (Gas) + flat estimate (Biomass)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2025 | `DIRECT` | Gas: mit_sp_nga_pow (supply price paid by power generators) from Georgia_fuel-subsidies_2022.xlsx 'data' sheet, country=AZE. 2024 = 4.225 USD/GJ = 4.46 USD/MMBtu (converted at 1 GJ = 0.9478 MMBtu). This is NOT the domestic subsidized retail price (~$1.5-2/MMBtu) NOR the full export/opportunity cost (~$6-8/MMBtu) — it is the actual price charged to power generators by SOCAR.
 |
| 2026–2053 | `EXTRAPOLATED` | Growth trajectory applied from Armenia (tyndp_iea_weo_2022): same year-on-year increments (+0.033/yr to 2040, +0.022/yr beyond). Biomass: 0.50 USD/MMBtu flat (Balakhani landfill gas, quasi-free fuel).
 |

*Confidence: [MEDIUM] · Last updated: 2026-06-05*


<a id="azerbaijan-pavailabilitycustom"></a>

### `pAvailabilityCustom`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: WB EPM Georgia v8.5 (2022) — Technical Parameters (`wb_epm_georgia_v85`)

> ⚠ **Needs review**: Proxy from Georgia Enguri calibration — no Azerbaijan-specific hydrological data used. Should be replaced with actual Mingechevir seasonal generation data when available (e.g., GRDC Kura discharge at Mingechevir gauge, or AzerEnerji annual generation reports). Mingechevir has 16 km3 storage capacity — actual seasonal dispatch also depends on irrigation releases and operational decisions, not just inflows.


**Method**: PROXY from Georgia Kura River calibration — same river system

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `PROXY_Georgia_Kura` | 5 ReservoirHydro plants (Mingechevir, Shamkir, Yenikend, Khudafarin, Gyz Galasy): all on Kura River downstream of Georgia. Seasonal pattern proxied from WB EPM Georgia v8.5 Enguri calibration (Q1=0.21, Q2=0.48, Q3=0.52, Q4=0.20) — physically justified: same catchment, same snowmelt-driven seasonal cycle. All 5 plants share same seasonal CF (no plant-specific data available). Thermal/Biomass/VRE: no custom — generic defaults apply.
 |

*Confidence: [LOW] · Last updated: 2026-06-05*


---
