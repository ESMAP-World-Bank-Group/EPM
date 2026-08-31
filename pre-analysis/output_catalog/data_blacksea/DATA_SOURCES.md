# Data Sources — EPM — Black Sea 2026

*Generated 2026-08-31*

---

## Model overview

**Countries**: Turkiye, Armenia, Georgia, Azerbaijan, iran_swap, AzerbaijanMain, Nakhchivan, Romania, Bulgaria  
**Data horizon**: 2024–2053 · step: 1 year


| Category | Item | Parameter | Description | Turkiye | Armenia | Georgia | Azerbaijan | iran_swap | AzerbaijanMain | Nakhchivan | Romania | Bulgaria |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Load | Annual demand forecast | `pDemandForecast` | Historical and projected electricity demand (GWh and MW peak) by year | — | CESI (World Bank consultant) /… (2022) | ⚠ World Bank (internal) (2022) | ⚠ Our World in Data (OWID) (2025) + [SSC](https://statistika.nmr.az/) | ⚠ Modeller expert judgment (2026) | ⚠ Our World in Data (OWID) (2025) + [SSC](https://statistika.nmr.az/) | ⚠ SSC + [Our World in Data (OWID)](https://ourworldindata.org/energy) | ⚠ Our World in Data (OWID) (2025) | World Bank Bulgaria CCDR (2026) + [Our World in Data (OWID)](https://ourworldindata.org/energy) |
| Load | Hourly demand profile | `pDemandProfile` | Typical hourly load curve (8760 h) for a representative year | — | ⚠ proxy of Turkiye/EastAna | World Bank (internal) (2022) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | ⚠ proxy of Turkiye (ENTSO-E hourly shape, scaled to AZ energy) | Modeller expert judgment (2026) | ⚠ Proxy load profiles (Azerbaija… (2026) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | Proxy load profiles (Azerbaija… (2026) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | ENTSO-E Transparency Platform (2025) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | ⚠ ENTSO-E Transparency Platform (2025) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) |
| Supply | Generator database | `pGenDataInput` | Existing, committed, and candidate plants: name, technology, capacity (MW), COD, CAPEX, O&M, operating constraints | ⚠ Observed annual capacity addit… (2026) + World Bank EPM Türkiye Least-Cost Model v7 (2025) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | CESI (World Bank consultant) /… (2022) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + EPSO + [Renewable resource potential](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment) + [RE candidate annual build-rate limits](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | ⚠ SESA (Georgian Power Sector An… (2022-07-01) + Georgia Power Sector Data Repository (WB Internal) + World Bank EPM Georgia v8.5 (2022, internal model) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + [Renewable resource potential](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment) + [RE candidate annual build-rate limits](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/) + [GSE Ten-Year Network Development Plan of Georgia (TYNDP 2023-2033)](https://www.gse.com.ge/komunikacia/publikaciebi/saqartvelos-gadamcemi-qselis-ganvitarebis-atwliani-gegma) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | ⚠ Global Energy Monitor (GEM) (2025-09) + [SSC Azerbaijan](https://stat.gov.az/source/balance_energy/) + [SSC](https://statistika.nmr.az/) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + [Renewable resource potential](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment) + [RE candidate annual build-rate limits](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | — | ⚠ Global Energy Monitor (GEM) (2025-09) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + [SSC](https://statistika.nmr.az/) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | ⚠ SSC + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + [Renewable resource potential](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment) + [RE candidate annual build-rate limits](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | ⚠ World Bank EPM Romania v8.5 (2… (2024) + [Global Energy Monitor (GEM)](https://globalenergymonitor.org/projects/global-integrated-power-tracker/) | ⚠ Global Energy Monitor (GEM) (2025-09) + World Bank Bulgaria CCDR + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) |
| Supply | Fuel prices | `pFuelPrice` | Gas, coal, diesel, HFO trajectory 2025–2050 ($/GJ) | Türkiye gas import-parity cost… (2026-08) + World Bank EPM Türkiye Least-Cost Model v7 (2025) | South Caucasus bilateral gas c… (2026-08) + [TYNDP / IEA World Energy Outlook 2022](https://www.iea.org/reports/world-energy-outlook-2022) | South Caucasus bilateral gas c… (2026-08) + IMF Energy Subsidies Database | IMF (2022) + [TYNDP / IEA World Energy Outlook 2022](https://www.iea.org/reports/world-energy-outlook-2022) | — | IMF (2022) + [TYNDP / IEA World Energy Outlook 2022](https://www.iea.org/reports/world-energy-outlook-2022) | — | ⚠ World Bank EPM Romania v8.5 (2… (2024) | World Bank Bulgaria CCDR (2026) + World Bank EPM Romania v8.5 (2024, internal model) |
| Supply | Plant availability | `pAvailabilityCustom` | Seasonal capacity factors for thermal, hydro, and other dispatchable units | — | ⚠ World Nuclear Association (updated annually) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + EPSO | ⚠ World Bank EPM Georgia v8.5 (2… (2022) + Georgia Hourly Generation Profiles by Technology 2019–2022 | ⚠ World Bank EPM Georgia v8.5 (2… (2022) + [SSC Azerbaijan](https://stat.gov.az/source/balance_energy/) + [SSC](https://statistika.nmr.az/) | — | ⚠ World Bank EPM Georgia v8.5 (2… (2022) + [SSC Azerbaijan](https://stat.gov.az/source/balance_energy/) | ⚠ SSC + World Bank EPM Georgia v8.5 (2022, internal model) | World Bank EPM Romania v8.5 (2… (2024) | ⚠ Bulgarian quarterly availabili… (2026) + [World Nuclear Association](https://world-nuclear.org/nuclear-reactor-database/) + [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) + World Bank Bulgaria CCDR |
| Supply | Storage assumptions | `pStorageDataInput` | For BESS and PSH: capacity, duration, efficiency, cost assumptions | Observed annual capacity addit… (2026) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | EPM Generic Defaults | ⚠ EPM Generic Defaults | — | — | EPM Generic Defaults | EPM Generic Defaults | — | — |
| Supply | VRE and hydro profiles | `pVREProfile` | Hourly capacity factor profiles for solar PV, wind, and run-of-river hydro (normalised 0–1) | — | ⚠ Renewables Ninja (2018–2023) + TEİAŞ + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | ⚠ World Bank EPM Georgia 2022 (i… (2022) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | — | — | — | — | Global Energy Monitor (GEM) (2025-09) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | ⚠ Renewables Ninja (2018–2023) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) |
| Resources | Maximum installable capacity | `pMaxGenerationByFuel` | Maximum new capacity by technology (resource potential and spatial constraints) | — | — | — | — | — | — | — | — | — |
| Resources | VRE integration assumptions | `pSettings` | VRE curtailment, variability handling, and balancing cost assumptions | — | — | — | — | — | — | — | — | — |
| Trade | Cross-border transmission | `pTransferLimit` | Existing and planned cross-border interconnectors: capacity (MW), year, routing options | — | — | — | ⚠ Black Sea Cross-Border Lines D… (2026) + Modeller expert judgment | Modeller expert judgment (2026) | — | ⚠ Black Sea Cross-Border Lines D… (2026) + Modeller expert judgment | — | — |
| Trade | Transmission losses | `pLossFactorInternal` | Cross-border interconnector losses (% by corridor) | — | — | — | — | — | — | — | — | — |
| Trade | Trade prices | `pTradePrice` | Import/export prices with temporal variability ($/MWh) — external zones | — | — | — | Kazakh border price for the Tr… (2026) + Modeller expert judgment | — | — | — | — | — |
| Reserves | Reserve margin | `pPlanningReserveMargin` | Planning reserve margin (%) and operating reserve assumptions | — | — | — | — | — | — | — | — | — |
| Other | Carbon pricing | `pCarbonPrice` | Carbon price or emission constraint applied in planning (NDC, ETS membership) | — | — | — | — | — | — | — | — | — |
| Other | Fuel and import limits | `pMaxFuelLimit` | Caps or floors on fuel use or electricity imports (e.g. gas import quotas) | — | — | ⚠ Modeller expert judgment (2026) | — | — | — | — | — | — |

---

<a id="toc"></a>

## Contents

- [Turkiye](#turkiye) — [`pFuelPrice`](#turkiye-pfuelprice) · [`pGenDataInput`](#turkiye-pgendatainput) · [`pStorageDataInput`](#turkiye-pstoragedatainput)
- [Armenia](#armenia) — [`pStorageDataInput`](#armenia-pstoragedatainput) · [`pDemandForecast`](#armenia-pdemandforecast) · [`pDemandProfile`](#armenia-pdemandprofile) · [`pVREProfile`](#armenia-pvreprofile) · [`pAvailabilityCustom`](#armenia-pavailabilitycustom) · [`pGenDataInput`](#armenia-pgendatainput) · [`pFuelPrice`](#armenia-pfuelprice)
- [Georgia](#georgia) — [`pStorageDataInput`](#georgia-pstoragedatainput) · [`pGenDataInput`](#georgia-pgendatainput) · [`pDemandForecast`](#georgia-pdemandforecast) · [`pDemandProfile`](#georgia-pdemandprofile) · [`pVREProfile`](#georgia-pvreprofile) · [`pFuelPrice`](#georgia-pfuelprice) · [`pAvailabilityCustom`](#georgia-pavailabilitycustom) · [`pMaxFuelLimit`](#georgia-pmaxfuellimit)
- [Azerbaijan](#azerbaijan) — [`pGenDataInput`](#azerbaijan-pgendatainput) · [`pDemandForecast`](#azerbaijan-pdemandforecast) · [`pDemandProfile`](#azerbaijan-pdemandprofile) · [`pFuelPrice`](#azerbaijan-pfuelprice) · [`pAvailabilityCustom`](#azerbaijan-pavailabilitycustom) · [`pTransferLimit`](#azerbaijan-ptransferlimit) · [`pTradePrice`](#azerbaijan-ptradeprice) · [`pTradePriceExport`](#azerbaijan-ptradepriceexport) · [`pMaxAnnualExternalTradeShare`](#azerbaijan-pmaxannualexternaltradeshare)
- [iran_swap](#iran-swap) — [`pDemandForecast`](#iran-swap-pdemandforecast) · [`pDemandProfile`](#iran-swap-pdemandprofile) · [`pTransferLimit`](#iran-swap-ptransferlimit)
- [AzerbaijanMain](#azerbaijanmain) — [`pStorageDataInput`](#azerbaijanmain-pstoragedatainput) · [`pGenDataInput`](#azerbaijanmain-pgendatainput) · [`pDemandForecast`](#azerbaijanmain-pdemandforecast) · [`pDemandProfile`](#azerbaijanmain-pdemandprofile) · [`pFuelPrice`](#azerbaijanmain-pfuelprice) · [`pAvailabilityCustom`](#azerbaijanmain-pavailabilitycustom)
- [Nakhchivan](#nakhchivan) — [`pStorageDataInput`](#nakhchivan-pstoragedatainput) · [`pGenDataInput`](#nakhchivan-pgendatainput) · [`pDemandForecast`](#nakhchivan-pdemandforecast) · [`pDemandProfile`](#nakhchivan-pdemandprofile) · [`pAvailabilityCustom`](#nakhchivan-pavailabilitycustom) · [`pTransferLimit`](#nakhchivan-ptransferlimit)
- [Romania](#romania) — [`pGenDataInput`](#romania-pgendatainput) · [`pDemandForecast`](#romania-pdemandforecast) · [`pDemandProfile`](#romania-pdemandprofile) · [`pVREProfile`](#romania-pvreprofile) · [`pFuelPrice`](#romania-pfuelprice) · [`pAvailabilityCustom`](#romania-pavailabilitycustom)
- [Bulgaria](#bulgaria) — [`pDemandForecast`](#bulgaria-pdemandforecast) · [`pDemandProfile`](#bulgaria-pdemandprofile) · [`pVREProfile`](#bulgaria-pvreprofile) · [`pGenDataInput`](#bulgaria-pgendatainput) · [`pFuelPrice`](#bulgaria-pfuelprice) · [`pAvailabilityCustom`](#bulgaria-pavailabilitycustom)

---

<a id="turkiye"></a>

## Turkiye

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pGenDataInput`](#turkiye-pgendatainput) | Observed annual capacity addit… (2026) + World Bank EPM Türkiye Least-Cost Model v7 (2025) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] ⚠ |
| [`pFuelPrice`](#turkiye-pfuelprice) | Türkiye gas import-parity cost… (2026-08) + World Bank EPM Türkiye Least-Cost Model v7 (2025) | [MEDIUM] |
| [`pStorageDataInput`](#turkiye-pstoragedatainput) | Observed annual capacity addit… (2026) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] |

<a id="turkiye-pfuelprice"></a>

### `pFuelPrice`

[&#8593; Turkiye](#turkiye)

**Source**: Türkiye gas import-parity cost, 2026 review (World Bank CMO April 2026 + BOTAŞ contract indexation) (`turkiye_gas_import_parity_2026`)

**Data / file**: A constructed trajectory, not a published series. Türkiye imports ~99% of its gas, so its delivered cost is an import-parity blend of the two indices its supply contracts are written against. Two publ…

**Also uses**: World Bank EPM Türkiye Least-Cost Model v7 (2025) — fuel prices on IEA WEO scenarios (`wb_epm_turkiye_v7`)

**Method**: Gas: CONSTRUCTED import-parity blend (40% TTF / 60% oil-indexed) from WB CMO April 2026. All other Turkiye fuels DIRECT from WB EPM Turkiye v7, sheet 'Fuel Prices', scenario STEPS.

| Period | Method | Notes |
|--------|--------|-------|
| 2024-2053 | `DIRECT` | ImportedCoal 2.942 flat; DomesticCoal 3.152 flat; HFO 19.36 flat; Uranium 0.97 flat; Biomass 5.0 flat. Copied verbatim from input_epm_Turkiye_v7.xlsx, sheet 'Fuel Prices', rows tagged scenario STEPS, zone Turkiye. Values are already in $/MMBtu, so no conversion was applied. pFuelPrice is keyed by country via zcmap.csv, so one row prices all nine Turkish zones (WestMed, WestAna, EastMed, EastAna, SouthEast, NorthWest, CenterAna, CenterBlack, Trakia). |
| 2024-2025 | `DIRECT` | Gas 11.3 $/MMBtu, unchanged. Corresponds to the BOTAS power-sector tariff of ~12,000 TL/1000m3 and sits just below the WB CMO April 2026 print for European gas of 12.0 in 2025, so the base-year calibration is sound and is NOT affected by the 2026-08-25 change. |
| 2026-2053 | `ASSUMPTION` | Gas glides linearly over five years to a 10.0 $/MMBtu plateau in 2030 and holds it to 2053: 11.04 / 10.78 / 10.52 / 10.26 / 10.0. Derived as 0.40 x TTF + 0.60 x oil-indexed = 0.40 x 12.0 + 0.60 x (0.12 x 70) = 9.84, rounded to 10.0. See source turkiye_gas_import_parity_2026 for the full derivation and the published inputs. |

> CHANGED 2026-08-25. The Turkiye gas price is the largest single driver of regional flows in this deployment, and the default trajectory was replaced. WHAT CHANGED. config.csv now points pFuelPrice at supply/pFuelPrice_tr_gas_flat.csv instead of supply/pFuelPrice.csv. The two files differ in exactly one row, Turkiye/Gas. The default is now 11.3 (2024-25) -> 10.0 (2030) -> 10.0 (2053); the previous default was 11.3 -> 6.5 (2030) -> 7.6 (2040) -> 7.7 (2050+). Every other country, fuel and year is byte-identical between the two files, so no other assumption in the deployment moved. WHY. The old trajectory is IEA WEO STEPS as carried by the WB EPM Turkiye v7 workbook. STEPS is the highest of the three WEO scenarios (APS 6.0 in 2030, NZE 4.4) because the more the world decarbonises the cheaper gas gets, so the model was already on the most expensive trajectory available off the shelf and a gas-stays-expensive case had to be built by hand. It now has been. The 6.5-in-2030 figure is of a pre-2022 vintage - the World Bank's own April 2021 long-term forecast carried 6.5 for 2035 - and it is roughly half the CMO April 2026 realised print of 12.0 for European gas in 2025. It also requires Turkiye's delivered cost to fall below the European hub price at the same time as the hub-indexed share of its import contracts is rising (40% TTF / 60% oil in 2024, with 55% of contract volume expiring by end-2026). WHICH PRICE LEVEL THIS IS. An import-parity cost, not the BOTAS tariff. Turkiye imports ~99% of its gas (Russia, Azerbaijan, Iran, LNG; Sakarya Black Sea output is still under 10% of need) and BOTAS, the state importer, resells to power plants at a regulated tariff historically below its own cost. Pricing the resource rather than the tariff is the correct economic-cost convention and matches the treatment of Azerbaijan. WHAT IT DOES TO THE RESULTS. The marginal Turkish plant is a generic new CCGT at 5.88 MMBtu/MWh, so the Turkish zonal price moves ~5.9 $/MWh per 1 $/MMBtu of gas. Under the old default the Turkish price halved from 82 to 43 $/MWh between 2025 and 2030, which is what flipped the Georgia-Turkiye corridor around 2029 and made Turkiye a net supplier of the Caucasus for 2030-2033. At 10.0 the implied Turkish price is ~60 $/MWh against ~42 in Georgia, and that flip is not expected to survive. CAVEAT ON READING THE NEW RESULTS. The Georgia-EastAna corridor is rated 700 MW in both directions, which is 6,132 GWh at full load - exactly the flow observed in 2025-2026 under the old default, i.e. the line was already saturated whenever the Turkish price was high. Beyond roughly 9-10 $/MMBtu the model stops arbitrating on price and simply reports the line as full. Results at this gas level are therefore informative about interconnection capacity, not about prices. SENSITIVITY (the old default): supply/pFuelPrice.csv still holds the STEPS trajectory falling to 6.5 by 2030. It is no longer wired into config.csv or into any scenarios.csv column; add it as a pFuelPrice row value on a scenario column to test the gas-relief case. NOTE that the LC_TRGasFlat column of scenarios.csv now points at the file that config.csv already uses, so that column is a no-op until it is repointed at supply/pFuelPrice.csv.

*Confidence: [MEDIUM] · Last updated: 2026-08-25*


<a id="turkiye-pgendatainput"></a>

### `pGenDataInput`

[&#8593; Turkiye](#turkiye)

**Source**: Observed annual capacity additions and growth-constraint benchmarks (build-limit calibration) (`build_rate_benchmarks_2026`)

**Data / file**: Calibration basis for BuildLimitperYear in pGenDataInput and pStorageDataInput. The parameter is a PHYSICAL PLAUSIBILITY GUARD (supply chain, permitting, grid connection, workforce), not a policy targ…

**Also uses**: World Bank EPM Türkiye Least-Cost Model v7 (2025) — fuel prices on IEA WEO scenarios (`wb_epm_turkiye_v7`)

**Also uses**: [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

> ⚠ **Needs review**: This block documents the ANNUAL BUILD LIMITS only. The Turkish fleet itself - its capacities, heat rates and costs - comes from WB EPM Turkiye v7 and is not yet documented here. Two items found while rebuilding the limits and NOT fixed by this edit: the generic Turkish candidates have staggered CODs (PV 2027, biomass 2028, gas 2029, onshore and offshore wind 2030), so PV gets a three-year head start under perfect foresight and saturates before wind exists - the 45.8 GW of PV built in 2027 is driven as much by that asymmetry as by the limit; and Generic Offshore Wind Romania carries tech OnshoreWind, so it shares the Romanian onshore limit instead of having an offshore one of its own.


**Method**: Fleet DIRECT from WB EPM Turkiye v7; ANNUAL BUILD LIMITS constructed, see below

| Period | Method | Notes |
|--------|--------|-------|
| ramp and minimum generation, 2026-08-30 | `CONSTRUCTED` | Ramp rates and minimum generation put on the CCDR default table and switched on. WHY. fApplyRampConstraint and fApplyMinGenShareAllHours were both 0, so neither constraint was in the model, and the values sitting in the columns could not have been used as they stood. The full rule, the equations that read each parameter, the ones that cannot be reached at fDispatchMode = 0, and the reason for each share are in [ramp_mingen_ccdr_2026]. THE RULE. MinGenShareAllHours(z,tech,fuel) = min[ m(tech,fuel) , lowest seasonal availability of the units that (z,tech,fuel) governs ]. The cap can only lower a value, never raise one: eMinGen forces a floor in every hour while base.gms:770 caps seasonal energy at availability x capacity, so a share above a unit's own availability is INFEASIBLE and not merely expensive. The previous table put 0.55 on Turkish domestic lignite against a pAvailabilityCustom of 0.45, 72 units and 12,288 MW, and 0.45 on CCGT in every hour, 148 percent of Armenia's night trough. Ramp rates are the CCDR generic values unchanged: Nuclear 0.15, ST 0.50, the rest 1.00 per hour, PV and wind outside the constraint. The 196 Turkish cells at 0.05 and 0.08 were the textbook per-MINUTE figures loaded into a per-hour parameter; at 0.05 per hour a 600 MW unit needs 20 hours to reach full output. FLAGS. pSettings.csv fApplyRampConstraint and fApplyMinGenShareAllHours set to 1. fApplyMinGenCommitment stays 0 and MinGenCommitment, minUT, minDT and StUpCost stay unread: they need fDispatchMode = 1. Those columns are left in place. Every m is an ASSUMPTION. Method slide: blacksea_2026/RampMinGen_method.pptx, built by RampMinGen_method_slide.py from the deployment's own inputs. RAMP. 196 explicit cells cleared in Turkiye: ST at 0.05 (102), CCGT at 0.05 (52), OCGT at 0.08 (21), Nuclear at 0.05 (15), ICE at 0.05 (6). Every one was an override of the regional default and the region now runs on one table. MIN GEN. MinGenShareAllHours in pGenDataInputDefault, before -> after:<br>  CCGT       Gas            0.45     -> 0.10<br>  ICE        Diesel         blank    -> 0.00<br>  ICE        Gas            blank    -> 0.00<br>  ICE        HFO            blank    -> 0.00<br>  Nuclear    Uranium        0.75     -> 0.70<br>  ST         Coal           0.3      -> 0.25<br>  ST         DomesticCoal   0.55     -> 0.25<br>  ST         Gas            blank    -> 0.00<br>  ST         Geothermal     blank    -> 0.55<br>  ST         ImportedCoal   0.3      -> 0.25<br>  ST         Lignite        0.55     -> 0.25 |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear rebuilt on a stated rule, every country and every generic technology at once. WHY. The column had been filled ad hoc and no longer bound anything: Turkiye carried 90,000 MW/yr of PV nationally (9 zones x 10,000), 45,000 MW/yr each of CCGT and OCGT against 2,933 MW installed, and 440,000 MW/yr of batteries (8 zones x 5 durations x 10,000) against a 64 GW peak. EPM applies the limit PER ROW (main.gms:722, vBuild.up = pGenData(BuildLimitperYear) * pWeightYear), so a generic candidate replicated across nine zones multiplies the national headroom by nine, and five duration variants multiply it again. With y.csv = 2025..2040 consecutive, pWeightYear = 1 and the value is literally MW per year. THE RULE. BuildLimitperYear is a physical plausibility guard - supply chain, permitting, grid connection, workforce - not a policy target and not an economic parameter. It is the growth-constraint form used by the capacity-expansion models that treat this explicitly: PyPSA (max_relative_growth, with max_growth as the seed), MESSAGEix (NEW_CAPACITY_CONSTRAINT_UP, with growth_new_capacity_up and initial_new_capacity_up), and ReEDS, which applies the same idea as a growth penalty rather than a hard bound.<br>    L(c,t,y) = min[ b0(c,t) * gamma_t^(y-2025) , plateau(c,t,y) ]<br>    b0(c,t)  = max( additions observed in 2025 in c , sigma_t * Peak2025(c) )<br>b0 is the rate the country has itself demonstrated, which is the only quantity that transfers between a 75 GW system and a 1.7 GW one: 2025 solar additions ran from 3.9 percent of peak in Azerbaijan to 36 percent in Armenia, a factor of nine, so no single share-of-peak coefficient can serve as the limit. That is what the earlier max[s x Peak, g x Fleet] form got wrong; it gave Turkiye 3.2x its record year and capped Armenia below what Armenia had just installed. sigma is the standing-start seed that lets a technology with no installed base begin; gamma is the annual growth of the build rate; the plateau is the logistic saturation, expressed in the metric of the empirical diffusion literature, annual additions as a share of national electricity supply. The plateau is never set below b0: a country is not capped below a rate it has already achieved. COEFFICIENTS (gamma per year / sigma, share of peak / plateau): PV +20% / 2.0% / 3% of supply; OnshoreWind +20% / 2.0% / 3% of supply; OffshoreWind +20% / 1.0% / 2% of supply; Storage +20% / 2.0% / 5% of peak; CCGT and OCGT +10% / 1.0% / 8% of peak; BiomassPlant +5% / 0.5% / 2% of peak. The observed 2025 additions and the literature anchors are in [build_rate_benchmarks_2026]. Every gamma and every sigma is an ASSUMPTION. EXCLUSIONS. Named projects keep BuildLimit = Capacity, their lead time being already carried by StYr. Hydro, pumped storage, nuclear and geothermal are outside the formula: they are limited by sites, not by build rate. NOTE that tech "ST" holds GEOTHERMAL in this deployment and not coal - the only ST candidates are Geo_Candidate_WestMed, _WestAna and _CenterAna, fuel Geothermal, 265 MW in total - so their limits are unchanged. An earlier draft of this rule pinned ST to zero as a no-new-coal assumption, which would have deleted Turkish geothermal instead. ZONE SPLIT. EPM has no country-level build constraint, so the national limit is divided over the zones carrying a generic candidate. Key = zone share of the pHours-weighted capacity factor from pVREProfile for PV and wind, zone share of peak demand otherwise, floored at 5 percent so no zone is locked out by a marginal resource, then renormalised. Where several rows sit in one zone for one technology they divide that zone's allowance. Keys sum to 1: the national total is exact and no slack factor is applied. TIME. The parameter has no year dimension, one value governs 2025-2040, so the growth path is flattened to its MEAN over the build years 2026-2040. The mean is the flattening that preserves the cumulative headroom the path allows; only the shape is lost. Making the limit genuinely time-varying without touching the GAMS would mean splitting every generic candidate into vintage tranches, about 120 extra rows across pGenDataInput, pStorageDataInput, pAvailabilityCustom and pCapexTrajectoriesCustom. Rejected: the deployment is already near its memory ceiling at 16 years. WHAT THE RULE IS NOT. It is not a forecast. It leaves Turkiye 8,707 MW/yr of PV and 5,448 MW/yr of onshore wind, against a national energy plan pace to 2035 of about 3.2 and 1.7 GW/yr. It is symmetric - it tightens the absurd values and loosens the over-constrained ones - which is what distinguishes it from a targeted cut of Turkiye. Method slide: blacksea_2026/BuildLimit_method.pptx, built by BuildLimit_method_slide.py from the deployment's own inputs. The same compute() writes these values into the CSVs, so the slide and the data cannot diverge. MW/yr, before -> after:<br>  BiomassPlant       900 ->      486<br>  CCGT             45000 ->     6532<br>  OCGT             45000 ->     1496<br>  OffshoreWind      6000 ->     2089<br>  OnshoreWind      15500 ->     5449<br>  PV               90000 ->     8708 |

*Confidence: [MEDIUM] · Last updated: 2026-08-30*


<a id="turkiye-pstoragedatainput"></a>

### `pStorageDataInput`

[&#8593; Turkiye](#turkiye)

**Source**: Observed annual capacity additions and growth-constraint benchmarks (build-limit calibration) (`build_rate_benchmarks_2026`)

**Data / file**: Calibration basis for BuildLimitperYear in pGenDataInput and pStorageDataInput. The parameter is a PHYSICAL PLAUSIBILITY GUARD (supply chain, permitting, grid connection, workforce), not a policy targ…

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

**Method**: GENERIC BESS candidates, 8 zones x 5 durations; build limits constructed

| Period | Method | Notes |
|--------|--------|-------|
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear of the BATTERY candidates recomputed under the same growth rule as pGenDataInput: L(y) = min[ b0 * 1.20^(y-2025) , 5 percent of national peak ], b0 = max( additions observed in 2025 , 2.0 percent of the 2025 peak ), flattened to its mean over the build years 2026-2040, split across zones by peak share and divided between the duration variants of a zone. The rule is stated in full in the pGenDataInput entry of the same date and in [build_rate_benchmarks_2026]. PUMPED HYDRO is carried under tech "Storage" in this deployment but is site-driven, so it is identified by name and excluded from the formula together with the rest of hydro: its build limits are unchanged. NOTE that input_treatment.merge_storage_into_gendata gives this file the last word on a unit present in both files, so storage build limits must be edited here and not in pGenDataInput; Georgia_BESS_Cand sits in both and the pGenDataInput twin is inert. MW/yr, before -> after:<br>  Storage         440000 ->     3734 |

*Confidence: [MEDIUM] · Last updated: 2026-08-30*


---

<a id="armenia"></a>

## Armenia

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pDemandForecast`](#armenia-pdemandforecast) | CESI (World Bank consultant) /… (2022) | [MEDIUM] |
| [`pDemandProfile`](#armenia-pdemandprofile) | proxy of Turkiye/EastAna | [LOW] ⚠ |
| [`pGenDataInput`](#armenia-pgendatainput) | CESI (World Bank consultant) /… (2022) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + EPSO + [Renewable resource potential](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment) + [RE candidate annual build-rate limits](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] |
| [`pFuelPrice`](#armenia-pfuelprice) | South Caucasus bilateral gas c… (2026-08) + [TYNDP / IEA World Energy Outlook 2022](https://www.iea.org/reports/world-energy-outlook-2022) | [MEDIUM] |
| [`pAvailabilityCustom`](#armenia-pavailabilitycustom) | World Nuclear Association (updated annually) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + EPSO | [MEDIUM] ⚠ |
| [`pStorageDataInput`](#armenia-pstoragedatainput) | EPM Generic Defaults | [LOW] |
| [`pVREProfile`](#armenia-pvreprofile) | Renewables Ninja (2018–2023) + TEİAŞ + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | [MEDIUM] ⚠ |

<a id="armenia-pstoragedatainput"></a>

### `pStorageDataInput`

[&#8593; Armenia](#armenia)

**Source**: EPM Generic Defaults (`epm_generic_defaults`)

**Data / file**: Default technical parameters by technology/fuel combination, applied automatically when fields are left blank in pGenDataInput, pAvailabilityCustom, pCapexTrajectories. Stored in epm/resources/pGenDat…

**Method**: GENERIC — candidate storage anchors (no national target yet)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `ASSUMPTION` | BESS 200 MW/4h (Status 3, 2028) + generic PSH 250 MW/8h (2032) — firm candidate wind 500 + PV 300. Generic WB cost/efficiency (BESS 4h, eff 0.85, CapexMWh 250; PSH 8h, eff 0.80). Fixed-capacity anchors (Georgia style), to refine with national storage targets. NB: Türkiye uses unbounded generic BESS. |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear of the BATTERY candidates recomputed under the same growth rule as pGenDataInput: L(y) = min[ b0 * 1.20^(y-2025) , 5 percent of national peak ], b0 = max( additions observed in 2025 , 2.0 percent of the 2025 peak ), flattened to its mean over the build years 2026-2040, split across zones by peak share and divided between the duration variants of a zone. The rule is stated in full in the pGenDataInput entry of the same date and in [build_rate_benchmarks_2026]. PUMPED HYDRO is carried under tech "Storage" in this deployment but is site-driven, so it is identified by name and excluded from the formula together with the rest of hydro: its build limits are unchanged. NOTE that input_treatment.merge_storage_into_gendata gives this file the last word on a unit present in both files, so storage build limits must be edited here and not in pGenDataInput; Georgia_BESS_Cand sits in both and the pGenDataInput twin is inert. MW/yr, before -> after:<br>  Storage            100 ->       77 |

*Confidence: [LOW]*


<a id="armenia-pdemandforecast"></a>

### `pDemandForecast`

[&#8593; Armenia](#armenia)

**Source**: CESI (World Bank consultant) / EPSO-G (Georgia TSO) — Armenia Power Sector PLEXOS Study (2022) (`epso_armenia_plexos_2022`)

**Data / file**: Available in project data folder: Data/Armenia/reply from EPSO/

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

**Data / file**: https://www.renewables.ninja/ API-based hourly capacity factor time series at arbitrary lat/lon. Solar: fixed-tilt 35°, azimuth 180°, 10% system losses, MERRA-2 reanalysis. Wind: Gamesa G114-2000 turb…

**Also uses**: TEİAŞ — Turkiye hourly load data (likely) (`teias_hourly_load`)

**Also uses**: [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/)

> ⚠ **Needs review**: Within-season variability RESOLVED 2026-07-06 — representative-days pipeline rerun; daytypes d1–d7 now carry distinct hourly profiles (verified in pVREProfile.csv). Residual: ROR still proxied from EastAna; confirm underlying PV/Wind data vintage after the rebuild.

**Method**: DIRECT (PV, Wind, Renewables Ninja) — PROXY_EastAna (ROR); mapped onto 28 representative days

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | PV + Wind: Renewables Ninja at Armenia centroid (44.5°E, 40.2°N), mapped onto 28 representative days (7 daytypes d1–d7 × 4 seasons). Since the 2026-07-06 rebuild each daytype carries a distinct hourly profile (within-season variability restored). |
| 2024–2053 | `PROXY_EastAna` | ROR: EastAna Turkiye zone rows copied verbatim — nearest zone, similar snowmelt hydrology |

> Within-season variability restored 2026-07-06 (d1–d7 daytypes now differentiated — previously all daytypes shared one seasonal mean). Residual caveat: ROR proxied from EastAna (no Armenia-specific run-of-river data).

*Confidence: [MEDIUM] · Last updated: 2026-07-06*


<a id="armenia-pavailabilitycustom"></a>

### `pAvailabilityCustom`

[&#8593; Armenia](#armenia)

**Source**: World Nuclear Association — Reactor Database (`wna_reactor_database`)

**Data / file**: https://world-nuclear.org/nuclear-reactor-database/ Per-reactor page: load factor, energy availability, electricity supplied by year. Use Energy Availability (not Load Factor) for pAvailabilityCustom…

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

**Also uses**: EPSO — Armenia 2021–2040 Electricity Power Generation Forecast (`epso_armenia_generation_forecast_2021_2040`)

> ⚠ **Needs review**: TENSION 2026-07-09 — Armenia_ANPP availability raised 0.70→0.84 to reproduce EPSO 2024 actual generation (~3010 GWh). This exceeds the WNA Energy Availability Factor (67–70%): at 407.5 MW net × EAF 0.70 the plant yields ~2510 GWh, not 3010. Likely a post-2021 uprate (~+40 MW, real gross ~448 MW) not captured in the modeled 407.5 MW, or realized output above pure availability. Cleaner long-term fix: reconcile capacity vs availability rather than carry 0.84 on the un-uprated 407.5 MW.


**Method**: DIRECT (Armenia_ANPP) + seasonal reservoir hydro (EPSO-calibrated) — EPM generic for other techs

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Armenia_ANPP: 0.84 flat (Q1–Q4) — CALIBRATION OVERRIDE (2026-07-09) set to reproduce EPSO 2024 actual generation (~3010 GWh, implied CF 0.84 on 407.5 MW net). Was 0.70 (mean of WNA Energy Availability Factor 2023=67.3%, 2024=70.4%); the 0.84 exceeds the WNA EAF — see review note below. Reservoir hydro (Armenia_SHC, Armenia_Vorotan, Armenia_Hydro_Cand) now have explicit seasonal availability (Q1–Q4) instead of the flat 0.85 generic — calibrated to EPSO 2021–2040 annual energy: SHC ~390 GWh (CF ~0.08, irrigation-limited by Lake Sevan releases, concentrated Q2–Q3); Vorotan ~950 GWh (CF ~0.27); Hydro_Cand ~830 GWh. Snowmelt shape proxied from EastAna (adjacent Armenian Highlands, matching regime & CF). Small hydro (ROR) shaped via pVREProfile (EastAna profile). Other gens: generic (CCGT=0.85, PV/Wind=1.0). |

> Nuclear excluded from pAvailabilityGeneric — availability is plant-specific (aging VVER-270). Other countries with nuclear plants should add their own custom entry referencing wna_reactor_database.

*Confidence: [MEDIUM] · Last updated: 2026-07-09*


<a id="armenia-pgendatainput"></a>

### `pGenDataInput`

[&#8593; Armenia](#armenia)

**Source**: CESI (World Bank consultant) / EPSO-G (Georgia TSO) — Armenia Power Sector PLEXOS Study (2022) (`epso_armenia_plexos_2022`)

**Data / file**: Available in project data folder: Data/Armenia/reply from EPSO/

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

**Also uses**: EPSO — Armenia 2021–2040 Electricity Power Generation Forecast (`epso_armenia_generation_forecast_2021_2040`)

**Also uses**: [Renewable resource potential — South Caucasus (solar & wind)](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment)

**Also uses**: [RE candidate annual build-rate limits — Caucasus (modeller assumption)](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/)

**Also uses**: [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

**Method**: DIRECT (capacity, dates from CESI) — tech params from EPM defaults

| Period | Method | Notes |
|--------|--------|-------|
| ramp and minimum generation, 2026-08-30 | `CONSTRUCTED` | Ramp rates and minimum generation put on the CCDR default table and switched on. WHY. fApplyRampConstraint and fApplyMinGenShareAllHours were both 0, so neither constraint was in the model, and the values sitting in the columns could not have been used as they stood. The full rule, the equations that read each parameter, the ones that cannot be reached at fDispatchMode = 0, and the reason for each share are in [ramp_mingen_ccdr_2026]. THE RULE. MinGenShareAllHours(z,tech,fuel) = min[ m(tech,fuel) , lowest seasonal availability of the units that (z,tech,fuel) governs ]. The cap can only lower a value, never raise one: eMinGen forces a floor in every hour while base.gms:770 caps seasonal energy at availability x capacity, so a share above a unit's own availability is INFEASIBLE and not merely expensive. The previous table put 0.55 on Turkish domestic lignite against a pAvailabilityCustom of 0.45, 72 units and 12,288 MW, and 0.45 on CCGT in every hour, 148 percent of Armenia's night trough. Ramp rates are the CCDR generic values unchanged: Nuclear 0.15, ST 0.50, the rest 1.00 per hour, PV and wind outside the constraint. The 196 Turkish cells at 0.05 and 0.08 were the textbook per-MINUTE figures loaded into a per-hour parameter; at 0.05 per hour a 600 MW unit needs 20 hours to reach full output. FLAGS. pSettings.csv fApplyRampConstraint and fApplyMinGenShareAllHours set to 1. fApplyMinGenCommitment stays 0 and MinGenCommitment, minUT, minDT and StUpCost stay unread: they need fDispatchMode = 1. Those columns are left in place. Every m is an ASSUMPTION. Method slide: blacksea_2026/RampMinGen_method.pptx, built by RampMinGen_method_slide.py from the deployment's own inputs. RAMP. 1 explicit cell cleared in Armenia: Nuclear at 0.8 (1). Every one was an override of the regional default and the region now runs on one table. MIN GEN. MinGenShareAllHours in pGenDataInputDefault, before -> after:<br>  CCGT       Gas            0.45     -> 0.10<br>  ICE        Diesel         blank    -> 0.00<br>  ICE        Gas            blank    -> 0.00<br>  ICE        HFO            blank    -> 0.00<br>  Nuclear    Uranium        0.75     -> 0.70<br>  ST         Coal           0.3      -> 0.25<br>  ST         DomesticCoal   0.55     -> 0.25<br>  ST         Gas            blank    -> 0.00<br>  ST         Geothermal     blank    -> 0.55<br>  ST         ImportedCoal   0.3      -> 0.25<br>  ST         Lignite        0.55     -> 0.25 |
| 2024–2053 | `DIRECT` | Capacity and dates (StYr, RetrYr) from CESI PLEXOS study slides. HeatRate, RampUpRate, RampDnRate, ResLimShare, FOMperMW, VOM left blank — filled automatically at runtime from pGenDataInputGeneric (EPM parameter guide: https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/). Nuclear and ST/Gas defaults added to pGenDataInputGeneric for this deployment. |
| plant lifetimes, 2026-08-29 | `ASSUMPTION` | RetrYr cleanup. Two implicit conventions coexisted in pGenDataInput.csv: a blank RetrYr, and a mechanical StYr + 25/30/50 with Life left blank. Both are unsafe. A blank RetrYr does NOT mean "never retires": main.gms:832 only pins vCap when RetrYr >= y, so the plant keeps a free vCap and an unbounded vRetire and can be retired at zero cost even though fEnableEconomicRetirement = 0. The StYr + N dates were a placeholder formula, not a published schedule, and were retiring the Caucasus gas fleet at ages no operator applies. RULE APPLIED (Life and RetrYr columns only, Status 1 and 2 only): a row is touched if Life is blank AND RetrYr - StYr is exactly 25, 30 or 50 AND RetrYr <= 2040, or if RetrYr is blank. It then receives an explicit Life — CCGT 40 yr (mid-life hot-gas- path retrofit), OCGT 35 yr, hydro 80 yr — and RetrYr = StYr + Life. Hydro is floored at RetrYr 2060, beyond the 2025-2040 horizon: the civil works are the asset and every pre-1980 unit in these fleets operates today after rehabilitation, so StYr + 80 alone would have retired plants in the past (Zahesi 1927 -> 2007, Rioni 1933 -> 2013). A row whose RetrYr is a published plan date, or whose age already exceeds the assumed life, is left untouched. PV and wind keep 25 yr, their real design life. The lifetimes themselves are modeller assumptions — see [epm_expert_judgment]; no decommissioning schedule is published for these fleets. Replace on sight if one is. THIS COUNTRY: Armenia_Yerevan_CCGT1 228.6 MW 2040 -> 2050 (Life 40) and Armenia_Vorotan 404.2 MW 2039 -> 2069 (Life 80). Available capacity in 2040 +632.8 MW. Untouched: Armenia_ANPP 407.5 MW retires 2036 (Life 60 explicit, Rosatom life extension); Armenia_Hrazdan_ST 2030 (mothballed, Capacity 0); Armenia_SHC 2047 (post-ADB rehabilitation life extension). |
| candidate set completion, 2026-08-29 | `ASSUMPTION` | Missing expansion options. The Caucasus zones had no dispatchable candidate at all (Georgia and Armenia) or a single named project as their only one (Azerbaijan), so their import dependence and their generation mix were imposed by the candidate list rather than produced by an economic arbitration — a result that cannot be defended in review. Options added below; each is an OPTION the optimiser may decline, not a committed project. Sizing follows the existing fleet, not a published plan: ceilings are deliberately modest and the annual build limit is the binding constraint, as elsewhere in this deployment. Tech and cost parameters are inherited from pGenDataInputDefault (CCGT capex 0.9 M$/MW, HeatRate 6.4, Life 30; OCGT 0.8 and 9.0; offshore wind 3.0 M$/MW, FOM 70000). Candidate Life stays at the generic 30 yr and is NOT aligned with the 40 yr technical life adopted for the existing CCGT fleet in the plant-lifetimes entry: the former drives annuitisation, the latter retirement, and changing the annuity is an economic decision outside this edit. Open item. THIS COUNTRY: Armenia_CCGT_Cand 250 MW, COD 2030, build limit 250 MW/yr — sized on the existing Yerevan CCGT (228.6 MW). Armenia previously had exactly one dispatchable candidate, Armenia_Nuclear_Cand 300 MW at COD 2036, so between 2030 and 2036 the only new firm capacity available to the model was storage. If the SMR proves costly or late, the model now has a gas alternative to price it against rather than being forced into imports. |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear rebuilt on a stated rule, every country and every generic technology at once. WHY. The column had been filled ad hoc and no longer bound anything: Turkiye carried 90,000 MW/yr of PV nationally (9 zones x 10,000), 45,000 MW/yr each of CCGT and OCGT against 2,933 MW installed, and 440,000 MW/yr of batteries (8 zones x 5 durations x 10,000) against a 64 GW peak. EPM applies the limit PER ROW (main.gms:722, vBuild.up = pGenData(BuildLimitperYear) * pWeightYear), so a generic candidate replicated across nine zones multiplies the national headroom by nine, and five duration variants multiply it again. With y.csv = 2025..2040 consecutive, pWeightYear = 1 and the value is literally MW per year. THE RULE. BuildLimitperYear is a physical plausibility guard - supply chain, permitting, grid connection, workforce - not a policy target and not an economic parameter. It is the growth-constraint form used by the capacity-expansion models that treat this explicitly: PyPSA (max_relative_growth, with max_growth as the seed), MESSAGEix (NEW_CAPACITY_CONSTRAINT_UP, with growth_new_capacity_up and initial_new_capacity_up), and ReEDS, which applies the same idea as a growth penalty rather than a hard bound.<br>    L(c,t,y) = min[ b0(c,t) * gamma_t^(y-2025) , plateau(c,t,y) ]<br>    b0(c,t)  = max( additions observed in 2025 in c , sigma_t * Peak2025(c) )<br>b0 is the rate the country has itself demonstrated, which is the only quantity that transfers between a 75 GW system and a 1.7 GW one: 2025 solar additions ran from 3.9 percent of peak in Azerbaijan to 36 percent in Armenia, a factor of nine, so no single share-of-peak coefficient can serve as the limit. That is what the earlier max[s x Peak, g x Fleet] form got wrong; it gave Turkiye 3.2x its record year and capped Armenia below what Armenia had just installed. sigma is the standing-start seed that lets a technology with no installed base begin; gamma is the annual growth of the build rate; the plateau is the logistic saturation, expressed in the metric of the empirical diffusion literature, annual additions as a share of national electricity supply. The plateau is never set below b0: a country is not capped below a rate it has already achieved. COEFFICIENTS (gamma per year / sigma, share of peak / plateau): PV +20% / 2.0% / 3% of supply; OnshoreWind +20% / 2.0% / 3% of supply; OffshoreWind +20% / 1.0% / 2% of supply; Storage +20% / 2.0% / 5% of peak; CCGT and OCGT +10% / 1.0% / 8% of peak; BiomassPlant +5% / 0.5% / 2% of peak. The observed 2025 additions and the literature anchors are in [build_rate_benchmarks_2026]. Every gamma and every sigma is an ASSUMPTION. EXCLUSIONS. Named projects keep BuildLimit = Capacity, their lead time being already carried by StYr. Hydro, pumped storage, nuclear and geothermal are outside the formula: they are limited by sites, not by build rate. NOTE that tech "ST" holds GEOTHERMAL in this deployment and not coal - the only ST candidates are Geo_Candidate_WestMed, _WestAna and _CenterAna, fuel Geothermal, 265 MW in total - so their limits are unchanged. An earlier draft of this rule pinned ST to zero as a no-new-coal assumption, which would have deleted Turkish geothermal instead. ZONE SPLIT. EPM has no country-level build constraint, so the national limit is divided over the zones carrying a generic candidate. Key = zone share of the pHours-weighted capacity factor from pVREProfile for PV and wind, zone share of peak demand otherwise, floored at 5 percent so no zone is locked out by a marginal resource, then renormalised. Where several rows sit in one zone for one technology they divide that zone's allowance. Keys sum to 1: the national total is exact and no slack factor is applied. TIME. The parameter has no year dimension, one value governs 2025-2040, so the growth path is flattened to its MEAN over the build years 2026-2040. The mean is the flattening that preserves the cumulative headroom the path allows; only the shape is lost. Making the limit genuinely time-varying without touching the GAMS would mean splitting every generic candidate into vintage tranches, about 120 extra rows across pGenDataInput, pStorageDataInput, pAvailabilityCustom and pCapexTrajectoriesCustom. Rejected: the deployment is already near its memory ceiling at 16 years. WHAT THE RULE IS NOT. It is not a forecast. It leaves Turkiye 8,707 MW/yr of PV and 5,448 MW/yr of onshore wind, against a national energy plan pace to 2035 of about 3.2 and 1.7 GW/yr. It is symmetric - it tightens the absurd values and loosens the over-constrained ones - which is what distinguishes it from a targeted cut of Turkiye. Method slide: blacksea_2026/BuildLimit_method.pptx, built by BuildLimit_method_slide.py from the deployment's own inputs. The same compute() writes these values into the CSVs, so the slide and the data cannot diverge. MW/yr, before -> after:<br>  CCGT               250 ->      467<br>  OnshoreWind        100 ->      121<br>  PV                 400 ->      615 |

> 15 generators: 12 Status-1 (existing), 2 Status-2 (committed), 4 Status-3 (candidates). DOUBT — Armenia_Hrazdan_ST: units 1–4 all mothballed per GEM (2024); Capacity set to 0 MW pending confirmation. CESI shows 300–410 MW nameplate but none dispatchable. DOUBT — Armenia_SHC RetrYr=2047: no explicit retirement date found; estimated from post-ADB rehabilitation life extension (loan matures 2029). Commissioned 1960–1962. NUCLEAR POST-2036 — Armenia_Nuclear_Cand: Metsamor/ANPP (407.5 MW VVER-440) is life- extended to 2036 (Rosatom $65M contract Dec-2024, plant shut Apr-2026 for upgrades) and retires 2036 in the model. Added a 300 MW Nuclear candidate (Status 3, COD 2036, Life 50) to replace it. Sized as one Western SMR (BWRX-300/AP300 class): in 2026 Armenia decided the successor will be a SMR (US/RU/CN/KR/FR bids, final choice by 2027), dropping the earlier 1000–1200 MW large-reactor plan. 300 MW @ CF~0.9 ≈ 2.4 TWh, matching current nuclear output. NB: EPSO 2021-2040 forecast shows ANPP energy jumping to ~7 TWh from 2036 (~1000 MW) — that reflects the now-abandoned large-reactor plan, NOT modelled in base; keep as a high-nuclear NDP variant if needed. Cost/tech params from Generic_Nuclear template (HeatRate 12.5, Capex 8.0, FOM 150000, VOM 3.5). GENERIC RE CANDIDATE DESIGN (applies to Armenia, Georgia, AzerbaijanMain, Nakhchivan; see also each country's pGenDataInput/pStorageDataInput block): named/committed projects stay as fixed candidates; generic solar/wind/BESS expansion candidates are UNBOUNDED by a resource ceiling (Capacity = sourced technical/economic potential, see [re_resource_potential_caucasus]) and constrained only by an annual BuildLimitperYear (see [re_candidate_build_limits]). Hydro & nuclear stay bounded (finite sites / lumpy). NDP target trajectories are imposed via scenarios.csv, NOT as least-cost caps. Adopted ceilings (MW): Armenia PV 8000 / Wind 3000; Georgia PV 4500 / Wind 4000 (GSE conservative); AzerbaijanMain PV 23000 / Wind 3000; Nakhchivan PV 2000 / Wind 1000 (assumption). BESS candidates converted to unbounded (Capacity 2000/5000/500, BuildLimit 100/200/30 for ARM/AZE/Nakh). FIXED 2026-08-24: Armenia_Hydro_Cand (350 MW, COD 2035) had a blank BuildLimitperYear and was therefore unbuildable (vBuild.up = 0, main.gms:722); set to 350 = its Capacity. Its Capex was also blank and, unlike the Georgian cases, had no donor plant for EPM_FILL_HYDRO_CAPEX to average, so it stayed undefined; set explicitly to 3.3, the Armenia ReservoirHydro default. This makes a 350 MW candidate available to the model that was silently absent from every previous run, so Armenian results will move.

*Confidence: [MEDIUM] · Last updated: 2026-08-30*


<a id="armenia-pfuelprice"></a>

### `pFuelPrice`

[&#8593; Armenia](#armenia)

**Source**: South Caucasus bilateral gas contract prices (Armenia–Gazprom, Georgia marginal import), 2026 review (`caucasus_gas_contracts_2026`)

**Data / file**: Not a downloadable dataset. Contract prices compiled from public reporting and trade statistics during the 2026-08 gas assumptions review for the Black Sea study. ARMENIA: the Armenia–Gazprom border p…

**Also uses**: [TYNDP / IEA World Energy Outlook 2022 — commodity prices](https://www.iea.org/reports/world-energy-outlook-2022)

**Method**: Gas: DIRECT from the Armenia-Gazprom border contract, stepwise. HFO/Uranium unchanged.

| Period | Method | Notes |
|--------|--------|-------|
| 2024-2025 | `DIRECT` | Gas: 165 USD/1000m3, the Armenia-Gazprom border price, frozen at that level from 2019 through 2025. At 36 MMBtu/1000m3 (HHV) this is 4.583 USD/MMBtu. |
| 2026-2030 | `DIRECT` | Gas: 177.5 USD/1000m3, the 2026 contract revision = 4.931 USD/MMBtu, held flat to 2030. The contract price moves in negotiated steps, not along a slope. |
| 2031-2053 | `ASSUMPTION` | Gas: +5% per five-year block applied to the 2026 contract price, mirroring the historical renegotiation cadence. 2031-35 = 5.177, 2036-40 = 5.436, 2041-45 = 5.708, 2046-50 = 5.993, 2051-53 = 6.293 USD/MMBtu. Assumes the Gazprom political discount to Armenia erodes slowly rather than converging on European hub prices. |
| 2024-2053 | `DIRECT` | HFO (6.468 flat) and Uranium (0.97, proxied flat from Turkiye) are unchanged from the previous TYNDP/IEA WEO basis. Only Gas was rebased. Coal and Lignite excluded, no coal generation in Armenia existing or planned. |

> REBASED 2026-08-24. Gas previously came from tyndp_iea_weo_2022, a European market projection: 4.715 in 2024 rising to 5.533 by 2053. The LEVEL happened to be right, 4.715 modelled against 4.583 actually paid, but the MECHANISM was wrong. Armenia buys under a bilateral Gazprom contract, not on a European market, and that contract moves in steps. The previous note acknowledged the divergence itself. The correction barely moves the numbers; its purpose is that the assumption now be verifiable in review, since the border price is public whereas the TYNDP trajectory describes a market Armenia is not connected to. Armenia also barters with Iran at 1 m3 of gas for 3 kWh of electricity (Hrazdan-5, Yerevan CCGT). The implied heat rate of 12 is the Iranian margin on the swap, not a plant efficiency, and is represented through the iran_swap zone rather than here. UNITS: 36 MMBtu/1000m3 is an HHV basis, consistent with the EPM convention efficiency = 3.412/HeatRate. On LHV (~33.5) the same contract price gives 4.93 in 2024, a +7.5% systematic shift on every gas SRMC. The HHV/LHV convention is an open item model-wide, not specific to Armenia.

*Confidence: [MEDIUM] · Last updated: 2026-08-24*


---

<a id="georgia"></a>

## Georgia

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pDemandForecast`](#georgia-pdemandforecast) | World Bank (internal) (2022) | [MEDIUM] ⚠ |
| [`pDemandProfile`](#georgia-pdemandprofile) | World Bank (internal) (2022) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | [MEDIUM] |
| [`pGenDataInput`](#georgia-pgendatainput) | SESA (Georgian Power Sector An… (2022-07-01) + Georgia Power Sector Data Repository (WB Internal) + World Bank EPM Georgia v8.5 (2022, internal model) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + [Renewable resource potential](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment) + [RE candidate annual build-rate limits](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/) + [GSE Ten-Year Network Development Plan of Georgia (TYNDP 2023-2033)](https://www.gse.com.ge/komunikacia/publikaciebi/saqartvelos-gadamcemi-qselis-ganvitarebis-atwliani-gegma) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] ⚠ |
| [`pFuelPrice`](#georgia-pfuelprice) | South Caucasus bilateral gas c… (2026-08) + IMF Energy Subsidies Database | [MEDIUM] |
| [`pAvailabilityCustom`](#georgia-pavailabilitycustom) | World Bank EPM Georgia v8.5 (2… (2022) + Georgia Hourly Generation Profiles by Technology 2019–2022 | [MEDIUM] ⚠ |
| [`pStorageDataInput`](#georgia-pstoragedatainput) | EPM Generic Defaults | [LOW] ⚠ |
| [`pVREProfile`](#georgia-pvreprofile) | World Bank EPM Georgia 2022 (i… (2022) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | [MEDIUM] ⚠ |
| [`pMaxFuelLimit`](#georgia-pmaxfuellimit) | Modeller expert judgment (2026) | [LOW] ⚠ |

<a id="georgia-pstoragedatainput"></a>

### `pStorageDataInput`

[&#8593; Georgia](#georgia)

**Source**: EPM Generic Defaults (`epm_generic_defaults`)

**Data / file**: Default technical parameters by technology/fuel combination, applied automatically when fields are left blank in pGenDataInput, pAvailabilityCustom, pCapexTrajectories. Stored in epm/resources/pGenDat…

> ⚠ **Needs review**: Storage ceilings across the four Caucasus zones are anchors chosen for internal consistency, not national targets. Replace with GSE / Ministry of Energy storage planning figures when published. The Georgian PSH candidate in particular should be matched to a real site (the Enguri cascade is the obvious screening candidate) before any result depending on it is reported.


**Method**: GENERIC — candidate storage anchors (no national target yet)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `ASSUMPTION` | BESS 4h (Status 3, 2028) + generic PSH 250 MW/8h (2033). Generic WB cost/efficiency (BESS 4h, eff 0.85, CapexMWh 250; PSH 8h, eff 0.80). Fixed-capacity anchors, to refine with a national storage target. NB: Türkiye uses unbounded generic BESS. |
| candidate set completion, 2026-08-29 | `ASSUMPTION` | Georgia was the least-equipped zone for balancing while being the one that needs it most: BESS ceiling 200 MW at 50 MW/yr against 2000 MW / 100 MW-yr in Armenia and 5000 MW / 200 MW-yr in AzerbaijanMain, and no pumped-storage candidate at all while Armenia had one and Türkiye four. The asymmetry was not sourced. BESS raised to 1000 MW / 100 MW-yr (4000 MWh, 4h kept) and Georgia_PSH_Cand 250 MW / 2000 MWh added at COD 2033, a copy of Armenia_PSH_Cand with the same generic costs. Both are options the optimiser may decline. The PSH siting is generic: no specific Georgian scheme is implied, and the 250 MW is an anchor, not a project. |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear of the BATTERY candidates recomputed under the same growth rule as pGenDataInput: L(y) = min[ b0 * 1.20^(y-2025) , 5 percent of national peak ], b0 = max( additions observed in 2025 , 2.0 percent of the 2025 peak ), flattened to its mean over the build years 2026-2040, split across zones by peak share and divided between the duration variants of a zone. The rule is stated in full in the pGenDataInput entry of the same date and in [build_rate_benchmarks_2026]. PUMPED HYDRO is carried under tech "Storage" in this deployment but is site-driven, so it is identified by name and excluded from the formula together with the rest of hydro: its build limits are unchanged. NOTE that input_treatment.merge_storage_into_gendata gives this file the last word on a unit present in both files, so storage build limits must be edited here and not in pGenDataInput; Georgia_BESS_Cand sits in both and the pGenDataInput twin is inert. MW/yr, before -> after:<br>  Storage            100 ->      148 |

*Confidence: [LOW] · Last updated: 2026-08-30*


<a id="georgia-pgendatainput"></a>

### `pGenDataInput`

[&#8593; Georgia](#georgia)

**Source**: SESA (Georgian Power Sector Analysis System) / World Bank — Georgia Power Plant Inventory (July 2022) (`sesa_georgia_2022`)

**Data / file**: EPM_Georgia/2022/1. Data/Generation_1.07.2022.xlsx

**Also uses**: Georgia Power Sector Data Repository (WB Internal) (`ge_power_sector_data_repository`)

**Also uses**: World Bank EPM Georgia v8.5 (2022, internal model) — primary data sources not documented (`wb_epm_georgia_v85`)

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

**Also uses**: [Renewable resource potential — South Caucasus (solar & wind)](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment)

**Also uses**: [RE candidate annual build-rate limits — Caucasus (modeller assumption)](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/)

**Also uses**: [GSE Ten-Year Network Development Plan of Georgia (TYNDP 2023-2033) — generation pipeline](https://www.gse.com.ge/komunikacia/publikaciebi/saqartvelos-gadamcemi-qselis-ganvitarebis-atwliani-gegma)

**Also uses**: [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

> ⚠ **Needs review**: (1) RESOLVED 2026-07-02 — Large hydro reclassified using gse_tyndp_georgia + 2025-26 news: Khudoni (702, absent from TYNDP, zombie since 1989) and Namakhvani (433, ENKA contract terminated + arbitration, TYNDP COD 2035) DOWNGRADED to candidates (COD 2035). Nenskra (280, only project with real sponsor K-water + IFI finance) kept committed, COD 2029->2032. (2) Tbilsresi CCGT (1963): 60+ year old plant, RetrYr=2027 estimated — confirm operational status with CESI/GSE. (3) Kirnati capacity discrepancy: sesa_georgia_2022 shows 27.47 MW, ge_power_sector_data_repository shows 51.22 MW — used sesa_georgia_2022 value. (4) Mtkvari VOM=0.06 $/MWh from wb_epm_georgia_v85 is unusually low — verify. (5) Tbilsresi labeled CCGT in data sources but 1963 vintage — likely old steam turbine. (6) RESOLVED — DomesticCoal for Tkibuli now priced in pFuelPrice (Georgia/DomesticCoal = 3.82 USD/MMBtu, DIRECT from georgia_fuel_subsidies_2022, power-sector coal price 2021). (7) OPEN 2026-08-24 — candidate RoR Capex is the EPM generic ROR default (2.8), not a per-project estimate. A capex sweep (2.8 / 2.3 / 2.0) is planned to report the threshold at which the pipeline builds; per-project costs would be better. (8) OPEN 2026-08-24 — BuildLimitperYear = Capacity lets any single project be built in one year. Fine per project, but nothing stops all 32 being built the same year. If a realistic national build rate matters, add pAnnualMaxBuildZ (present in extras/ but not wired into config.csv).


**Method**: DIRECT (capacity, dates, tech) — old EPM for HeatRate thermal — generic for all other params

| Period | Method | Notes |
|--------|--------|-------|
| ramp and minimum generation, 2026-08-30 | `CONSTRUCTED` | Ramp rates and minimum generation put on the CCDR default table and switched on. WHY. fApplyRampConstraint and fApplyMinGenShareAllHours were both 0, so neither constraint was in the model, and the values sitting in the columns could not have been used as they stood. The full rule, the equations that read each parameter, the ones that cannot be reached at fDispatchMode = 0, and the reason for each share are in [ramp_mingen_ccdr_2026]. THE RULE. MinGenShareAllHours(z,tech,fuel) = min[ m(tech,fuel) , lowest seasonal availability of the units that (z,tech,fuel) governs ]. The cap can only lower a value, never raise one: eMinGen forces a floor in every hour while base.gms:770 caps seasonal energy at availability x capacity, so a share above a unit's own availability is INFEASIBLE and not merely expensive. The previous table put 0.55 on Turkish domestic lignite against a pAvailabilityCustom of 0.45, 72 units and 12,288 MW, and 0.45 on CCGT in every hour, 148 percent of Armenia's night trough. Ramp rates are the CCDR generic values unchanged: Nuclear 0.15, ST 0.50, the rest 1.00 per hour, PV and wind outside the constraint. The 196 Turkish cells at 0.05 and 0.08 were the textbook per-MINUTE figures loaded into a per-hour parameter; at 0.05 per hour a 600 MW unit needs 20 hours to reach full output. FLAGS. pSettings.csv fApplyRampConstraint and fApplyMinGenShareAllHours set to 1. fApplyMinGenCommitment stays 0 and MinGenCommitment, minUT, minDT and StUpCost stay unread: they need fDispatchMode = 1. Those columns are left in place. Every m is an ASSUMPTION. Method slide: blacksea_2026/RampMinGen_method.pptx, built by RampMinGen_method_slide.py from the deployment's own inputs. RAMP. No explicit cell in Georgia; the zone reads pGenDataInputDefault. MIN GEN. MinGenShareAllHours in pGenDataInputDefault, before -> after:<br>  CCGT       Gas            0.45     -> 0.10<br>  ICE        Diesel         blank    -> 0.00<br>  ICE        Gas            blank    -> 0.00<br>  ICE        HFO            blank    -> 0.00<br>  Nuclear    Uranium        0.75     -> 0.70<br>  ST         Coal           0.3      -> 0.25<br>  ST         DomesticCoal   0.55     -> 0.00<br>  ST         Gas            blank    -> 0.00<br>  ST         Geothermal     blank    -> 0.55<br>  ST         ImportedCoal   0.3      -> 0.25<br>  ST         Lignite        0.55     -> 0.25 |
| 2024–2053 | `DIRECT` | 113 plants from sesa_georgia_2022 reduced to 46 rows: plants ≥10 MW kept individual; plants <10 MW aggregated into Georgia_AGG_SmallHydro (~224 MW). Capacity: sesa_georgia_2022. StYr: ge_power_sector_data_repository (commissioning year per plant). tech: mapped from Status column (with Reservoir→ReservoirHydro, Seasonal/Small→ROR) cross-checked with ge_power_sector_data_repository type column. HeatRate for Mtkvari (10.3 MMBtu/MWh) and Gardabani CCGT (6.93 MMBtu/MWh) from wb_epm_georgia_v85. All other technical params (VOM, FOM, Capex, RampRate, ResLimShare, Life) left blank → filled at runtime from pGenDataInputGeneric (EPM generic defaults). |
| committed | `DIRECT` | Committed rows (Status=2): Nenskra 280 MW (StYr=2032) and Georgia_HydroSHP_Com 549 MW aggregate (near-term hydro in construction, confirmed by gse_tyndp_georgia hydro-<=2027 = 552 MW). UPDATED 2026-07-02: Khudoni (702) and Namakhvani (433) were DOWNGRADED from committed to candidates (see candidates period + review_note); Nenskra COD moved 2029->2032. |
| candidates | `DIRECT` | RE candidates now debridaged (see re_resource_potential_caucasus / re_candidate_build_limits): generic Wind (cap 4000, build 150/yr) & PV (cap 4500, build 400/yr); BESS 200 MW. HYDRO candidates rebuilt from gse_tyndp_georgia (TYNDP 2023-2033, revised CODs): the old generic SmallHydro_Cand (300 MW) was REPLACED by 31 named RoR/Seasonal projects >=10 MW (COD 2028-2033) + one <10 MW aggregate (Georgia_AGG_SmallHydro_Cand) = ~1025 MW, Capex 3.3, tech RoR->ROR / Seasonal->ReservoirHydro. Capex was 3.3 M$/MW for all of them until 2026-08-24, now 2.8 - see the capex sweep period below. Plus the two downgraded large reservoirs available as candidates: Khudoni 702 (COD 2035), Namakhvani 433 (COD 2035). Wind pipeline NOT imported (TYNDP wind = identical template placeholders) -> generic kept. |
| candidate hydro capex + build limits, 2026-08-24 | `ASSUMPTION` | Two defects found together while preparing a capex sweep, both silent. (A) BUILD LIMITS. All 32 Georgian hydro candidates carried a blank BuildLimitperYear. main.gms:722 sets vBuild.up = BuildLimitperYear * pWeightYear, and the Georgia ROR / ReservoirHydro rows of pGenDataInputDefault.csv also carry 0, so the upper bound was pinned at zero: NONE of the pipeline could be built, at any capex. A sweep would have returned "nothing builds" at every rung and been read as "Georgian hydro is uneconomic". FIXED 2026-08-24: BuildLimitperYear = Capacity on all 32, the convention already used by Khudoni/Namakhvani, by every Turkish hydro candidate, and by preprocessing.perform_sensitivity when it removes a build-rate limit. vCap.up = Capacity still caps the total, so each project can be built once, in one year, whenever it is economic - which matches the stated design in this block (hydro bounded by a finite site list, not by an annual rate). The guard that would have caught this, input_verification._check_candidate_build_limits, was commented out; it has been sharpened (error only for Status 3 rows that declare a Capacity; committed rows are exempt because main.gms:727-729 ignores their build limit) and re-enabled. It aborts the run only where the deployment opts in through the new EPM_STRICT_BUILD_LIMITS setting, which every data_blacksea pSettings variant now sets to 1; elsewhere it only warns, because other deployments in the repo (data_test among them) ship candidates with blank build limits and this study is not the place to break their runs. (B) CAPEX. The 30 named RoR candidates carried Capex 3.3 M$/MW - the generic ReservoirHydro value - although pGenDataInputDefault.csv gives ROR 2.8 in every zone of the deployment. CHANGED to 2.8, i.e. they now sit on the deployment own generic ROR capex rather than on a value belonging to another technology. Three further rows had a BLANK Capex and were being filled at runtime by EPM_FILL_HYDRO_CAPEX with the MEAN capex of the other plants of the same (zone, tech) - for Georgia ReservoirHydro that meant the mean of Khudoni 1.6, Namakhvani 1.9 and Nenskra 3.5 = 2.333, a number with no meaning, and it coupled a committed plant to the candidates so any sweep moved its cost too. All three are now explicit: Georgia_TekhuraCascade_Cand and Georgia_IltoAlazani_Cand at 3.3 (the Georgia ReservoirHydro default, their own tech), Georgia_HydroSHP_Com at 2.8 (the Georgia ROR default). Every hydro row in the deployment now carries an explicit Capex, so the auto-fill has nothing left to infer. NOT TOUCHED: Khudoni 1.6, Namakhvani 1.9, Nenskra 3.5, which have real project costs behind them, and the pGenDataInputDefault Georgia rows themselves. |
| plant lifetimes, 2026-08-29 | `ASSUMPTION` | RetrYr cleanup. Two implicit conventions coexisted in pGenDataInput.csv: a blank RetrYr, and a mechanical StYr + 25/30/50 with Life left blank. Both are unsafe. A blank RetrYr does NOT mean "never retires": main.gms:832 only pins vCap when RetrYr >= y, so the plant keeps a free vCap and an unbounded vRetire and can be retired at zero cost even though fEnableEconomicRetirement = 0. The StYr + N dates were a placeholder formula, not a published schedule, and were retiring the Caucasus gas fleet at ages no operator applies. RULE APPLIED (Life and RetrYr columns only, Status 1 and 2 only): a row is touched if Life is blank AND RetrYr - StYr is exactly 25, 30 or 50 AND RetrYr <= 2040, or if RetrYr is blank. It then receives an explicit Life — CCGT 40 yr (mid-life hot-gas- path retrofit), OCGT 35 yr, hydro 80 yr — and RetrYr = StYr + Life. Hydro is floored at RetrYr 2060, beyond the 2025-2040 horizon: the civil works are the asset and every pre-1980 unit in these fleets operates today after rehabilitation, so StYr + 80 alone would have retired plants in the past (Zahesi 1927 -> 2007, Rioni 1933 -> 2013). A row whose RetrYr is a published plan date, or whose age already exceeds the assumed life, is left untouched. PV and wind keep 25 yr, their real design life. The lifetimes themselves are modeller assumptions — see [epm_expert_judgment]; no decommissioning schedule is published for these fleets. Replace on sight if one is. THIS COUNTRY: 33 hydro rows carried a blank RetrYr and now carry Life 80 and an explicit RetrYr, the pre-1980 fleet at the 2060 floor (Enguri 1300 MW, Vardnili 220, Vartsikhe 184, AGG_SmallHydro 224, Khrami 1 and 2, Lajanuri, Gumati, Zahesi, Rioni, Ortachala, Tetrikhevi, Dzevruli, Shaori, Bjuja, Chitakhevi, Atshesi). No Georgian capacity moves inside the horizon: none of these was retiring anyway, the edit only closes the free-retirement hole described above. Untouched: Georgia_Tbilsresi_CCGT 270 MW retires 2027 (age 64) and Georgia_Mtkvari_CCGT 300 MW 2030 (age 39), both well past 40 yr; Georgia_GPower_OCGT 110 MW 2031 and Georgia_Tkibuli_ST 13.2 MW 2036, which already carried an explicit Life 25. |
| candidate set completion, 2026-08-29 | `ASSUMPTION` | Missing expansion options. The Caucasus zones had no dispatchable candidate at all (Georgia and Armenia) or a single named project as their only one (Azerbaijan), so their import dependence and their generation mix were imposed by the candidate list rather than produced by an economic arbitration — a result that cannot be defended in review. Options added below; each is an OPTION the optimiser may decline, not a committed project. Sizing follows the existing fleet, not a published plan: ceilings are deliberately modest and the annual build limit is the binding constraint, as elsewhere in this deployment. Tech and cost parameters are inherited from pGenDataInputDefault (CCGT capex 0.9 M$/MW, HeatRate 6.4, Life 30; OCGT 0.8 and 9.0; offshore wind 3.0 M$/MW, FOM 70000). Candidate Life stays at the generic 30 yr and is NOT aligned with the 40 yr technical life adopted for the existing CCGT fleet in the plant-lifetimes entry: the former drives annuitisation, the latter retirement, and changing the annuity is an economic decision outside this edit. Open item. THIS COUNTRY: Georgia_CCGT_Cand 300 MW, COD 2029 (gauge: Gardabani 1 and 2, 230 MW each) and Georgia_OCGT_Cand 100 MW, COD 2029, both with build limit equal to the ceiling. Georgia had NO thermal candidate whatsoever while losing Tbilsresi 270 MW (2027), Mtkvari 300 MW (2030) and GPower 110 MW (2031) — 680 MW of dispatchable capacity that the model was structurally unable to replace. The 300 MW ceiling is a near-term bound, not a resource limit: it lets the model price one Gardabani-class unit against imports and against hydro, and should be revisited if it binds in the results. |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear rebuilt on a stated rule, every country and every generic technology at once. WHY. The column had been filled ad hoc and no longer bound anything: Turkiye carried 90,000 MW/yr of PV nationally (9 zones x 10,000), 45,000 MW/yr each of CCGT and OCGT against 2,933 MW installed, and 440,000 MW/yr of batteries (8 zones x 5 durations x 10,000) against a 64 GW peak. EPM applies the limit PER ROW (main.gms:722, vBuild.up = pGenData(BuildLimitperYear) * pWeightYear), so a generic candidate replicated across nine zones multiplies the national headroom by nine, and five duration variants multiply it again. With y.csv = 2025..2040 consecutive, pWeightYear = 1 and the value is literally MW per year. THE RULE. BuildLimitperYear is a physical plausibility guard - supply chain, permitting, grid connection, workforce - not a policy target and not an economic parameter. It is the growth-constraint form used by the capacity-expansion models that treat this explicitly: PyPSA (max_relative_growth, with max_growth as the seed), MESSAGEix (NEW_CAPACITY_CONSTRAINT_UP, with growth_new_capacity_up and initial_new_capacity_up), and ReEDS, which applies the same idea as a growth penalty rather than a hard bound.<br>    L(c,t,y) = min[ b0(c,t) * gamma_t^(y-2025) , plateau(c,t,y) ]<br>    b0(c,t)  = max( additions observed in 2025 in c , sigma_t * Peak2025(c) )<br>b0 is the rate the country has itself demonstrated, which is the only quantity that transfers between a 75 GW system and a 1.7 GW one: 2025 solar additions ran from 3.9 percent of peak in Azerbaijan to 36 percent in Armenia, a factor of nine, so no single share-of-peak coefficient can serve as the limit. That is what the earlier max[s x Peak, g x Fleet] form got wrong; it gave Turkiye 3.2x its record year and capped Armenia below what Armenia had just installed. sigma is the standing-start seed that lets a technology with no installed base begin; gamma is the annual growth of the build rate; the plateau is the logistic saturation, expressed in the metric of the empirical diffusion literature, annual additions as a share of national electricity supply. The plateau is never set below b0: a country is not capped below a rate it has already achieved. COEFFICIENTS (gamma per year / sigma, share of peak / plateau): PV +20% / 2.0% / 3% of supply; OnshoreWind +20% / 2.0% / 3% of supply; OffshoreWind +20% / 1.0% / 2% of supply; Storage +20% / 2.0% / 5% of peak; CCGT and OCGT +10% / 1.0% / 8% of peak; BiomassPlant +5% / 0.5% / 2% of peak. The observed 2025 additions and the literature anchors are in [build_rate_benchmarks_2026]. Every gamma and every sigma is an ASSUMPTION. EXCLUSIONS. Named projects keep BuildLimit = Capacity, their lead time being already carried by StYr. Hydro, pumped storage, nuclear and geothermal are outside the formula: they are limited by sites, not by build rate. NOTE that tech "ST" holds GEOTHERMAL in this deployment and not coal - the only ST candidates are Geo_Candidate_WestMed, _WestAna and _CenterAna, fuel Geothermal, 265 MW in total - so their limits are unchanged. An earlier draft of this rule pinned ST to zero as a no-new-coal assumption, which would have deleted Turkish geothermal instead. ZONE SPLIT. EPM has no country-level build constraint, so the national limit is divided over the zones carrying a generic candidate. Key = zone share of the pHours-weighted capacity factor from pVREProfile for PV and wind, zone share of peak demand otherwise, floored at 5 percent so no zone is locked out by a marginal resource, then renormalised. Where several rows sit in one zone for one technology they divide that zone's allowance. Keys sum to 1: the national total is exact and no slack factor is applied. TIME. The parameter has no year dimension, one value governs 2025-2040, so the growth path is flattened to its MEAN over the build years 2026-2040. The mean is the flattening that preserves the cumulative headroom the path allows; only the shape is lost. Making the limit genuinely time-varying without touching the GAMS would mean splitting every generic candidate into vintage tranches, about 120 extra rows across pGenDataInput, pStorageDataInput, pAvailabilityCustom and pCapexTrajectoriesCustom. Rejected: the deployment is already near its memory ceiling at 16 years. WHAT THE RULE IS NOT. It is not a forecast. It leaves Turkiye 8,707 MW/yr of PV and 5,448 MW/yr of onshore wind, against a national energy plan pace to 2035 of about 3.2 and 1.7 GW/yr. It is symmetric - it tightens the absurd values and loosens the over-constrained ones - which is what distinguishes it from a targeted cut of Turkiye. Method slide: blacksea_2026/BuildLimit_method.pptx, built by BuildLimit_method_slide.py from the deployment's own inputs. The same compute() writes these values into the CSVs, so the slide and the data cannot diverge. MW/yr, before -> after:<br>  CCGT               300 ->      273<br>  OCGT               100 ->       60<br>  OnshoreWind        150 ->      256<br>  PV                 400 ->      251 |

*Confidence: [MEDIUM] · Last updated: 2026-08-30*


<a id="georgia-pdemandforecast"></a>

### `pDemandForecast`

[&#8593; Georgia](#georgia)

**Source**: World Bank (internal) — Georgia Hourly Load Profile, 3% Annual Growth (2021–2040); original source undocumented (`georgia_demand_load_2022`)

**Data / file**: Team/Av. 3% Load growth (hourly profiles) 2021-2040.xlsx

> ⚠ **Needs review**: Peak demand (MW) has no independent cross-validation — only georgia_demand_load_2022 provides peak figures. Energy figures validated against historical balance (error <1% for 2023–2024). Growth rate of 3%/yr is undocumented — confirm with GSE/GNERC official load forecasts. Obtain electrification scenario for post-2030 period (EV, heat pumps) as 3%/yr may underestimate long-term growth.


**Method**: DIRECT (2024–2040 from hourly file) + EXTRAP (2041–2053 at 3%/yr)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2040 | `DIRECT` | Annual peak (MW) = max hourly value per year from Av. 3% Load growth file. Annual energy (GWh) = sum of hourly values / 1000 per year. File covers 2021–2040; 2024 is the first model-relevant year. |
| 2041–2053 | `EXTRAP` | Extrapolation at 3%/yr from 2040 base (same growth rate as file assumption). |

*Confidence: [MEDIUM] · Last updated: 2026-06-04*


<a id="georgia-pdemandprofile"></a>

### `pDemandProfile`

[&#8593; Georgia](#georgia)

**Source**: World Bank (internal) — Georgia Hourly Load Profile, 3% Annual Growth (2021–2040); original source undocumented (`georgia_demand_load_2022`)

**Data / file**: Team/Av. 3% Load growth (hourly profiles) 2021-2040.xlsx

**Also uses**: [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/)

**Method**: DIRECT seasonal mean from 2025 hourly data, normalized by peak

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Hourly load profile from 2025 data in Av. 3% Load growth file (2025 = first model year), normalized by max seasonal-mean value (2105 MW in Q4 evening peak), mapped onto 28 representative days (7 daytypes × 4 seasons). Since the 2026-07-06 rebuild each daytype (d1–d7) carries a distinct hourly profile. |

*Confidence: [MEDIUM] · Last updated: 2026-07-06*


<a id="georgia-pvreprofile"></a>

### `pVREProfile`

[&#8593; Georgia](#georgia)

**Source**: World Bank EPM Georgia 2022 (internal) — VRE Timeseries; PV/Wind original source undocumented (`wb_epm_georgia_timeseries`)

**Data / file**: EPM_Georgia/2022/1. Data/Timeseries all data.xlsx

**Also uses**: [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/)

> ⚠ **Needs review**: (1) Single typical year — no multi-year average. (2) Wind profile: Timeseries mean CF ~0.27 vs actual Qartli 2021 CF ~0.46 — Timeseries likely represents a generic Georgian wind site, not Qartli's specific high-wind location. Existing Georgia_Qartli_Wind may be under-dispatched in the model; consider a separate pVREProfile entry or pAvailabilityCustom override for Qartli. (3) PV data origin undocumented — replace with Renewables Ninja multi-year average when running representative days pipeline for Georgia. (4) Within-season variability RESOLVED 2026-07-06 — rep-days pipeline rerun, daytypes d1–d7 now distinct (verified in pVREProfile.csv). Items (1)–(3) still open.


**Method**: DIRECT seasonal mean from typical-year hourly CFs, normalized by tech peak

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Three techs (ROR, OnshoreWind, PV) from Timeseries all data.xlsx (sheet RE data). Seasonal mean computed per (season, hour) for 8,760 hourly CF values. Normalized by the maximum seasonal-mean hourly value across all seasons/hours for each tech separately, then mapped onto 28 representative days (d1–d7 × 4 seasons); since the 2026-07-06 rebuild each daytype carries a distinct hourly profile. Seasonal CF characteristics: ROR — Q2 peak (spring snowmelt) cf_mean=0.977, Q4 minimum cf_mean=0.566. Wind — Q2 highest cf_mean=0.902, Q3 lowest cf_mean=0.738. PV — Q3 highest (more sun hours), Q1 lowest. |

*Confidence: [MEDIUM] · Last updated: 2026-07-06*


<a id="georgia-pfuelprice"></a>

### `pFuelPrice`

[&#8593; Georgia](#georgia)

**Source**: South Caucasus bilateral gas contract prices (Armenia–Gazprom, Georgia marginal import), 2026 review (`caucasus_gas_contracts_2026`)

**Data / file**: Not a downloadable dataset. Contract prices compiled from public reporting and trade statistics during the 2026-08 gas assumptions review for the Black Sea study. ARMENIA: the Armenia–Gazprom border p…

**Also uses**: IMF Energy Subsidies Database — South Caucasus extract (Georgia file, 2022 vintage) (`georgia_fuel_subsidies_2022`)

**Method**: Gas: DIRECT from the marginal Russian import price + real escalation. DomesticCoal: DIRECT.

| Period | Method | Notes |
|--------|--------|-------|
| 2024-2053 | `DIRECT` | Gas: base 5.50 USD/MMBtu in 2024 = ~198 USD/1000m3 at 36 MMBtu/1000m3 (HHV), the mid-point of the 185-215 USD/1000m3 band at which Georgian thermal plants buy marginal Russian gas. Escalated +0.63%/yr in real terms: 2030 = 5.711, 2040 = 6.082, 2053 = 6.600 USD/MMBtu. Rationale: since 2024 Georgian TPPs are supplied close to the actual market price. The cheap blended Azeri import (~150 USD/1000m3) is the SOCIAL gas channel serving households, it is not what Gardabani burns. |
| 2024-2053 | `DIRECT` | DomesticCoal: 3.82 USD/MMBtu flat (= 3.62 USD/GJ, georgia_fuel_subsidies_2022 power-sector coal supply cost, 2021). Tkibuli domestic coal, minimal variation. |

> CORRECTED 2026-08-24. The previous value, 4.50 in 2024 rising to 5.32, was a South Caucasus wholesale proxy built on the ~150 USD/1000m3 Azeri price with the Armenian growth rate bolted on. It had no contract behind it and was itself flagged confidence: low, needs_review: true. It understated the Gardabani fuel cost by about 1 USD/MMBtu, i.e. ~7 USD/MWh of SRMC at HR 6.93, which made the plant look comfortably infra-marginal against Russian electricity imports across the whole horizon. Corrected SRMC: 33 to 40 USD/MWh. PIVOT: at 6.20 USD/MMBtu the Gardabani SRMC reaches the ~45 USD/MWh Russian import price. On the corrected trajectory that happens around 2043, so the plant stays infra-marginal but on a much thinner margin. This is the number to watch in any Georgian gas sensitivity. ON THE IMF FILE: georgia_fuel_subsidies_2022 DOES carry a Georgian power-sector gas price, 11.22 USD/GJ in 2024 in column mit_sp_nga_pow, and it is deliberately not used. That figure is ~383 USD/1000m3, which matches the Georgian COMMERCIAL gas tariff rather than the dedicated supply Georgian TPPs receive; the IMF appears to have attributed a generic non-household tariff to the power sector. Note this is a sector-attribution problem, NOT a wholesale-versus-retail one: 11.99 (2022) is the supply cost column and the retail column, 11.62, sits below it. The earlier review note here claimed the opposite and has been corrected. See the Azerbaijan pFuelPrice note for why the same IMF column IS used there.

*Confidence: [MEDIUM] · Last updated: 2026-08-24*


<a id="georgia-pavailabilitycustom"></a>

### `pAvailabilityCustom`

[&#8593; Georgia](#georgia)

**Source**: World Bank EPM Georgia v8.5 (2022, internal model) — primary data sources not documented (`wb_epm_georgia_v85`)

**Data / file**: EPM_Georgia2022/Baseline/WB_EPM_v8_5.xlsb (binary Excel)

**Also uses**: Georgia Hourly Generation Profiles by Technology 2019–2022 (`georgia_generation_profiles_2019_2022`)

> ⚠ **Needs review**: (1) Dzevruli Q3=0.04 and Shaori Q1=0.56 are unusual patterns from old model calibration — verify against actual plant hydrology with CESI. (2) Committed large reservoir (Khudoni/Namakhvani/Nenskra): proxy from Enguri — no plant-specific hydrological data available. (3) ROR aggregate: individual plant availability variability lost (Rioni ~0.70 flat vs Vartsikhe Q3=0.35 in old model). Acceptable for planning model.


**Method**: DIRECT from WB EPM v8.5 GenAvailability (ReservoirHydro) + AGGREGATE 2019-2022 (ROR)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | 7 existing ReservoirHydro plants: quarterly CFs from WB_EPM_v8_5.xlsb GenAvailability sheet (monthly factors m1-m12 averaged to quarters). Plant-specific hydrological calibration from WB EPM Georgia 2022 study: Enguri Q1=0.21/Q2=0.48/Q3=0.52/Q4=0.20; Vardnili Q1=0.30/Q2=0.49/Q3=0.48/Q4=0.28; Khrami-1&2 Q1=0.12/Q2=0.31/Q3=0.27/Q4=0.32; Jinvali Q1=0.28/Q2=0.30/Q3=0.22/Q4=0.18; Dzevruli Q1=0.38/Q2=0.17/Q3=0.04/Q4=0.17 (very low summer — specific hydrology); Shaori Q1=0.56/Q2=0.30/Q3=0.14/Q4=0.44 (peaks in winter — specific regime). 3 committed reservoir plants (Khudoni/Namakhvani/Nenskra): proxy from Enguri (western Georgia large reservoir profile). |
| 2024–2053 | `DIRECT_aggregate` | 26 ROR plants (23 individual ≥10 MW + AGG_SmallHydro + committed + candidate): uniform Q1=0.45, Q2=0.81, Q3=0.54, Q4=0.40 from georgia_generation_profiles_2019_2022. 4-year average (2019-2022) of total RoR hourly generation / installed RoR capacity. pVREProfile for ROR set to 1.0 flat — all seasonal variation in pAvailabilityCustom. |

*Confidence: [MEDIUM] · Last updated: 2026-06-04*


<a id="georgia-pmaxfuellimit"></a>

### `pMaxFuelLimit`

[&#8593; Georgia](#georgia)

**Source**: Modeller expert judgment — Black Sea 2026 assumptions (`epm_expert_judgment`)

**Data / file**: Not an external source. Groups the values set by the modeller where no measured data exists, so that those values are traceable like any other and are not mistaken for observed data.

> ⚠ **Needs review**: The 28 (MMBtu×1e6) cap is an expert-judgment physical proxy for SCP spare capacity, not a contracted figure — refine with actual SOCAR/Gazprom winter import volumes.


**Method**: Physical cap on Georgian gas-for-power ≈ South Caucasus Pipeline spare / domestic allocation

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2040 | `DIRECT` | Georgia Gas ≈ 28 (annual, MMBtu×1e6) ≈ ~4 TWh of gas-fired generation. Added 2026-07-09 (Phase 2 calibration) to stop the model over-running domestic gas: Georgia in reality leans on winter imports (Russia/Azerbaijan) rather than unlimited domestic gas. Paired with a Georgia←Russia winter import (400 MW, Q1/Q4) in pExtTransferLimit. Activated by fApplyFuelConstraint=1. |

*Confidence: [LOW] · Last updated: 2026-07-09*


---

<a id="azerbaijan"></a>

## Azerbaijan

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pDemandForecast`](#azerbaijan-pdemandforecast) | Our World in Data (OWID) (2025) + [SSC](https://statistika.nmr.az/) | [MEDIUM] ⚠ |
| [`pDemandProfile`](#azerbaijan-pdemandprofile) | proxy of Turkiye (ENTSO-E hourly shape, scaled to AZ energy) | [LOW] ⚠ |
| [`pGenDataInput`](#azerbaijan-pgendatainput) | Global Energy Monitor (GEM) (2025-09) + [SSC Azerbaijan](https://stat.gov.az/source/balance_energy/) + [SSC](https://statistika.nmr.az/) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + [Renewable resource potential](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment) + [RE candidate annual build-rate limits](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] ⚠ |
| [`pFuelPrice`](#azerbaijan-pfuelprice) | IMF (2022) + [TYNDP / IEA World Energy Outlook 2022](https://www.iea.org/reports/world-energy-outlook-2022) | [MEDIUM] |
| [`pAvailabilityCustom`](#azerbaijan-pavailabilitycustom) | World Bank EPM Georgia v8.5 (2… (2022) + [SSC Azerbaijan](https://stat.gov.az/source/balance_energy/) + [SSC](https://statistika.nmr.az/) | [LOW] ⚠ |
| [`pTransferLimit`](#azerbaijan-ptransferlimit) | Black Sea Cross-Border Lines D… (2026) + Modeller expert judgment | [MEDIUM] ⚠ |
| [`pTradePrice`](#azerbaijan-ptradeprice) | Kazakh border price for the Tr… (2026) + Modeller expert judgment | [MEDIUM] |

<a id="azerbaijan-pgendatainput"></a>

### `pGenDataInput`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: Global Energy Monitor (GEM) — Global Integrated Power Tracker (GIPT) (`gem_gipt`)

**Data / file**: Global Energy Monitor — Global Integrated Power Tracker, September 2025 download. Covers power plants worldwide: technology, installed capacity (MW), status (operating / construction / announced / ret…

**Also uses**: [SSC Azerbaijan — Annual Energy Statistics (1913–2024)](https://stat.gov.az/source/balance_energy/)

**Also uses**: [SSC — Nakhchivan AR: capacity, generation mix, GDP/electricity (2003–2022)](https://statistika.nmr.az/)

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

**Also uses**: [Renewable resource potential — South Caucasus (solar & wind)](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment)

**Also uses**: [RE candidate annual build-rate limits — Caucasus (modeller assumption)](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/)

**Also uses**: [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

> ⚠ **Needs review**: Further data collection needed

**Method**: AzerbaijanMain: GEM/GIPT 39 rows (calibrated vs SSC 2024) + Nakhchivan: NSU capacity table 4 rows

| Period | Method | Notes |
|--------|--------|-------|
| ramp and minimum generation, 2026-08-30 | `CONSTRUCTED` | Ramp rates and minimum generation put on the CCDR default table and switched on. WHY. fApplyRampConstraint and fApplyMinGenShareAllHours were both 0, so neither constraint was in the model, and the values sitting in the columns could not have been used as they stood. The full rule, the equations that read each parameter, the ones that cannot be reached at fDispatchMode = 0, and the reason for each share are in [ramp_mingen_ccdr_2026]. THE RULE. MinGenShareAllHours(z,tech,fuel) = min[ m(tech,fuel) , lowest seasonal availability of the units that (z,tech,fuel) governs ]. The cap can only lower a value, never raise one: eMinGen forces a floor in every hour while base.gms:770 caps seasonal energy at availability x capacity, so a share above a unit's own availability is INFEASIBLE and not merely expensive. The previous table put 0.55 on Turkish domestic lignite against a pAvailabilityCustom of 0.45, 72 units and 12,288 MW, and 0.45 on CCGT in every hour, 148 percent of Armenia's night trough. Ramp rates are the CCDR generic values unchanged: Nuclear 0.15, ST 0.50, the rest 1.00 per hour, PV and wind outside the constraint. The 196 Turkish cells at 0.05 and 0.08 were the textbook per-MINUTE figures loaded into a per-hour parameter; at 0.05 per hour a 600 MW unit needs 20 hours to reach full output. FLAGS. pSettings.csv fApplyRampConstraint and fApplyMinGenShareAllHours set to 1. fApplyMinGenCommitment stays 0 and MinGenCommitment, minUT, minDT and StUpCost stay unread: they need fDispatchMode = 1. Those columns are left in place. Every m is an ASSUMPTION. Method slide: blacksea_2026/RampMinGen_method.pptx, built by RampMinGen_method_slide.py from the deployment's own inputs. RAMP. No explicit cell in Azerbaijan; the zone reads pGenDataInputDefault. MIN GEN. MinGenShareAllHours in pGenDataInputDefault, before -> after:<br>  CCGT       Gas            0.45     -> 0.10<br>  ICE        Diesel         blank    -> 0.00<br>  ICE        Gas            blank    -> 0.00<br>  ICE        HFO            blank    -> 0.00<br>  Nuclear    Uranium        0.75     -> 0.70<br>  ST         Coal           0.3      -> 0.25<br>  ST         DomesticCoal   0.55     -> 0.25<br>  ST         Gas            blank    -> 0.00<br>  ST         Geothermal     blank    -> 0.55<br>  ST         ImportedCoal   0.3      -> 0.25<br>  ST         Lignite        0.55     -> 0.25 |
| 2024–2053 | `DIRECT` | AzerbaijanMain (39 generators): GEM GIPT September 2025. Gas vintage pre-2000 → ST, post-2000 → CCGT. HeatRate calibrated from SSC fuel-consumption data. Committed: 4×Mingecevir CCGT 320 MW (StYr=2025). Bridge: Azerbaijan_CHP_Legacy 985 MW (retires 2029, covers 4,300 GWh CHP gap absent from GEM). Calibration corrections June 2026 vs SSC 005_3en/005_4en. Nakhchivan (4 generators from ssc_az_nakhchivan_2022): CCGT 87 MW, OCGT 50 MW, Arpachay ReservoirHydro 48.4 MW, Solar PV 20 MW (NSU/statistika.nmr.az, ~2022). Tech params (HeatRate, VOM, FOM, Capex) from epm_generic_defaults. |
| plant lifetimes, 2026-08-29 | `ASSUMPTION` | RetrYr cleanup. Two implicit conventions coexisted in pGenDataInput.csv: a blank RetrYr, and a mechanical StYr + 25/30/50 with Life left blank. Both are unsafe. A blank RetrYr does NOT mean "never retires": main.gms:832 only pins vCap when RetrYr >= y, so the plant keeps a free vCap and an unbounded vRetire and can be retired at zero cost even though fEnableEconomicRetirement = 0. The StYr + N dates were a placeholder formula, not a published schedule, and were retiring the Caucasus gas fleet at ages no operator applies. RULE APPLIED (Life and RetrYr columns only, Status 1 and 2 only): a row is touched if Life is blank AND RetrYr - StYr is exactly 25, 30 or 50 AND RetrYr <= 2040, or if RetrYr is blank. It then receives an explicit Life — CCGT 40 yr (mid-life hot-gas- path retrofit), OCGT 35 yr, hydro 80 yr — and RetrYr = StYr + Life. Hydro is floored at RetrYr 2060, beyond the 2025-2040 horizon: the civil works are the asset and every pre-1980 unit in these fleets operates today after rehabilitation, so StYr + 80 alone would have retired plants in the past (Zahesi 1927 -> 2007, Rioni 1933 -> 2013). A row whose RetrYr is a published plan date, or whose age already exceeds the assumed life, is left untouched. PV and wind keep 25 yr, their real design life. The lifetimes themselves are modeller assumptions — see [epm_expert_judgment]; no decommissioning schedule is published for these fleets. Replace on sight if one is. THIS COUNTRY: ten AzerbaijanMain CCGTs move to Life 40 (Baku and Baku_1 2031 -> 2041, Shimal_1 400 MW 2032 -> 2042, Astara and Khachmaz 2036 -> 2046, Baku_2 and Shaki 2037 -> 2047, Sangachal 307.8 MW 2038 -> 2048, Shahdagh and Sumgayit 525 MW 2039 -> 2049), plus Nakhchivan_CCGT 87 MW 2037 -> 2047. Nakhchivan_GasTurbine_OCGT 50 MW goes 2030 -> 2040 (Life 35) and so still retires in the last horizon year. Seven hydro rows with a blank RetrYr are filled: Mingechevir 424 MW and Sarsang 50 MW and Khrami-era units at the 2060 floor, Shamkir 380 MW 2063, Yenikend 150 MW 2080, Arpachay 48.4 MW 2083, Gyz Galasy 80 MW and Khudafarin 102 MW 2106. Available capacity in 2040 +1809.6 MW in AzerbaijanMain and +87.0 MW in Nakhchivan; the gas fleet retirement share over 2025-2040 drops from 39% to 14% (AzerbaijanMain) and from 76% to 34% (Nakhchivan). Untouched: Azerbaijan_CHP_Legacy 985 MW retires 2029 (bridge asset for the CHP gap, age 44); Azerbaijan_Khizi_Wind 14 MW 2034 and Nakhchivan_Solar_PV 2040, where 25 yr is the real life. |
| candidate set completion, 2026-08-29 | `ASSUMPTION` | Missing expansion options. The Caucasus zones had no dispatchable candidate at all (Georgia and Armenia) or a single named project as their only one (Azerbaijan), so their import dependence and their generation mix were imposed by the candidate list rather than produced by an economic arbitration — a result that cannot be defended in review. Options added below; each is an OPTION the optimiser may decline, not a committed project. Sizing follows the existing fleet, not a published plan: ceilings are deliberately modest and the annual build limit is the binding constraint, as elsewhere in this deployment. Tech and cost parameters are inherited from pGenDataInputDefault (CCGT capex 0.9 M$/MW, HeatRate 6.4, Life 30; OCGT 0.8 and 9.0; offshore wind 3.0 M$/MW, FOM 70000). Candidate Life stays at the generic 30 yr and is NOT aligned with the 40 yr technical life adopted for the existing CCGT fleet in the plant-lifetimes entry: the former drives annuitisation, the latter retirement, and changing the annuity is an economic decision outside this edit. Open item. THIS COUNTRY: AzerbaijanMain_CCGT_Generic 3000 MW ceiling, 400 MW/yr, COD 2030. The only previous gas candidate was Azerbaijan_Yashma_CCGT, a NAMED 500 MW project at 100 MW/yr; that row is kept as it is, and the generic candidate is added alongside it exactly as AzerbaijanMain_PV_Generic sits alongside the named PV projects. The 400 MW/yr pace is demonstrated: Mingecevir 4x320 MW is under construction now. Nakhchivan_OCGT_Cand 100 MW, COD 2030 — the exclave had no dispatchable candidate at all and its existing OCGT retires in 2040. CASPIAN OFFSHORE WIND: AzerbaijanMain_Offshore_Generic 3000 MW ceiling, 300 MW/yr, COD 2032. The 157 GW Caspian technical potential in [re_resource_potential_caucasus] was previously parked and no offshore candidate existed. The 3000 MW ceiling is a deployment-horizon bound and NOT that technical potential; it needs reconciling with the WB/ESMAP Azerbaijan offshore wind roadmap, which is why this block is flagged needs_review. Adding the candidate required fixing the profile first — see the pVREProfile entry: OffshoreWind was a byte-identical copy of OnshoreWind, so offshore ran at the onshore capacity factor against offshore capex and could never be built. ALSO: Azerbaijan_AGG_SmallPV (8 MW, existing) had a blank RetrYr, the same free- retirement hazard closed for hydro in the plant-lifetimes entry. Set to 2060, i.e. beyond the horizon. Its StYr is still blank and no commissioning date was invented. |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear rebuilt on a stated rule, every country and every generic technology at once. WHY. The column had been filled ad hoc and no longer bound anything: Turkiye carried 90,000 MW/yr of PV nationally (9 zones x 10,000), 45,000 MW/yr each of CCGT and OCGT against 2,933 MW installed, and 440,000 MW/yr of batteries (8 zones x 5 durations x 10,000) against a 64 GW peak. EPM applies the limit PER ROW (main.gms:722, vBuild.up = pGenData(BuildLimitperYear) * pWeightYear), so a generic candidate replicated across nine zones multiplies the national headroom by nine, and five duration variants multiply it again. With y.csv = 2025..2040 consecutive, pWeightYear = 1 and the value is literally MW per year. THE RULE. BuildLimitperYear is a physical plausibility guard - supply chain, permitting, grid connection, workforce - not a policy target and not an economic parameter. It is the growth-constraint form used by the capacity-expansion models that treat this explicitly: PyPSA (max_relative_growth, with max_growth as the seed), MESSAGEix (NEW_CAPACITY_CONSTRAINT_UP, with growth_new_capacity_up and initial_new_capacity_up), and ReEDS, which applies the same idea as a growth penalty rather than a hard bound.<br>    L(c,t,y) = min[ b0(c,t) * gamma_t^(y-2025) , plateau(c,t,y) ]<br>    b0(c,t)  = max( additions observed in 2025 in c , sigma_t * Peak2025(c) )<br>b0 is the rate the country has itself demonstrated, which is the only quantity that transfers between a 75 GW system and a 1.7 GW one: 2025 solar additions ran from 3.9 percent of peak in Azerbaijan to 36 percent in Armenia, a factor of nine, so no single share-of-peak coefficient can serve as the limit. That is what the earlier max[s x Peak, g x Fleet] form got wrong; it gave Turkiye 3.2x its record year and capped Armenia below what Armenia had just installed. sigma is the standing-start seed that lets a technology with no installed base begin; gamma is the annual growth of the build rate; the plateau is the logistic saturation, expressed in the metric of the empirical diffusion literature, annual additions as a share of national electricity supply. The plateau is never set below b0: a country is not capped below a rate it has already achieved. COEFFICIENTS (gamma per year / sigma, share of peak / plateau): PV +20% / 2.0% / 3% of supply; OnshoreWind +20% / 2.0% / 3% of supply; OffshoreWind +20% / 1.0% / 2% of supply; Storage +20% / 2.0% / 5% of peak; CCGT and OCGT +10% / 1.0% / 8% of peak; BiomassPlant +5% / 0.5% / 2% of peak. The observed 2025 additions and the literature anchors are in [build_rate_benchmarks_2026]. Every gamma and every sigma is an ASSUMPTION. EXCLUSIONS. Named projects keep BuildLimit = Capacity, their lead time being already carried by StYr. Hydro, pumped storage, nuclear and geothermal are outside the formula: they are limited by sites, not by build rate. NOTE that tech "ST" holds GEOTHERMAL in this deployment and not coal - the only ST candidates are Geo_Candidate_WestMed, _WestAna and _CenterAna, fuel Geothermal, 265 MW in total - so their limits are unchanged. An earlier draft of this rule pinned ST to zero as a no-new-coal assumption, which would have deleted Turkish geothermal instead. ZONE SPLIT. EPM has no country-level build constraint, so the national limit is divided over the zones carrying a generic candidate. Key = zone share of the pHours-weighted capacity factor from pVREProfile for PV and wind, zone share of peak demand otherwise, floored at 5 percent so no zone is locked out by a marginal resource, then renormalised. Where several rows sit in one zone for one technology they divide that zone's allowance. Keys sum to 1: the national total is exact and no slack factor is applied. TIME. The parameter has no year dimension, one value governs 2025-2040, so the growth path is flattened to its MEAN over the build years 2026-2040. The mean is the flattening that preserves the cumulative headroom the path allows; only the shape is lost. Making the limit genuinely time-varying without touching the GAMS would mean splitting every generic candidate into vintage tranches, about 120 extra rows across pGenDataInput, pStorageDataInput, pAvailabilityCustom and pCapexTrajectoriesCustom. Rejected: the deployment is already near its memory ceiling at 16 years. WHAT THE RULE IS NOT. It is not a forecast. It leaves Turkiye 8,707 MW/yr of PV and 5,448 MW/yr of onshore wind, against a national energy plan pace to 2035 of about 3.2 and 1.7 GW/yr. It is symmetric - it tightens the absurd values and loosens the over-constrained ones - which is what distinguishes it from a targeted cut of Turkiye. Method slide: blacksea_2026/BuildLimit_method.pptx, built by BuildLimit_method_slide.py from the deployment's own inputs. The same compute() writes these values into the CSVs, so the slide and the data cannot diverge. MW/yr, before -> after:<br>  CCGT               400 ->     1280<br>  OCGT               100 ->      125<br>  OffshoreWind       300 ->      152<br>  OnshoreWind        330 ->      488<br>  PV                 850 ->      584 |

*Confidence: [MEDIUM] · Last updated: 2026-08-30*


<a id="azerbaijan-pdemandforecast"></a>

### `pDemandForecast`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: Our World in Data (OWID) — Energy Dataset (IEA source) (`owid_energy_data`)

**Data / file**: Our World in Data — Energy dataset, downloaded 2025. Primary underlying source: International Energy Agency (IEA) — World Energy Statistics and Balances. Full CSV available on OWID GitHub: https://git…

**Also uses**: [SSC — Nakhchivan AR: capacity, generation mix, GDP/electricity (2003–2022)](https://statistika.nmr.az/)

> ⚠ **Needs review**: Peak estimated from load factor (0.58) — no independent peak data available. Nakhchivan split inferred from generation balance; no official demand statistics.


**Method**: OWID/IEA 2025 base + 1.9%/yr CAGR; Nakhchivan split ~500 GWh / 84 MW

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT_EXTRAPOLATED` | Anchor: OWID electricity_demand 2025 = 27.17 TWh (includes CHP self-consumption). CAGR 1.9%/yr from OWID 2020–2025 trend. Peak via load_factor=0.58. Nakhchivan split: 500 GWh / 84 MW (NSU generation balance, 2021). AzerbaijanMain 2024: Energy=26,165 GWh, Peak=5,164 MW. Both zones grow proportionally. |

*Confidence: [MEDIUM] · Last updated: 2026-06-10*


<a id="azerbaijan-pdemandprofile"></a>

### `pDemandProfile`

[&#8593; Azerbaijan](#azerbaijan)

**Proxied from**: Turkiye (ENTSO-E hourly shape, scaled to AZ energy)  
**Original source**: TEİAŞ — Turkiye hourly load data (likely)

> ⚠ **Needs review**: No AZ hourly load data. Replace with AZENERGY/TANAP SCADA when available.

**Method**: PROXY Turkey ENTSO-E hourly shape, seasonal mean per quarter

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `PROXY_Turkey` | Turkey ENTSO-E hourly load profile scaled to AZ annual energy. Seasonal mean per (season, hour), normalized by peak. Q1=0.737 (winter), Q3=0.651 (summer). Same profile applied to AzerbaijanMain and Nakhchivan. No Azerbaijan-specific SCADA data available. |

*Confidence: [LOW] · Last updated: 2026-06-05*


<a id="azerbaijan-pfuelprice"></a>

### `pFuelPrice`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: IMF — Energy Subsidies Database (IMF/World Bank methodology, 2022) (`imf_energy_subsidies`)

**Data / file**: Team/Georgia_fuel-subsidies_2022.xlsx

**Also uses**: [TYNDP / IEA World Energy Outlook 2022 — commodity prices](https://www.iea.org/reports/world-energy-outlook-2022)

**Method**: IMF supply cost Gas 2024 + Armenia TYNDP/IEA WEO growth trajectory.

| Period | Method | Notes |
|--------|--------|-------|
| 2024-2053 | `DIRECT` | Gas: 4.225 USD/GJ (2024) = 4.458 USD/MMBtu at 1 GJ = 0.9478 MMBtu. IMF supply cost for AZ power generators, from Georgia_fuel-subsidies_2022.xlsx sheet 'data', countrycode = AZE, column mit_sp_nga_pow. Forward: Armenia TYNDP/IEA WEO trajectory (+0.033/yr to 2040, +0.022/yr beyond). pFuelPrice is keyed by country via zcmap.csv, so this row prices BOTH AzerbaijanMain and Nakhchivan (the latter supplied via Iran swap or Turkiye). |
| 2024-2053 | `DIRECT` | Biomass 0.50 USD/MMBtu flat (Balakhani landfill gas). |

> UNCHANGED 2026-08-24, but the reasoning is now documented. WHY THE IMF COLUMN IS USED HERE AND NOT FOR GEORGIA OR ARMENIA. The IMF supply cost is a constructed figure whose construction differs by country type. For Azerbaijan, a gas producer and net exporter, the series is stable across the 2021-22 European gas crisis (3.5 to 4.4 USD/GJ), which is the signature of a genuine domestic production cost. For net importers the same column behaves as an international import-parity construct: Georgia jumps 7.82 to 12.36 and Armenia 6.41 to 9.52 over the same crisis, and both land at roughly double the bilateral contract price actually paid. Hence kept for AZ, rejected for GE and AM. WHY THE SUPPLY COST AND NOT THE TARIFF. The IMF retail price for AZ power generators is 2.712 USD/GJ = 2.861 USD/MMBtu, the subsidised Azerenerji tariff. The gap to 4.46 is an explicit subsidy, i.e. a transfer, not a resource cost. A least-cost study must price the resource; otherwise the model builds gas plant the country will pay more for than the model assumed. THIS IS A SYSTEM-BOUNDARY CHOICE, NOT A DATA PROBLEM, and it is the most consequential gas assumption in this deployment: it decides the DIRECTION of Georgia-Azerbaijan trade rather than merely its volume. SRMC of the marginal fleet (capacity-weighted HR 10.2, VOM 2):
  financial / Azerenerji tariff       2.86 USD/MMBtu ->  31 USD/MWh
  economic supply cost   (IN USE)     4.46 USD/MMBtu ->  48 USD/MWh
  export opportunity cost via TANAP   6.50 USD/MMBtu ->  68 USD/MWh
The IEA has urged Azerbaijan to abolish the domestic subsidy, so the economic and opportunity-cost readings both have a policy case. Sensitivity files are not shipped: the two bounds are obtained by scaling the Azerbaijan,Gas row of pFuelPrice.csv by 0.642 and 1.458 respectively.

*Confidence: [MEDIUM] · Last updated: 2026-08-24*


<a id="azerbaijan-pavailabilitycustom"></a>

### `pAvailabilityCustom`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: World Bank EPM Georgia v8.5 (2022, internal model) — primary data sources not documented (`wb_epm_georgia_v85`)

**Data / file**: EPM_Georgia2022/Baseline/WB_EPM_v8_5.xlsb (binary Excel)

**Also uses**: [SSC Azerbaijan — Annual Energy Statistics (1913–2024)](https://stat.gov.az/source/balance_energy/)

**Also uses**: [SSC — Nakhchivan AR: capacity, generation mix, GDP/electricity (2003–2022)](https://statistika.nmr.az/)

> ⚠ **Needs review**: Hydro seasonal profiles are proxies — no AZ-specific gauge or seasonal generation data. Replace with Mingechevir seasonal data (AzerEnerji annual reports) when available.


**Method**: Kura hydro: Georgia Enguri proxy (WB EPM v8.5). Solar: SSC 2024 CF. Nakhchivan Arpachay: NSU data.

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `PROXY_Georgia_Kura` | 6 AzMain ReservoirHydro (Kura/Tartar system): World Bank EPM Georgia v8.5 Enguri seasonal profile (Q1=0.21, Q2=0.48, Q3=0.52, Q4=0.20). Same snowmelt catchment. Nakhchivan Arpachay ReservoirHydro: Q1=0.20, Q2=0.50, Q3=0.30, Q4=0.20 (from NSU generation data 2020–2021, mountain snowmelt logic). Solar PV (Garadagh 230 MW + small): CF=0.246 flat quarterly (SSC 2024: 556.4 GWh / 257.6 MW installed / 8,760 h = 24.6%). |

*Confidence: [LOW] · Last updated: 2026-06-10*


<a id="azerbaijan-ptransferlimit"></a>

### `pTransferLimit`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: Black Sea Cross-Border Lines Database v6 (`blacksea_crossborder_lines_v6`)

**Data / file**: Internal database of cross-border transmission infrastructure for the Black Sea region. Compiled from ENTSO-E, national TSO publications, project documentation, and expert knowledge. Local file: Data/…

**Also uses**: Modeller expert judgment — Black Sea 2026 assumptions (`epm_expert_judgment`)

> ⚠ **Needs review**: Zangezur corridor COD target 2027–2028; modeled as 2028. Run sensitivity with 2029–2030.


**Method**: DIRECT from crossborder infrastructure database + committed Zangezur corridor (2028)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Nakhchivan ↔ EastAna (Türkiye): 50 MW flat (Sederek→Iğdır 154 kV). Nakhchivan ↔ Iran (external zone): 50 MW import+export (Babek→Khoy 132/220 kV). AzerbaijanMain ↔ Nakhchivan (Zangezur corridor): 0 MW 2024–2027, 1,000 MW from 2028. AzerbaijanMain ↔ Armenia / Georgia: from blacksea_crossborder_lines_v6. |

*Confidence: [MEDIUM] · Last updated: 2026-06-10*


<a id="azerbaijan-ptradeprice"></a>

### `pTradePrice`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: Kazakh border price for the TransCaspian anchor (CASA model dual, KAZ_N) (`kazakhstan_border_price_2026`)

**Data / file**: Basis for the Kazakhstan rows of pTradePrice and pTradePriceExport in data_blacksea. Applied 2026-08-30. Replaces a flat 70 USD/MWh carried in both directions and in every hour of every year.

**Also uses**: Modeller expert judgment — Black Sea 2026 assumptions (`epm_expert_judgment`)

**Method**: MODELLED - Kazakh anchor from the CASA model dual (KAZ_N); EU anchors from the netback pipeline; Iran, Iraq, Syria and Russia unchanged

| Period | Method | Notes |
|--------|--------|-------|
| Kazakhstan, 2024-2053, applied 2026-08-30 | `MODELLED` | The Kazakhstan anchor was flat at 70 USD/MWh in both directions and in every hour of every year. It is now a seasonal series, four values per year, taken from the Central Asia deployment's own dual (ca_2026, run simulations_run_20260825_173849, scenario baseline, zone KAZ_N) and remapped on to the Black Sea season grid by shared calendar hours. Buy side = marginal cost x 1.05. 2035, USD/MWh: Q1 71.7, Q2 39.7, Q3 28.1, Q4 44.5, against 70 flat before. WHY. EPM prices external trade in two directions. pTradePriceExport is what the counterparty pays US, so it has to be the counterparty's own marginal cost; at a flat 70 the model had Kazakhstan buying at 70 the power it makes at 20, and the sale was profitable in all 8,760 hours against an Azerbaijani marginal cost that never exceeds 51. The TransCaspian cable therefore saturated by construction. The full derivation, the year mapping, the KAZ_N choice and the reason the 2045 and 2050 CASA years are excluded are in [kazakhstan_border_price_2026]. NO INTRADAY SHAPE, stated rather than decorated: CASA runs at fDispatchMode = 0 on three representative days, so its dual is constant within the day. The series is the same on all seven representative days and all 24 hours of a season. The flow decision is still hourly, because the Azerbaijani side of the comparison is. WHERE IT IS WRITTEN. Once, into trade/pTradePrice.csv, which build_trade_prices.py uses as its template and copies non-EU zones through from. promote_trade_prices.py then copies staging into the fifteen live pTradePrice_eu_*.csv and pTradePriceExport_eu_*.csv files. Editing a live file directly would be undone by the next pipeline run. NOT EXTENDED to Iran, Iraq, Syria or Russia, which stay flat at 40, 40, 40 and 45 and carry the same class of defect on the export side. No comparable dual exists for them, so correcting them would mean inventing a cost. Open item. |
| Bulgaria, Greece and Romania, 2024-2053 | `CONSTRUCTED` | Unchanged by the 2026-08-30 edit. Built by the EU netback pipeline; see the Bulgaria and Romania sections and pre-analysis/output_prices/staging/DIFF.md. |

*Confidence: [MEDIUM] · Last updated: 2026-08-30*


<a id="azerbaijan-ptradepriceexport"></a>

### `pTradePriceExport`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: Kazakh border price for the TransCaspian anchor (CASA model dual, KAZ_N) (`kazakhstan_border_price_2026`)

**Data / file**: Basis for the Kazakhstan rows of pTradePrice and pTradePriceExport in data_blacksea. Applied 2026-08-30. Replaces a flat 70 USD/MWh carried in both directions and in every hour of every year.

**Method**: DERIVED from pTradePrice by a seller-side deduction

| Period | Method | Notes |
|--------|--------|-------|
| Kazakhstan, 2024-2053, applied 2026-08-30 | `CONSTRUCTED` | Not written by hand. build_trade_prices.apply_seller_loss() derives it from the buy side through NON_EU_SELLER_LOSS, where Kazakhstan carries 1 - 0.95/1.05 = 9.5238 percent. That closes a symmetric +/-5 percent band around the Kazakh marginal cost: buy = MC x 1.05, sell = MC x 0.95. 2035, USD/MWh: Q1 64.9, Q2 35.9, Q3 25.4, Q4 40.3, against 70 flat before. THE 5 PERCENT IS THE ONLY UNSOURCED NUMBER in the Kazakh series. It stands for KEGOC wheeling, converter losses on the HVDC crossing and a trader margin. Russia already carried 2.5 percent on the same mechanism, so the shape of the treatment is not new. ASSUMPTION. EFFECT. Against the Azerbaijani marginal cost of 43.9 / 36.2 / 49.4 / 51.2 (Black Sea baseline run, zone AzerbaijanMain, no cable), the corridor goes from 8,760 hours of export to 2,284 hours of export, 4,416 of import and 2,060 of no flow: export in Q1, nothing in Q2 beyond 124 hours of solar surplus, import in Q3 and Q4. Gate G5b in build_trade_prices.py was widened from the EU zones to every zone at the same time, so a sign error in the deduction would now be caught. |

*Confidence: [MEDIUM] · Last updated: 2026-08-30*


<a id="azerbaijan-pmaxannualexternaltradeshare"></a>

### `pMaxAnnualExternalTradeShare`

[&#8593; Azerbaijan](#azerbaijan)

**Source**: Modeller expert judgment — Black Sea 2026 assumptions (`epm_expert_judgment`)

**Data / file**: Not an external source. Groups the values set by the modeller where no measured data exists, so that those values are traceable like any other and are not mistaken for observed data.

**Method**: UNITS FIX - 10 to 1, all countries and all years

| Period | Method | Notes |
|--------|--------|-------|
| 2024-2053, applied 2026-08-30 | `ASSUMPTION` | The column held 10 for Turkiye, Armenia, Georgia, Azerbaijan and iran_swap. base.gms:954-960 multiplies annual demand by this value RAW, with no division by 100 anywhere in the code path, and input_readers.gms reads it as a bare parameter. Ten therefore capped external trade at 1,000 percent of national demand and bound nothing, in either direction, for any country. config.csv already documented the unit as a per-unit share, and data_test and data_casa both use 1. Set to 1, which is 100 percent of national demand in each direction. It is a units fix, not a policy cap; data_test and data_casa both use 1. IT IS NOT UNIFORMLY SLACK, and that has to be said. Comparing the physical ceiling of the external corridors, limit x pHours summed over all external zones of a country, against that country's own annual demand:<br>  baseline and transcaspian - the largest is Azerbaijan 2035 at 27 percent of<br>    demand in each direction, so 1 binds nothing.<br>  allprojects - Georgia 2040 reaches 196 percent on export and 202 percent on<br>    import, because of the 5,200 MW Black Sea submarine cable. There the cap at<br>    1 BINDS and holds Georgian external trade to 24,092 GWh where the lines could<br>    carry 47,304.<br>Whether that is right is a real question and it has not been decided. A transit country moving more than its own consumption across its borders is not absurd, so a Georgia-specific value above 1 may be the honest answer. Left at 1 for now because the alternative, leaving 10, capped nothing anywhere and hid the question entirely. Flag this before publishing any allprojects result for 2040. |

*Confidence: [HIGH] · Last updated: 2026-08-30*


---

<a id="iran-swap"></a>

## iran_swap

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pDemandForecast`](#iran-swap-pdemandforecast) | Modeller expert judgment (2026) | [LOW] ⚠ |
| [`pDemandProfile`](#iran-swap-pdemandprofile) | Modeller expert judgment (2026) | [LOW] |
| [`pTransferLimit`](#iran-swap-ptransferlimit) | Modeller expert judgment (2026) | [LOW] |

<a id="iran-swap-pdemandforecast"></a>

### `pDemandForecast`

[&#8593; iran_swap](#iran-swap)

**Source**: Modeller expert judgment — Black Sea 2026 assumptions (`epm_expert_judgment`)

**Data / file**: Not an external source. Groups the values set by the modeller where no measured data exists, so that those values are traceable like any other and are not mistaken for observed data.

> ⚠ **Needs review**: Volume (~1.1 TWh) is an order-of-magnitude estimate of the historical swap; refine with EPSO actual annual swap export data. Peak = 350 MW is NOT a measured peak: it was copied from the Armenia↔iran_swap line rating in pTransferLimit. Energy 1100 GWh is the binding figure; the profile carries the shape and implies a 214.8 MW peak (see the pDemandProfile 2026-08-27 entry). Peak is left at 350 because under fUseSimplifiedDemand only the product profile x Peak enters the load, so rescaling the profile instead of Peak keeps the contractual energy exact.


**Method**: SYNTHETIC — swap volume from the Armenia–Iran electricity-for-gas barter

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `SYNTHETIC` | iran_swap Energy = 1100 GWh, Peak = 350 MW. Represents the Iranian side of the Armenia–Iran swap (barter: ~1 m3 gas = 3 kWh): Armenia exports ~1.1 TWh/yr to Iran, matched by Iranian gas imports. Placed in a mini-zone so the flow is INTERNAL — no priced export revenue for Armenia — and the demand belongs to Iran, not Armenia. |

*Confidence: [LOW] · Last updated: 2026-08-27*


<a id="iran-swap-pdemandprofile"></a>

### `pDemandProfile`

[&#8593; iran_swap](#iran-swap)

**Source**: Modeller expert judgment — Black Sea 2026 assumptions (`epm_expert_judgment`)

**Data / file**: Not an external source. Groups the values set by the modeller where no measured data exists, so that those values are traceable like any other and are not mistaken for observed data.

**Method**: SYNTHETIC — summer-heavy shape (Iran cooling-season imports)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `SYNTHETIC` | Q2/Q3 = 0.6138393, Q1/Q4 = 0.1023065 — the swap is summer-weighted (Iran imports Armenian surplus in the cooling season). The 6:1 seasonal ratio is the modelling assumption; the absolute levels are arithmetic, see the 2026-08-27 entry. |
| rescale, 2026-08-27 | `SYNTHETIC` | Levels were 0.90 and 0.15 and produced NEGATIVE demand. Under fUseSimplifiedDemand generate_demand.gms rebuilds the load as profile x Peak and then spreads the residual pdiff = Energy x 1e3 - sum(profile x Peak x hours) over the hours, weighted by (pmax - profile). With 0.90 for half the year, profile x Peak implied 1612.8 GWh against a stated 1100 GWh, so pdiff = -512.8 GWh was dumped entirely on the 0.15 hours — a two-valued profile leaves nowhere else — driving them to -64.9 MW. Negative load is a free generator, and the zone was exporting ~284 GWh/yr to Armenia against the direction of the swap. Both levels multiplied by 0.6820436508 = 1100000 / (350 x 4608), preserving the 6:1 ratio and the contractual 1100 GWh. Implied peak falls to 214.8 MW, which is the correct reading: the 350 MW in pDemandForecast is the line rating copied from pTransferLimit, not a measured peak. Lowest hour is now +35.8 MW. All 14 internal zones verified positive over 2025-2040. Guard added: input_verification.py replays the closed form and aborts under EPM_STRICT_DEMAND = 1 (set in pSettings.csv). |

*Confidence: [LOW] · Last updated: 2026-08-27*


<a id="iran-swap-ptransferlimit"></a>

### `pTransferLimit`

[&#8593; iran_swap](#iran-swap)

**Source**: Modeller expert judgment — Black Sea 2026 assumptions (`epm_expert_judgment`)

**Data / file**: Not an external source. Groups the values set by the modeller where no measured data exists, so that those values are traceable like any other and are not mistaken for observed data.

**Method**: SYNTHETIC — Armenia↔iran_swap internal link (Armenia–Iran corridor proxy)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Armenia↔iran_swap = 350 MW both directions (proxy for the Armenia–Iran 220 kV Meghri corridor / 3rd-line capacity). Internal link so the swap earns no external trade revenue. Value corrected here on 2026-08-27: this block said 400 MW, the CSV has always carried 350 MW. |

*Confidence: [LOW] · Last updated: 2026-08-27*


---

<a id="azerbaijanmain"></a>

## AzerbaijanMain

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pDemandForecast`](#azerbaijanmain-pdemandforecast) | Our World in Data (OWID) (2025) + [SSC](https://statistika.nmr.az/) | [MEDIUM] ⚠ |
| [`pDemandProfile`](#azerbaijanmain-pdemandprofile) | Proxy load profiles (Azerbaija… (2026) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | [LOW] ⚠ |
| [`pGenDataInput`](#azerbaijanmain-pgendatainput) | Global Energy Monitor (GEM) (2025-09) + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + [SSC](https://statistika.nmr.az/) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] ⚠ |
| [`pFuelPrice`](#azerbaijanmain-pfuelprice) | IMF (2022) + [TYNDP / IEA World Energy Outlook 2022](https://www.iea.org/reports/world-energy-outlook-2022) | [MEDIUM] |
| [`pAvailabilityCustom`](#azerbaijanmain-pavailabilitycustom) | World Bank EPM Georgia v8.5 (2… (2022) + [SSC Azerbaijan](https://stat.gov.az/source/balance_energy/) | [LOW] ⚠ |
| [`pStorageDataInput`](#azerbaijanmain-pstoragedatainput) | EPM Generic Defaults | [LOW] |

<a id="azerbaijanmain-pstoragedatainput"></a>

### `pStorageDataInput`

[&#8593; AzerbaijanMain](#azerbaijanmain)

**Source**: EPM Generic Defaults (`epm_generic_defaults`)

**Data / file**: Default technical parameters by technology/fuel combination, applied automatically when fields are left blank in pGenDataInput, pAvailabilityCustom, pCapexTrajectories. Stored in epm/resources/pGenDat…

**Method**: GENERIC — candidate storage anchors (no national target yet)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `ASSUMPTION` | BESS 500 MW/4h (Status 3, 2028) — firm the ~1.7 GW AZURE candidate RE buildout. Generic WB cost/efficiency (BESS 4h, eff 0.85, CapexMWh 250; PSH 8h, eff 0.80). Fixed-capacity anchors (Georgia style), to refine with national storage targets. NB: Türkiye uses unbounded generic BESS. |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear of the BATTERY candidates recomputed under the same growth rule as pGenDataInput: L(y) = min[ b0 * 1.20^(y-2025) , 5 percent of national peak ], b0 = max( additions observed in 2025 , 2.0 percent of the 2025 peak ), flattened to its mean over the build years 2026-2040, split across zones by peak share and divided between the duration variants of a zone. The rule is stated in full in the pGenDataInput entry of the same date and in [build_rate_benchmarks_2026]. PUMPED HYDRO is carried under tech "Storage" in this deployment but is site-driven, so it is identified by name and excluded from the formula together with the rest of hydro: its build limits are unchanged. NOTE that input_treatment.merge_storage_into_gendata gives this file the last word on a unit present in both files, so storage build limits must be edited here and not in pGenDataInput; Georgia_BESS_Cand sits in both and the pGenDataInput twin is inert. MW/yr, before -> after:<br>  Storage            200 ->      267 |

*Confidence: [LOW]*


<a id="azerbaijanmain-pgendatainput"></a>

### `pGenDataInput`

[&#8593; AzerbaijanMain](#azerbaijanmain)

**Source**: Global Energy Monitor (GEM) — Global Integrated Power Tracker (GIPT) (`gem_gipt`)

**Data / file**: Global Energy Monitor — Global Integrated Power Tracker, September 2025 download. Covers power plants worldwide: technology, installed capacity (MW), status (operating / construction / announced / ret…

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

**Also uses**: [SSC — Nakhchivan AR: capacity, generation mix, GDP/electricity (2003–2022)](https://statistika.nmr.az/)

**Also uses**: [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

> ⚠ **Needs review**: HeatRate includes gas consumed for district heat (CHP) per SSC definition — the fuel/electricity ratio (10.2 GJ/MWh for old plants) reflects actual AZ power sector gas cost per MWh, not nameplate design efficiency. Modern plants (Janub/Shimal/Gobu) use design efficiency estimate (7.5 GJ/MWh), not SSC data (individual plant fuel data not available from SSC). Khudafarin + Gyz Galasy commissioning year (2026) is an estimate — confirm with AZENERGY / ADB project status when available.


**Method**: DIRECT from GEM GIPT September 2025; Nakhchivan capacity removed to Nakhchivan zone

| Period | Method | Notes |
|--------|--------|-------|
| ramp and minimum generation, 2026-08-30 | `CONSTRUCTED` | Ramp rates and minimum generation put on the CCDR default table and switched on. WHY. fApplyRampConstraint and fApplyMinGenShareAllHours were both 0, so neither constraint was in the model, and the values sitting in the columns could not have been used as they stood. The full rule, the equations that read each parameter, the ones that cannot be reached at fDispatchMode = 0, and the reason for each share are in [ramp_mingen_ccdr_2026]. THE RULE. MinGenShareAllHours(z,tech,fuel) = min[ m(tech,fuel) , lowest seasonal availability of the units that (z,tech,fuel) governs ]. The cap can only lower a value, never raise one: eMinGen forces a floor in every hour while base.gms:770 caps seasonal energy at availability x capacity, so a share above a unit's own availability is INFEASIBLE and not merely expensive. The previous table put 0.55 on Turkish domestic lignite against a pAvailabilityCustom of 0.45, 72 units and 12,288 MW, and 0.45 on CCGT in every hour, 148 percent of Armenia's night trough. Ramp rates are the CCDR generic values unchanged: Nuclear 0.15, ST 0.50, the rest 1.00 per hour, PV and wind outside the constraint. The 196 Turkish cells at 0.05 and 0.08 were the textbook per-MINUTE figures loaded into a per-hour parameter; at 0.05 per hour a 600 MW unit needs 20 hours to reach full output. FLAGS. pSettings.csv fApplyRampConstraint and fApplyMinGenShareAllHours set to 1. fApplyMinGenCommitment stays 0 and MinGenCommitment, minUT, minDT and StUpCost stay unread: they need fDispatchMode = 1. Those columns are left in place. Every m is an ASSUMPTION. Method slide: blacksea_2026/RampMinGen_method.pptx, built by RampMinGen_method_slide.py from the deployment's own inputs. RAMP. No explicit cell in AzerbaijanMain; the zone reads pGenDataInputDefault. MIN GEN. MinGenShareAllHours in pGenDataInputDefault, before -> after:<br>  CCGT       Gas            0.45     -> 0.10<br>  ICE        Diesel         blank    -> 0.00<br>  ICE        Gas            blank    -> 0.00<br>  ICE        HFO            blank    -> 0.00<br>  Nuclear    Uranium        0.75     -> 0.70<br>  ST         Coal           0.3      -> 0.25<br>  ST         DomesticCoal   0.55     -> 0.25<br>  ST         Gas            blank    -> 0.00<br>  ST         Geothermal     blank    -> 0.55<br>  ST         ImportedCoal   0.3      -> 0.25<br>  ST         Lignite        0.55     -> 0.25 |
| 2024–2053 | `DIRECT` | 57 GEM/GIPT plants mapped to 39 EPM rows (plants with RetrYr<2025 excluded). Gas: year<2000 -> ST, year>=2000 -> CCGT. Hydro -> ReservoirHydro. Technical params (HeatRate, VOM, FOM, Capex) from epm_generic_defaults. Committed: Mingecevir CCGT 4x320 MW (StYr=2025). Note: Nakhchivan_CCGT (87 MW) and Nakhchivan_Solar_PV (20 MW) previously listed as Azerbaijan have been moved to the Nakhchivan zone. Nakhchivan capacity no longer appears in AzerbaijanMain supply. Sarsang ReservoirHydro (50 MW, Tartar River, Karabakh) retained in AzerbaijanMain. HeatRate (GJ/MWh) calibrated from SSC 002_53-55en ÷ 005_4en by vintage:<br>  Old CCGTs (StYr 2001-2009): 10.2 GJ/MWh — matches SSC 2020-2023 system<br>  average of 10.18 GJ/MWh (gas consumed in TJ / thermal electricity in MWh).<br>  Modern CCGTs (Janub 2013, Shimal 2019, Gobu 2022): 7.5 GJ/MWh — estimated<br>  from GE Frame 9/Siemens SGT5 design specifications (~48% LHV efficiency).<br>  Committed CCGTs (Mingecevir 4x320 MW, Status=2): 7.0 GJ/MWh (new build).<br>  Nakhchivan_CCGT (2007 vintage): 10.2 GJ/MWh (same as old AZ fleet).<br>  Nakhchivan_GasTurbine_OCGT (2005): 12.0 GJ/MWh (open cycle, ~30% eff).<br>Calibration corrections (2026-06-10):<br>  Azerbaijan_CHP_Legacy (985 MW, StYr=1985, RetrYr=2029): aggregate of CHP<br>  and district-heating plants that generate ~4300 GWh/yr in 2024 and are not<br>  captured in GEM/GIPT utility plant list. HeatRate=12.5 GJ/MWh (old CHP,<br>  ~28% efficiency). Derived from SSC 003_1.18en CHP row vs SSC 005_3en<br>  thermal capacity.<br>  Azerbaijan_Khudafarin_Khoda_Afa_ReservoirHydro and Gyz_Galasy_ReservoirHydro<br>  StYr shifted 2024→2026: SSC 005_3en shows hydro capacity dropping 1209→1062 MW<br>  in 2024, inconsistent with commissioning. Plants not yet counted by SSC.<br>  Azerbaijan_AGG_SmallPV reduced 40.5→8 MW: SSC 2024 non-Garadagh solar = 27.6 MW<br>  total (257.6 - 230 Garadagh). After allocating ~20 MW to Nakhchivan, ~8 MW<br>  residual for mainland small PV.<br>  Nakhchivan_Solar_PV reduced 35→20 MW: SSC Nakhchivan capacity data (2022) and<br>  SSC 003_1.18en solar residual consistent with ~20 MW installed.<br>  Azerbaijan_Khizi_Wind added 14 MW (StYr=2009, RetrYr=2034): SSC wind 64 MW vs<br>  model 50 MW (Yeni Yashma); residual ~14 MW attributed to small Khizi-area<br>  turbines predating the Khizi-Absheron committed project.<br>Demand perimeter note: pDemandForecast is calibrated to OWID total supply (~27 TWh = production + imports - exports), which includes CHP/autoproducer self-consumption. As CHP declines (−34% in 2020–2024), utility capacity must grow to replace it. Azerbaijan_CHP_Legacy explicitly bridges this transition: it covers the CHP gap in 2024–2029 and retires as Mingecevir CCGTs (4×320 MW, StYr=2025) come online. |
| plant lifetimes, 2026-08-29 | `ASSUMPTION` | RetrYr cleanup. Two implicit conventions coexisted in pGenDataInput.csv: a blank RetrYr, and a mechanical StYr + 25/30/50 with Life left blank. Both are unsafe. A blank RetrYr does NOT mean "never retires": main.gms:832 only pins vCap when RetrYr >= y, so the plant keeps a free vCap and an unbounded vRetire and can be retired at zero cost even though fEnableEconomicRetirement = 0. The StYr + N dates were a placeholder formula, not a published schedule, and were retiring the Caucasus gas fleet at ages no operator applies. RULE APPLIED (Life and RetrYr columns only, Status 1 and 2 only): a row is touched if Life is blank AND RetrYr - StYr is exactly 25, 30 or 50 AND RetrYr <= 2040, or if RetrYr is blank. It then receives an explicit Life — CCGT 40 yr (mid-life hot-gas- path retrofit), OCGT 35 yr, hydro 80 yr — and RetrYr = StYr + Life. Hydro is floored at RetrYr 2060, beyond the 2025-2040 horizon: the civil works are the asset and every pre-1980 unit in these fleets operates today after rehabilitation, so StYr + 80 alone would have retired plants in the past (Zahesi 1927 -> 2007, Rioni 1933 -> 2013). A row whose RetrYr is a published plan date, or whose age already exceeds the assumed life, is left untouched. PV and wind keep 25 yr, their real design life. The lifetimes themselves are modeller assumptions — see [epm_expert_judgment]; no decommissioning schedule is published for these fleets. Replace on sight if one is. THIS ZONE: see the same entry under the Azerbaijan country section for the full rule and the plant-by-plant list. Ten CCGTs move to Life 40 and six hydro rows with a blank RetrYr are filled; available capacity in 2040 rises by 1809.6 MW. |
| candidate set completion, 2026-08-29 | `ASSUMPTION` | Missing expansion options. The Caucasus zones had no dispatchable candidate at all (Georgia and Armenia) or a single named project as their only one (Azerbaijan), so their import dependence and their generation mix were imposed by the candidate list rather than produced by an economic arbitration — a result that cannot be defended in review. Options added below; each is an OPTION the optimiser may decline, not a committed project. Sizing follows the existing fleet, not a published plan: ceilings are deliberately modest and the annual build limit is the binding constraint, as elsewhere in this deployment. Tech and cost parameters are inherited from pGenDataInputDefault (CCGT capex 0.9 M$/MW, HeatRate 6.4, Life 30; OCGT 0.8 and 9.0; offshore wind 3.0 M$/MW, FOM 70000). Candidate Life stays at the generic 30 yr and is NOT aligned with the 40 yr technical life adopted for the existing CCGT fleet in the plant-lifetimes entry: the former drives annuitisation, the latter retirement, and changing the annuity is an economic decision outside this edit. Open item. THIS ZONE: see the same entry under the Azerbaijan country section. Adds AzerbaijanMain_CCGT_Generic 3000 MW / 400 MW-yr COD 2030 alongside the named Yashma project, and AzerbaijanMain_Offshore_Generic 3000 MW / 300 MW-yr COD 2032. Azerbaijan_AGG_SmallPV RetrYr blank -> 2060. |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear rebuilt on a stated rule, every country and every generic technology at once. WHY. The column had been filled ad hoc and no longer bound anything: Turkiye carried 90,000 MW/yr of PV nationally (9 zones x 10,000), 45,000 MW/yr each of CCGT and OCGT against 2,933 MW installed, and 440,000 MW/yr of batteries (8 zones x 5 durations x 10,000) against a 64 GW peak. EPM applies the limit PER ROW (main.gms:722, vBuild.up = pGenData(BuildLimitperYear) * pWeightYear), so a generic candidate replicated across nine zones multiplies the national headroom by nine, and five duration variants multiply it again. With y.csv = 2025..2040 consecutive, pWeightYear = 1 and the value is literally MW per year. THE RULE. BuildLimitperYear is a physical plausibility guard - supply chain, permitting, grid connection, workforce - not a policy target and not an economic parameter. It is the growth-constraint form used by the capacity-expansion models that treat this explicitly: PyPSA (max_relative_growth, with max_growth as the seed), MESSAGEix (NEW_CAPACITY_CONSTRAINT_UP, with growth_new_capacity_up and initial_new_capacity_up), and ReEDS, which applies the same idea as a growth penalty rather than a hard bound.<br>    L(c,t,y) = min[ b0(c,t) * gamma_t^(y-2025) , plateau(c,t,y) ]<br>    b0(c,t)  = max( additions observed in 2025 in c , sigma_t * Peak2025(c) )<br>b0 is the rate the country has itself demonstrated, which is the only quantity that transfers between a 75 GW system and a 1.7 GW one: 2025 solar additions ran from 3.9 percent of peak in Azerbaijan to 36 percent in Armenia, a factor of nine, so no single share-of-peak coefficient can serve as the limit. That is what the earlier max[s x Peak, g x Fleet] form got wrong; it gave Turkiye 3.2x its record year and capped Armenia below what Armenia had just installed. sigma is the standing-start seed that lets a technology with no installed base begin; gamma is the annual growth of the build rate; the plateau is the logistic saturation, expressed in the metric of the empirical diffusion literature, annual additions as a share of national electricity supply. The plateau is never set below b0: a country is not capped below a rate it has already achieved. COEFFICIENTS (gamma per year / sigma, share of peak / plateau): PV +20% / 2.0% / 3% of supply; OnshoreWind +20% / 2.0% / 3% of supply; OffshoreWind +20% / 1.0% / 2% of supply; Storage +20% / 2.0% / 5% of peak; CCGT and OCGT +10% / 1.0% / 8% of peak; BiomassPlant +5% / 0.5% / 2% of peak. The observed 2025 additions and the literature anchors are in [build_rate_benchmarks_2026]. Every gamma and every sigma is an ASSUMPTION. EXCLUSIONS. Named projects keep BuildLimit = Capacity, their lead time being already carried by StYr. Hydro, pumped storage, nuclear and geothermal are outside the formula: they are limited by sites, not by build rate. NOTE that tech "ST" holds GEOTHERMAL in this deployment and not coal - the only ST candidates are Geo_Candidate_WestMed, _WestAna and _CenterAna, fuel Geothermal, 265 MW in total - so their limits are unchanged. An earlier draft of this rule pinned ST to zero as a no-new-coal assumption, which would have deleted Turkish geothermal instead. ZONE SPLIT. EPM has no country-level build constraint, so the national limit is divided over the zones carrying a generic candidate. Key = zone share of the pHours-weighted capacity factor from pVREProfile for PV and wind, zone share of peak demand otherwise, floored at 5 percent so no zone is locked out by a marginal resource, then renormalised. Where several rows sit in one zone for one technology they divide that zone's allowance. Keys sum to 1: the national total is exact and no slack factor is applied. TIME. The parameter has no year dimension, one value governs 2025-2040, so the growth path is flattened to its MEAN over the build years 2026-2040. The mean is the flattening that preserves the cumulative headroom the path allows; only the shape is lost. Making the limit genuinely time-varying without touching the GAMS would mean splitting every generic candidate into vintage tranches, about 120 extra rows across pGenDataInput, pStorageDataInput, pAvailabilityCustom and pCapexTrajectoriesCustom. Rejected: the deployment is already near its memory ceiling at 16 years. WHAT THE RULE IS NOT. It is not a forecast. It leaves Turkiye 8,707 MW/yr of PV and 5,448 MW/yr of onshore wind, against a national energy plan pace to 2035 of about 3.2 and 1.7 GW/yr. It is symmetric - it tightens the absurd values and loosens the over-constrained ones - which is what distinguishes it from a targeted cut of Turkiye. Method slide: blacksea_2026/BuildLimit_method.pptx, built by BuildLimit_method_slide.py from the deployment's own inputs. The same compute() writes these values into the CSVs, so the slide and the data cannot diverge. MW/yr, before -> after:<br>  CCGT               400 ->     1280<br>  OffshoreWind       300 ->      152<br>  OnshoreWind        300 ->      244<br>  PV                 800 ->      292 |

*Confidence: [MEDIUM] · Last updated: 2026-08-30*


<a id="azerbaijanmain-pdemandforecast"></a>

### `pDemandForecast`

[&#8593; AzerbaijanMain](#azerbaijanmain)

**Source**: Our World in Data (OWID) — Energy Dataset (IEA source) (`owid_energy_data`)

**Data / file**: Our World in Data — Energy dataset, downloaded 2025. Primary underlying source: International Energy Agency (IEA) — World Energy Statistics and Balances. Full CSV available on OWID GitHub: https://git…

**Also uses**: [SSC — Nakhchivan AR: capacity, generation mix, GDP/electricity (2003–2022)](https://statistika.nmr.az/)

> ⚠ **Needs review**: Peak demand estimated from energy via load factor (0.58) — no independent peak data source. Nakhchivan split based on generation balance method (Nakh_generation_mix_2003-2021.csv: 2021 total generation 444.8 GWh, net Iran swap ~0, so consumption ≈ 480-500 GWh). Peak estimated from load factor assuming similar shape to main AZ (load factor ~0.68).


**Method**: DIRECT from OWID + CAGR, minus Nakhchivan (~500 GWh / 84 MW split to Nakhchivan zone)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT_EXTRAPOLATED` | Anchor: OWID electricity_demand 2025 = 27.17 TWh (net consumption) for all AZ. CAGR = 1.9%/yr computed from OWID 2020-2025 trend. Peak estimated via load_factor=0.58. Nakhchivan split: 500 GWh / 84 MW (1.876% energy, 1.60% peak) moved to Nakhchivan zone. AzerbaijanMain 2024: Energy=26165 GWh, Peak=5164 MW. Both zones grow at the same proportional rate as the OWID AZ trajectory. |

*Confidence: [MEDIUM] · Last updated: 2026-06-10*


<a id="azerbaijanmain-pdemandprofile"></a>

### `pDemandProfile`

[&#8593; AzerbaijanMain](#azerbaijanmain)

**Source**: Proxy load profiles (Azerbaijan, Nakhchivan) — built by run_blacksea_data.py (`run_blacksea_data_proxy`)

**Data / file**: Hourly profiles for the zones where NO national hourly data exists. Built by pre-analysis/studies/blacksea_2026/run_blacksea_data.py, then reduced to representative days by pre-analysis/representative…

**Also uses**: [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/)

> ⚠ **Needs review**: PROXY — no Azerbaijan-specific hourly load data. Turkey shape used as proxy (similar climate: continental, hot summers, cold winters). TO RECOMPUTE: run full representative-days pipeline for all Black Sea countries (including AZ) with VRE profiles when all country data is available. Command: python run_blacksea_data.py, then rerun compute_epm_demand.py --profile.


**Method**: PROXY Turkey shape from ENTSO-E, scaled to AZ demand, seasonal mean

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `PROXY_Turkey` | Load shape: Turkey ENTSO-E hourly profile scaled to AZ annual energy (from run_blacksea_data.py, AZE_ANNUAL_MWH=29.3 TWh), mapped onto 28 representative days (d1–d7 × 4 seasons) — each daytype distinct since the 2026-07-06 rebuild. Q1_mean=0.737 (winter heating peak), Q3_mean=0.651 (summer). Computed via compute_epm_demand.py --country AZE --profile. |

*Confidence: [LOW] · Last updated: 2026-06-05*


<a id="azerbaijanmain-pfuelprice"></a>

### `pFuelPrice`

[&#8593; AzerbaijanMain](#azerbaijanmain)

**Source**: IMF — Energy Subsidies Database (IMF/World Bank methodology, 2022) (`imf_energy_subsidies`)

**Data / file**: Team/Georgia_fuel-subsidies_2022.xlsx

**Also uses**: [TYNDP / IEA World Energy Outlook 2022 — commodity prices](https://www.iea.org/reports/world-energy-outlook-2022)

**Method**: IMF supply cost Gas 2024 + Armenia TYNDP/IEA WEO growth trajectory.

| Period | Method | Notes |
|--------|--------|-------|
| 2024-2053 | `DIRECT` | Gas: 4.225 USD/GJ (2024) = 4.458 USD/MMBtu at 1 GJ = 0.9478 MMBtu. IMF supply cost for AZ power generators, from Georgia_fuel-subsidies_2022.xlsx sheet 'data', countrycode = AZE, column mit_sp_nga_pow. Forward: Armenia TYNDP/IEA WEO trajectory (+0.033/yr to 2040, +0.022/yr beyond). pFuelPrice is keyed by country via zcmap.csv, so this row prices BOTH AzerbaijanMain and Nakhchivan (the latter supplied via Iran swap or Turkiye). |
| 2024-2053 | `DIRECT` | Biomass 0.50 USD/MMBtu flat (Balakhani landfill gas). |

> UNCHANGED 2026-08-24, but the reasoning is now documented. WHY THE IMF COLUMN IS USED HERE AND NOT FOR GEORGIA OR ARMENIA. The IMF supply cost is a constructed figure whose construction differs by country type. For Azerbaijan, a gas producer and net exporter, the series is stable across the 2021-22 European gas crisis (3.5 to 4.4 USD/GJ), which is the signature of a genuine domestic production cost. For net importers the same column behaves as an international import-parity construct: Georgia jumps 7.82 to 12.36 and Armenia 6.41 to 9.52 over the same crisis, and both land at roughly double the bilateral contract price actually paid. Hence kept for AZ, rejected for GE and AM. WHY THE SUPPLY COST AND NOT THE TARIFF. The IMF retail price for AZ power generators is 2.712 USD/GJ = 2.861 USD/MMBtu, the subsidised Azerenerji tariff. The gap to 4.46 is an explicit subsidy, i.e. a transfer, not a resource cost. A least-cost study must price the resource; otherwise the model builds gas plant the country will pay more for than the model assumed. THIS IS A SYSTEM-BOUNDARY CHOICE, NOT A DATA PROBLEM, and it is the most consequential gas assumption in this deployment: it decides the DIRECTION of Georgia-Azerbaijan trade rather than merely its volume. SRMC of the marginal fleet (capacity-weighted HR 10.2, VOM 2):
  financial / Azerenerji tariff       2.86 USD/MMBtu ->  31 USD/MWh
  economic supply cost   (IN USE)     4.46 USD/MMBtu ->  48 USD/MWh
  export opportunity cost via TANAP   6.50 USD/MMBtu ->  68 USD/MWh
The IEA has urged Azerbaijan to abolish the domestic subsidy, so the economic and opportunity-cost readings both have a policy case. Sensitivity files are not shipped: the two bounds are obtained by scaling the Azerbaijan,Gas row of pFuelPrice.csv by 0.642 and 1.458 respectively.

*Confidence: [MEDIUM] · Last updated: 2026-08-24*


<a id="azerbaijanmain-pavailabilitycustom"></a>

### `pAvailabilityCustom`

[&#8593; AzerbaijanMain](#azerbaijanmain)

**Source**: World Bank EPM Georgia v8.5 (2022, internal model) — primary data sources not documented (`wb_epm_georgia_v85`)

**Data / file**: EPM_Georgia2022/Baseline/WB_EPM_v8_5.xlsb (binary Excel)

**Also uses**: [SSC Azerbaijan — Annual Energy Statistics (1913–2024)](https://stat.gov.az/source/balance_energy/)

> ⚠ **Needs review**: Kura hydro proxy from Georgia Enguri calibration — no AZ-specific hydrological data. Replace with Mingechevir actual seasonal generation data when available (AzerEnerji annual reports, GRDC Kura gauge). Solar (2026-07-09): double-discount fixed (avail 0.246→1.0); remaining calibration step is to raise the AzerbaijanMain PV VRE profile average (0.183 → ~0.246) to reach the 556 GWh SSC-2024 actual. Excludes ~52 GWh Nakhchivan solar (split out). Wind CF (~9% apparent from SSC 50.9 GWh / 66 MW) is suspect; needs review before adding custom entry (may indicate curtailment or data gap).


**Method**: Kura River proxy (hydro) + SSC solar CF calibration

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `PROXY_Georgia_Kura` | 6 ReservoirHydro plants (Mingechevir, Shamkir, Yenikend, Khudafarin, Gyz Galasy, Sarsang): all on Kura/Tartar River system. Seasonal pattern proxied from WB EPM Georgia v8.5 Enguri calibration (Q1=0.21, Q2=0.48, Q3=0.52, Q4=0.20) — physically justified: same catchment, same snowmelt-driven seasonal cycle. All 6 plants share same seasonal CF (no plant-specific data available). |
| 2024–2053 | `DIRECT` | Solar PV (Garadagh 230 MW, AGG_SmallPV 40.5 MW): availability = 1.0 since 2026-07-09 (was 0.246). The 0.246 was the intended annual CF (556.4 GWh SSC 2024 / 257.6 MW / 8760 = 24.6%), but EPM MULTIPLIES availability by the AzerbaijanMain PV hourly VRE profile (avg 0.183) → effective CF 0.045, only ~95 GWh (double-discount). Setting availability to 1.0 lets the VRE profile alone drive output → CF 0.183, ~413 GWh. Residual gap to the 556 GWh actual: the AzerbaijanMain PV profile averages 0.183 < realized 0.246 — uplift the profile to fully calibrate solar. |

*Confidence: [LOW] · Last updated: 2026-07-09*


---

<a id="nakhchivan"></a>

## Nakhchivan

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pDemandForecast`](#nakhchivan-pdemandforecast) | SSC + [Our World in Data (OWID)](https://ourworldindata.org/energy) | [LOW] ⚠ |
| [`pDemandProfile`](#nakhchivan-pdemandprofile) | Proxy load profiles (Azerbaija… (2026) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | [LOW] |
| [`pGenDataInput`](#nakhchivan-pgendatainput) | SSC + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) + [Renewable resource potential](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment) + [RE candidate annual build-rate limits](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/) + [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] ⚠ |
| [`pAvailabilityCustom`](#nakhchivan-pavailabilitycustom) | SSC + World Bank EPM Georgia v8.5 (2022, internal model) | [LOW] ⚠ |
| [`pStorageDataInput`](#nakhchivan-pstoragedatainput) | EPM Generic Defaults | [LOW] |
| [`pTransferLimit`](#nakhchivan-ptransferlimit) | Black Sea Cross-Border Lines D… (2026) + Modeller expert judgment | [MEDIUM] ⚠ |

<a id="nakhchivan-pstoragedatainput"></a>

### `pStorageDataInput`

[&#8593; Nakhchivan](#nakhchivan)

**Source**: EPM Generic Defaults (`epm_generic_defaults`)

**Data / file**: Default technical parameters by technology/fuel combination, applied automatically when fields are left blank in pGenDataInput, pAvailabilityCustom, pCapexTrajectories. Stored in epm/resources/pGenDat…

**Method**: GENERIC — candidate storage anchors (no national target yet)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `ASSUMPTION` | BESS 50 MW/4h (Status 3, 2028) — small isolated exclave. Generic WB cost/efficiency (BESS 4h, eff 0.85, CapexMWh 250; PSH 8h, eff 0.80). Fixed-capacity anchors (Georgia style), to refine with national storage targets. NB: Türkiye uses unbounded generic BESS. |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear of the BATTERY candidates recomputed under the same growth rule as pGenDataInput: L(y) = min[ b0 * 1.20^(y-2025) , 5 percent of national peak ], b0 = max( additions observed in 2025 , 2.0 percent of the 2025 peak ), flattened to its mean over the build years 2026-2040, split across zones by peak share and divided between the duration variants of a zone. The rule is stated in full in the pGenDataInput entry of the same date and in [build_rate_benchmarks_2026]. PUMPED HYDRO is carried under tech "Storage" in this deployment but is site-driven, so it is identified by name and excluded from the formula together with the rest of hydro: its build limits are unchanged. NOTE that input_treatment.merge_storage_into_gendata gives this file the last word on a unit present in both files, so storage build limits must be edited here and not in pGenDataInput; Georgia_BESS_Cand sits in both and the pGenDataInput twin is inert. MW/yr, before -> after:<br>  Storage             30 ->       14 |

*Confidence: [LOW]*


<a id="nakhchivan-pgendatainput"></a>

### `pGenDataInput`

[&#8593; Nakhchivan](#nakhchivan)

**Source**: SSC — Nakhchivan AR: capacity, generation mix, GDP/electricity (2003–2022) (`ssc_az_nakhchivan_2022`)

**Data / file**: Data extracted from: Mammadova, Aytan (Nakhchivan State University / NSU), "Alternative Energy Production in Nakhchivan Autonomous Republic" — tables sourced from statistika.nmr.az (Statistical Commit…

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

**Also uses**: [Renewable resource potential — South Caucasus (solar & wind)](https://www.irena.org/Publications/2025/May/Investment-opportunities-for-utility-scale-solar-and-wind-areas-Georgia-zoning-assessment)

**Also uses**: [RE candidate annual build-rate limits — Caucasus (modeller assumption)](https://www.pv-magazine.com/2026/02/05/armenia-adds-around-615-mw-of-solar-in-2025/)

**Also uses**: [Ramp rates and minimum generation shares (CCDR parameter set, reachable subset)](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

> ⚠ **Needs review**: Capacity snapshot is ~2022. No updates to 2024 available. Nakhchivan_CCGT retirement year (2037) may be extended — modular plant could be refurbished. Confirm with NAR energy authority. Nakhchivan_GasTurbine_OCGT retirement 2030 is uncertain (age ~20 yr by then). No CAPEX/FOM/VOM specific to Nakhchivan — generic defaults used.


**Method**: DIRECT from Mammadova (NSU) / statistika.nmr.az capacity table (~late 2022)

| Period | Method | Notes |
|--------|--------|-------|
| ramp and minimum generation, 2026-08-30 | `CONSTRUCTED` | Ramp rates and minimum generation put on the CCDR default table and switched on. WHY. fApplyRampConstraint and fApplyMinGenShareAllHours were both 0, so neither constraint was in the model, and the values sitting in the columns could not have been used as they stood. The full rule, the equations that read each parameter, the ones that cannot be reached at fDispatchMode = 0, and the reason for each share are in [ramp_mingen_ccdr_2026]. THE RULE. MinGenShareAllHours(z,tech,fuel) = min[ m(tech,fuel) , lowest seasonal availability of the units that (z,tech,fuel) governs ]. The cap can only lower a value, never raise one: eMinGen forces a floor in every hour while base.gms:770 caps seasonal energy at availability x capacity, so a share above a unit's own availability is INFEASIBLE and not merely expensive. The previous table put 0.55 on Turkish domestic lignite against a pAvailabilityCustom of 0.45, 72 units and 12,288 MW, and 0.45 on CCGT in every hour, 148 percent of Armenia's night trough. Ramp rates are the CCDR generic values unchanged: Nuclear 0.15, ST 0.50, the rest 1.00 per hour, PV and wind outside the constraint. The 196 Turkish cells at 0.05 and 0.08 were the textbook per-MINUTE figures loaded into a per-hour parameter; at 0.05 per hour a 600 MW unit needs 20 hours to reach full output. FLAGS. pSettings.csv fApplyRampConstraint and fApplyMinGenShareAllHours set to 1. fApplyMinGenCommitment stays 0 and MinGenCommitment, minUT, minDT and StUpCost stay unread: they need fDispatchMode = 1. Those columns are left in place. Every m is an ASSUMPTION. Method slide: blacksea_2026/RampMinGen_method.pptx, built by RampMinGen_method_slide.py from the deployment's own inputs. RAMP. No explicit cell in Nakhchivan; the zone reads pGenDataInputDefault. MIN GEN. MinGenShareAllHours in pGenDataInputDefault, before -> after:<br>  CCGT       Gas            0.45     -> 0.10<br>  ICE        Diesel         blank    -> 0.00<br>  ICE        Gas            blank    -> 0.00<br>  ICE        HFO            blank    -> 0.00<br>  Nuclear    Uranium        0.75     -> 0.70<br>  ST         Coal           0.3      -> 0.25<br>  ST         DomesticCoal   0.55     -> 0.25<br>  ST         Gas            blank    -> 0.00<br>  ST         Geothermal     blank    -> 0.55<br>  ST         ImportedCoal   0.3      -> 0.25<br>  ST         Lignite        0.55     -> 0.25 |
| 2024–2053 | `DIRECT` | Source: Nakh_capacity_installed_2022.csv (NSU paper via statistika.nmr.az). 12 power plants, total 248.8 MW (snapshot ~Oct 2022). Modeled as 4 EPM generators: (1) Nakhchivan_CCGT: 87 MW CCGT Gas (Modular PP, StYr=2007, RetrYr=2037).<br>    Formerly Azerbaijan_Nakhchivan_CCGT in Azerbaijan zone.<br>(2) Nakhchivan_GasTurbine_OCGT: 50 MW OCGT Gas (Gas-Turbine PS, StYr=2005,<br>    RetrYr=2030). New entry — not previously in model.<br>(3) Nakhchivan_Arpachay_ReservoirHydro: 48.4 MW (Heydar Aliyev 4.5 MW +<br>    Bilav 22 MW + Arpachay-1 20.5 MW + Arpachay-2 1.4 MW, StYr=2003).<br>    New entry — not previously in model.<br>(4) Nakhchivan_Solar_PV: 35 MW PV (Nakhchivan SPP 22 MW + Kangarli 5 MW +<br>    Sharur 8 MW, StYr=2015, RetrYr=2040). Updated from 20 MW.<br>Wind (0.3 MW + 1.1 MW hybrid) not modeled (negligible capacity). Tech params from epm_generic_defaults. |
| plant lifetimes, 2026-08-29 | `ASSUMPTION` | RetrYr cleanup. Two implicit conventions coexisted in pGenDataInput.csv: a blank RetrYr, and a mechanical StYr + 25/30/50 with Life left blank. Both are unsafe. A blank RetrYr does NOT mean "never retires": main.gms:832 only pins vCap when RetrYr >= y, so the plant keeps a free vCap and an unbounded vRetire and can be retired at zero cost even though fEnableEconomicRetirement = 0. The StYr + N dates were a placeholder formula, not a published schedule, and were retiring the Caucasus gas fleet at ages no operator applies. RULE APPLIED (Life and RetrYr columns only, Status 1 and 2 only): a row is touched if Life is blank AND RetrYr - StYr is exactly 25, 30 or 50 AND RetrYr <= 2040, or if RetrYr is blank. It then receives an explicit Life — CCGT 40 yr (mid-life hot-gas- path retrofit), OCGT 35 yr, hydro 80 yr — and RetrYr = StYr + Life. Hydro is floored at RetrYr 2060, beyond the 2025-2040 horizon: the civil works are the asset and every pre-1980 unit in these fleets operates today after rehabilitation, so StYr + 80 alone would have retired plants in the past (Zahesi 1927 -> 2007, Rioni 1933 -> 2013). A row whose RetrYr is a published plan date, or whose age already exceeds the assumed life, is left untouched. PV and wind keep 25 yr, their real design life. The lifetimes themselves are modeller assumptions — see [epm_expert_judgment]; no decommissioning schedule is published for these fleets. Replace on sight if one is. THIS ZONE: see the same entry under the Azerbaijan country section for the full rule. Nakhchivan_CCGT 87 MW 2037 -> 2047 (Life 40); Nakhchivan_GasTurbine_OCGT 50 MW 2030 -> 2040 (Life 35, still retires in the last horizon year); Nakhchivan_Arpachay_ReservoirHydro blank -> 2083 (Life 80). Nakhchivan_Solar_PV keeps its 2040 date, 25 yr being the real design life. The review_note below on the CCGT and OCGT retirement years is answered by this entry. |
| candidate set completion, 2026-08-29 | `ASSUMPTION` | Missing expansion options. The Caucasus zones had no dispatchable candidate at all (Georgia and Armenia) or a single named project as their only one (Azerbaijan), so their import dependence and their generation mix were imposed by the candidate list rather than produced by an economic arbitration — a result that cannot be defended in review. Options added below; each is an OPTION the optimiser may decline, not a committed project. Sizing follows the existing fleet, not a published plan: ceilings are deliberately modest and the annual build limit is the binding constraint, as elsewhere in this deployment. Tech and cost parameters are inherited from pGenDataInputDefault (CCGT capex 0.9 M$/MW, HeatRate 6.4, Life 30; OCGT 0.8 and 9.0; offshore wind 3.0 M$/MW, FOM 70000). Candidate Life stays at the generic 30 yr and is NOT aligned with the 40 yr technical life adopted for the existing CCGT fleet in the plant-lifetimes entry: the former drives annuitisation, the latter retirement, and changing the annuity is an economic decision outside this edit. Open item. THIS ZONE: Nakhchivan_OCGT_Cand 100 MW, COD 2030, build limit 100 MW/yr. The exclave had wind, PV and BESS candidates but nothing dispatchable, while its existing OCGT (50 MW) retires in 2040 and its CCGT (87 MW) in 2047. |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear rebuilt on a stated rule, every country and every generic technology at once. WHY. The column had been filled ad hoc and no longer bound anything: Turkiye carried 90,000 MW/yr of PV nationally (9 zones x 10,000), 45,000 MW/yr each of CCGT and OCGT against 2,933 MW installed, and 440,000 MW/yr of batteries (8 zones x 5 durations x 10,000) against a 64 GW peak. EPM applies the limit PER ROW (main.gms:722, vBuild.up = pGenData(BuildLimitperYear) * pWeightYear), so a generic candidate replicated across nine zones multiplies the national headroom by nine, and five duration variants multiply it again. With y.csv = 2025..2040 consecutive, pWeightYear = 1 and the value is literally MW per year. THE RULE. BuildLimitperYear is a physical plausibility guard - supply chain, permitting, grid connection, workforce - not a policy target and not an economic parameter. It is the growth-constraint form used by the capacity-expansion models that treat this explicitly: PyPSA (max_relative_growth, with max_growth as the seed), MESSAGEix (NEW_CAPACITY_CONSTRAINT_UP, with growth_new_capacity_up and initial_new_capacity_up), and ReEDS, which applies the same idea as a growth penalty rather than a hard bound.<br>    L(c,t,y) = min[ b0(c,t) * gamma_t^(y-2025) , plateau(c,t,y) ]<br>    b0(c,t)  = max( additions observed in 2025 in c , sigma_t * Peak2025(c) )<br>b0 is the rate the country has itself demonstrated, which is the only quantity that transfers between a 75 GW system and a 1.7 GW one: 2025 solar additions ran from 3.9 percent of peak in Azerbaijan to 36 percent in Armenia, a factor of nine, so no single share-of-peak coefficient can serve as the limit. That is what the earlier max[s x Peak, g x Fleet] form got wrong; it gave Turkiye 3.2x its record year and capped Armenia below what Armenia had just installed. sigma is the standing-start seed that lets a technology with no installed base begin; gamma is the annual growth of the build rate; the plateau is the logistic saturation, expressed in the metric of the empirical diffusion literature, annual additions as a share of national electricity supply. The plateau is never set below b0: a country is not capped below a rate it has already achieved. COEFFICIENTS (gamma per year / sigma, share of peak / plateau): PV +20% / 2.0% / 3% of supply; OnshoreWind +20% / 2.0% / 3% of supply; OffshoreWind +20% / 1.0% / 2% of supply; Storage +20% / 2.0% / 5% of peak; CCGT and OCGT +10% / 1.0% / 8% of peak; BiomassPlant +5% / 0.5% / 2% of peak. The observed 2025 additions and the literature anchors are in [build_rate_benchmarks_2026]. Every gamma and every sigma is an ASSUMPTION. EXCLUSIONS. Named projects keep BuildLimit = Capacity, their lead time being already carried by StYr. Hydro, pumped storage, nuclear and geothermal are outside the formula: they are limited by sites, not by build rate. NOTE that tech "ST" holds GEOTHERMAL in this deployment and not coal - the only ST candidates are Geo_Candidate_WestMed, _WestAna and _CenterAna, fuel Geothermal, 265 MW in total - so their limits are unchanged. An earlier draft of this rule pinned ST to zero as a no-new-coal assumption, which would have deleted Turkish geothermal instead. ZONE SPLIT. EPM has no country-level build constraint, so the national limit is divided over the zones carrying a generic candidate. Key = zone share of the pHours-weighted capacity factor from pVREProfile for PV and wind, zone share of peak demand otherwise, floored at 5 percent so no zone is locked out by a marginal resource, then renormalised. Where several rows sit in one zone for one technology they divide that zone's allowance. Keys sum to 1: the national total is exact and no slack factor is applied. TIME. The parameter has no year dimension, one value governs 2025-2040, so the growth path is flattened to its MEAN over the build years 2026-2040. The mean is the flattening that preserves the cumulative headroom the path allows; only the shape is lost. Making the limit genuinely time-varying without touching the GAMS would mean splitting every generic candidate into vintage tranches, about 120 extra rows across pGenDataInput, pStorageDataInput, pAvailabilityCustom and pCapexTrajectoriesCustom. Rejected: the deployment is already near its memory ceiling at 16 years. WHAT THE RULE IS NOT. It is not a forecast. It leaves Turkiye 8,707 MW/yr of PV and 5,448 MW/yr of onshore wind, against a national energy plan pace to 2035 of about 3.2 and 1.7 GW/yr. It is symmetric - it tightens the absurd values and loosens the over-constrained ones - which is what distinguishes it from a targeted cut of Turkiye. Method slide: blacksea_2026/BuildLimit_method.pptx, built by BuildLimit_method_slide.py from the deployment's own inputs. The same compute() writes these values into the CSVs, so the slide and the data cannot diverge. MW/yr, before -> after:<br>  OCGT               100 ->      125<br>  OnshoreWind         30 ->      244<br>  PV                  50 ->      292 |

*Confidence: [MEDIUM] · Last updated: 2026-08-30*


<a id="nakhchivan-pdemandforecast"></a>

### `pDemandForecast`

[&#8593; Nakhchivan](#nakhchivan)

**Source**: SSC — Nakhchivan AR: capacity, generation mix, GDP/electricity (2003–2022) (`ssc_az_nakhchivan_2022`)

**Data / file**: Data extracted from: Mammadova, Aytan (Nakhchivan State University / NSU), "Alternative Energy Production in Nakhchivan Autonomous Republic" — tables sourced from statistika.nmr.az (Statistical Commit…

**Also uses**: [Our World in Data (OWID) — Energy Dataset (IEA source)](https://ourworldindata.org/energy)

> ⚠ **Needs review**: No official Nakhchivan demand statistics — consumption inferred from generation balance. Iran swap volumes (~33 GWh each way) assumed symmetric. CAGR 1.9%/yr from main AZ applied — may overestimate if Nakhchivan growth is slower (historical 2015-2021: ~0.5%/yr stagnation period). Recalibrate once AZ Ministry publishes Nakhchivan-specific demand data.


**Method**: Generation-balance estimate from Nakh generation mix + proportional CAGR

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DERIVED_GENERATION_BALANCE` | Anchor 2024: Energy=500 GWh, Peak=84 MW. Methodology: Nakhchivan consumption ≈ local generation + net imports. Generation data (Nakh_generation_mix_2003-2021.csv): 2021 total=444.8 GWh. Net Iran swap flow ~0 (symmetric swap, ~33 GWh each direction per blacksea_crossborder_lines_v6.xlsx). Consumption estimate ≈ 480–500 GWh. Anchor set at 500 GWh (2024, extrapolating 2021 at ~1.5%/yr CAGR). Peak: load factor ~0.68 → Peak = 500e6/(0.68×8760) ≈ 84 MW. Forward trajectory: proportional to AzerbaijanMain OWID CAGR (1.9%/yr). 2053: Energy≈861 GWh, Peak≈145 MW. |

*Confidence: [LOW] · Last updated: 2026-06-10*


<a id="nakhchivan-pdemandprofile"></a>

### `pDemandProfile`

[&#8593; Nakhchivan](#nakhchivan)

**Source**: Proxy load profiles (Azerbaijan, Nakhchivan) — built by run_blacksea_data.py (`run_blacksea_data_proxy`)

**Data / file**: Hourly profiles for the zones where NO national hourly data exists. Built by pre-analysis/studies/blacksea_2026/run_blacksea_data.py, then reduced to representative days by pre-analysis/representative…

**Also uses**: [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/)

**Method**: PROXY from AzerbaijanMain load shape (same Turkey ENTSO-E origin)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `PROXY_AzerbaijanMain` | Same 24-hour × 4-quarter × 6-daytype profile as AzerbaijanMain. Nakhchivan has similar climate to southern AZ (continental, hot summers). No Nakhchivan-specific hourly load data available. |

*Confidence: [LOW] · Last updated: 2026-06-10*


<a id="nakhchivan-pavailabilitycustom"></a>

### `pAvailabilityCustom`

[&#8593; Nakhchivan](#nakhchivan)

**Source**: SSC — Nakhchivan AR: capacity, generation mix, GDP/electricity (2003–2022) (`ssc_az_nakhchivan_2022`)

**Data / file**: Data extracted from: Mammadova, Aytan (Nakhchivan State University / NSU), "Alternative Energy Production in Nakhchivan Autonomous Republic" — tables sourced from statistika.nmr.az (Statistical Commit…

**Also uses**: World Bank EPM Georgia v8.5 (2022, internal model) — primary data sources not documented (`wb_epm_georgia_v85`)

> ⚠ **Needs review**: Arpachay hydro profile is a simplified 4-quarter proxy. No gauge data used — inferred from total annual generation and mountain snowmelt logic. Seasonal profile should be validated against GRDC Araz River discharge or statistika.nmr.az monthly generation data if available. Solar CF (17%) is from 2021 data; capacity has grown since.


**Method**: Arpachay seasonal proxy (snowmelt) + SSC solar CF calibration

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DERIVED_HYDRO_SEASONAL` | Nakhchivan_Arpachay_ReservoirHydro (48.4 MW): Arpachay/Araz rivers, Nakhchivan AR. Seasonal profile: Q1=0.20, Q2=0.50, Q3=0.30, Q4=0.20. Derived from generation data (Nakh_generation_mix_2003-2021.csv): 2020 hydro CF = 157.1/(48.4×8760) = 37.1% annual. 2021 hydro CF = 129.6/(48.4×8760) = 30.6% annual. Quarterly distribution: mountain snowmelt peak in spring/early summer (Q2), lower Q1/Q4, moderate Q3. Distinct from Kura River (Enguri proxy) which peaks Q2/Q3; Arpachay drains smaller catchment at higher altitude. |
| 2024–2053 | `DERIVED_SSC` | Nakhchivan_Solar_PV (35 MW): CF=0.17 flat quarterly. Derived from Nakh_generation_mix_2003-2021.csv 2021 data: 50.8 GWh / (35 MW × 8760 h) = 16.6% ≈ 17%. (In 2021, installed solar was ~27 MW; 50.8/27×8760 = 21.5% CF — 35 MW denominator gives 17% as conservative planning estimate). |

*Confidence: [LOW] · Last updated: 2026-06-10*


<a id="nakhchivan-ptransferlimit"></a>

### `pTransferLimit`

[&#8593; Nakhchivan](#nakhchivan)

**Source**: Black Sea Cross-Border Lines Database v6 (`blacksea_crossborder_lines_v6`)

**Data / file**: Internal database of cross-border transmission infrastructure for the Black Sea region. Compiled from ENTSO-E, national TSO publications, project documentation, and expert knowledge. Local file: Data/…

**Also uses**: Modeller expert judgment — Black Sea 2026 assumptions (`epm_expert_judgment`)

> ⚠ **Needs review**: Nakhchivan↔Türkiye (50 MW) and ↔Iran (50 MW) capacities from infrastructure database — actual operational limits may differ from rated capacity (line aging, operational constraints). Zangezur corridor COD 2027-2028 is the official target; modeled as available from 2028. COD slippage possible — sensitivity test with 2029-2030 COD recommended. HVDC Nakhchivan↔Türkiye (1000 MW, post-2030, no FID) not modeled.


**Method**: DIRECT from crossborder infrastructure database + committed Zangezur corridor

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Three connections for Nakhchivan: (1) Nakhchivan ↔ EastAna (Türkiye): 50 MW flat all years.<br>    Source: Sederek/Babek → Iğdır, 154 kV line (~50 MW).<br>    blacksea_crossborder_lines_v6.xlsx.<br>(2) Nakhchivan ↔ Inter_Nakhchivan_Iran (external): 50 MW Import+Export.<br>    Source: Babek/Julfa → Khoy/Marand, 132/220 kV (~50 MW, 2005 swap).<br>    Historical: ~33 GWh each direction/year (near capacity at 50 MW peak).<br>    Modeled as external zone (pExtTransferLimit) — Iran not in model.<br>(3) AzerbaijanMain ↔ Nakhchivan (Zangezur corridor):<br>    0 MW 2024–2027; 1000 MW from 2028 (all quarters).<br>    Source: 330 kV double-circuit line, construction launched Jan 2026,<br>    COD 2027–2028 per official statements. Committed project (Status=2).<br>    Capacity: 1000 MW rated (2×330 kV circuits, each ~500 MW). |

*Confidence: [MEDIUM] · Last updated: 2026-06-10*


---

<a id="romania"></a>

## Romania

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pDemandForecast`](#romania-pdemandforecast) | Our World in Data (OWID) (2025) | [LOW] ⚠ |
| [`pDemandProfile`](#romania-pdemandprofile) | ENTSO-E Transparency Platform (2025) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | [HIGH] |
| [`pGenDataInput`](#romania-pgendatainput) | World Bank EPM Romania v8.5 (2… (2024) + [Global Energy Monitor (GEM)](https://globalenergymonitor.org/projects/global-integrated-power-tracker/) | [MEDIUM] ⚠ |
| [`pFuelPrice`](#romania-pfuelprice) | World Bank EPM Romania v8.5 (2… (2024) | [MEDIUM] ⚠ |
| [`pAvailabilityCustom`](#romania-pavailabilitycustom) | World Bank EPM Romania v8.5 (2… (2024) | [HIGH] |
| [`pVREProfile`](#romania-pvreprofile) | Global Energy Monitor (GEM) (2025-09) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | [HIGH] |

<a id="romania-pgendatainput"></a>

### `pGenDataInput`

[&#8593; Romania](#romania)

**Source**: World Bank EPM Romania v8.5 (2024, internal model) — primary data sources partially documented (`wb_epm_romania_v8`)

**Data / file**: WB_EPM_RO_12_42.xlsb (binary Excel format, 12 zones → 42 years). Extracted via pre-analysis/extract_epm_excel.py --country Romania. Zone name in xlsb: RomaniaZ…

**Also uses**: [Global Energy Monitor (GEM) — Global Integrated Power Tracker (GIPT)](https://globalenergymonitor.org/projects/global-integrated-power-tracker/)

> ⚠ **Needs review**: Cernavoda-3 and Cernavoda-4 (Status=2, committed, 720 MW each): timing (StYr 2035/2036) from WB EPM v8.5. Current status uncertain; Romania–Korea nuclear agreement 2024 suggests construction start ~2028, COD ~2035–2037. Verify with NUCLEARELECTRICA. HPP Portile de Fier I (1166 MW) retires 2026 (life=50 from 1972) → extension "HPP Portile de Fier I_ext" starts 2027. Real extension pending Romania–Serbia feasibility study; recommend sensitivity scenario. Gas price trajectory: xlsb 18.35 USD/GJ in 2024 → 9.81 flat from 2025. The 2024 value reflects high 2022 spot price; 9.81 from 2025 may be too low or too high depending on TTF evolution. Cross-check with IEA WEO 2024 NPS. Onshore wind capacity 2,968 MW existing appears low vs RWEA data (~4 GW by end 2023) — some parks may have commissioned after xlsb calibration date.


**Method**: DIRECT from WB EPM Romania v8.5 xlsb (cleaned + fuel-normalised); GEM GIPT cross-check

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | 162 generators extracted and cleaned from WB_EPM_RO_12_42.xlsb via pre-analysis/extract_epm_excel.py + pre-analysis/prepare_romania_for_blacksea.py. Key fuel normalisation: Coal→DomesticCoal (domestic lignite), WindLow1/2/3 & WindMed1/2/3→Wind (multi-tier resource atlas consolidated to single Wind profile). GasCCS → Gas (CCS plants simplified). Geothermal, Hydrogen/H2 candidates dropped. 9 Generic Onshore Wind resource-class entries collapsed to:<br>  "Generic Onshore Wind Romania" (120 GW potential, 2 GW/yr build limit).<br>  "Generic Offshore Wind Romania" (94 GW Black Sea potential, 500 MW/yr build limit).<br>TPP Mintia capacity set to 495 MW (2 remaining units as of 2024). R-Coal aggregate dropped (no documented capacity). GEM GIPT (September 2025) used for cross-check: key differences noted in nuclear commissioning (Cernavoda-3&4 Status=2), gas committed pipeline, and Wind/PV installed capacity growth 2024–2025. |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear rebuilt on a stated rule, every country and every generic technology at once. WHY. The column had been filled ad hoc and no longer bound anything: Turkiye carried 90,000 MW/yr of PV nationally (9 zones x 10,000), 45,000 MW/yr each of CCGT and OCGT against 2,933 MW installed, and 440,000 MW/yr of batteries (8 zones x 5 durations x 10,000) against a 64 GW peak. EPM applies the limit PER ROW (main.gms:722, vBuild.up = pGenData(BuildLimitperYear) * pWeightYear), so a generic candidate replicated across nine zones multiplies the national headroom by nine, and five duration variants multiply it again. With y.csv = 2025..2040 consecutive, pWeightYear = 1 and the value is literally MW per year. THE RULE. BuildLimitperYear is a physical plausibility guard - supply chain, permitting, grid connection, workforce - not a policy target and not an economic parameter. It is the growth-constraint form used by the capacity-expansion models that treat this explicitly: PyPSA (max_relative_growth, with max_growth as the seed), MESSAGEix (NEW_CAPACITY_CONSTRAINT_UP, with growth_new_capacity_up and initial_new_capacity_up), and ReEDS, which applies the same idea as a growth penalty rather than a hard bound.<br>    L(c,t,y) = min[ b0(c,t) * gamma_t^(y-2025) , plateau(c,t,y) ]<br>    b0(c,t)  = max( additions observed in 2025 in c , sigma_t * Peak2025(c) )<br>b0 is the rate the country has itself demonstrated, which is the only quantity that transfers between a 75 GW system and a 1.7 GW one: 2025 solar additions ran from 3.9 percent of peak in Azerbaijan to 36 percent in Armenia, a factor of nine, so no single share-of-peak coefficient can serve as the limit. That is what the earlier max[s x Peak, g x Fleet] form got wrong; it gave Turkiye 3.2x its record year and capped Armenia below what Armenia had just installed. sigma is the standing-start seed that lets a technology with no installed base begin; gamma is the annual growth of the build rate; the plateau is the logistic saturation, expressed in the metric of the empirical diffusion literature, annual additions as a share of national electricity supply. The plateau is never set below b0: a country is not capped below a rate it has already achieved. COEFFICIENTS (gamma per year / sigma, share of peak / plateau): PV +20% / 2.0% / 3% of supply; OnshoreWind +20% / 2.0% / 3% of supply; OffshoreWind +20% / 1.0% / 2% of supply; Storage +20% / 2.0% / 5% of peak; CCGT and OCGT +10% / 1.0% / 8% of peak; BiomassPlant +5% / 0.5% / 2% of peak. The observed 2025 additions and the literature anchors are in [build_rate_benchmarks_2026]. Every gamma and every sigma is an ASSUMPTION. EXCLUSIONS. Named projects keep BuildLimit = Capacity, their lead time being already carried by StYr. Hydro, pumped storage, nuclear and geothermal are outside the formula: they are limited by sites, not by build rate. NOTE that tech "ST" holds GEOTHERMAL in this deployment and not coal - the only ST candidates are Geo_Candidate_WestMed, _WestAna and _CenterAna, fuel Geothermal, 265 MW in total - so their limits are unchanged. An earlier draft of this rule pinned ST to zero as a no-new-coal assumption, which would have deleted Turkish geothermal instead. ZONE SPLIT. EPM has no country-level build constraint, so the national limit is divided over the zones carrying a generic candidate. Key = zone share of the pHours-weighted capacity factor from pVREProfile for PV and wind, zone share of peak demand otherwise, floored at 5 percent so no zone is locked out by a marginal resource, then renormalised. Where several rows sit in one zone for one technology they divide that zone's allowance. Keys sum to 1: the national total is exact and no slack factor is applied. TIME. The parameter has no year dimension, one value governs 2025-2040, so the growth path is flattened to its MEAN over the build years 2026-2040. The mean is the flattening that preserves the cumulative headroom the path allows; only the shape is lost. Making the limit genuinely time-varying without touching the GAMS would mean splitting every generic candidate into vintage tranches, about 120 extra rows across pGenDataInput, pStorageDataInput, pAvailabilityCustom and pCapexTrajectoriesCustom. Rejected: the deployment is already near its memory ceiling at 16 years. WHAT THE RULE IS NOT. It is not a forecast. It leaves Turkiye 8,707 MW/yr of PV and 5,448 MW/yr of onshore wind, against a national energy plan pace to 2035 of about 3.2 and 1.7 GW/yr. It is symmetric - it tightens the absurd values and loosens the over-constrained ones - which is what distinguishes it from a targeted cut of Turkiye. Method slide: blacksea_2026/BuildLimit_method.pptx, built by BuildLimit_method_slide.py from the deployment's own inputs. The same compute() writes these values into the CSVs, so the slide and the data cannot diverge. MW/yr, before -> after: |

*Confidence: [MEDIUM] · Last updated: 2026-08-30*


<a id="romania-pdemandforecast"></a>

### `pDemandForecast`

[&#8593; Romania](#romania)

**Source**: Our World in Data (OWID) — Energy Dataset (IEA source) (`owid_energy_data`)

**Data / file**: Our World in Data — Energy dataset, downloaded 2025. Primary underlying source: International Energy Agency (IEA) — World Energy Statistics and Balances. Full CSV available on OWID GitHub: https://git…

> ⚠ **Needs review**: CAGR of -1.3%/yr reflects recent historical decline — not appropriate as a planning scenario. Romania is expected to have significant electrification (EVs, heat pumps, green hydrogen) that would reverse this trend. Override recommended: use --growth 0.02 for moderate-growth scenario or replace with Romania energy strategy official forecast (ANRE/MoE). 2053 value (37,952 GWh) is likely understated by 30-50% for a transition scenario. Flag for revision before model submission.


**Method**: OWID/IEA base 54.4 TWh (2025) + CAGR -1.3%/yr (5-yr historical trend)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT_EXTRAPOLATED` | Anchor: OWID electricity_demand 2025 = 54.4 TWh. CAGR = -1.3%/yr from OWID 2020–2025 trend (industrial contraction + efficiency gains). Peak estimated via load_factor=0.58. 2024: Energy=55,094 GWh, Peak=10,843 MW. 2030: Energy=51,005 GWh (declining). |

*Confidence: [LOW] · Last updated: 2026-06-11*


<a id="romania-pdemandprofile"></a>

### `pDemandProfile`

[&#8593; Romania](#romania)

**Source**: ENTSO-E Transparency Platform — Actual Total Load (hourly) (`entsoe_hourly_load`)

**Data / file**: ENTSO-E Transparency Platform, dataset: Actual Total Load per Bidding Zone. Downloaded via entsoe-py Python client (pre-analysis/studies/blacksea_2026/run_blacksea_data.py, step_entsoe_download()). Co…

**Also uses**: [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/)

**Method**: DIRECT seasonal mean from ENTSO-E Romania hourly load, all d1-d6 daytypes

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Romania zone from blacksea_run1/reprdays_input/Load.csv (ENTSO-E hourly load, RO bidding zone, 2018–2024). Seasonal mean per (season, hour), normalized by peak. Q1_mean=0.737 (winter heating peak), Q2_mean=0.631 (spring), Q3_mean=0.651 (summer AC), Q4_mean=0.698. All d1–d6 daytypes within a season share the same seasonal mean profile. Computed via compute_epm_demand.py --country ROU --profile. |

*Confidence: [HIGH] · Last updated: 2026-06-11*


<a id="romania-pvreprofile"></a>

### `pVREProfile`

[&#8593; Romania](#romania)

**Source**: Global Energy Monitor (GEM) — Global Integrated Power Tracker (GIPT) (`gem_gipt`)

**Data / file**: Global Energy Monitor — Global Integrated Power Tracker, September 2025 download. Covers power plants worldwide: technology, installed capacity (MW), status (operating / construction / announced / ret…

**Also uses**: [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/)

**Method**: Renewables Ninja multi-year seasonal mean CF (2010-2019), Romania centroid

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | PV and OnshoreWind profiles from Renewables Ninja API output (blacksea_run1/ninja/ directory, Romania centroid coordinates). Multi-year seasonal mean per (season, hour), normalized by tech maximum. PV: Q1=0.204, Q2=0.313 (spring solar), Q3=0.326 (summer peak), Q4=0.183. OnshoreWind: Q1=0.910 (winter wind), Q2=0.666, Q3=0.592 (summer low), Q4=0.805. All d1–d6 daytypes share the same seasonal mean (simplified). Computed via compute_epm_vre.py --country ROU. |

*Confidence: [HIGH] · Last updated: 2026-06-11*


<a id="romania-pfuelprice"></a>

### `pFuelPrice`

[&#8593; Romania](#romania)

**Source**: World Bank EPM Romania v8.5 (2024, internal model) — primary data sources partially documented (`wb_epm_romania_v8`)

**Data / file**: WB_EPM_RO_12_42.xlsb (binary Excel format, 12 zones → 42 years). Extracted via pre-analysis/extract_epm_excel.py --country Romania. Zone name in xlsb: RomaniaZ…

> ⚠ **Needs review**: UNIT LABEL SETTLED 2026-08-24: these figures were previously labelled USD/GJ, but they are copied unconverted into pFuelPrice.csv, which EPM reads as $/MMBtu, and the source is itself a WB EPM model natively in $/MMBtu. The label was wrong, the values are right as loaded. NO VALUE WAS CHANGED. Two substantive issues remain open, both latent: (1) 18.35 in 2024 is the 2022 TTF peak, not a 2024 price; (2) the 2025+ level of 9.81 sits above European forwards of 7-9 $/MMBtu, and
    Romania ends up 4.2 $/MMBtu more expensive than Bulgaria even though Romania is
    the producer of the two. That spread has the sign backwards.
NOT FIXED because Romania is listed in zext.csv and the zcmap row of scenarios.csv is empty in all three scenarios, so zcmap.csv applies and Romania is external: it is priced by pTradePrice and never dispatched. These fuel prices and the 181 Romanian generators are inert in every current run. Fix both items BEFORE any run that wires in zcmap_robg.csv. DomesticCoal: 4.4 is a benchmark; the actual CE Oltenia generator fuel cost should be verified against ANRE tariff filings.


**Method**: DIRECT from WB EPM Romania v8.5 xlsb (Gas trajectory + Uranium trajectory)

| Period | Method | Notes |
|--------|--------|-------|
| 2024 | `DIRECT` | Gas: 18.35 $/MMBtu (high 2022 spot price still in the WB EPM v8.5 2024 calibration). This value reflects the TTF peak period and should be revised downward before Romania is ever dispatched. |
| 2025-2053 | `DIRECT` | Gas: 9.81 $/MMBtu flat (WB EPM v8.5 baseline European gas assumption from 2025). DomesticCoal (lignite): 4.4 $/MMBtu flat (Oltenia Basin domestic lignite). Uranium: 1.5 $/MMBtu (2024) -> 5.6 $/MMBtu (2053), nuclear fuel cycle cost escalation (WB EPM v8.5 assumption). Biomass: 5.0 $/MMBtu flat (European pellet market price estimate). |

*Confidence: [MEDIUM] · Last updated: 2026-08-24*


<a id="romania-pavailabilitycustom"></a>

### `pAvailabilityCustom`

[&#8593; Romania](#romania)

**Source**: World Bank EPM Romania v8.5 (2024, internal model) — primary data sources partially documented (`wb_epm_romania_v8`)

**Data / file**: WB_EPM_RO_12_42.xlsb (binary Excel format, 12 zones → 42 years). Extracted via pre-analysis/extract_epm_excel.py --country Romania. Zone name in xlsb: RomaniaZ…

**Method**: DIRECT from WB EPM Romania v8.5 GenAvailability (Cernavoda + hydro + gas)

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | 181 generators with quarterly CFs from GenAvailability sheet of WB_EPM_RO_12_42.xlsb. Key entries: Cernavoda-1: Q1=0.823, Q2=0.755, Q3=0.823, Q4=0.823 (annual outage in Q2). Cernavoda-1_refurbished: Q1=0.915, Q2=0.821, Q3=0.892, Q4=0.892 (post-refurb CF). Cernavoda-2: Q1=0.900, Q2=0.798, Q3=0.900, Q4=0.900 (planned outage Q2). Cernavoda-3&4: Q1=Q2=Q3=Q4=0.930 (new build assumption). Hydro seasonal CFs: individual plant-specific quarterly availability (12 ROR + 12 ReservoirHydro plants, reflecting Olt/Danube/Bistrita seasonal regimes). Gas plants: 0.88–0.95 flat quarterly (plant-specific from WB EPM model). |

*Confidence: [HIGH] · Last updated: 2026-06-11*


---

<a id="bulgaria"></a>

## Bulgaria

[&#8593; Contents](#toc)

### Summary

| Parameter | Source | Confidence |
|---|---|---|
| [`pDemandForecast`](#bulgaria-pdemandforecast) | World Bank Bulgaria CCDR (2026) + [Our World in Data (OWID)](https://ourworldindata.org/energy) | [MEDIUM] |
| [`pDemandProfile`](#bulgaria-pdemandprofile) | ENTSO-E Transparency Platform (2025) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | [HIGH] ⚠ |
| [`pGenDataInput`](#bulgaria-pgendatainput) | Global Energy Monitor (GEM) (2025-09) + World Bank Bulgaria CCDR + [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/) | [MEDIUM] ⚠ |
| [`pFuelPrice`](#bulgaria-pfuelprice) | World Bank Bulgaria CCDR (2026) + World Bank EPM Romania v8.5 (2024, internal model) | [MEDIUM] |
| [`pAvailabilityCustom`](#bulgaria-pavailabilitycustom) | Bulgarian quarterly availabili… (2026) + [World Nuclear Association](https://world-nuclear.org/nuclear-reactor-database/) + [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) + World Bank Bulgaria CCDR | [MEDIUM] ⚠ |
| [`pVREProfile`](#bulgaria-pvreprofile) | Renewables Ninja (2018–2023) + [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/) | [HIGH] ⚠ |

<a id="bulgaria-pdemandforecast"></a>

### `pDemandForecast`

[&#8593; Bulgaria](#bulgaria)

**Source**: World Bank Bulgaria CCDR — Kinesys/TIMES Energy Model (2026) (`bg_ccdr_kinesys_2026`)

**Data / file**: BG-CCDR - Kinesys_2101_b2.20.26.xlsx. Local path: Data/Bulgaria/BG-CCDR - Kinesys_2101_b2.20.26.xlsx. Two main scenarios:

**Also uses**: [Our World in Data (OWID) — Energy Dataset (IEA source)](https://ourworldindata.org/energy)

**Method**: Kinesys WEM scenario FEC calibrated to OWID 2025 anchor, interpolated annually

| Period | Method | Notes |
|--------|--------|-------|
| 2025–2050 | `DIRECT` | Kinesys WEM scenario (cl_wb7-WEM.Nuc-Y.Clim-HDNucRet-Y), KeyCommBalance sheet. Total electricity FEC = sum of demand sectors (Agriculture, Commercial, Energy, Hydrogen, Industry, Mining, Residential, Transport) in PJ, converted to TWh. Calibration factor applied to align 2025 FEC (31.84 TWh) with OWID gross supply anchor (36.5 TWh), implying ~14.8% T&D loss margin. Factor=1.146 applied flat to all years. 5-year milestones: 2025=36.5, 2030=35.8, 2035=36.4, 2040=39.4, 2045=41.7, 2050=43.7 TWh (gross). Demand driven by transport electrification from 2030+. |
| 2024 | `EXTRAP` | Extrapolated backward from 2025–2030 slope (−0.13 TWh/yr). |
| 2026–2053 | `INTERP_EXTRAP` | Linear interpolation between 5-year Kinesys anchors. 2051–2053 extrapolated at 2045–2050 rate (+0.41 TWh/yr). Peak = Energy_GWh × 1e6 / (8760 × 0.58 load_factor). |

*Confidence: [MEDIUM] · Last updated: 2026-06-12*


<a id="bulgaria-pdemandprofile"></a>

### `pDemandProfile`

[&#8593; Bulgaria](#bulgaria)

**Source**: ENTSO-E Transparency Platform — Actual Total Load (hourly) (`entsoe_hourly_load`)

**Data / file**: ENTSO-E Transparency Platform, dataset: Actual Total Load per Bidding Zone. Downloaded via entsoe-py Python client (pre-analysis/studies/blacksea_2026/run_blacksea_data.py, step_entsoe_download()). Co…

**Also uses**: [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/)

> ⚠ **Needs review**: Profile to be recalculated once Bulgaria is fully integrated in the representative days pipeline (run_blacksea_data.py). Current profiles are valid ENTSO-E data but within-season variability (d1–d6 differentiation) is lost — all daytypes share the same seasonal mean. Rerun when all Black Sea countries are integrated.


**Method**: DIRECT seasonal mean from ENTSO-E Bulgaria hourly load, all d1-d6 daytypes

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | Bulgaria zone from blacksea_run1/reprdays_input/Load.csv (ENTSO-E hourly load, BG bidding zone, 2018–2024). Seasonal mean per (season, hour), normalized by peak. Q1_mean=0.691 (winter), Q2_mean=0.518 (spring), Q3_mean=0.533 (summer), Q4_mean=0.624. All d1–d6 daytypes share the same seasonal mean profile. Computed via compute_epm_demand.py --country BGR --profile. |

*Confidence: [HIGH] · Last updated: 2026-06-12*


<a id="bulgaria-pvreprofile"></a>

### `pVREProfile`

[&#8593; Bulgaria](#bulgaria)

**Source**: Renewables Ninja — PV and Wind capacity factors (`renewables_ninja`)

**Data / file**: https://www.renewables.ninja/ API-based hourly capacity factor time series at arbitrary lat/lon. Solar: fixed-tilt 35°, azimuth 180°, 10% system losses, MERRA-2 reanalysis. Wind: Gamesa G114-2000 turb…

**Also uses**: [Black Sea hourly load + VRE, representative-days pipeline (ENTSO-E · EPİAŞ · Renewables Ninja)](https://transparency.entsoe.eu/)

> ⚠ **Needs review**: Single centroid point for entire Bulgaria — does not capture regional variation (Black Sea coast wind is stronger than interior; Rhodope/Rila have different solar). Within-season variability (d1–d6) lost — all daytypes share seasonal mean. Offshore wind not included (Bulgaria has Black Sea offshore potential, not yet modeled). Rerun with per-daytype segmentation once full representative days pipeline is updated for all Black Sea countries.


**Method**: Renewables Ninja multi-year seasonal mean CF (2018–2023), Bulgaria centroid

| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `DIRECT` | PV and OnshoreWind from Renewables Ninja API output (blacksea_run1/ninja/ directory, Bulgaria centroid 25.5°E, 42.7°N). 6-year mean (2018–2023) seasonal mean per (season, hour), normalized by tech maximum. PV: Q1=0.219, Q2=0.307 (spring), Q3=0.323 (summer peak), Q4=0.191. OnshoreWind: Q1=0.943 (strong winter), Q2=0.706, Q3=0.702, Q4=0.757. All d1–d6 daytypes share the same seasonal mean (simplified approach). Computed via compute_epm_vre.py --country BGR. |

*Confidence: [HIGH] · Last updated: 2026-06-12*


<a id="bulgaria-pgendatainput"></a>

### `pGenDataInput`

[&#8593; Bulgaria](#bulgaria)

**Source**: Global Energy Monitor (GEM) — Global Integrated Power Tracker (GIPT) (`gem_gipt`)

**Data / file**: Global Energy Monitor — Global Integrated Power Tracker, September 2025 download. Covers power plants worldwide: technology, installed capacity (MW), status (operating / construction / announced / ret…

**Also uses**: World Bank Bulgaria CCDR — Kinesys/TIMES Energy Model (2026) (`bg_ccdr_kinesys_2026`)

**Also uses**: [EPM Generic Defaults](https://esmap-world-bank-group.github.io/EPM/input/input_parameter_guide/)

> ⚠ **Needs review**: (1) Lignite/coal aggregates use approximate StYr=2000 — replace with weighted-average commissioning year if unit-level data becomes available from ESO Bulgaria or EUROSTAT. (2) Kozloduy 7+8 commissioned year estimates (~2033/2036) from WNA and Kinesys WEM; update when official COD confirmed by NEC/BNRA licensing decision. (3) GEM hydro candidates (Turnu Magurele 840 MW, Batak 800 MW, Dospat 800 MW) conflict with Kinesys flat hydro trajectory — model will build these if economically viable; consider removing if not in national energy plan. (4) Generic Solar/Wind slots (10,000 MW each) are unconstrained expansion; calibrate BuildLimitperYear against national NECP targets once available. (5) Chaira PSP capacity 864 MW (GEM) vs Kinesys "Stg-Pumped" 216 MW — discrepancy unresolved; likely Kinesys models Chaira under Hydro and the 216 MW is a separate small pumped scheme. EPM uses 864 MW as Storage/Water (generation capacity).


**Method**: GEM GIPT plant-level (solar, wind, hydro, nuclear, gas) + Kinesys WEM aggregates (lignite, coal, old gas, battery) + Kinesys-derived heat rates

| Period | Method | Notes |
|--------|--------|-------|
| 2025–2053 | `DIRECT_GEM` | Individual plants from GEM GIPT September 2025 snapshot (BGR): Solar PV: 40+ operating plants (10–250 MW each) + AGG_SmallPV (841 MW). 2 committed solar farms (Tenovo 238 MW + Bobov Dol 100 MW). 5 candidate solar projects (160–800 MW) + Generic_Solar_Bulgaria (10,000 MW slot). Wind: 20 operating plants (10–156 MW). 3 candidate wind projects (40–250 MW) + Generic_Wind_Bulgaria (10,000 MW slot). Hydro: 13 reservoir hydro plants (60–375 MW), 3 large candidate projects (800–840 MW). Chaira PSP (864 MW) reclassified from ReservoirHydro → Storage/Water. Nuclear: Kozloduy 5 (1040 MW, StYr=1993) and 6 (1040 MW, StYr=1988) as existing. Kozloduy 7 (1000 MW, StYr=2033) and 8 (1000 MW, StYr=2036) as Committed (Status=2, upgraded from GEM "planned" per Kinesys WEM confirmed pipeline). Gas CCGTs: Plovdiv North (50 MW), Varna (210 MW), Toplofikacia Pleven (43 MW). 3 gas CCGT candidates (42–276 MW) from GEM pipeline. |
| 2025–2053 | `KINESYS_AGGREGATE` | Kinesys WEM 2025 used as authoritative source for plants not (or incompletely) captured by GEM due to retirement filter or CHP misclassification: Bulgaria_Agg_Lignite: 3966 MW ST/DomesticCoal, StYr=2000, RetrYr=2033.<br>  Fuel mapped to DomesticCoal (not a separate Lignite type) for consistency<br>  with Romania and to ensure pEmissionFactor lookup works in EPM.<br>  Kinesys WEM shows 3966 MW lignite operating in 2025, phasing to 925 MW<br>  by 2030 and 0 by 2035. RetrYr=2033 represents midpoint of phase-out.<br>Bulgaria_Agg_Coal: 669 MW ST/ImportedCoal, StYr=2000, RetrYr=2035.<br>  Hard coal is imported (no domestic bituminous coal reserves in Bulgaria).<br>  Kinesys WEM coal 2025=669 MW, declining to 559 MW (2030) and 110 MW (2035).<br>Bulgaria_Agg_Gas_Old: 1054 MW OCGT/Gas, StYr=2000, RetrYr=2040.<br>  Gap between Kinesys 2025 total gas (1357 MW) and GEM modern CCGTs (303 MW).<br>  Represents old gas turbines and CHP units not in GIPT or retired by GIPT filter.<br>Bulgaria_Battery: 728 MW Storage/Battery, StYr=2024, RetrYr=2034.<br>  Kinesys WEM 2025 shows 728 MW battery storage; absent from GIPT.<br>  10-year life assumed (battery replacement cycle). |
| 2025–2053 | `KINESYS_HEAT_RATES` | Heat rates derived from Kinesys WEM 2025 PowerUtility-Cons sheet (fuel consumption PJ) divided by Ele generation sheet (TWh): ST/DomesticCoal (lignite): 97.26 PJ / 8.34 TWh = 11.7 GJ/MWh (old Maritza East fleet,<br>  higher than generic 10.3 GJ/MWh due to low-grade lignite quality).<br>Nuclear VVER-1000: 164.77 PJ / 14.65 TWh = 11.3 GJ/MWh (Kozloduy actual<br>  operating efficiency, better than generic 12.5 GJ/MWh).<br>CCGT Gas: 15.70 PJ / 2.64 TWh = 5.9 GJ/MWh (modern Bulgarian CCGTs,<br>  slightly better than generic 6.4 GJ/MWh).<br>Coal heat rate from Kinesys (4.5 GJ/MWh) rejected — CHP attribution artifact (Deven CFB is cogenerating; fuel allocated partly to heat sector). OCGT/Gas old: generic 9.0 GJ/MWh (no Kinesys signal for old units). All other params (FOM, VOM, Life, Ramp, ResLimShare) left blank → model reads from pGenDataInputGeneric.csv (epm_generic_defaults). |
| build limits, 2026-08-30 | `CONSTRUCTED` | BuildLimitperYear rebuilt on a stated rule, every country and every generic technology at once. WHY. The column had been filled ad hoc and no longer bound anything: Turkiye carried 90,000 MW/yr of PV nationally (9 zones x 10,000), 45,000 MW/yr each of CCGT and OCGT against 2,933 MW installed, and 440,000 MW/yr of batteries (8 zones x 5 durations x 10,000) against a 64 GW peak. EPM applies the limit PER ROW (main.gms:722, vBuild.up = pGenData(BuildLimitperYear) * pWeightYear), so a generic candidate replicated across nine zones multiplies the national headroom by nine, and five duration variants multiply it again. With y.csv = 2025..2040 consecutive, pWeightYear = 1 and the value is literally MW per year. THE RULE. BuildLimitperYear is a physical plausibility guard - supply chain, permitting, grid connection, workforce - not a policy target and not an economic parameter. It is the growth-constraint form used by the capacity-expansion models that treat this explicitly: PyPSA (max_relative_growth, with max_growth as the seed), MESSAGEix (NEW_CAPACITY_CONSTRAINT_UP, with growth_new_capacity_up and initial_new_capacity_up), and ReEDS, which applies the same idea as a growth penalty rather than a hard bound.<br>    L(c,t,y) = min[ b0(c,t) * gamma_t^(y-2025) , plateau(c,t,y) ]<br>    b0(c,t)  = max( additions observed in 2025 in c , sigma_t * Peak2025(c) )<br>b0 is the rate the country has itself demonstrated, which is the only quantity that transfers between a 75 GW system and a 1.7 GW one: 2025 solar additions ran from 3.9 percent of peak in Azerbaijan to 36 percent in Armenia, a factor of nine, so no single share-of-peak coefficient can serve as the limit. That is what the earlier max[s x Peak, g x Fleet] form got wrong; it gave Turkiye 3.2x its record year and capped Armenia below what Armenia had just installed. sigma is the standing-start seed that lets a technology with no installed base begin; gamma is the annual growth of the build rate; the plateau is the logistic saturation, expressed in the metric of the empirical diffusion literature, annual additions as a share of national electricity supply. The plateau is never set below b0: a country is not capped below a rate it has already achieved. COEFFICIENTS (gamma per year / sigma, share of peak / plateau): PV +20% / 2.0% / 3% of supply; OnshoreWind +20% / 2.0% / 3% of supply; OffshoreWind +20% / 1.0% / 2% of supply; Storage +20% / 2.0% / 5% of peak; CCGT and OCGT +10% / 1.0% / 8% of peak; BiomassPlant +5% / 0.5% / 2% of peak. The observed 2025 additions and the literature anchors are in [build_rate_benchmarks_2026]. Every gamma and every sigma is an ASSUMPTION. EXCLUSIONS. Named projects keep BuildLimit = Capacity, their lead time being already carried by StYr. Hydro, pumped storage, nuclear and geothermal are outside the formula: they are limited by sites, not by build rate. NOTE that tech "ST" holds GEOTHERMAL in this deployment and not coal - the only ST candidates are Geo_Candidate_WestMed, _WestAna and _CenterAna, fuel Geothermal, 265 MW in total - so their limits are unchanged. An earlier draft of this rule pinned ST to zero as a no-new-coal assumption, which would have deleted Turkish geothermal instead. ZONE SPLIT. EPM has no country-level build constraint, so the national limit is divided over the zones carrying a generic candidate. Key = zone share of the pHours-weighted capacity factor from pVREProfile for PV and wind, zone share of peak demand otherwise, floored at 5 percent so no zone is locked out by a marginal resource, then renormalised. Where several rows sit in one zone for one technology they divide that zone's allowance. Keys sum to 1: the national total is exact and no slack factor is applied. TIME. The parameter has no year dimension, one value governs 2025-2040, so the growth path is flattened to its MEAN over the build years 2026-2040. The mean is the flattening that preserves the cumulative headroom the path allows; only the shape is lost. Making the limit genuinely time-varying without touching the GAMS would mean splitting every generic candidate into vintage tranches, about 120 extra rows across pGenDataInput, pStorageDataInput, pAvailabilityCustom and pCapexTrajectoriesCustom. Rejected: the deployment is already near its memory ceiling at 16 years. WHAT THE RULE IS NOT. It is not a forecast. It leaves Turkiye 8,707 MW/yr of PV and 5,448 MW/yr of onshore wind, against a national energy plan pace to 2035 of about 3.2 and 1.7 GW/yr. It is symmetric - it tightens the absurd values and loosens the over-constrained ones - which is what distinguishes it from a targeted cut of Turkiye. Method slide: blacksea_2026/BuildLimit_method.pptx, built by BuildLimit_method_slide.py from the deployment's own inputs. The same compute() writes these values into the CSVs, so the slide and the data cannot diverge. MW/yr, before -> after: |

*Confidence: [MEDIUM] · Last updated: 2026-08-30*


<a id="bulgaria-pfuelprice"></a>

### `pFuelPrice`

[&#8593; Bulgaria](#bulgaria)

**Source**: World Bank Bulgaria CCDR — Kinesys/TIMES Energy Model (2026) (`bg_ccdr_kinesys_2026`)

**Data / file**: BG-CCDR - Kinesys_2101_b2.20.26.xlsx. Local path: Data/Bulgaria/BG-CCDR - Kinesys_2101_b2.20.26.xlsx. Two main scenarios:

**Also uses**: World Bank EPM Romania v8.5 (2024, internal model) — primary data sources partially documented (`wb_epm_romania_v8`)

**Method**: Kinesys WEM ELCFuel sheet (scenario cl_wb7-WEM.Nuc-Y.Clim-HDNucRet-Y), 5-year milestones linearly interpolated to annual 2024-2053. Kinesys units assumed $/GJ (IEA/TIMES convention) → converted ×1.055 to $/MMBtu. Uranium trajectory copied from Romania (same study); Biomass flat at 5.0 $/MMBtu.


| Period | Method | Notes |
|--------|--------|-------|
| 2024–2053 | `KINESYS_WEM` | Gas (ELCNGA): 5.62 (2025) -> 6.31 (2030) -> 5.70 (2035) -> 6.56 (2040) -> 8.55 (2050)<br>  $/MMBtu. Dip in 2035 reflects WEM policy scenario (diversification of gas supply,<br>  increased LNG/interconnector use offset by carbon pricing effects).<br>DomesticCoal / lignite (ELCCOB): 2.26 (2025) -> 2.32 (2030) $/MMBtu, flat thereafter.<br>  Lignite priced as DomesticCoal to match pEmissionFactor table; price reflects<br>  Kinesys ELCCOB (brown coal/lignite fuel cost). Flat post-2030 because Kinesys shows<br>  near-zero lignite consumption by 2033 and does not model the price reliably after<br>  Bulgaria_Agg_Lignite retires (RetrYr=2033).<br>ImportedCoal (ELCCOA): 4.09 (2025) -> 3.93 (2030) -> 3.85 (2050) $/MMBtu.<br>  Slight declining trend in Kinesys WEM consistent with IEA coal price assumptions<br>  under stated policies.<br>Uranium: copied from Romania pFuelPrice trajectory (same BG-CCDR study).<br>  Kinesys ELCNUC (0.014 $/GJ) rejected — near-zero fuel cost is a modeling<br>  convention in Kinesys where all nuclear costs are in capacity/fixed charges;<br>  EPM requires a non-zero fuel price for dispatch economics.<br>  Romania trajectory (1.50->3.40->5.61 $/MMBtu) used as proxy.<br>Biomass: flat 5.0 $/MMBtu (Romania reference value, no Bulgaria-specific data). |

*Confidence: [MEDIUM] · Last updated: 2026-06-12*


<a id="bulgaria-pavailabilitycustom"></a>

### `pAvailabilityCustom`

[&#8593; Bulgaria](#bulgaria)

**Source**: Bulgarian quarterly availabilities — reconstructed by cross-referencing (ENTSO-E · PRIS · Kinesys · Turkish patterns) (`bulgaria_availability_reconstructed`)

**Data / file**: A composite source, not a document. Bulgarian quarterly availabilities were not collected: they were reconstructed through four distinct routes, each with its own reliability. The numeric detail is in…

**Also uses**: [World Nuclear Association — Reactor Database](https://world-nuclear.org/nuclear-reactor-database/)

**Also uses**: [ENTSO-E Transparency Platform — Actual Generation per Production Type, Bulgaria (hourly 2019-2023)](https://transparency.entsoe.eu/)

**Also uses**: World Bank Bulgaria CCDR — Kinesys/TIMES Energy Model (2026) (`bg_ccdr_kinesys_2026`)

> ⚠ **Needs review**: Nuclear: K5/K6 share same seasonal shape (only combined BG nuclear in ENTSO-E). Individual refueling schedules differ — verify with NEK (Kozloduy operator) annual outage plans. Q2 dip may shift to Q1 or Q3 in specific years. Hydro: high inter-annual variability (std ~0.07 across 2019-2023). Single profile approximation may mis-dispatch hydro in dry vs wet year scenarios. Confirm with ESO (Bulgarian grid operator) or NEK hydro generation data if available.


**Method**: PATTERN (lignite/coal) + SCALED (nuclear: PRIS EAF x ENTSO-E shape) + SCALED (hydro: ENTSO-E shape x Kinesys level)

| Period | Method | Notes |
|--------|--------|-------|
| 2024-2053 | `PATTERN` | Bulgaria_Agg_Lignite: Q1=0.50, Q2=0.45, Q3=0.50, Q4=0.50. Copied from Trakia_Agg_Lignite (all Turkiye zone lignite entries are identical). Reflects technical availability of ageing steam turbine fleet on domestic lignite. Generic default (0.65) is too optimistic for old Soviet-era plant. Q2 dip represents spring maintenance outage season. |
| 2024-2053 | `PATTERN` | Bulgaria_Agg_Coal: Q1=0.85, Q2=0.60, Q3=0.85, Q4=0.85. Copied from SouthEast_Agg_ImpCoal. Hard coal plants are more reliable than lignite (newer, better maintained). Generic default (0.65) is too conservative. Q2 dip reflects spring maintenance period. |
| 2024-2053 | `SCALED` | Bulgaria_Kozloduy_5: Q1=1.00, Q2=0.761, Q3=0.943, Q4=0.830 (mean=0.883). Bulgaria_Kozloduy_6: Q1=0.987, Q2=0.749, Q3=0.928, Q4=0.817 (mean=0.870). Method: ENTSO-E 2019-2023 actual BG nuclear generation (hourly, combined K5+K6) averaged to quarterly shape, then scaled so annual mean matches IAEA PRIS EAF. PRIS EAF (2015-2024 average): K5=88.4%, K6=87.0% (world-nuclear.org reactor database). Q2 deep dip (to ~0.75) reflects annual refueling outage — confirmed by ENTSO-E data (much deeper than Cernavoda CANDU pattern ~10%; VVER-1000 refueling ~25% reduction). Q1 theoretical CF slightly >1.0 from ENTSO-E (uprating artefact: actual combined capacity ~2006 MW not 2080 MW); capped at 1.00 for K5, 0.987 for K6 after scaling. Kozloduy_7/8 (committed VVER-1200): no custom entry — Generic_Nuclear 0.85 flat used. |
| 2024-2053 | `SCALED` | All 13 existing + 3 candidate ReservoirHydro plants: uniform profile Q1=0.214, Q2=0.248, Q3=0.160, Q4=0.156 (mean=0.195). CRITICAL: EPM has no separate hydro energy budget constraint (no pHydroEnergy). pAvailabilityCustom is the ONLY quarterly water availability cap for reservoir hydro. Without custom entry, model dispatches at Generic default 0.85 — ~4x actual output. Method: ENTSO-E 2019-2023 quarterly shape (Hydro Water Reservoir, 1719 MW basis, shape Q1:Q2:Q3:Q4 = 1.10:1.27:0.82:0.80) scaled so mean = Kinesys 2025 CF (0.195). ENTSO-E historical mean = 0.146 (drier-than-average 2019-2023); Kinesys forward projection used for level as more representative of future water availability. Seasonal pattern: spring peak (Q2, snowmelt Rhodopes/Arda/Vacha systems), summer-autumn low (Q3/Q4). Consistent with Bulgarian hydrology. Limitation: single uniform profile for all plants; individual plant hydrological data not available. High inter-annual variability (CF range 0.12-0.29 across years). |

*Confidence: [MEDIUM] · Last updated: 2026-06-12*


---
