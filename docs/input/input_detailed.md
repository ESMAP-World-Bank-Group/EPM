# Input Catalog

Complete documentation for all EPM input files. For how the inputs fit together, see [Setup](input_setup.md). For typical parameter values, see [Typical Values](input_parameter_guide.md).

---

## Quick Reference

| Section | Files |
|---|---|
| [Configuration](#configuration) | `pSettings.csv` · `y.csv` · `zcmap.csv` |
| [Resources](#resources) | `ftfindex.csv` · `pTechData.csv` · `pFuelCarbonContent.csv` |
| [Load](#load) | `pDemandProfile.csv` · `pDemandForecast.csv` · `pDemandData.csv` · `pEnergyEfficiencyFactor.csv` · `sRelevant.csv` |
| [Supply](#supply) | `pGenDataInputCustom.csv` · `pGenDataInputDefault.csv` · `pAvailabilityCustom.csv` · `pAvailabilityDefault.csv` · `pVREProfile.csv` · `pVREgenProfile.csv` · `pCapexTrajectoriesCustom.csv` · `pCapexTrajectoriesDefault.csv` · `pFuelPrice.csv` |
| [Constraints](#constraints) | `pCarbonPrice.csv` · `pEmissionsCountry.csv` · `pEmissionsTotal.csv` · `pMaxFuelLimit.csv` |
| [Reserves](#reserves) | `pPlanningReserveMargin.csv` · `pSpinningReserveReqCountry.csv` · `pSpinningReserveReqSystem.csv` |
| [Trade](#trade) | `pTransferLimit.csv` · `pNewTransmission.csv` · `pLossFactorInternal.csv` · `zext.csv` · `pExtTransferLimit.csv` · `pTradePrice.csv` · `pMaxAnnualExternalTradeShare.csv` |
| [Hydrogen](#hydrogen) | See `h2/` subfolder |

---

## Configuration

??? "pSettings.csv"

    Controls model behavior through feature flags and economic parameters. Located in `config/pSettings.csv`.

    !!! warning
        The validated `pSettings.csv` template includes the full set of required parameters. Starting from EPM v2024.09, always copy the latest file to avoid parameters defaulting to zero.

    **Structure**: `Parameter`, `Abbreviation`, `Value` — each row has a human-readable label, the GAMS abbreviation, and the value. All percentages are stored as fractions (e.g., `0.06` = 6%). Toggle parameters use `0` / `1`.

    **Example**: [pSettings.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test/config/pSettings.csv)

    | CSV label | Abbreviation | Default | Units |
    |---|---|---|---|
    | **Core** | | | |
    | Weighted Average Cost of Capital, % | `WACC` | 0.06 | Fraction |
    | Discount rate, % | `DR` | 0.06 | Fraction |
    | Cost of unserved energy | `VoLL` | 1000 | $/MWh |
    | Cost of reserve shortfall | `ReserveVoLL` | 60000 | $/MW |
    | Spin Reserve VoLL | `SpinReserveVoLL` | 60 | $/MWh |
    | Cost of surplus power | `CostSurplus` | 0 | $/MWh |
    | Cost of curtailment | `CostCurtail` | 0 | $/MWh |
    | CO₂ backstop penalty | `CO2backstop` | 300 | $/tCO₂ |
    | H2 unserved cost | `H2UnservedCost` | 3000 | $/tCO₂ |
    | **Interconnection** | | | |
    | Activate internal zone exchange | `fEnableInternalExchange` | 1 | Toggle |
    | Remove internal transfer limits | `fRemoveInternalTransferLimit` | 0 | Toggle |
    | Allow transmission expansion | `fAllowTransferExpansion` | 1 | Toggle |
    | Activate external trade | `fEnableExternalExchange` | 1 | Toggle |
    | Max external import share | `sMaxHourlyImportExternalShare` | 1 | Fraction |
    | Max external export share | `sMaxHourlyExportExternalShare` | 1 | Fraction |
    | **Optional Features** | | | |
    | Include carbon price | `fEnableCarbonPrice` | 0 | Toggle |
    | Include energy efficiency | `fEnableEnergyEfficiency` | 0 | Toggle |
    | Include CSP optimization | `fEnableCSP` | 0 | Toggle |
    | Include storage | `fEnableStorage` | 1 | Toggle |
    | Allow economic retirement | `fEnableEconomicRetirement` | 0 | Toggle |
    | Use simplified demand | `fUseSimplifiedDemand` | 1 | Toggle |
    | **Planning Reserves** | | | |
    | Count transmission for planning reserves | `fCountIntercoForReserves` | 1 | Toggle |
    | Apply planning reserve constraint | `fApplyPlanningReserveConstraint` | 1 | Toggle |
    | System planning reserve margin | `sReserveMarginPct` | 0.10 | Fraction |
    | **Spinning Reserves** | | | |
    | Apply country spinning reserve | `fApplyCountrySpinReserveConstraint` | 1 | Toggle |
    | Apply system spinning reserve | `fApplySystemSpinReserveConstraint` | 0 | Toggle |
    | VRE forecast error for spinning reserve | `sVREForecastErrorPct` | 0.15 | Fraction |
    | Transmission contribution to spinning reserve | `sIntercoReserveContributionPct` | 0 | Fraction |
    | **Policy** | | | |
    | Apply country CO₂ constraint | `fApplyCountryCo2Constraint` | 0 | Toggle |
    | Apply system CO₂ constraint | `fApplySystemCo2Constraint` | 0 | Toggle |
    | Minimum RE share | `sMinRenewableSharePct` | 0 | Fraction |
    | RE target year | `sRenewableTargetYear` | — | Year |
    | Apply fuel constraints | `fApplyFuelConstraint` | 0 | Toggle |
    | Apply capital budget constraint | `fApplyCapitalConstraint` | 0 | Toggle |
    | Max capital investment | `sMaxCapitalInvestment` | — | Billion $ |
    | **Plant Operations** | | | |
    | Apply min generation constraint | `fApplyMinGenCommitment` | 0 | Toggle |
    | Apply ramp constraints | `fApplyRampConstraint` | 0 | Toggle |
    | Apply startup costs *(Dispatch Mode only)* | `fApplyStartupCost` | 0 | Toggle |
    | **Hydrogen** | | | |
    | Allow H2 capex trajectory | `fEnableCapexTrajectoryH2` | 0 | Toggle |
    | Include H2 production | `fEnableH2Production` | 0 | Toggle |

??? "y.csv"

    Defines the years included in the intertemporal optimization.

    - **Structure**: Single column `y` — list of integer years.
    - **Example**: [y.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/y.csv)

??? "zcmap.csv"

    Defines the zones and countries included in the model.

    - **Structure**: Two columns — `zone`, `country`.
    - **Example**: [zcmap.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/zcmap.csv)

---

## Resources

??? "ftfindex.csv"

    List of fuels recognized by EPM: Coal, Gas, Water, Solar, Wind, Import, HFO, LFO, Uranium, CSP, Battery, Diesel, Biomass, Geothermal, LNG.

    - **Structure**: `Fuel`, `Index` (index column is deprecated and will be removed).
    - **Location**: `epm/resources/headers/ftfindex.csv`

??? "pTechData.csv"

    List of technologies recognized by EPM.

    - **Structure**: Technology name · HourlyVariation (1 = VRE) · RETechnology (1 = renewable).
    - **Location**: `epm/resources/headers/pTechData.csv`

    | Technology | Description |
    |---|---|
    | `OCGT` | Open Cycle Gas Turbine |
    | `CCGT` | Combined Cycle Gas Turbine |
    | `ST` | Steam Turbine |
    | `ICE` | Internal Combustion Engine |
    | `OnshoreWind` | Onshore wind |
    | `OffshoreWind` | Offshore wind |
    | `PV` | Photovoltaic |
    | `PVwSTO` | PV with battery storage |
    | `CSPPlant` | Concentrated Solar Power |
    | `Storage` | Battery storage |
    | `ReservoirHydro` | Reservoir hydro |
    | `ROR` | Run-of-river hydro |
    | `BiomassPlant` | Biomass |
    | `CHP` | Combined Heat and Power |
    | `ImportTransmission` | Imports from external zones |

??? "pFuelCarbonContent.csv"

    Carbon emission factors by fuel type.

    - **Structure**: `Fuel`, `value` — values in tCO₂/MMBtu.
    - **Location**: `epm/resources/headers/pFuelCarbonContent.csv`

---

## Load

??? "pDemandProfile.csv"

    Normalized hourly demand shape used to distribute annual energy across time steps.

    - **Structure**: Wide format — `z, q, d, t1, t2, ..., t24`. Zone, season, day as row index; hours as columns. Values normalized [0–1].
    - **Example**: [pDemandProfile.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pDemandProfile.csv)

??? "pDemandForecast.csv"

    Annual demand forecast by zone — total energy (GWh) and peak (MW).

    - **Structure**: Wide format — `z, type, 2025, 2030, ..., 2050`. Zone and type as row index; years as columns.
    - **Example**: [pDemandForecast.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pDemandForecast.csv)

??? "pDemandData.csv"

    Alternative full demand definition combining profile and forecast in one file.

    - **Structure**: Wide format — `z, q, d, y, t1, t2, ..., t24`. Zone, season, day, year as row index; hours as columns.
    - **Example**: [pDemandData.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pDemandData.csv)

??? "pEnergyEfficiencyFactor.csv"

    Annual efficiency improvement factors by zone. Scales demand downward over time when `fEnableEnergyEfficiency = 1`.

    - **Structure**: `zone`, `year` → efficiency factor (fraction; 1 = no improvement).
    - **Example**: [pEnergyEfficiencyFactor.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pEnergyEfficiencyFactor.csv)

??? "sRelevant.csv"

    Filters which demand scenarios from `pDemandForecast.csv` or `pDemandData.csv` are loaded.

    - **Structure**: Single column listing active scenario names.
    - **Example**: [sRelevant.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/sRelevant.csv)

---

## Supply

??? "pGenDataInputCustom.csv"

    Main plant-level data table defining all power plants with their technical and economic characteristics.

    - **Example**: [pGenDataInputCustom.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pGenDataInputCustom.csv)

    | Column | Type | Description |
    |---|---|---|
    | `gen` | string | Plant name |
    | `zone` | string | Zone (as defined in `zcmap.csv`) |
    | `tech` | string | Technology (as defined in `pTechData.csv`) |
    | `fuel` | string | Primary fuel (as defined in `ftfindex.csv`) |
    | `StYr` | year | Start year of operation |
    | `RetrYr` | year | Retirement year |
    | `Capacity` | MW | Installed capacity |
    | `Status` | int | 1 = existing · 2 = committed · 3 = candidate |
    | `MinLimitShare` | fraction | Minimum generation as share of capacity |
    | `HeatRate` | MMBtu/MWh | Heat rate on primary fuel |
    | `RampUpRate` | fraction/h | Maximum ramp-up rate |
    | `RampDnRate` | fraction/h | Maximum ramp-down rate |
    | `OverLoadFactor` | fraction | Overload capability factor |
    | `ResLimShare` | fraction | Max share of capacity available for reserves |
    | `Capex` | M$/MW | Capital expenditure per MW |
    | `FOMperMW` | $/MW-yr | Fixed O&M cost per MW |
    | `VOM` | $/MWh | Variable O&M cost per MWh |
    | `ReserveCost` | $/MW-yr | Additional cost for reserve capacity |
    | `Life` | years | Operational lifetime |
    | `UnitSize` | MW | Size of a single unit |
    | `fuel2` | string | Secondary fuel (optional) |
    | `HeatRate2` | MMBtu/MWh | Heat rate on secondary fuel |
    | `DescreteCap` | MW | Discrete capacity increment for expansion |
    | `BuildLimitperYear` | MW/yr | Max capacity addition per year |
    | `MaxTotalBuild` | MW | Max total capacity for this plant |

??? "pGenDataInputDefault.csv"

    Default parameters by zone · technology · fuel. Fills missing values in `pGenDataInputCustom.csv`. Same column structure.

    - **Source**: Initially based on CCDR guidelines — see [Parameter Guide](input_parameter_guide.md).
    - **Example**: [pGenDataInputDefault.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pGenDataInputDefault.csv)

??? "pAvailabilityCustom.csv"

    Seasonal availability factors for individual plants, accounting for maintenance or hydro storage constraints.

    - **Structure**: Wide format — `g, Q1, Q2, Q3, Q4`. Plant as row index; seasons as columns. Values [0–1].
    - **Example**: [pAvailabilityCustom.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pAvailabilityCustom.csv)

??? "pAvailabilityDefault.csv"

    Default seasonal availability by zone · technology · fuel. Fills missing values in `pAvailabilityCustom.csv`.

    - **Structure**: `zone, tech, fuel` as row index; seasons as columns. Values [0–1].
    - **Example**: [pAvailabilityDefault.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pAvailabilityDefault.csv)

??? "pVREProfile.csv"

    VRE generation profiles at the **technology level**. Automatically populates `pVREgenProfile.csv` for all plants of that technology.

    - **Structure**: Wide format — `z, tech, q, d, t1, t2, ..., t24`. Zone, technology, season, day as row index; hours as columns. Values [0–1].
    - **Example**: [pVREProfile.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pVREProfile.csv)

??? "pVREgenProfile.csv"

    VRE generation profiles at the **plant level**. Takes priority over `pVREProfile.csv` when specified.

    - **Structure**: Wide format — `g, q, d, t1, t2, ..., t24`. Plant, season, day as row index; hours as columns. Values [0–1].
    - **Example**: [pVREgenProfile.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pVREgenProfile.csv)

??? "pCapexTrajectoriesCustom.csv"

    Plant-level CAPEX cost trajectory. A factor of 0.5 in 2050 means 50% of the initial CAPEX value in `pGenDataInput.csv`.

    - **Structure**: Wide format — `gen, 2023, 2024, ..., 2050`. Plant as row index; years as columns. Values [0–1].
    - **Example**: [pCapexTrajectoriesCustom.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pCapexTrajectoriesCustom.csv)

??? "pCapexTrajectoriesDefault.csv"

    Default CAPEX trajectories by zone · technology · fuel. Fills missing values in `pCapexTrajectoriesCustom.csv`.

    - **Structure**: `zone, tech, fuel` as row index; years as columns. Values [0–1].
    - **Example**: [pCapexTrajectoriesDefault.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pCapexTrajectoriesDefault.csv)

??? "pFuelPrice.csv"

    Fuel price projections by country and year.

    - **Structure**: `country, fuel` as row index; years as columns. Values in $/MMBtu.
    - **Example**: [pFuelPrice.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pFuelPrice.csv)

---

## Constraints

??? "pCarbonPrice.csv"

    Carbon price trajectory over time.

    - **Structure**: `year` → carbon price ($/tCO₂).
    - **Enabled by**: `fEnableCarbonPrice = 1`
    - **Example**: [pCarbonPrice.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/constraint/pCarbonPrice.csv)

??? "pEmissionsCountry.csv"

    Country-level CO₂ emission caps by year.

    - **Structure**: `zone, year` → CO₂ limit (tCO₂).
    - **Enabled by**: `fApplyCountryCo2Constraint = 1`
    - **Example**: [pEmissionsCountry.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pEmissionsCountry.csv)

??? "pEmissionsTotal.csv"

    System-wide CO₂ emission cap by year.

    - **Structure**: `year` → CO₂ limit (tCO₂).
    - **Enabled by**: `fApplySystemCo2Constraint = 1`
    - **Example**: [pEmissionsTotal.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pEmissionsTotal.csv)

??? "pMaxFuelLimit.csv"

    Maximum fuel consumption limits by zone and year.

    - **Structure**: `zone, fuel, year` → max consumption (MMBtu).
    - **Enabled by**: `fApplyFuelConstraint = 1`
    - **Example**: [pMaxFuelLimit.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pMaxFuelLimit.csv)

---

## Reserves

!!! note
    Spinning reserve requirements combine the values in these files **plus** the VRE forecast error defined by `sVREForecastErrorPct` in `pSettings.csv`.

??? "pPlanningReserveMargin.csv"

    Minimum planning reserve margin as a share of peak demand, by country and year.

    - **Structure**: Wide format — `c, 2025, 2030, ..., 2050`. Country as row index; years as columns. Values are fractions.
    - **Enabled by**: `fApplyPlanningReserveConstraint = 1`
    - **Example**: [pPlanningReserveMargin.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/reserve/pPlanningReserveMargin.csv)

??? "pSpinningReserveReqCountry.csv"

    Minimum spinning reserve requirement at the country level, by year.

    - **Structure**: Wide format — `c, 2025, 2030, ..., 2050`. Country as row index; years as columns. Values in MW.
    - **Enabled by**: `fApplyCountrySpinReserveConstraint = 1`
    - **Example**: [pSpinningReserveReqCountry.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/reserve/pSpinningReserveReqCountry.csv)

??? "pSpinningReserveReqSystem.csv"

    Total system-wide spinning reserve requirement, by year.

    - **Structure**: Wide format — single row; years as columns. Values in MW.
    - **Enabled by**: `fApplySystemSpinReserveConstraint = 1`
    - **Example**: [pSpinningReserveReqSystem.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/reserve/pSpinningReserveReqSystem.csv)

---

## Trade

EPM models two types of exchange:

| Type | Scope | Files |
|---|---|---|
| **Internal** | Between modeled zones, explicit network | `pTransferLimit` · `pNewTransmission` · `pLossFactorInternal` |
| **External** | Price-driven trade outside the model | `zext` · `pExtTransferLimit` · `pTradePrice` · `pMaxAnnualExternalTradeShare` |

??? "pTransferLimit.csv"

    Available transfer capacity between internal zones, defining the network topology.

    - **Structure**: `from, to, season` as row index; years as columns. Values in MW.
    - **Example**: [pTransferLimit.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test/trade/pTransferLimit.csv)

??? "pNewTransmission.csv"

    Candidate transmission lines for network expansion. Used when `fAllowTransferExpansion = 1`.

    !!! warning
        Do not include lines from `pNewTransmission.csv` in `pTransferLimit.csv` — they will be double-counted. Each line must be specified only once (order of From/To does not matter).

    | Column | Description |
    |---|---|
    | `From` / `To` | Origin and destination zones |
    | `EarliestEntry` | Earliest year the line can be built |
    | `MaximumNumOfLines` | Max number of lines (whole units only) |
    | `CapacityPerLine` | Capacity per line (MW) |
    | `CostPerLine` | Investment cost per line |
    | `Life` | Lifespan (years) |
    | `Status` | 2 = committed · 3 = candidate |

    - **Example**: [pNewTransmission.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test/trade/pNewTransmission.csv)

??? "pLossFactorInternal.csv"

    Transmission loss factors for each internal line. Required when `fEnableInternalExchange = 1`.

??? "zext.csv"

    Lists external zones available for trade. Not modeled explicitly — only their prices and capacity limits are used.

    - **Structure**: Single column `zone`.
    - **Example**: [zext.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test/trade/zext.csv)

??? "pExtTransferLimit.csv"

    Seasonal transfer capacity between internal and external zones.

    - **Structure**: Wide format — `Internal zone, External zone, Seasons, Import-Export, 2024, ..., 2050`. Years as columns. Values in MW.
    - **Example**: [pExtTransferLimit.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/trade/pExtTransferLimit.csv)

??? "pTradePrice.csv"

    Hourly import/export prices from external zones.

    - **Structure**: Wide format — `zext, q, d, y, t1, t2, ..., t24`. External zone, season, day, year as row index; hours as columns. Values in $/MWh.
    - **Example**: [pTradePrice.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test/trade/pTradePrice.csv)

??? "pMaxAnnualExternalTradeShare.csv"

    Maximum share of total country demand that imports and exports can represent.

    - **Structure**: `year` as row index; `country` as columns. Values are fractions [0–1].
    - **Example**: [pMaxAnnualExternalTradeShare.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test/trade/pMaxAnnualExternalTradeShare.csv)

---

## Hydrogen

Hydrogen modeling is enabled via `fEnableH2Production = 1` in `pSettings.csv`. Input files are in the `h2/` subfolder. Full documentation is in progress — see the [h2 folder](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/h2) for current file examples.
