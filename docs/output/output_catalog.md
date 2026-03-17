# Output Catalog

Full reference of all EPM output variables — name, description, dimensions, units, and whether they are exported to CSV by default or available in GDX only.

---

## Quick Reference

| Category | Key variables |
|---|---|
| [Capacity & Transmission](#capacity-transmission) | `pCapacityPlant` · `pCapacityFuel` · `pCapacitySummary` · `pTransmissionCapacity` |
| [Energy & Dispatch](#energy-dispatch) | `pEnergyFuel` · `pEnergyBalance` · `pDispatchFuel` |
| [Costs & Investment](#costs-investment) | `pCapexInvestment` · `pCosts` · `pNetPresentCostSystem` · `pHourlyPrice` |
| [Utilization](#utilization) | `pUtilizationPlant` · `pUtilizationFuel` · `pUtilizationTechFuel` |
| [Reserves](#reserves) | `pReserveSpinningFuelZone` · `pCapacityCredit` |
| [Interconnection](#interconnection) | `pInterchange` · `pInterconUtilization` · `pCongestionShare` |
| [Emissions](#emissions) | `pEmissionsZone` · `pEmissionsIntensityZone` |
| [Financial & Fuel](#financial-fuel) | `pPlantAnnualLCOE` · `pNetPresentCostPerMWh` · `pFuelCosts` |

---

## Capacity & Transmission

??? "Capacity & Transmission variables"

    All locations relative to the scenario folder (`output/<run_id>/<scenario>/`).

    | Variable | Description | Dimensions | Units | CSV |
    |---|---|---|---|---|
    | `pCapacityPlant` | Installed capacity by generator, fuel, and year | plant · fuel · year | MW | Yes |
    | `pCapacityFuel` | Installed capacity aggregated by fuel and zone | fuel · zone · year | MW | Yes |
    | `pCapacityFuelCountry` | Installed capacity aggregated by fuel and country | fuel · country · year | MW | Yes |
    | `pNewCapacityFuel` | New capacity builds by fuel and zone | fuel · zone · year | MW | Yes |
    | `pNewCapacityFuelCountry` | New capacity builds by fuel and country | fuel · country · year | MW | Yes |
    | `pCapacitySummary` | Capacity balance by zone and year (existing, new, retirements) | zone · year | MW | Yes |
    | `pCapacitySummaryCountry` | Capacity balance aggregated to country level | country · year | MW | Yes |
    | `pTransmissionCapacity` | Total available transfer capacity between zones | zone pair · year | MW | Yes |
    | `pNewTransmissionCapacity` | Incremental transfer capacity required on each interconnector | zone pair · year | MW | Yes |

---

## Energy & Dispatch

??? "Energy & Dispatch variables"

    | Variable | Description | Dimensions | Units | CSV |
    |---|---|---|---|---|
    | `pEnergyPlant` | Annual generation by plant | plant · year | GWh | Yes |
    | `pEnergyFuel` | Annual generation by fuel and zone | fuel · zone · year | GWh | Yes |
    | `pEnergyFuelCountry` | Annual generation by fuel and country | fuel · country · year | GWh | Yes |
    | `pEnergyBalance` | Zone-level energy balance (demand, generation, imports, losses) | zone · year | GWh | Yes |
    | `pDispatchFuel` | Dispatch by fuel, zone, and operating period | fuel · zone · period | MW | Yes |
    | `pDispatch` | System dispatch by attribute (supply, demand, net trade) | attribute · period | MW | Yes |

---

## Costs & Investment

??? "Costs & Investment variables"

    | Variable | Description | Dimensions | Units | CSV |
    |---|---|---|---|---|
    | `pCapexInvestment` | System capital expenditures by component | component · year | M$ | Yes |
    | `pCapexInvestmentComponent` | CAPEX broken down by investment component | component · year | M$ | Yes |
    | `pCapexInvestmentPlant` | Capital spending allocated to individual plants | plant · year | M$ | Yes |
    | `pCapexInvestmentTransmission` | Capital spending on transmission assets | zone pair · year | M$ | Yes |
    | `pCostsPlant` | Operating costs per plant (fixed, variable, fuel) | plant · year | M$ | Yes |
    | `pCosts` | Annual cost breakdown by zone and cost attribute | zone · attribute · year | M$ | Yes |
    | `pCostsCountry` | Annual cost breakdown aggregated to country level | country · attribute · year | M$ | Yes |
    | `pNetPresentCostSystem` | Total discounted system cost components | component | M$ | Yes |
    | `pNetPresentCostSystemPerMWh` | System cost normalized by energy | — | $/MWh | Yes |
    | `pHourlyPrice` | Marginal price signal by zone and period | zone · period | $/MWh | Yes |

---

## Utilization

??? "Utilization variables"

    | Variable | Description | Dimensions | Units | CSV |
    |---|---|---|---|---|
    | `pUtilizationPlant` | Capacity factor of each plant by year | plant · year | % | Yes |
    | `pUtilizationFuel` | Capacity factor aggregated by fuel and zone | fuel · zone · year | % | Yes |
    | `pUtilizationFuelCountry` | Capacity factor aggregated by fuel and country | fuel · country · year | % | Yes |
    | `pUtilizationTechFuel` | Capacity factor by technology-fuel combination and zone | tech · fuel · zone · year | % | Yes |
    | `pUtilizationTechFuelCountry` | Capacity factor by technology-fuel combination and country | tech · fuel · country · year | % | Yes |

---

## Reserves

??? "Reserves variables"

    | Variable | Description | Dimensions | Units | CSV |
    |---|---|---|---|---|
    | `pReserveSpinningPlantZone` | Spinning reserve provision by plant and zone | plant · zone · period | MW | Yes |
    | `pReserveSpinningPlantCountry` | Spinning reserve provision by plant and country | plant · country · period | MW | Yes |
    | `pReserveSpinningFuelZone` | Spinning reserve provision by fuel and zone | fuel · zone · period | MW | Yes |
    | `pCapacityCredit` | Capacity credits assigned to generators by zone | plant · zone · year | MW | Yes |

---

## Interconnection

??? "Interconnection variables"

    | Variable | Description | Dimensions | Units | CSV |
    |---|---|---|---|---|
    | `pInterchange` | Interzonal power flows by direction and period | zone pair · period | MW | Yes |
    | `pInterchangeCountry` | Cross-border exchange aggregated by country | country pair · period | MW | Yes |
    | `pInterconUtilization` | Utilization rate of each interconnector | zone pair · year | % | Yes |
    | `pCongestionShare` | Share of congestion rents per interconnection | zone pair · year | — | Yes |

---

## Emissions

??? "Emissions variables"

    | Variable | Description | Dimensions | Units | CSV |
    |---|---|---|---|---|
    | `pEmissionsZone` | Annual CO₂ emissions by zone | zone · year | kt CO₂e | Yes |
    | `pEmissionsIntensityZone` | Emissions intensity of supply by zone | zone · year | t CO₂/MWh | Yes |

---

## Financial & Fuel

??? "Financial & Fuel variables"

    | Variable | Description | Dimensions | Units | CSV |
    |---|---|---|---|---|
    | `pPlantAnnualLCOE` | Levelized cost of electricity per plant | plant · year | $/MWh | Yes |
    | `pNetPresentCostPerMWh` | Average total system cost per zone | zone | $/MWh | Yes |
    | `pFuelCosts` | Fuel price trajectories by fuel and year | fuel · year | $/MMBtu | Yes |
    | `pFuelCostsCountry` | Fuel price trajectories with country adjustments | fuel · country · year | $/MMBtu | Yes |
    | `pSettings` | Scenario settings captured at run time | — | — | Yes |
    | `zcmap` | Zone-to-country mapping | zone · country | — | Yes |

---

## GDX-only outputs

The following are written to `epmresults.gdx` but not exported to CSV by default. Use `postprocessing.utils.gdx_to_csv` or `extract_epm_folder(..., save_to_csv=True)` to convert.

??? "GDX-only variables"

    | Group | Variables |
    |---|---|
    | **Capacity detail** | `pCapacityTechFuel` · `pCapacityTechFuelCountry` · `pCapacityPlantH2` |
    | **Retirements** | `pRetirementsPlant` · `pRetirementsFuel` · `pRetirementsFuelCountry` · `pRetirementsCountry` |
    | **New capacity by tech** | `pNewCapacityTech` · `pNewCapacityTechCountry` |
    | **Generation detail** | `pEnergyTechFuel` · `pEnergyTechFuelCountry` · `pDispatchPlant` · `pUtilizationTech` |
    | **Energy balance ext.** | `pEnergyBalanceCountry` · `pEnergyBalanceH2` · `pEnergyBalanceCountryH2` |
    | **Fuel consumption** | `pFuelConsumption` · `pFuelConsumptionCountry` |
    | **Reserve margins** | `pReserveMargin` · `pReserveMarginCountry` |
    | **Interconnection detail** | `pLossesTransmission` · `pHourlyInterchangeExternal` · `pYearlyInterchangeExternal` · `pInterchangeExternalExports/Imports` · `pInterconUtilizationExternal*` |
    | **Trade costs** | `pImportCostsInternal` · `pExportRevenuesInternal` · `pTradeSharedBenefits` · `pCongestionRevenues` |
    | **Emissions detail** | `pEmissionsCountrySummary` · `pEmissionsIntensityCountry` · `pEmissionMarginalCosts*` |
    | **Price detail** | `pPrice` · `pPriceExport` · `pPriceImport` · `pPriceHub` · `pPriceCountry*` |
    | **Technology balances** | `pCSPBalance` · `pPVwSTOBalance` · `pStorageBalance` · `pSolarPower` · `pSolarEnergyZone` |
    | **System metrics** | `pNetPresentCostCountryPerMWh` · `pDiscountedDemand*` · `pCostsSystem` · `pVarCost` · `pSolverParameters` |
