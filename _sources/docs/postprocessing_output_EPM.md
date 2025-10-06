# Postprocessing Output Overview

## 1. CAPACITY

| Variable | Description |
|----------|-------------|
| pCapacityPlant | Installed capacity \[MW\] by plant, zone, and year |
| pCapacityTechFuel | Installed capacity \[MW\] by technology, fuel, and zone |
| pCapacityFuel | Installed capacity \[MW\] by fuel and zone |
| pCapacityTechFuelCountry | Installed capacity \[MW\] by technology, fuel, and country |
| pCapacityFuelCountry | Installed capacity \[MW\] by fuel and country |
| pRetirementsPlant | Retired capacity \[MW\] by plant, zone, and year |
| pRetirementsFuel | Retired capacity \[MW\] by fuel and zone |
| pRetirementsCountry | Total retired capacity \[MW\] by country and year |
| pRetirementsFuelCountry | Retired capacity \[MW\] by fuel and country |
| pNewCapacityFuel | Newly added capacity \[MW\] by fuel and zone |
| pNewCapacityTech | Newly added capacity \[MW\] by technology and zone |
| pNewCapacityFuelCountry | Newly added capacity \[MW\] by fuel and country |
| pNewCapacityTechCountry | Newly added capacity \[MW\] by technology and country |
| pAnnualTransmissionCapacity | Total available transmission capacity \[MW\] between internal zones |
| pAdditionalCapacity | Additional transmission capacity \[MW\] between internal zones |
| pCapacitySummary | Summary of capacity indicators \[MW\] by zone and year |
| pCapacitySummaryCountry | Summary of capacity indicators \[MW\] by country and year |
| pCapacityPlantH2 | Installed electrolyzer capacity \[MW\] by zone and year |

## 2. COSTS

| Variable | Description |
|----------|-------------|
| pCostsPlant | Yearly cost breakdown by plant and year |
| pCapexInvestment | Annual CAPEX investment in USD by zone and year |
| pPrice | Marginal cost \[USD/MWh\] by zone, time, and year |
| pImportCostsInternal | Import costs from internal zones \[USD\] by zone and year |
| pExportRevenuesInternal | Export revenues to internal zones \[USD\] by zone and year |
| pCongestionRevenues | Congestion rents \[USD\] from saturated internal lines by year |
| pTradeSharedBenefits | Shared congestion rents \[USD\] allocated between zones |
| pCostZone | Annual cost summary \[million USD\] by zone and year |
| pCostCountry | Annual cost summary \[million USD\] by country and year |
| pCostAverageCountry | Average annual cost \[million USD\] by country (undiscounted) |
| pCostSystem | System-level cost summary \[million USD\], weighted and discounted |
| pFuelCosts | Annual fuel costs \[million USD\] by fuel, zone, and year |
| pFuelCostsCountry | Annual fuel costs \[million USD\] by fuel, country, and year |
| pFuelConsumption | Annual fuel consumption \[MMBtu\] by fuel, zone, and year |
| pFuelConsumptionCountry | Annual fuel consumption \[MMBtu\] by fuel, country, and year |

## 3. ENERGY BALANCE

| Variable | Description |
|----------|-------------|
| pEnergyPlant | Annual energy generation \[GWh\] by plant, zone, and year |
| pEnergyFuel | Annual energy generation \[GWh\] by fuel and zone |
| pEnergyFuelCountry | Annual energy generation \[GWh\] by fuel and country |
| pEnergyTechFuel | Annual energy generation \[GWh\] by technology, fuel, and zone |
| pEnergyTechFuelCountry | Annual energy generation \[GWh\] by technology, fuel, and country |
| pEnergyBalance | Annual supply–demand balance \[GWh\] by zone |
| pEnergyBalanceCountry | Annual supply–demand balance \[GWh\] by country |
| pEnergyBalanceH2 | Annual hydrogen supply–demand balance \[mmBTU\] by zone |
| pEnergyBalanceCountryH2 | Annual hydrogen supply–demand balance \[mmBTU\] by country |
| pUtilizationPlant | Annual plant utilization factor |
| pUtilizationTech | Annual technology utilization factor |
| pUtilizationFuel | Annual average capacity factor by fuel |
| pUtilizationTechFuel | Annual average capacity factor by technology and fuel |
| pUtilizationFuelCountry | Annual average capacity factor by fuel and country |
| pUtilizationTechFuelCountry | Annual average capacity factor by technology, fuel, and country |

## 4. ENERGY DISPATCH

| Variable | Description |
|----------|-------------|
| pDispatchPlant | Plant-level hourly dispatch and reserve \[MW\] by zone, plant, time, and year |
| pDispatchFuel | Fuel-level hourly dispatch \[MW\] by zone, fuel, time, and year |
| pDispatch | Zone-level hourly dispatch and flows \[MW\] (imports, exports, unmet demand, storage charge, demand) |

## 5. RESERVES

| Variable | Description |
|----------|-------------|
| pReserveSpinningPlantZone | Annual spinning reserve provided by plant \[GWh\] per zone and year |
| pReserveSpinningFuelZone | Annual spinning reserve provided by fuel \[GWh\] per zone and year |
| pReserveSpinningPlantCountry | Annual spinning reserve provided by plant \[GWh\] per country and year |
| pReserveMargin | Reserve margin indicators by zone and year (peak demand, total firm capacity, reserve margin) |
| pReserveMarginCountry | Reserve margin indicators by country and year |

## 6. INTERCONNECTIONS

| Variable | Description |
|----------|-------------|
| pInterchange | Annual energy exchanged \[GWh\] between internal zones |
| pInterconUtilization | Interconnection utilization \[%\] between internal zones |
| pLossesTransmission | Transmission losses \[MWh\] per internal zone |
| pInterchangeCountry | Annual energy exchanged \[GWh\] between countries |
| pLossesTransmissionCountry | Transmission losses \[MWh\] per country |
| isCongested | Congestion indicator (binary) for line z–z2 per timestep |
| pCongestionShare | Share of time congested \[%\] for each internal line |
| pHourlyInterchangeExternal | Hourly external trade \[MW\] per zone |
| pYearlyInterchangeExternal | Annual external trade \[GWh\] per zone |
| pYearlyInterchangeExternalCountry | Annual external trade \[GWh\] per country |
| pHourlyInterchangeExternalCountry | Hourly external trade \[MW\] per country |
| pInterchangeExternalExports | Annual exports \[GWh\] from zone to external zone |
| pInterchangeExternalImports | Annual imports \[GWh\] from external zone to zone |
| pInterconUtilizationExternalExports | External export line utilization \[%\] |
| pInterconUtilizationExternalImports | External import line utilization \[%\] |

## 7. EMISSIONS

| Variable | Description |
|----------|-------------|
| pEmissionsZone | CO2 emissions \[Mt\] by zone and year |
| pEmissionsIntensityZone | CO2 intensity \[tCO2/GWh\] by zone and year |
| pEmissionsCountrySummary | CO2 emissions and backstop emissions \[Mt\] by country and year |
| pEmissionsIntensityCountry | CO2 intensity \[tCO2/GWh\] by country and year |
| pEmissionMarginalCosts | Marginal cost of system emission constraint \[USD/tCO2\] |
| pEmissionMarginalCostsCountry | Marginal cost of country emission constraint \[USD/tCO2\] |

## 8. PRICES

| Variable | Description |
|----------|-------------|
| pYearlyPrice | Demand-weighted average electricity price \[USD/MWh\] by zone and year |
| pYearlyPriceExport | Flow-weighted average export price \[USD/MWh\] by zone and year |
| pYearlyPriceImport | Flow-weighted average import price \[USD/MWh\] by zone and year |
| pYearlyPriceHub | Flow-weighted hub price \[USD/MWh\] by zone and year |
| pYearlyPriceCountry | Demand-weighted average electricity price \[USD/MWh\] by country and year |
| pYearlyPriceExportCountry | Flow-weighted average export price \[USD/MWh\] by country and year |
| pYearlyPriceImportCountry | Flow-weighted average import price \[USD/MWh\] by country and year |
| pFlowMWSum | Sum of hourly MW flows over the year (used for weights) |
| pFlowMWh | Annual energy flow \[MWh\] between zones |
| pCountryExportFlowMWh | Annual exported energy \[MWh\] from country |
| pCountryImportFlowMWh | Annual imported energy \[MWh\] into country |

## 9. SPECIAL TECHNOLOGIES

| Variable | Description |
|----------|-------------|
| pCSPBalance | CSP hourly output by type \[MW\] (thermal, storage input/output, power) |
| pCSPComponents | CSP installed components and metrics (thermal field, storage, power block, solar multiple, storage hours) |
| pPVwSTOBalance | PV+Storage hourly output by type \[MW\] |
| pPVwSTOComponents | PV+Storage installed components and metrics |
| pStorageBalance | Generic storage hourly output \[MW\] and storage level |
| pStorageComponents | Generic storage installed capacity \[MW, MWh\] and storage hours |
| pSolarPower | Solar hourly output \[MWh\] |
| pSolarEnergyZone | Annual solar energy \[MWh\] by zone |
| pSolarValueZone | Average market value of solar \[USD/MWh\] by zone |
| pSolarCost | Levelized cost of solar \[USD/MWh\] by zone |


## 10. METRICS

| Variable | Description |
|----------|-------------|
| pPlantAnnualLCOE | Plant-level LCOE \[USD/MWh\] by year |
| pZonalAverageCost | Zone average total cost \[USD/MWh\] by year |
| pZonalAverageGenCost | Zone average generation cost \[USD/MWh\] by year |
| pCountryAverageCost | Country average total cost \[USD/MWh\] by year |
| pCountryAverageGenCost | Country average generation cost \[USD/MWh\] by year |
| pSystemAverageCost | System average cost \[USD/MWh\] by year |
| pZoneTradeCost | Combined internal/external trade costs by zone and year |
| pZoneGenCost | Generation-only cost by zone and year |
| pZoneTotalCost | Total system cost by zone and year |
| pZoneEnergyMWh | Annual energy output by zone \[MWh\] |
| pCountryEnergyMWh | Annual energy output by country \[MWh\] |
| pZoneCostEnergyBasis | Energy basis for cost normalization by zone \[MWh\] |
| pCountryCostEnergyBasis | Energy basis for cost normalization by country \[MWh\] |
| pPlantEnergyMWh | Annual energy production by plant \[MWh\] |


## 11. SOLVER PARAMETERS

| Variable | Description |
|----------|-------------|
| pSolverParameters | Solver status, time, and gap diagnostics |
| "Solver Status" | Model status code |
| "Solver Time: ms" | Solution time in milliseconds |
| "Absolute gap" | Absolute optimality gap |
| "Relative gap" | Relative optimality gap |


## OTHER OUTPUTS & AUXILIARY VARIABLES

| Variable | Description |
|----------|-------------|
| pSettings | Model configuration and run settings |
| zcmap | Zone-to-country mapping |
| pVarCost | Variable cost inputs by generator/fuel |
| pCapacityCredit | Capacity credit factors by generator |