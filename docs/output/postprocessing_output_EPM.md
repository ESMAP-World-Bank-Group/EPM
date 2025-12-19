# EPM Output Overview

The postprocessing pipeline reads the GAMS run artifacts located under each scenario folder (for example `output/<run_id>/<scenario>/`). Two types of result files are produced when the model is executed with `REPORTSHORT = 0`:

- `epmresults.gdx` – the complete data dump written by the `execute_unload` statement in `generate_report.gms`.
- Comma-separated files created by the embedded GAMS Connect block in `generate_report.gms`. Each CSV mirrors the identically named GDX symbol and is stored next to the scenario GDX file.

Unless stated otherwise, the columns `epmresults.gdx` and `CSV export` indicate whether each symbol is produced by default when `REPORTSHORT = 0`.

## Tips

- Large CSVs can be slimmed down using `reduce_definition_csv` in `postprocessing.py` (keeps first, middle, last year).
- When `REPORTSHORT = 1`, only `pYearlyCostsZone`, `pYearlyCostsZoneFull`, and `pEnergyBalance` are exported; the tables below assume the default detailed reporting.

## 1. CAPACITY

| Variable                    | Description                                                         | epmresults.gdx | CSV export |
| --------------------------- | ------------------------------------------------------------------- | -------------- | ---------- |
| pCapacityPlant              | Installed capacity \[MW\] by plant, zone, and year                  | Yes            | Yes        |
| pCapacityTechFuel           | Installed capacity \[MW\] by technology, fuel, and zone             | Yes            | No         |
| pCapacityFuel               | Installed capacity \[MW\] by fuel and zone                          | Yes            | Yes        |
| pCapacityTechFuelCountry    | Installed capacity \[MW\] by technology, fuel, and country          | Yes            | No         |
| pCapacityFuelCountry        | Installed capacity \[MW\] by fuel and country                       | Yes            | Yes        |
| pCapacityPlantH2            | Installed electrolyzer capacity \[MW\] by zone and year             | Yes            | No         |
| pRetirementsPlant           | Retired capacity \[MW\] by plant, zone, and year                    | Yes            | No         |
| pRetirementsFuel            | Retired capacity \[MW\] by fuel and zone                            | Yes            | No         |
| pRetirementsCountry         | Total retired capacity \[MW\] by country and year                   | Yes            | No         |
| pRetirementsFuelCountry     | Retired capacity \[MW\] by fuel and country                         | Yes            | No         |
| pNewCapacityFuel            | Newly added capacity \[MW\] by fuel and zone                        | Yes            | Yes        |
| pNewCapacityTech            | Newly added capacity \[MW\] by technology and zone                  | Yes            | No         |
| pNewCapacityFuelCountry     | Newly added capacity \[MW\] by fuel and country                     | Yes            | Yes        |
| pNewCapacityTechCountry     | Newly added capacity \[MW\] by technology and country               | Yes            | No         |
| pAnnualTransmissionCapacity | Total available transmission capacity \[MW\] between internal zones | Yes            | Yes        |
| pNewTransmissionCapacity    | Additional transmission capacity \[MW\] between internal zones      | Yes            | Yes        |
| pCapacitySummary            | Summary of capacity indicators \[MW\] by zone and year              | Yes            | Yes        |
| pCapacitySummaryCountry     | Summary of capacity indicators \[MW\] by country and year           | Yes            | Yes        |

## 2. COSTS

| Variable                     | Description                                                          | epmresults.gdx | CSV export |
| ---------------------------- | -------------------------------------------------------------------- | -------------- | ---------- |
| pCostsPlant                  | Yearly cost breakdown by plant and year                              | Yes            | Yes        |
| pCapexInvestment             | Annual CAPEX investment \[USD\] by zone and year                     | Yes            | Yes        |
| pCapexInvestmentPlant        | CAPEX investment \[USD\] by plant and component                      | Yes            | Yes        |
| pCapexInvestmentTransmission | Transmission CAPEX \[USD\] by line and year                          | Yes            | Yes        |
| pCapexInvestmentComponent    | Annual CAPEX investment \[USD\] by component and zone                | Yes            | Yes        |
| pPrice                       | Marginal cost \[USD/MWh\] by zone, time, and year                    | Yes            | Yes        |
| pImportCostsInternal         | Import costs with internal zones \[USD\] by zone and year            | Yes            | No         |
| pExportRevenuesInternal      | Export revenues with internal zones \[USD\] by zone and year         | Yes            | No         |
| pCongestionRevenues          | Congestion rents \[USD\] from saturated internal lines by year       | Yes            | No         |
| pTradeSharedBenefits         | Congestion rents shared equally between zones \[USD\]                | Yes            | No         |
| pYearlyCostsZone             | Annual cost summary \[million USD\] by zone and year                 | Yes            | Yes        |
| pYearlyCostsCountry          | Annual cost summary \[million USD\] by country and year              | Yes            | Yes        |
| pCostsZone                   | Total cost \[million USD\] by zone and cost category                 | Yes            | No         |
| pCostsSystem                 | System-level cost summary \[million USD\], weighted and discounted   | Yes            | Yes        |
| pCostsSystemPerMWh           | System-level cost summary \[USD/MWh\], weighted and discounted       | Yes            | Yes        |
| pYearlyCostsSystem           | Annual system cost summary \[million USD\] by cost category and year | Yes            | No         |
| pFuelCosts                   | Annual fuel costs \[million USD\] by fuel, zone, and year            | Yes            | Yes        |
| pFuelCostsCountry            | Annual fuel costs \[million USD\] by fuel, country, and year         | Yes            | Yes        |
| pFuelConsumption             | Annual fuel consumption \[MMBtu\] by fuel, zone, and year            | Yes            | No         |
| pFuelConsumptionCountry      | Annual fuel consumption \[MMBtu\] by fuel, country, and year         | Yes            | No         |

## 3. ENERGY BALANCE

| Variable                    | Description                                                       | epmresults.gdx | CSV export |
| --------------------------- | ----------------------------------------------------------------- | -------------- | ---------- |
| pEnergyPlant                | Annual energy generation \[GWh\] by plant, zone, and year         | Yes            | Yes        |
| pEnergyTechFuel             | Annual energy generation \[GWh\] by technology, fuel, and zone    | Yes            | No         |
| pEnergyFuel                 | Annual energy generation \[GWh\] by fuel and zone                 | Yes            | Yes        |
| pEnergyTechFuelCountry      | Annual energy generation \[GWh\] by technology, fuel, and country | Yes            | No         |
| pEnergyFuelCountry          | Annual energy generation \[GWh\] by fuel and country              | Yes            | Yes        |
| pEnergyBalance              | Annual supply-demand balance \[GWh\] by zone                      | Yes            | Yes        |
| pEnergyBalanceCountry       | Annual supply-demand balance \[GWh\] by country                   | Yes            | No         |
| pEnergyBalanceH2            | Annual hydrogen supply-demand balance \[mmBTU\] by zone           | Yes            | No         |
| pEnergyBalanceCountryH2     | Annual hydrogen supply-demand balance \[mmBTU\] by country        | Yes            | No         |
| pUtilizationPlant           | Annual plant utilization factor                                   | Yes            | Yes        |
| pUtilizationTech            | Annual technology utilization factor                              | Yes            | No         |
| pUtilizationFuel            | Annual average capacity factor by fuel                            | Yes            | Yes        |
| pUtilizationTechFuel        | Annual average capacity factor by technology and fuel             | Yes            | Yes        |
| pUtilizationFuelCountry     | Annual average capacity factor by fuel and country                | Yes            | Yes        |
| pUtilizationTechFuelCountry | Annual average capacity factor by technology, fuel, and country   | Yes            | Yes        |

## 4. ENERGY DISPATCH

| Variable       | Description                                    | epmresults.gdx | CSV export |
| -------------- | ---------------------------------------------- | -------------- | ---------- |
| pDispatchPlant | Plant-level hourly dispatch and reserve \[MW\] | Yes            | No         |
| pDispatchFuel  | Fuel-level hourly dispatch \[MW\]              | Yes            | Yes        |
| pDispatch      | Zone-level hourly dispatch and flows \[MW\]    | Yes            | Yes        |

## 5. RESERVES

| Variable                     | Description                                                     | epmresults.gdx | CSV export |
| ---------------------------- | --------------------------------------------------------------- | -------------- | ---------- |
| pReserveSpinningPlantZone    | Spinning reserve provided by plant \[MWh\] per zone and year    | Yes            | Yes        |
| pReserveSpinningFuelZone     | Spinning reserve provided by fuel \[MWh\] per zone and year     | Yes            | Yes        |
| pReserveSpinningPlantCountry | Spinning reserve provided by plant \[MWh\] per country and year | Yes            | Yes        |
| pReserveMargin               | Reserve margin indicators by zone and year                      | Yes            | No         |
| pReserveMarginCountry        | Reserve margin indicators by country and year                   | Yes            | No         |

## 6. INTERCONNECTIONS

| Variable                            | Description                                              | epmresults.gdx | CSV export |
| ----------------------------------- | -------------------------------------------------------- | -------------- | ---------- |
| pInterchange                        | Annual energy exchanged \[GWh\] between internal zones   | Yes            | Yes        |
| pInterconUtilization                | Interconnection utilization \[%\] between internal zones | Yes            | Yes        |
| pLossesTransmission                 | Transmission losses \[MWh\] per internal zone            | Yes            | No         |
| pInterchangeCountry                 | Annual energy exchanged \[GWh\] between countries        | Yes            | Yes        |
| pLossesTransmissionCountry          | Transmission losses \[MWh\] per country                  | Yes            | No         |
| pCongestionShare                    | Share of time congested \[%\] for line z–z2              | Yes            | Yes        |
| pHourlyInterchangeExternal          | Hourly external trade \[MW\] per zone                    | Yes            | No         |
| pYearlyInterchangeExternal          | Annual external trade \[GWh\] per zone                   | Yes            | No         |
| pYearlyInterchangeExternalCountry   | Annual external trade \[GWh\] per country                | Yes            | No         |
| pHourlyInterchangeExternalCountry   | Hourly external trade \[MW\] per country                 | Yes            | No         |
| pInterchangeExternalExports         | Annual exports \[GWh\] from zone to external zone        | Yes            | No         |
| pInterchangeExternalImports         | Annual imports \[GWh\] from external zone                | Yes            | No         |
| pInterconUtilizationExternalExports | External export line utilization \[%\]                   | Yes            | No         |
| pInterconUtilizationExternalImports | External import line utilization \[%\]                   | Yes            | No         |

## 7. EMISSIONS

| Variable                      | Description                                               | epmresults.gdx | CSV export |
| ----------------------------- | --------------------------------------------------------- | -------------- | ---------- |
| pEmissionsZone                | CO₂ emissions \[Mt\] by zone and year                     | Yes            | Yes        |
| pEmissionsIntensityZone       | CO₂ intensity \[tCO₂/GWh\] by zone and year               | Yes            | Yes        |
| pEmissionsCountrySummary      | CO₂ emissions \[Mt\] by country, type, and year           | Yes            | No         |
| pEmissionsIntensityCountry    | CO₂ intensity \[tCO₂/GWh\] by country and year            | Yes            | No         |
| pEmissionMarginalCosts        | Marginal cost of system emission constraint \[USD/tCO₂\]  | Yes            | No         |
| pEmissionMarginalCostsCountry | Marginal cost of country emission constraint \[USD/tCO₂\] | Yes            | No         |

## 8. PRICES

| Variable                  | Description                                                               | epmresults.gdx | CSV export |
| ------------------------- | ------------------------------------------------------------------------- | -------------- | ---------- |
| pYearlyPrice              | Demand-weighted average electricity price \[USD/MWh\] by zone and year    | Yes            | No         |
| pYearlyPriceExport        | Flow-weighted average export price \[USD/MWh\] by zone and year           | Yes            | No         |
| pYearlyPriceImport        | Flow-weighted average import price \[USD/MWh\] by zone and year           | Yes            | No         |
| pYearlyPriceHub           | Flow-weighted hub price \[USD/MWh\] by zone and year                      | Yes            | No         |
| pYearlyPriceCountry       | Demand-weighted average electricity price \[USD/MWh\] by country and year | Yes            | No         |
| pYearlyPriceExportCountry | Flow-weighted average export price \[USD/MWh\] by country and year        | Yes            | No         |
| pYearlyPriceImportCountry | Flow-weighted average import price \[USD/MWh\] by country and year        | Yes            | No         |

## 9. SPECIAL TECHNOLOGIES

| Variable           | Description                                                      | epmresults.gdx | CSV export |
| ------------------ | ---------------------------------------------------------------- | -------------- | ---------- |
| pCSPBalance        | CSP hourly output by type \[MW\]                                 | Yes            | No         |
| pCSPComponents     | CSP installed components and metrics                             | Yes            | No         |
| pPVwSTOBalance     | PV+Storage hourly output by type \[MW\]                          | Yes            | No         |
| pPVwSTOComponents  | PV+Storage installed components and metrics                      | Yes            | No         |
| pStorageBalance    | Generic storage hourly output \[MW\] and storage level           | Yes            | No         |
| pStorageComponents | Generic storage installed capacity \[MW, MWh\] and storage hours | Yes            | No         |
| pSolarPower        | Solar hourly output \[MWh\]                                      | Yes            | No         |
| pSolarEnergyZone   | Annual solar energy \[MWh\] by zone                              | Yes            | No         |
| pSolarValueZone    | Average market value of solar \[USD/MWh\] by zone                | Yes            | No         |
| pSolarCost         | Levelized cost of solar \[USD/MWh\] by zone                      | Yes            | No         |

## 10. METRICS

| Variable                    | Description                                                      | epmresults.gdx | CSV export |
| --------------------------- | ---------------------------------------------------------------- | -------------- | ---------- |
| pPlantAnnualLCOE            | Plant-level LCOE \[USD/MWh\] by year                             | Yes            | Yes        |
| pCostsZonePerMWh            | Zone discounted average cost by component \[USD/MWh\]            | Yes            | Yes        |
| pCostsCountryPerMWh         | Country discounted average cost by component \[USD/MWh\]         | Yes            | No         |
| pDiscountedDemandZoneMWh    | Discounted electricity demand denominator \[MWh\] by zone        | Yes            | No         |
| pDiscountedDemandCountryMWh | Discounted electricity demand denominator \[MWh\] by country     | Yes            | No         |
| pDiscountedDemandSystemMWh  | Discounted electricity demand denominator \[MWh\] for the system | Yes            | No         |
| pYearlyCostsSystemPerMWh     | System average cost \[USD/MWh\] by year                          | Yes            | No         |

## 11. modeltype PARAMETERS

| Variable             | Description                                 | epmresults.gdx | CSV export |
| -------------------- | ------------------------------------------- | -------------- | ---------- |
| pSolverParameters | modeltype status, time, and gap diagnostics | Yes            | No         |

`pSolverParameters` includes entries such as `modeltype Status`, `modeltype Time: ms`, `Absolute gap`, and `Relative gap`.

## OTHER OUTPUTS & AUXILIARY VARIABLES

| Variable        | Description                            | epmresults.gdx | CSV export |
| --------------- | -------------------------------------- | -------------- | ---------- |
| pSettings       | Model configuration and run settings   | Yes            | Yes        |
| zcmap           | Zone-to-country mapping                | No             | Yes        |
| pVarCost        | Variable cost inputs by generator/fuel | Yes            | No         |
| pCapacityCredit | Capacity credit factors by generator   | Yes            | Yes        |

