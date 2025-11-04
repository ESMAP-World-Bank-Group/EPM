# Postprocessing Output Overview

The postprocessing pipeline reads the GAMS run artifacts located under each scenario folder (for example `output/<run_id>/<scenario>/`). Two types of result files are produced when the model is executed with `REPORTSHORT = 0`:

- `epmresults.gdx` – the complete data dump written by the `execute_unload` statement in `generate_report.gms`.
- Comma-separated files created by the embedded GAMS Connect block in `generate_report.gms`. Each CSV mirrors the identically named GDX symbol and is stored next to the scenario GDX file.

The tables below document which metrics are sent to both outputs by default and which ones remain GDX-only unless manually converted.

## Outputs exported to both `epmresults.gdx` and CSV

These symbols are always included in the GDX dump and are also written to CSV by the Connect workflow. All locations are relative to the scenario folder (`output/<run_id>/<scenario>/`).

### Capacity and Transmission

| Variable                      | Description                                                             | epmresults.gdx | CSV export    |
| ----------------------------- | ----------------------------------------------------------------------- | -------------- | ------------- |
| `pCapacityPlant`              | Installed capacity by generator, fuel, and year (MW).                   | Yes            | Yes (default) |
| `pCapacityFuel`               | Installed capacity aggregated by fuel and zone (MW).                    | Yes            | Yes (default) |
| `pCapacityFuelCountry`        | Installed capacity aggregated by fuel and country (MW).                 | Yes            | Yes (default) |
| `pNewCapacityFuel`            | New capacity builds by fuel and zone (MW).                              | Yes            | Yes (default) |
| `pNewCapacityFuelCountry`     | New capacity builds by fuel and country (MW).                           | Yes            | Yes (default) |
| `pCapacitySummary`            | Summary capacity balance by zone and year (existing, new, retirements). | Yes            | Yes (default) |
| `pCapacitySummaryCountry`     | Summary capacity balance aggregated to the country level.               | Yes            | Yes (default) |
| `pAnnualTransmissionCapacity` | Total available annual transfer capacity between zones (MW).            | Yes            | Yes (default) |
| `pNewTransmissionCapacity`    | Incremental transfer capacity required on each interconnector (MW).     | Yes            | Yes (default) |

### Costs and Investment

| Variable                       | Description                                                               | epmresults.gdx | CSV export    |
| ------------------------------ | ------------------------------------------------------------------------- | -------------- | ------------- |
| `pCapexInvestment`             | System capital expenditures by component (M$).                            | Yes            | Yes (default) |
| `pCapexInvestmentComponent`    | Capital expenditures broken down by investment component (M$).            | Yes            | Yes (default) |
| `pCapexInvestmentPlant`        | Capital spending allocated to individual plants (M$).                     | Yes            | Yes (default) |
| `pCapexInvestmentTransmission` | Capital spending on transmission assets (M$).                             | Yes            | Yes (default) |
| `pCostsPlant`                  | Operating costs per plant including fixed, variable, and fuel costs (M$). | Yes            | Yes (default) |
| `pCostsSystem`                 | Total discounted system cost components (M$).                             | Yes            | Yes (default) |
| `pCostsSystemPerMWh`           | System cost normalized by energy basis ($/MWh).                           | Yes            | Yes (default) |
| `pYearlyCostsZone`             | Annual cost breakdown by zone and cost attribute (M$).                    | Yes            | Yes (default) |
| `pYearlyCostsCountry`          | Annual cost breakdown aggregated to the country level (M$).               | Yes            | Yes (default) |
| `pPrice`                       | Marginal price signal by zone and period ($/MWh).                         | Yes            | Yes (default) |

### Fuel Inputs

| Variable            | Description                                                | epmresults.gdx | CSV export    |
| ------------------- | ---------------------------------------------------------- | -------------- | ------------- |
| `pFuelCosts`        | Fuel price trajectories by fuel and year.                  | Yes            | Yes (default) |
| `pFuelCostsCountry` | Fuel price trajectories with country-specific adjustments. | Yes            | Yes (default) |

### Energy and Dispatch

| Variable             | Description                                                                        | epmresults.gdx | CSV export    |
| -------------------- | ---------------------------------------------------------------------------------- | -------------- | ------------- |
| `pEnergyPlant`       | Annual generation by plant (GWh).                                                  | Yes            | Yes (default) |
| `pEnergyFuel`        | Annual generation aggregated by fuel and zone (GWh).                               | Yes            | Yes (default) |
| `pEnergyFuelCountry` | Annual generation aggregated by fuel and country (GWh).                            | Yes            | Yes (default) |
| `pEnergyBalance`     | Zone-level energy balance including demand, generation, imports, and losses (GWh). | Yes            | Yes (default) |
| `pDispatchFuel`      | Dispatch by fuel, zone, and operating period (MW).                                 | Yes            | Yes (default) |
| `pDispatch`          | System dispatch by attribute (supply, demand, net trades) and operating period.    | Yes            | Yes (default) |

### Utilization Metrics

| Variable                      | Description                                                             | epmresults.gdx | CSV export    |
| ----------------------------- | ----------------------------------------------------------------------- | -------------- | ------------- |
| `pUtilizationPlant`           | Capacity factor of each plant by year.                                  | Yes            | Yes (default) |
| `pUtilizationFuel`            | Capacity factor aggregated by fuel and zone.                            | Yes            | Yes (default) |
| `pUtilizationFuelCountry`     | Capacity factor aggregated by fuel and country.                         | Yes            | Yes (default) |
| `pUtilizationTechFuel`        | Capacity factor by technology-fuel combination and zone.                | Yes            | Yes (default) |
| `pUtilizationTechFuelCountry` | Capacity factor by technology-fuel combination aggregated to countries. | Yes            | Yes (default) |

### Reserves and Capacity Credits

| Variable                       | Description                                                      | epmresults.gdx | CSV export    |
| ------------------------------ | ---------------------------------------------------------------- | -------------- | ------------- |
| `pReserveSpinningPlantZone`    | Spinning reserve provision by plant and zone (MW).               | Yes            | Yes (default) |
| `pReserveSpinningPlantCountry` | Spinning reserve provision aggregated by plant and country (MW). | Yes            | Yes (default) |
| `pReserveSpinningFuelZone`     | Spinning reserve provision aggregated by fuel and zone (MW).     | Yes            | Yes (default) |
| `pCapacityCredit`              | Capacity credits assigned to generators by zone (MW).            | Yes            | Yes (default) |

### Interconnection and Congestion

| Variable               | Description                                                  | epmresults.gdx | CSV export    |
| ---------------------- | ------------------------------------------------------------ | -------------- | ------------- |
| `pInterchange`         | Interzonal power flows by direction and period (MW).         | Yes            | Yes (default) |
| `pInterchangeCountry`  | Cross-border power exchange aggregated by country (MW).      | Yes            | Yes (default) |
| `pInterconUtilization` | Utilization rate of each interconnector (% of capacity).     | Yes            | Yes (default) |
| `pCongestionShare`     | Share of congestion rents allocated to each interconnection. | Yes            | Yes (default) |

### Emissions

| Variable                  | Description                                        | epmresults.gdx | CSV export    |
| ------------------------- | -------------------------------------------------- | -------------- | ------------- |
| `pEmissionsZone`          | Annual emissions by zone (kt CO2e).                | Yes            | Yes (default) |
| `pEmissionsIntensityZone` | Emissions intensity of supply by zone (t CO2/MWh). | Yes            | Yes (default) |

### Financial Metrics

| Variable           | Description                                           | epmresults.gdx | CSV export    |
| ------------------ | ----------------------------------------------------- | -------------- | ------------- |
| `pPlantAnnualLCOE` | Levelized cost of electricity for each plant ($/MWh). | Yes            | Yes (default) |
| `pCostsZonePerMWh` | Average total system cost per zone ($/MWh).           | Yes            | Yes (default) |

### Reference Tables

| Variable    | Description                                                      | epmresults.gdx | CSV export    |
| ----------- | ---------------------------------------------------------------- | -------------- | ------------- |
| `pSettings` | Scenario settings captured from pSettings (model configuration). | Yes            | Yes (default) |
| `zcmap`     | Zone-to-country mapping table.                                   | Yes            | Yes (default) |

## GDX-only outputs (no default CSV)

The following symbols are written to `epmresults.gdx` when detailed reporting is enabled, but they are not exported to CSV by default. Use `postprocessing.utils.gdx_to_csv` or `extract_epm_folder_by_scenario(..., save_to_csv=True)` if you need a flat file.

- **Capacity detail** (`pCapacityTechFuel`, `pCapacityTechFuelCountry`, `pCapacityPlantH2`) - Capacity by technology and hydrogen variants beyond the fuel aggregation. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **Retirements** (`pRetirementsPlant`, `pRetirementsFuel`, `pRetirementsFuelCountry`, `pRetirementsCountry`) - Retired capacity by plant, fuel, and country. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **New capacity by technology** (`pNewCapacityTech`, `pNewCapacityTechCountry`) - New builds reported by technology group. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **Generation detail by technology** (`pEnergyTechFuel`, `pEnergyTechFuelCountry`, `pDispatchPlant`, `pUtilizationTech`) - Generation and utilization indexed by technology as well as plant-level dispatch. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **Energy balance extensions** (`pEnergyBalanceCountry`, `pEnergyBalanceH2`, `pEnergyBalanceCountryH2`) - Country and hydrogen views of the energy balance. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **Fuel consumption** (`pFuelConsumption`, `pFuelConsumptionCountry`) - Fuel burn quantities aligned with fuel pricing assumptions. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **Reserve margins** (`pReserveMargin`, `pReserveMarginCountry`) - Planning reserve margin outcomes by zone and country. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **Interconnection detail** (`pLossesTransmission`, `pLossesTransmissionCountry`, `pHourlyInterchangeExternal`, `pHourlyInterchangeExternalCountry`, `pYearlyInterchangeExternal`, `pYearlyInterchangeExternalCountry`, `pInterchangeExternalExports`, `pInterchangeExternalImports`, `pInterconUtilizationExternalExports`, `pInterconUtilizationExternalImports`) - Detailed cross-border and external interchange metrics, including losses and utilization splits. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **Trade cost breakdown** (`pImportCostsInternal`, `pExportRevenuesInternal`, `pTradeSharedBenefits`, `pCongestionRevenues`) - Decomposition of trade-related monetary flows. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **Emissions detail** (`pEmissionsCountrySummary`, `pEmissionsIntensityCountry`, `pEmissionMarginalCosts`, `pEmissionMarginalCostsCountry`) - Country summaries and marginal emission costs. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **Price detail** (`pYearlyPrice`, `pYearlyPriceExport`, `pYearlyPriceImport`, `pYearlyPriceHub`, `pYearlyPriceCountry`, `pYearlyPriceExportCountry`, `pYearlyPriceImportCountry`) - Annual price blocks by hub, import, export, and country aggregation. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **Technology specific balances** (`pCSPBalance`, `pCSPComponents`, `pPVwSTOBalance`, `pPVwSTOComponents`, `pStorageBalance`, `pStorageComponents`, `pSolarPower`, `pSolarEnergyZone`, `pSolarValueZone`, `pSolarCost`) - Detailed tracking for CSP, PV with storage, and storage technologies. Available in `epmresults.gdx` only; convert manually if a CSV is needed.
- **System metrics** (`pCostsCountryPerMWh`, `pCostsZonePerMWh`, `pDiscountedDemandCountryMWh`, `pDiscountedDemandZoneMWh`, `pYearlyCostsSystemPerMWh`, `pYearlyCostsSystem`, `pCostsZone`, `pVarCost`, `pSolverParameters`) - Additional summary indicators and modeltype diagnostics. Available in `epmresults.gdx` only; convert manually if a CSV is needed.

## Tips

- Large CSVs can be slimmed down using `reduce_definition_csv` in `postprocessing.py` (keeps first, middle, last year).
- When `REPORTSHORT = 1`, only `pYearlyCostsZone`, `pYearlyCostsZoneFull`, and `pEnergyBalance` are exported; the tables above assume the default detailed reporting.
