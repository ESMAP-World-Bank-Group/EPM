# Data Structure Documentation

## Configuration Files

These files are direclty located in the `data` folder.

---

### **`pSettings.csv`**  


- **Description**:  
  This file contains **global model parameters and settings** that control various aspects of the **EPM model**, including **economic assumptions, penalties, optional features, and constraints**.  
  It plays a **critical role in defining the behavior of the optimization model**, enabling customization of financial assumptions, operational constraints, and system features.
- **Data Structure**:  
  - **Parameter** (*string*) – Full name of the parameter.  
  - **Abbreviation** (*string*) – Short name used in the model.  
  - **Value** (*varied units*) – The assigned numerical value or toggle (0/1) for the parameter.  
- **Example Link**: [pSettings.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_sapp/config/pSettings.csv)

**!! Important !!** Some parameters have been added in the latest EPM version. If you are using an older version of `pSettings.csv`, these new parameters will be missing and will therefore be set to zero by default in GAMS, which can significantly alter model results.
To ensure accurate outcomes, it is essential to update your `pSettings.csv` file to include all new parameters when running the latest version of EPM.

---

#### **Economic Assumptions**
These parameters define key **financial** and **economic** inputs.

| Parameter | Abbreviation | Value | Unit |
|-----------|-------------|-------|------|
| Weighted Average Cost of Capital (WACC) | `WACC` | 0.06 | Fraction |
| Discount rate | `DR` | 0.06 | Fraction |

---

#### **Penalties**
These define the **costs associated with system constraints** and **violations**.

| Parameter | Abbreviation | Value | Unit |
|-----------|-------------|-------|------|
| Cost of unserved energy per MWh | `VOLL` | 500 | $/MWh |
| Cost of reserve shortfall per MW | `ReserveVoLL` | 60000 | $/MW |
| Spinning reserve VoLL per MWh | `SpinReserveVoLL` | 25 | $/MWh |
| Cost of surplus power per MWh | `costSurplus` | 0 | $/MWh |
| Cost of curtailment per MWh | `costcurtail` | 0 | $/MWh |
| CO₂ backstop price | `CO2backstop` | 300 | $/tCO₂ |
| Cost of climate backstop technology per ton CO₂ | `H2UnservedCost` | 3000 | $/tCO₂ |

---

#### **Optional Features**
These parameters **toggle optional modeling features** (0 = disabled, 1 = enabled).

| Parameter                                                           | Abbreviation             | Value |
|---------------------------------------------------------------------|--------------------------|-------|
| Include carbon price                                                | `includeCarbonPrice`     | 0     |
| Include energy efficiency                                           | `includeEE`              | 0     |
| Include CSP optimization                                            | `includeCSP`             | 0     |
| Include storage operation                                           | `includeStorage`         | 0     |
| Show zero values                                                    | `show0`                  | 0     |
| Retire plants on economic grounds                                   | `econRetire`             | 0     |
| Run in interconnected mode                                          | `interconMode`           | 0     |
| Allow exports based on price                                        | `allowExports`           | 0     |
| Remove transfer limits                                              | `NoTransferLim`          | 0     |
| Allow expansion of transfer limits                                  | `pAllowHighTransfer`     | 0     |
| Use less detailed demand definition                                 | `altDemand`              | 1     |
| Allow CAPEX trajectory                                              | `Captraj`                | 0     |
| Include H₂ production                                               | `IncludeH2`              | 0     |
| Include transmission lines when assessing country planning reserves | `includeIntercoReserves` | 1     |

---

#### **Constraints**
These parameters define **model constraints**, limiting **emissions, fuel use, capacity investments, and operational reserves**.

| Parameter                                                               | Abbreviation                          | Value |
|-------------------------------------------------------------------------|---------------------------------------|-------|
| Apply system CO₂ constraints                                            | `system_co2_constraints`              | 0     |
| Apply fuel constraints                                                  | `fuel_constraints`                    | 0     |
| Apply maximum capital constraint                                        | `capital_constraints`                 | 0     |
| Apply minimum generation constraint                                     | `mingen_constraints`                  | 0     |
| Apply planning reserve constraint                                       | `planning_reserve_constraints`        | 0     |
| System planning reserve margin (%)                                      | `system_reserve_margin`               | 0.1   |
| Contribution of transmission lines to country spinning reserves need (%) | `interco_reserve_contribution`        | 1     |
| Apply ramp constraints                                                  | `ramp_constraints`                    | 0     |
| Apply system spinning reserve constraints                               | `system_spinning_reserve_constraints` | 0     |
| Apply zonal CO₂ constraint                                              | `zonal_co2_constraints`               | 0     |
| Apply zonal spinning reserve constraints                                | `zonal_spinning_reserve_constraints`  | 0     |
| Minimum share of RE (%)                                                 | `MinREshare`                          | 0     |
| RE share target year                                                    | `RETargetYr`                          | -     |
| Total maximum capital investments ($ billion)                           | `MaxCapital`                          | -     |
| Maximum share of imports (%)                                            | `MaxImports`                          | -     |
| Maximum share of exports (%)                                            | `MaxExports`                          | -     |
| Spinning reserve needs for VRE (%)                                      | `VREForecastError`                    | 0.15  |
| User-defined capacity credits                                           | `VRECapacityCredits`                  | -     |

---

#### **Reporting and Scenario Settings**
These parameters **control reporting options and fuel price scenarios**.

| Parameter | Abbreviation | Value |
|-----------|-------------|-------|
| Seasonal reporting | `Seasonalreporting` | - |
| System results file production | `Systemresultreporting` | - |
| Include decom and committed capacity in system results | `IncludeDecomCom` | - |
| Active fuel price scenario | `FuelPriceScenario` | - |

---

#### **Capacity Credit Calculation**
These parameters define **capacity credits** for reporting and system reliability evaluation.

| Parameter | Abbreviation | Value |
|-----------|-------------|-------|
| Capacity credits for reporting | `CapacityCredits` | - |
| Solar capacity credit | `CapCreditSolar` | - |
| Wind capacity credit | `CapCreditWind` | - |
| Max load fraction for capacity credit calculation | `MaxLoadFractionCCCalc` | - |

#### **Notes:**
- **All cost values are in USD ($).**  
- **All percentages are in fractional format (e.g., 0.06 for 6%).**  
- **Parameters with `0` or `1` act as toggles (0 = disabled, 1 = enabled).**  

---

### `y.csv`

Defines the years included in the intertemporal optimization. One-dimensional csv.

- **Data Structure**:  
  - **Index**
    - **y** (*int*) –  Years to be included in the model.

- **Example Link**: [y.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/y.csv)  

### **`zcmap.csv`**  
  - Two-dimensional, defines the zones and countries included in the model.  
  - Example: [zcmap.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/zcmap.csv)

## Resources

All files in this section are included in the `resources` folder. These datasets define parameters that do not necessarily change between projects.

---

### **`ftfindex.csv`**  
  - List of fuels recognized by EPM.  
  - 2 columns:  
    1. Fuel  
    2. Index (Note: Index is no longer used, and the structure is being simplified to keep only "Fuel").  
  - Example: [ftfindex.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/resources/ftfindex.csv)
  - List of fuel types recognized by EPM: 
    - **Coal**  
    - **Gas**  
    - **Water**  
    - **Hydro**  
    - **PV**  
    - **OnshoreWind**
    - **OffshoreWind**  
    - **Import**  
    - **HFO**  
    - **Uranium**  
    - **CSP**  
    - **Battery**
    - **Diesel**
    - **Biomass**
    - **Geothermal**
    - **LNG**
  - **Note**: Not clear why there is need of OnshoreWind and OffshoreWind.
---

### **`pTechData.csv`**  
  - List of technologies recognized by EPM.  
  - 4 columns:  
    1. Technology name  
    2. Construction period (in years)  
    3. Hourly variation (if the technology varies hourly)  
    4. RE Technology (if it is a renewable energy technology)  
  - Example: [pTechData.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/resources/pTechData.csv)
  - List of technologies recognized by EPM:  
    - **GT**: Gas Turbine
    - **CCGT**: Combined Cycle Gas Turbine
    - **ST**: Steam Turbine
    - **ICE**: Internal Combustion Engine
    - **COAL**: Coal
    - **WIND**: Wind
    - **PV**: Photovoltaic
    - **STORAGE**: Storage
    - **STOHY**: Storage Hydro
    - **ROR**: Run of River
    - **BIOGAS**: Biogas
    - **INT**: ??
    - **CHP**: Combined Heat and Power 
    - **BIOMAS**: Biomass
    - **CSP**: Concentrated Solar Power
    - **PVwSTO**: Photovoltaic with Storage
    - **STOPV**: Storage with Photovoltaic
    - **MPAOpt**: ??
---

### **`pFuelCarbonContent.csv`**  
  - 2 columns:  
    1. Fuel  
    2. Carbon content in gCO₂/kWh  
  - Example: [pFuelCarbonContent.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/resources/pFuelCarbonContent.csv)

---

## Load Data

All files in this section are included in the `load` folder.

---

### **`pDemandProfile.csv`**  
  - 4 dimensions:  
    1. Zone  
    2. Season  
    3. Days  
    4. Hours  
  - 1 value column (normalized between 0 and 1).  
  - Example: [pDemandProfile.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pDemandProfile.csv)

---

### **`pDemandForecast.csv`**  
  - 2 dimensions:  
    1. Zone  
    2. Type 
  - 1 value column (Energy in GWh and Peak in MW)  
  - Example: [pDemandForecast.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pDemandForecast.csv)

---

### **`pDemandData.csv`**  
  - Alternative way to define load demand.  
  - Example: [pDemandData.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pDemandData.csv)

---

### **`pEnergyEfficiencyFactor.csv`**  
  - TODO: Define usage.  
  - Example: [pEnergyEfficiencyFactor.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pEnergyEfficiencyFactor.csv)

---

### **`sRelevants.csv`**  
  - TODO: Define usage.  
  - Example: [sRelevants.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/sRelevants.csv)

---
## Supply Data

All files in this section are stored in the **`supply`** folder.

### `pGenDataExcelCustom.csv`
- **Description**: The main data table defining all power plants, including their **technical** and **economic** characteristics.
- **Data Structure**:
  - **`Plant`** (*string*) – Name of the power plant.  
  - **`Zone`** (*string*) – Zone where the plant is located, as defined in `zcmap.csv`.  
  - **`Type`** (*string*) – Technology type, as defined in `pTechData.csv`.  
  - **`fuel1`** (*string*) – Primary fuel name, as defined in `ftfindex.csv`.  
  - **`StYr`** (*year*) – Start year of plant operation.  
  - **`RetrYr`** (*year*) – Retirement year of the plant.  
  - **`Capacity`** (*MW*) – Installed capacity of the plant.  
  - **`UnitSize`** (*MW*) – Size of a single unit in the plant.  
  - **`Status`** (*integer*) – Operational status indicator (e.g., existing, planned, retired).  
  - **`fuel2`** (*string*) – Secondary fuel name, if applicable.  
  - **`HeatRate2`** (*MMBtu/MWh*) – Heat rate when operating on secondary fuel.  
  - **`DescreteCap`** (*MW*) – Discrete capacity increments allowed for expansion.  
  - **`BuildLimitperYear`** (*MW/year*) – Maximum allowable capacity additions per year.  
  - **`MaxTotalBuild`** (*MW*) – Maximum total capacity allowed for the plant.  
  - **`MinLimitShare`** (*fraction*) – Minimum share of total generation capacity that must be maintained.  
  - **`HeatRate`** (*MMBtu/MWh*) – Heat rate when operating on primary fuel.  
  - **`RampUpRate`** (*fraction/hour*) – Maximum rate at which generation can be increased.  
  - **`RampDnRate`** (*fraction/hour*) – Maximum rate at which generation can be decreased.  
  - **`OverLoadFactor`** (*fraction*) – Factor determining the plant’s overload capability.  
  - **`ResLimShare`** (*fraction*) – Maximum share of plant capacity that can be allocated for reserves.  
  - **`Capex`** (*$/kW*) – Capital expenditure per unit of installed capacity.  
  - **`FOMperMW`** (*$/MW-year*) – Fixed operation and maintenance cost per MW of capacity.  
  - **`VOM`** (*$/MWh*) – Variable operation and maintenance cost per unit of electricity generated.  
  - **`ReserveCost`** (*$/MW-year*) – Additional cost for maintaining reserve capacity.  
  - **`Life`** (*years*) – Expected operational lifetime of the plant.  

- **Example Link**: [pGenDataExcelCustom.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pGenDataExcelCustom.csv)

---

### `pGenDataExcelDefault.csv`
- **Description**:  
  A dataset with **default** characteristics for power generation technologies, reducing the need for repetitive entries in `pGenDataExcelCustom.csv`.  
  It provides **default values** based on **zone, technology name, and fuel name** for various plant characteristics.

- **Data Structure**:  
  - 4 dimensions:  
    1. **Zone** – The geographical area where the technology is applied.  
    2. **Type** – The generation technology name, as defined in `pTechData.csv`.  
    3. **Fuel** – The primary fuel name, as defined in `ftfindex.csv`.  
    4. **Characteristics** – Default technical and economic characteristics.  

- **Column Descriptions**:  
  The columns follow the same structure as `pGenDataExcelCustom.csv`.  
  Please refer to its documentation for details on each field, including **capacity, efficiency, operational constraints, and cost parameters**.

- **Source**:  
  The initial dataset is typically based on **CCDR (Country Climate and Development Report) guidelines**.

- **Example Link**: [pGenDataExcelDefault.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pGenDataExcelDefault.csv)
- **Link to Standard Data**: [pGenDataExcelDefaultStandard.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_rwanda/supply/pGenDataExcelDefaultStandard.csv) - See [Technology overview](https://esmap-world-bank-group.github.io/EPM/docs/input_technology_brief.html)

---

### `pAvailabilityCustom.csv`
- **Description**:  
  Defines the **seasonal availability** of power plants, accounting for **maintenance periods** or **hydro storage** constraints.  
  Availability values range between **0 and 1** and represent an average over the season.  
  The optimizer determines the best way to manage reduced availability.

- **Data Structure**:  
  - **Plant** (*string*) – Name of the power plant.  
  - **Season** (*string*) – The season for which availability is defined.  
  - **Availability Factor** (*fraction [0-1]*) – Defines the fraction of the plant's capacity available for use.  

- **Example Link**: [pAvailabilityCustom.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pAvailabilityCustom.csv)  

---

### `pAvailabilityDefault.csv`
- **Description**:  
  Provides **default seasonal availability** for power plants based on **zone, technology, and fuel name**.  
  This dataset enables automatic filling of seasonal availability values.  

- **Data Structure**:  
  - **Zone** (*string*) – The geographical area.  
  - **Technology** (*string*) – Name of power generation technology.  
  - **Fuel** (*string*) – Primary fuel name.  
  - **Season** (*string*) – The season for which availability is defined.  
  - **Availability Factor** (*fraction [0-1]*) – The default availability value for this configuration.  

- **Source**:  
  The initial dataset is typically based on **CCDR (Country Climate and Development Report) guidelines**.

- **Example Link**: [pAvailabilityDefault.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pAvailabilityDefault.csv)  

---

### `pVREgenProfile.csv`
- **Description**:  
  Defines the **generation profile** of variable renewable energy (VRE) plants over time.

- **Data Structure**:  
  - **Plant** (*string*) – Name of the VRE plant.  
  - **Season** (*string*) – The season considered.  
  - **Day** (*integer*) – The day within the season.  
  - **Time** (*hour*) – The time of day.  
  - **Generation Factor** (*fraction [0-1]*) – The fraction of installed capacity available at a given time.  

- **Example Link**: [pVREgenProfile.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pVREgenProfile.csv)  

---

### `pVREProfile.csv`
- **Description**:  
  Defines **VRE generation profiles** at the **technology level** and automatically populates `pVREgenProfile.csv`.  

- **Data Structure**:  
  - **Technology** (*string*) – Name of renewable technology (e.g., PV, WIND, ROR).  
  - **Season** (*string*) – The season considered.  
  - **Day** (*integer*) – The day within the season.  
  - **Time** (*hour*) – The time of day.  
  - **Generation Factor** (*fraction [0-1]*) – The expected generation fraction for this technology.  

- **Example Link**: [pVREProfile.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pVREProfile.csv)  

---

### `pCapexTrajectoriesCustom.csv`
- **Description**:  
  Defines the **capital expenditure (CAPEX) cost trajectory** for specific plants over time.  
  A factor of **0.5 in 2050** means the cost will be **50% of the initial CAPEX value** from `pGenData.csv` in 2050.

- **Data Structure**:  
  - **Plant** (*string*) – Name of the power plant.  
  - **Year** (*integer*) – Year of CAPEX adjustment.  
  - **CAPEX Factor** (*fraction [0-1]*) – Adjustment factor applied to the original CAPEX value.  

- **Example Link**: [pCapexTrajectoriesCustom.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pCapexTrajectoriesCustom.csv)  

---

### `pCapexTrajectoriesDefault.csv`
- **Description**:  
  Provides **default CAPEX cost trajectories** for different **zones, technology names, and fuels**, reducing the need for repeated entries.  

- **Data Structure**:  
  - **Zone** (*string*) – The geographical area.  
  - **Technology** (*string*) – Name of power generation technology.  
  - **Fuel** (*string*) – Primary fuel name.  
  - **Year** (*integer*) – Year of CAPEX adjustment.  
  - **CAPEX Factor** (*fraction [0-1]*) – Adjustment factor applied to the original CAPEX value.  

- **Source**:  
  The initial dataset is typically based on **CCDR (Country Climate and Development Report) guidelines**.

- **Example Link**: [pCapexTrajectoriesDefault.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pCapexTrajectoriesDefault.csv)  

---

### `pFuelPrice.csv`
- **Description**:  
  Provides **fuel price projections** by country and year, used for economic modeling of power generation costs.  

- **Data Structure**:  
  - **Country** (*string*) – Name of the country.  
  - **Fuel** (*string*) – Name of fuel (as defined in `ftfindex.csv`).  
  - **Year** (*integer*) – Year for the price data.  
  - **Fuel Price** (*$/MMBtu*) – Cost of fuel per million British thermal units (MMBtu).  

- **Example Link**: [pFuelPrice.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pFuelPrice.csv)  

--
### Others
`pCSPData.csv` and `pStorDataExcel.csv` are not included in the documentation, as we want to merge them into `pGenDataExcel.csv`.

- **Example Link**: 
  - [pCSPData.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pCSPData.csv)
  - [pStorDataExcel.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pStorDataExcel.csv)

## Constraint

All files in this section are stored in the **`constraint`** folder. 
Constraint are typically defined in the `pSettings.csv` file. Here, these files are used to define the constraints in more detail.

### `pCarbonPrice.csv`
- **Description**:  
  Defines the **carbon price trajectory** over time, used to account for carbon costs in power generation planning.

- **Data Structure**:  
  - **Year** (*integer*) – The year for which the carbon price applies.  
  - **Carbon Price** (*$/tCO₂*) – Cost of carbon emissions per metric ton of CO₂.  

- **Example Link**: [pCarbonPrice.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pCarbonPrice.csv)  

---

### `pEmissionsCountry.csv`
- **Description**:  
  Defines **total CO₂ emissions** at the **country level**, used for national carbon constraints.

- **Data Structure**:  
  - **Zone** (*string*) – The geographical area.  
  - **Year** (*integer*) – The year for which the emission limit applies.  
  - **CO₂ Emissions** (*tCO₂*) – The total allowed emissions for the given year.  

- **Example Link**: [pEmissionsCountry.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pEmissionsCountry.csv)  

---

### `pEmissionsTotal.csv`
- **Description**:  
  Defines the **total CO₂ emissions cap** for the entire system.

- **Data Structure**:  
  - **Year** (*integer*) – The year for which the emission limit applies.  
  - **CO₂ Emissions** (*tCO₂*) – The total allowed emissions for all zones combined.  

- **Example Link**: [pEmissionsTotal.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pEmissionsTotal.csv)  

---

### `pMaxFuelLimit.csv`
- **Description**:  
  Defines **maximum fuel consumption limits** for different zones, used to constrain fuel usage in the model.

- **Data Structure**:  
  - **Zone** (*string*) – The geographical area.  
  - **Fuel** (*string*) – Name of fuel (as defined in `ftfindex.csv`).  
  - **Year** (*integer*) – The year for which the limit applies.  
  - **Max Fuel Limit** (*MMBtu*) – Maximum allowable fuel consumption in million British thermal units.  

- **Example Link**: [pMaxFuelLimit.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pMaxFuelLimit.csv)  

## Reserve  

There are two types of reserves in **EPM**:  

1. **Spinning Reserve** – A real-time operational reserve that provides immediate response to sudden changes in demand or supply.  
2. **Planning Reserve** – A reserve margin that ensures sufficient generation capacity is available to meet peak demand and contingencies.  

**Note:**  
- Spinning reserve requirements depend on the values in this dataset **plus** the forecast error of VRE (Variable Renewable Energy), which is defined in `pSettings.csv`.  
- To be included in the model, reserve requirements must be properly **defined in `pSettings.csv`**.  

---

### `pPlanningReserveMargin.csv`
- **Description**:  
  Defines the **minimum required planning reserve margin** as a **share of total demand**, ensuring system adequacy.  

- **Data Structure**:  
  - **Zone** (*string*) – The geographical area.  
  - **Year** (*integer*) – The year for which the planning reserve requirement applies.  
  - **Reserve Margin** (*fraction*) – The percentage of total demand that must be reserved for planning purposes.  

- **Example Link**: [pPlanningReserveMargin.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/reserve/pPlanningReserveMargin.csv)  

---

### `pSpinningReserveReqCountry.csv`
- **Description**:  
  Defines the **spinning reserve requirement at the country level**, ensuring adequate capacity is available for sudden demand fluctuations or generation losses.  

- **Data Structure**:  
  - **Country** (*string*) – The country for which the reserve requirement applies.  
  - **Year** (*integer*) – The year for which the requirement applies.  
  - **Spinning Reserve Requirement** (*MW*) – The minimum spinning reserve capacity required in megawatts.  

- **Example Link**: [pSpinningReserveReqCountry.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/reserve/pSpinningReserveReqCountry.csv)  

---

### `pSpinningReserveReqTotal.csv`
- **Description**:  
  Defines the **total system-wide spinning reserve requirement**, summing up reserves required across all zones/countries.  

- **Data Structure**:  
  - **Year** (*integer*) – The year for which the requirement applies.  
  - **Total Spinning Reserve Requirement** (*MW*) – The total spinning reserve capacity required across the entire system.  

- **Example Link**: [pSpinningReserveReqTotal.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/reserve/pSpinningReserveReqTotal.csv)  

---

## Trade

Documentation in progress. Check the `trade` [folder](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/trade) for more details.

### `pExtTransferLimit.csv`

- **Description**:  

    Defines the available capacity for price-driven imports and exports on a seasonal basis. This input can be adjusted alongside the code to support finer time resolutions, such as hourly capacity definitions.

    Note: This input is only used when `pAllowExports` is set to 1 in `pSettings.csv`. 

- **Data Structure**:  
  - **Index**
    - **Internal zone** (*str*) –  Origin of the import/export.  
    - **External zone** (*str*) – Destination of the import/export.  
    - **Seasons** (*str*) – Season for which the capacity is specified.
    - **Import/Export** (*str*) –  Indicates whether the capacity applies to imports or exports. Allowed values: Export, Import.
  - **Columns**
    - **Year** (*int*) – Year for which the capacity applies.
  - **Value**
    - Capacity available for imports or exports.

- **Example Link**: [pExtTransferLimit.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/trade/pExtTransferLimit.csv)  


### `pMaxExchangeShare.csv`

- **Description**:  

    Specifies the maximum share of total country-level demand that imports and exports can represent.

    Note: This input is only used when `pAllowExports` is set to 1 in `pSettings.csv`. 

- **Data Structure**:  
  - **Index**
    - **Year** (*int*) –  Year considered  
  - **Columns**
    - **Country** (*str*) – Country considered  
  - **Value**
    - **Maximum exchange share** (*fraction [0-1]*) – Maximum percentage of total demand that imports and exports can reach. 

- **Example Link**: [pMaxExchangeShare.csv](https://github.com.mcas.ms/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_sapp/trade/pMaxExchangeShare.csv)

### `pNewTransmission.csv`

- **Description**:  
  Specifies the key characteristics of candidate transmission lines. Only active when `pAllowHighTransfer` is set to 1 in `pSettings.csv`. Each candidate line must only be specified once in this dataframe (e.g: only specify the line Angola-Namibia once, either as Namibia-Angola, or as Angola-Namibia, the order does not matter). The order in which the start location and the destination location are specified does not matter.

- **Data Structure**:  
  - **Index**
    - **From** (*str*) – Starting location of the transmission line.  
    - **To** (*str*) – Destination of the transmission line.  
  - **Columns**
    - **EarliestEntry** (*int*) – Earliest year the line can be built.
    - **MaximumNumOfLines** (*int*) – Maximum number of lines that can be constructed (lines must be built in whole units).
    - **CapacityPerLine** (*int*) – Capacity of a single transmission line.
    - **CostPerLine** (*float*) – Investment cost per line.
    - **Life** (*int*) – Expected lifespan of the transmission line.


- **Example Link**: [pNewTransmission.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_sapp/trade/pNewTransmission.csv)  


### `pTradePrice.csv`

- **Description**:  
  Trade price for imports and exports driven by price.

- **Data Structure**:  
  - **Index**
    - **zext** (*str*) – External zone that can trade.  
    - **Seasons** (*str*) – Season for which the trade price is specified.
    - **Day** (*str*) – Day for which the trade price is specified.
    - **Year** (*int*) – Year for which the trade price is specified.
  - **Columns**
    - **Time**(str) - Hour of the day for which the trade price is specified.
  - **Value**
    - Trade price (€/MWh)


- **Example Link**: [pTradePrice.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_sapp/trade/pTradePrice.csv)  

### `pTransferLimit.csv`

- **Description**:  
    Defines the available capacity for exchanges between internal zones. This dataframe is used to specify the network topology.

- **Data Structure**:  
  - **Index**
    - **From** (*str*) –  Origin of the import/export.  
    - **To** (*str*) – Destination of the import/export.  
    - **Seasons** (*str*) – Season for which the capacity is specified.
  - **Columns**
    - **Year** (*int*) – Year for which the capacity applies.
  - **Value**
    - Capacity available for imports or exports between internal zones.

- **Example Link**: [pTransferLimit.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_sapp/trade/pTransferLimit.csv)  

### `zext.csv`

- **Description**:  
  Lists external zones that can trade with the modeled zones. These external zones can only contribute to imports and exports based on predefined prices; their generation mix and supply availability are not modeled, as they are not explicitly modeled.  

- **Data Structure**:  
  - **Index**
    - **zone** (*str*) – Name of external zone

- **Example Link**: [zext.csv](https://github.com.mcas.ms/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_sapp/trade/zext.csv)

### `pLossFactor.csv`

- **Description**:  
    Defines the transmission losses for each transmission lines. 
    **Note**: when interconnected mode is activated (intercon = 1), the losses from each transmission line must be specified. If LossFactor is empty, then

---

## H2

Documentation in progress. Check the `h2` [h2](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/h2) for more details.
