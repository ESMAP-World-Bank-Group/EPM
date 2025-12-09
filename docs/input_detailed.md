# Input Description

Input files are located in the data_input folder, and are organized into several subfolders based on their type and purpose.

### **`pSettings.csv`**

- **Description**:  
  This file contains **global model parameters and settings** that control various aspects of the **EPM model**, including **economic assumptions, penalties, optional features, and constraints**.  
  It plays a **critical role in defining the behavior of the optimization model**, enabling customization of financial assumptions, operational constraints, and system features.
- **Data Structure**:
  - **Parameter** (_string_) – Full name of the parameter.
  - **Abbreviation** (_string_) – Short name used in the model.
  - **Value** (_varied units_) – The assigned numerical value or toggle (0/1) for the parameter.
- **Example Link**: [pSettings.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test_region/config/pSettings.csv)

**!! Important !!** The validated `pSettings.csv` template in the repository now includes the full set of required parameters. Starting from EPM v2024.09, the model expects this structure. Always copy the latest file (or extend your custom copy) to avoid missing parameters that would otherwise default to zero.

The CSV groups parameters by topical blocks. Each row exposes a human-readable label, the abbreviation used in GAMS, and the default value provided in the template.

---

#### **Core Parameters (`PARAMETERS` block)**

| CSV label                                           | Abbreviation     | Default value | Units / Notes                                  |
| --------------------------------------------------- | ---------------- | ------------- | ---------------------------------------------- |
| Weighted Average Cost of Capital (WACC), %          | `WACC`           | 0.06          | Fraction (6%)                                  |
| Discount rate, %                                    | `DR`             | 0.06          | Fraction (6%)                                  |
| Cost of unserved energy per MWh, $                  | `VoLL`           | 1000          | $/MWh                                          |
| Cost of reserve shortfall per MW, $                 | `ReserveVoLL`    | 60000         | $/MW                                           |
| Spin Reserve VoLL per MWh, $                        | `SpinReserveVoLL`| 60            | $/MWh                                          |
| Cost of surplus power per MWh, $                    | `CostSurplus`    | 0             | $/MWh                                          |
| Cost of curtailment per MWh, $                      | `CostCurtail`    | 0             | $/MWh                                          |
| Cost of curtailment per MWh, $¹                     | `CO2backstop`    | 300           | $/tCO₂ (label kept as-is in the CSV)           |
| Cost of climate backstop techno per ton CO₂, $      | `H2UnservedCost` | 3000          | $/tCO₂                                         |

¹CSV label duplicates the curtailment text but the abbreviation drives the CO₂ backstop penalty.

---

#### **Interconnection settings**

**Internal exchanges (`INTERNAL` block)**

| CSV label                                   | Abbreviation                   | Default value | Units / Notes                        |
| ------------------------------------------- | ------------------------------ | ------------- | ------------------------------------ |
| Activate exchange between internal zone     | `fEnableInternalExchange`      | 1             | Toggle (1 = enabled)                 |
| Remove transfer limits between internal zone| `fRemoveInternalTransferLimit` | 0             | Toggle                               |
| Allow expansion of transfer limits          | `fAllowTransferExpansion`      | 1             | Toggle                               |

**External exchanges (`EXTERNAL` block)**

| CSV label                                 | Abbreviation                       | Default value | Units / Notes                        |
| ----------------------------------------- | ---------------------------------- | ------------- | ------------------------------------ |
| Activate exchange between external zone   | `fEnableExternalExchange`          | 1             | Toggle                               |
| Maximum external import share, %          | `sMaxHourlyImportExternalShare`    | 1             | Fraction (1 = 100%)                  |
| Maximum external export share, %          | `sMaxHourlyExportExternalShare`    | 1             | Fraction (1 = 100%)                  |

---

#### **Optional features**

| CSV label                                                                 | Abbreviation                  | Default value | Units / Notes                                              |
| ------------------------------------------------------------------------- | ----------------------------- | ------------- | ---------------------------------------------------------- |
| Include carbon price                                                      | `fEnableCarbonPrice`          | 0             | Toggle                                                     |
| Include energy efficiency                                                 | `fEnableEnergyEfficiency`     | 0             | Toggle                                                     |
| Include CSP optimization                                                  | `fEnableCSP`                  | 0             | Toggle                                                     |
| Include Storage operation                                                 | `fEnableStorage`              | 1             | Toggle                                                     |
| Retire plants on economic grounds                                         | `fEnableEconomicRetirement`   | 0             | Toggle                                                     |
| Use less detailed demand definition                                       | `fUseSimplifiedDemand`        | 1             | Toggle                                                     |
| Fractional tolerance around system peak load used to identify ‘near-peak’ | `sPeakLoadProximityThreshold` | *(blank)*     | Provide a fraction when using the near-peak filter         |

---

#### **Planning reserves (`PLANNING RESERVES` block)**

| CSV label                                                       | Abbreviation                | Default value | Units / Notes                        |
| --------------------------------------------------------------- | --------------------------- | ------------- | ------------------------------------ |
| Include transmission lines when assessing country planning reserves | `fCountIntercoForReserves` | 1             | Toggle                               |
| Apply planning reserve constraint                               | `fApplyPlanningReserveConstraint` | 1       | Toggle                               |
| System planning reserve margin, %                               | `sReserveMarginPct`         | 0.1           | Fraction (10%)                       |

---

#### **Spinning reserves (`SPINNING RESERVES` block)**

| CSV label                                                                   | Abbreviation                         | Default value | Units / Notes               |
| --------------------------------------------------------------------------- | ------------------------------------ | ------------- | --------------------------- |
| Apply country spinning reserve constraints                                 | `fApplyCountrySpinReserveConstraint` | 1             | Toggle                      |
| Apply system spinning reserve constraints                                  | `fApplySystemSpinReserveConstraint`  | 0             | Toggle                      |
| Spinning (country and system) reserve needs for VRE, %                      | `sVREForecastErrorPct`               | 0.15          | Fraction (15%)              |
| Contribution of transmission lines to country spinning reserves need, %     | `sIntercoReserveContributionPct`     | 0             | Fraction                     |

---

#### **Hydrogen options (`H2` block)**

| CSV label                    | Abbreviation              | Default value | Units / Notes |
| ---------------------------- | ------------------------- | ------------- | ------------- |
| Allow capex trajectory H2    | `fEnableCapexTrajectoryH2`| 0             | Toggle        |
| Include H2 production        | `fEnableH2Production`     | 0             | Toggle        |

---

#### **Policy levers (`POLICY` block)**

| CSV label                            | Abbreviation                | Default value | Units / Notes        |
| ------------------------------------ | --------------------------- | ------------- | ---------------------|
| Apply country CO₂ constraint         | `fApplyCountryCo2Constraint`| 0             | Toggle               |
| Apply system CO₂ constraints         | `fApplySystemCo2Constraint` | 0             | Toggle               |
| Minimum share of RE, %               | `sMinRenewableSharePct`     | 0             | Fraction             |
| RE share target year                 | `sRenewableTargetYear`      | *(blank)*     | Year (leave blank if unused) |
| Apply fuel constraints               | `fApplyFuelConstraint`      | 0             | Toggle               |
| Apply max capital constraint         | `fApplyCapitalConstraint`   | 0             | Toggle               |
| Total maximum capital investments, $ billion | `sMaxCapitalInvestment` | *(blank)*     | Billion USD          |

---

#### **Plant operations (`PLANTS` block)**

| CSV label                     | Abbreviation                | Default value | Units / Notes |
| ----------------------------- | --------------------------- | ------------- | ------------- |
| Apply min generation constraint | `fApplyMinGenCommitment` | 0         | Toggle        |
| Apply ramp constraints        | `fApplyRampConstraint`      | 0             | Toggle        |

---

**Notes:**

- All cost values in the template are expressed in USD.
- Percentages are stored as fractions (e.g., `0.06` = 6%).
- Toggle parameters use `0` (disabled) and `1` (enabled).
- Blank values in the template indicate that you must provide project-specific numbers before running a study.

---

### `y.csv`

Defines the years included in the intertemporal optimization. One-dimensional csv.

- **Data Structure**:

  - **Index**
    - **y** (_int_) – Years to be included in the model.

- **Example Link**: [y.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/y.csv)

### **`zcmap.csv`**

- Two-dimensional, defines the zones and countries included in the model.
- Example: [zcmap.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/zcmap.csv)

## Resources

All files in this section are included in the `resources` folder. These datasets define parameters that do not necessarily change between projects.

---

### **`ftfindex.csv`**

TODO: needs to be updated !!

- List of fuels recognized by EPM.
- 2 columns:
  1. Fuel
  2. Index (Note: Index is no longer used, and the structure is being simplified to keep only "Fuel").
- Example: [ftfindex.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/resources/ftfindex.csv)
- List of fuel types recognized by EPM:
  - **Coal**
  - **Gas**
  - **Water**
  - **Solar**
  - **Wind**
  - **Import**
  - **HFO**
  - **LFO**
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
- 3 columns:
  1. Technology name
  2. Hourly variation (if the technology varies hourly)
  3. RE Technology (if it is a renewable energy technology)
- Example: [pTechData.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_test_region/resources/pTechData.csv)
- List of technologies recognized by EPM:
  - **OCGT**: Gas Turbine
  - **CCGT**: Combined Cycle Gas Turbine
  - **ST**: Steam Turbine
  - **ICE**: Internal Combustion Engine
  - **OnshoreWind**: Onshore wind power plant
  - **OffshoreWind**: Offshore wind power plant
  - **PV**: Photovoltaic power plant
  - **PVwSTO**: Photovoltaic with Storage
  - **CSPPlant**: Concentrated Solar power plant
  - **Storage**: Storage
  - **ReservoirHydro**: Storage hydro power plant
  - **ROR**: Run of River hydro power plant
  - **BiomassPlant**: Biomass power plant
  - **CHP**: Combined Heat and Power
  - **ImportTransmission**: Representing imports from external zones

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

### `pGenDataInputCustom.csv`

- **Description**: The main data table defining all power plants, including their **technical** and **economic** characteristics.
- **Data Structure**:

  - **`gen`** (_string_) – Name of the power plant.
  - **`zone`** (_string_) – Zone where the plant is located, as defined in `zcmap.csv`.
  - **`tech`** (_string_) – Technology type. Additional information on technologies can be found in `pTechData.csv`.
  - **`fuel`** (_string_) – Primary fuel name, as defined in `ftfindex.csv`.
  - **`StYr`** (_year_) – Start year of plant operation.
  - **`RetrYr`** (_year_) – Retirement year of the plant.
  - **`Capacity`** (_MW_) – Installed capacity of the plant.
  - **`Status`** (_integer_) – Operational status indicator (e.g., existing, planned, retired).
  - **`MinLimitShare`** (_fraction_) – Minimum share of total generation capacity that must be maintained.
  - **`HeatRate`** (_MMBtu/MWh_) – Heat rate when operating on primary fuel.
  - **`RampUpRate`** (_fraction/hour_) – Maximum rate at which generation can be increased.
  - **`RampDnRate`** (_fraction/hour_) – Maximum rate at which generation can be decreased.
  - **`OverLoadFactor`** (_fraction_) – Factor determining the plant’s overload capability.
  - **`ResLimShare`** (_fraction_) – Maximum share of plant capacity that can be allocated for reserves.
  - **`Capex`** (_$m/MW_) – Capital expenditure per unit of installed capacity.
  - **`FOMperMW`** (_$/MW-year_) – Fixed operation and maintenance cost per MW of capacity.
  - **`VOM`** (_$/MWh_) – Variable operation and maintenance cost per unit of electricity generated.
  - **`ReserveCost`** (_$/MW-year_) – Additional cost for maintaining reserve capacity.
  - **`Life`** (_years_) – Expected operational lifetime of the plant.
  - **`UnitSize`** (_MW_) – Size of a single unit in the plant.
  - **`fuel2`** (_string_) – Secondary fuel name, if applicable.
  - **`HeatRate2`** (_MMBtu/MWh_) – Heat rate when operating on secondary fuel.
  - **`DescreteCap`** (_MW_) – Discrete capacity increments allowed for expansion.
  - **`BuildLimitperYear`** (_MW/year_) – Maximum allowable capacity additions per year.
  - **`MaxTotalBuild`** (_MW_) – Maximum total capacity allowed for the plant.

- **Example Link**: [pGenDataInputCustom.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pGenDataInputCustom.csv)

---

### `pGenDataInputDefault.csv`

- **Description**:  
  A dataset with **default** characteristics for power generation technologies, reducing the need for repetitive entries in `pGenDataInputCustom.csv`.  
  It provides **default values** based on **zone, technology name, and fuel name** for various plant characteristics.

- **Data Structure**:

  - 4 dimensions:
    1. **Zone** – The geographical area where the technology is applied.
    2. **Type** – The generation technology name, as defined in `pTechData.csv`.
    3. **Fuel** – The primary fuel name, as defined in `ftfindex.csv`.
    4. **Characteristics** – Default technical and economic characteristics.

- **Column Descriptions**:  
  The columns follow the same structure as `pGenDataInputCustom.csv`.  
  Please refer to its documentation for details on each field, including **capacity, efficiency, operational constraints, and cost parameters**.

- **Source**:  
  The initial dataset is typically based on **CCDR (Country Climate and Development Report) guidelines**.

- **Example Link**: [pGenDataInputDefault.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pGenDataInputDefault.csv)
- **Link to Standard Data**: [pGenDataInputDefaultStandard.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_rwanda/supply/pGenDataInputDefaultStandard.csv) - See [Technology overview](https://esmap-world-bank-group.github.io/EPM/docs/input_technology_brief.html)

---

### `pAvailabilityCustom.csv`

- **Description**:  
  Defines the **seasonal availability** of power plants, accounting for **maintenance periods** or **hydro storage** constraints.  
  Availability values range between **0 and 1** and represent an average over the season.  
  The optimizer determines the best way to manage reduced availability.

- **Data Structure**:

  - **Plant** (_string_) – Name of the power plant.
  - **Season** (_string_) – The season for which availability is defined.
  - **Availability Factor** (_fraction [0-1]_) – Defines the fraction of the plant's capacity available for use.

- **Example Link**: [pAvailabilityCustom.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pAvailabilityCustom.csv)

---

### `pAvailabilityDefault.csv`

- **Description**:  
  Provides **default seasonal availability** for power plants based on **zone, technology, and fuel name**.  
  This dataset enables automatic filling of seasonal availability values.

- **Data Structure**:

  - **Zone** (_string_) – The geographical area.
  - **Technology** (_string_) – Name of power generation technology.
  - **Fuel** (_string_) – Primary fuel name.
  - **Season** (_string_) – The season for which availability is defined.
  - **Availability Factor** (_fraction [0-1]_) – The default availability value for this configuration.

- **Source**:  
  The initial dataset is typically based on **CCDR (Country Climate and Development Report) guidelines**.

- **Example Link**: [pAvailabilityDefault.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pAvailabilityDefault.csv)

---

### `pVREgenProfile.csv`

- **Description**:  
  Defines the **generation profile** of variable renewable energy (VRE) plants over time.

- **Data Structure**:

  - **Plant** (_string_) – Name of the VRE plant.
  - **Season** (_string_) – The season considered.
  - **Day** (_integer_) – The day within the season.
  - **Time** (_hour_) – The time of day.
  - **Generation Factor** (_fraction [0-1]_) – The fraction of installed capacity available at a given time.

- **Example Link**: [pVREgenProfile.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pVREgenProfile.csv)

---

### `pVREProfile.csv`

- **Description**:  
  Defines **VRE generation profiles** at the **technology level** and automatically populates `pVREgenProfile.csv`.

- **Data Structure**:

  - **Technology** (_string_) – Name of renewable technology (e.g., PV, WIND, ROR).
  - **Season** (_string_) – The season considered.
  - **Day** (_integer_) – The day within the season.
  - **Time** (_hour_) – The time of day.
  - **Generation Factor** (_fraction [0-1]_) – The expected generation fraction for this technology.

- **Example Link**: [pVREProfile.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pVREProfile.csv)

---

### `pCapexTrajectoriesCustom.csv`

- **Description**:  
  Defines the **capital expenditure (CAPEX) cost trajectory** for specific plants over time.  
  A factor of **0.5 in 2050** means the cost will be **50% of the initial CAPEX value** from `pGenData.csv` in 2050.

- **Data Structure**:

  - **Plant** (_string_) – Name of the power plant.
  - **Year** (_integer_) – Year of CAPEX adjustment.
  - **CAPEX Factor** (_fraction [0-1]_) – Adjustment factor applied to the original CAPEX value.

- **Example Link**: [pCapexTrajectoriesCustom.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pCapexTrajectoriesCustom.csv)

---

### `pCapexTrajectoriesDefault.csv`

- **Description**:  
  Provides **default CAPEX cost trajectories** for different **zones, technology names, and fuels**, reducing the need for repeated entries.

- **Data Structure**:

  - **Zone** (_string_) – The geographical area.
  - **Technology** (_string_) – Name of power generation technology.
  - **Fuel** (_string_) – Primary fuel name.
  - **Year** (_integer_) – Year of CAPEX adjustment.
  - **CAPEX Factor** (_fraction [0-1]_) – Adjustment factor applied to the original CAPEX value.

- **Source**:  
  The initial dataset is typically based on **CCDR (Country Climate and Development Report) guidelines**.

- **Example Link**: [pCapexTrajectoriesDefault.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pCapexTrajectoriesDefault.csv)

---

### `pFuelPrice.csv`

- **Description**:  
  Provides **fuel price projections** by country and year, used for economic modeling of power generation costs.

- **Data Structure**:

  - **Country** (_string_) – Name of the country.
  - **Fuel** (_string_) – Name of fuel (as defined in `ftfindex.csv`).
  - **Year** (_integer_) – Year for the price data.
  - **Fuel Price** (_$/MMBtu_) – Cost of fuel per million British thermal units (MMBtu).

- **Example Link**: [pFuelPrice.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pFuelPrice.csv)

--

### Others

`pCSPData.csv` and `pStorDataExcel.csv` are not included in the documentation, as we want to merge them into `pGenDataInput.csv`.

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

  - **Year** (_integer_) – The year for which the carbon price applies.
  - **Carbon Price** (_$/tCO₂_) – Cost of carbon emissions per metric ton of CO₂.

- **Example Link**: [pCarbonPrice.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pCarbonPrice.csv)

---

### `pEmissionsCountry.csv`

- **Description**:  
  Defines **total CO₂ emissions** at the **country level**, used for national carbon constraints.

- **Data Structure**:

  - **Zone** (_string_) – The geographical area.
  - **Year** (_integer_) – The year for which the emission limit applies.
  - **CO₂ Emissions** (_tCO₂_) – The total allowed emissions for the given year.

- **Example Link**: [pEmissionsCountry.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pEmissionsCountry.csv)

---

### `pEmissionsTotal.csv`

- **Description**:  
  Defines the **total CO₂ emissions cap** for the entire system.

- **Data Structure**:

  - **Year** (_integer_) – The year for which the emission limit applies.
  - **CO₂ Emissions** (_tCO₂_) – The total allowed emissions for all zones combined.

- **Example Link**: [pEmissionsTotal.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pEmissionsTotal.csv)

---

### `pMaxFuelLimit.csv`

- **Description**:  
  Defines **maximum fuel consumption limits** for different zones, used to constrain fuel usage in the model.

- **Data Structure**:

  - **Zone** (_string_) – The geographical area.
  - **Fuel** (_string_) – Name of fuel (as defined in `ftfindex.csv`).
  - **Year** (_integer_) – The year for which the limit applies.
  - **Max Fuel Limit** (_MMBtu_) – Maximum allowable fuel consumption in million British thermal units.

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

  - **Zone** (_string_) – The geographical area.
  - **Year** (_integer_) – The year for which the planning reserve requirement applies.
  - **Reserve Margin** (_fraction_) – The percentage of total demand that must be reserved for planning purposes.

- **Example Link**: [pPlanningReserveMargin.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/reserve/pPlanningReserveMargin.csv)

---

### `pSpinningReserveReqCountry.csv`

- **Description**:  
  Defines the **spinning reserve requirement at the country level**, ensuring adequate capacity is available for sudden demand fluctuations or generation losses.

- **Data Structure**:

  - **Country** (_string_) – The country for which the reserve requirement applies.
  - **Year** (_integer_) – The year for which the requirement applies.
  - **Spinning Reserve Requirement** (_MW_) – The minimum spinning reserve capacity required in megawatts.

- **Example Link**: [pSpinningReserveReqCountry.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/reserve/pSpinningReserveReqCountry.csv)

---

### `pSpinningReserveReqTotal.csv`

- **Description**:  
  Defines the **total system-wide spinning reserve requirement**, summing up reserves required across all zones/countries.

- **Data Structure**:

  - **Year** (_integer_) – The year for which the requirement applies.
  - **Total Spinning Reserve Requirement** (_MW_) – The total spinning reserve capacity required across the entire system.

- **Example Link**: [pSpinningReserveReqTotal.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/reserve/pSpinningReserveReqTotal.csv)

---

## Trade

Documentation in progress. Check the `trade` [folder](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/trade) for more details.

| **File**                               | **Purpose / Definition**                                                                                      | **Scope**                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **`pTransferLimit.csv`**               | Defines transfer capacities between internal zones and network topology.                                      | **Explicit – within region**  |
| **`pNewTransmission.csv`**             | Defines candidate internal transmission lines for potential expansion. Used if `fAllowTransferExpansion` = 1. | **Explicit – within region**  |
| **`pLossFactorInternal.csv`**          | Specifies transmission losses on each internal transmission line (required in interconnected mode).           | **Explicit – within region**  |
| **`zext.csv`**                         | Lists external zones available for trade. These zones are implicit (not modeled in detail).                   | **Implicit – external trade** |
| **`pExtTransferLimit.csv`**            | Defines seasonal transfer capacities between internal and external zones for price-driven imports/exports.    | **Implicit – external trade** |
| **`pTradePrice.csv`**                  | Sets import/export prices from/to external zones (by hour, season, day, year).                                | **Implicit – external trade** |
| **`pMaxAnnualExternalTradeShare.csv`** | Limits the maximum share of total demand that can be imported or exported by a country.                       | **Implicit – external trade** |

### `pTransferLimit.csv`

- **Description**:  
   Defines the available capacity for exchanges between internal zones. This dataframe is used to specify the network topology.

- **Data Structure**:

  - **Index**
    - **From** (_str_) – Origin of the import/export.
    - **To** (_str_) – Destination of the import/export.
    - **Seasons** (_str_) – Season for which the capacity is specified.
  - **Columns**
    - **Year** (_int_) – Year for which the capacity applies.
  - **Value**
    - Capacity available for imports or exports between internal zones.

- **Example Link**: [pTransferLimit.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test_region/trade/pTransferLimit.csv)

### `pNewTransmission.csv`

- **Description**:  
  Defines candidate transmission lines for potential expansion. **This file is only used when fAllowTransferExpansion is set to 1 in pSettings.csv.**

Each transmission line must be specified only once. The order of the From and To locations does not matter (e.g., specify either Angola–Namibia or Namibia–Angola, but not both).

- **Data Structure**:
  - **Index**
    - **From** (_str_) – Starting location of the transmission line.
    - **To** (_str_) – Destination of the transmission line.
  - **Columns**
    - **EarliestEntry** (_int_) – Earliest year the line can be built.
    - **MaximumNumOfLines** (_int_) – Maximum number of lines that can be constructed (lines must be built in whole units).
    - **CapacityPerLine** (_int_) – Capacity of a single transmission line.
    - **CostPerLine** (_float_) – Investment cost per line.
    - **Life** (_int_) – Expected lifespan of the transmission line.
    - **Status** (_int_) – Line status
      - `2`: committed line
      - `3`: candidate line

**Usage notes**

- Do not include lines listed in pNewTransmission.csv in `pTransferLimit.csv`, or they will be double-counted.
- For the model to consider lines in pNewTransmission.csv, the option `allowTransferExpansion` in `pSettings.csv` must be activated.

- **Example Link**: [pNewTransmission.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test_region/trade/pNewTransmission.csv)

### `pLossFactorInternal.csv`

- **Description**:  
   Defines the transmission losses for each transmission lines.
  **Note**: when interconnected mode is activated (intercon = 1), the losses from each transmission line must be specified.

### `zext.csv`

- **Description**:  
  Lists external zones that can trade with the modeled zones. These external zones can only contribute to imports and exports based on predefined prices; their generation mix and supply availability are not modeled, as they are not explicitly modeled.

- **Data Structure**:

  - **Index**
    - **zone** (_str_) – Name of external zone

- **Example Link**: [zext.csv](https://github.com.mcas.ms/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test_region/trade/zext.csv)

### `pExtTransferLimit.csv`

- **Description**:

  Defines the available capacity for price-driven imports and exports on a seasonal basis. This input can be adjusted alongside the code to support finer time resolutions, such as hourly capacity definitions.

  Note: This input is only used when `fAllowTransferExpansion` is set to 1 in `pSettings.csv`.

- **Data Structure**:

  - **Index**
    - **Internal zone** (_str_) – Origin of the import/export.
    - **External zone** (_str_) – Destination of the import/export.
    - **Seasons** (_str_) – Season for which the capacity is specified.
    - **Import/Export** (_str_) – Indicates whether the capacity applies to imports or exports. Allowed values: Export, Import.
  - **Columns**
    - **Year** (_int_) – Year for which the capacity applies.
  - **Value**
    - Capacity available for imports or exports.

- **Example Link**: [pExtTransferLimit.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/trade/pExtTransferLimit.csv)

### `pTradePrice.csv`

- **Description**:  
  Trade price for imports and exports driven by price.

- **Data Structure**:

  - **Index**
    - **zext** (_str_) – External zone that can trade.
    - **Seasons** (_str_) – Season for which the trade price is specified.
    - **Day** (_str_) – Day for which the trade price is specified.
    - **Year** (_int_) – Year for which the trade price is specified.
  - **Columns**
    - **Time**(str) - Hour of the day for which the trade price is specified.
  - **Value**
    - Trade price (€/MWh)

- **Example Link**: [pTradePrice.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test_region/trade/pTradePrice.csv)

### `pMaxAnnualExternalTradeShare.csv`

- **Description**:

  Specifies the maximum share of total country-level demand that imports and exports can represent.

  Note: This input is only used when `fAllowTransferExpansion` is set to 1 in `pSettings.csv`.

- **Data Structure**:

  - **Index**
    - **Year** (_int_) – Year considered
  - **Columns**
    - **Country** (_str_) – Country considered
  - **Value**
    - **Maximum exchange share** (_fraction [0-1]_) – Maximum percentage of total demand that imports and exports can reach.

- **Example Link**: [pMaxAnnualExternalTradeShare.csv](https://github.com.mcas.ms/ESMAP-World-Bank-Group/EPM/blob/main/epm/input/data_test_region/trade/pMaxAnnualExternalTradeShare.csv)

---

## H2

Documentation in progress. Check the `h2` [h2](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/h2) for more details.
