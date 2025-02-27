# Data Structure Documentation

## Configuration Files

These files are direclty located in the `data` folder.

---

### **`pSettings.csv`**  
  - 3 columns, but only the last two columns are read by the model.  
  - Example: [pSettings.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/pSettings.csv)


The **Settings** file allows users to configure the model:
- **General options**: Set Weighted Average Costs of Capital (WACC) for capital investments and the discount rate for present value calculations (in percentages).
- **Penalties**: Define penalties for unserved energy, curtailment, surplus power, etc.
- **Optional Features**: Enable or disable features such as:
  - Carbon prices
  - Energy efficiency
  - Storage optimization
  - CSP optimization
- **Constraints**: Apply or remove limits like:
  - CO2 emissions
  - Fuel constraints
  - Ramp constraints
  - Import share constraints
- **Output options**: Select results the model should produce (default is "yes" for all).
- **Active Fuel Price Scenario**: Select the fuel price scenario (scenarios are defined in the `FuelTechnologies.csv` file).
- **Capacity Credits**: Define capacity credits for variable renewable technologies. These apply only if the "User defined Capacity Credit" option is set to "Yes."

---

### **`y.csv`**  
  - One-dimensional, represents the years included in the intertemporal optimization.  
  - Example: [y.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/y.csv)

---

### **`zcmap.csv`**  
  - Two-dimensional, defines the zones and countries included in the model.  
  - Example: [zcmap.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/zcmap.csv)

## Resources

All files in this section are included in the `resources` folder. These datasets define parameters that do not necessarily change between projects.

---

### **`ftfindex.csv`**  
  - List of fuels recognized by EPM.  
  - 3 columns:  
    1. Fuel type  
    2. Fuel  
    3. Index (Note: Index is no longer used, and the structure is being simplified to keep only "Fuel").  
  - Example: [ftfindex.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/resources/ftfindex.csv)

---

### **`pTechData.csv`**  
  - List of technologies recognized by EPM.  
  - 4 columns:  
    1. Technology name  
    2. Construction period (in years)  
    3. Hourly variation (if the technology varies hourly)  
    4. RE Technology (if it is a renewable energy technology)  
  - Example: [pTechData.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/resources/pTechData.csv)

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
  - **`fuel1`** (*string*) – Primary fuel type, as defined in `ftfindex.csv`.  
  - **`StYr`** (*year*) – Start year of plant operation.  
  - **`RetrYr`** (*year*) – Retirement year of the plant.  
  - **`Capacity`** (*MW*) – Installed capacity of the plant.  
  - **`UnitSize`** (*MW*) – Size of a single unit in the plant.  
  - **`Status`** (*integer*) – Operational status indicator (e.g., existing, planned, retired).  
  - **`fuel2`** (*string*) – Secondary fuel type, if applicable.  
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
  It provides **default values** based on **zone, technology type, and fuel type** for various plant characteristics.

- **Data Structure**:  
  - 4 dimensions:  
    1. **Zone** – The geographical area where the technology is applied.  
    2. **Type** – The generation technology type, as defined in `pTechData.csv`.  
    3. **Fuel** – The primary fuel type, as defined in `ftfindex.csv`.  
    4. **Characteristics** – Default technical and economic characteristics.  

- **Column Descriptions**:  
  The columns follow the same structure as `pGenDataExcelCustom.csv`.  
  Please refer to its documentation for details on each field, including **capacity, efficiency, operational constraints, and cost parameters**.

- **Source**:  
  The initial dataset is typically based on **CCDR (Country Climate and Development Report) guidelines**.

- **Example Link**: [pGenDataExcelDefault.csv](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/supply/pGenDataExcelDefault.csv)

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
  Provides **default seasonal availability** for power plants based on **zone, technology, and fuel type**.  
  This dataset enables automatic filling of seasonal availability values.  

- **Data Structure**:  
  - **Zone** (*string*) – The geographical area.  
  - **Technology** (*string*) – Type of power generation technology.  
  - **Fuel** (*string*) – Primary fuel type.  
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
  - **Technology** (*string*) – Type of renewable technology (e.g., PV, WIND, ROR).  
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
  Provides **default CAPEX cost trajectories** for different **zones, technology types, and fuels**, reducing the need for repeated entries.  

- **Data Structure**:  
  - **Zone** (*string*) – The geographical area.  
  - **Technology** (*string*) – Type of power generation technology.  
  - **Fuel** (*string*) – Primary fuel type.  
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
  - **Fuel** (*string*) – Type of fuel (as defined in `ftfindex.csv`).  
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
  - **Fuel** (*string*) – Type of fuel (as defined in `ftfindex.csv`).  
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

---

## H2

Documentation in progress. Check the `h2` [h2](https://github.com/ESMAP-World-Bank-Group/EPM/blob/features/epm/input/data_gambia/h2) for more details.
