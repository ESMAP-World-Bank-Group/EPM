# Data Structure Documentation

## Configuration Files

These files are direclty located in the `data` folder.

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


### **`y.csv`**  
  - One-dimensional, represents the years included in the intertemporal optimization.  
  - Example: [y.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/y.csv)

### **`zcmap.csv`**  
  - Two-dimensional, defines the zones and countries included in the model.  
  - Example: [zcmap.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/zcmap.csv)

## Resources

All files in this section are included in the `resources` folder. These datasets define parameters that do not necessarily change between projects.

### **`ftfindex.csv`**  
  - List of fuels recognized by EPM.  
  - 3 columns:  
    1. Fuel type  
    2. Fuel  
    3. Index (Note: Index is no longer used, and the structure is being simplified to keep only "Fuel").  
  - Example: [ftfindex.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/resources/ftfindex.csv)

### **`pTechData.csv`**  
  - List of technologies recognized by EPM.  
  - 4 columns:  
    1. Technology name  
    2. Construction period (in years)  
    3. Hourly variation (if the technology varies hourly)  
    4. RE Technology (if it is a renewable energy technology)  
  - Example: [pTechData.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/resources/pTechData.csv)

### **`pFuelCarbonContent.csv`**  
  - 2 columns:  
    1. Fuel  
    2. Carbon content in gCO₂/kWh  
  - Example: [pFuelCarbonContent.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/resources/pFuelCarbonContent.csv)

## Load Data

All files in this section are included in the `load` folder.

### **`pDemandProfile.csv`**  
  - 4 dimensions:  
    1. Zone  
    2. Season  
    3. Days  
    4. Hours  
  - 1 value column (normalized between 0 and 1).  
  - Example: [pDemandProfile.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pDemandProfile.csv)

### **`pDemandForecast.csv`**  
  - 2 dimensions:  
    1. Zone  
    2. Type 
  - 1 value column (Energy in GWh and Peak in MW)  
  - Example: [pDemandForecast.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pDemandForecast.csv)

### **`pDemandData.csv`**  
  - Alternative way to define load demand.  
  - Example: [pDemandData.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pDemandData.csv)

### **`pEnergyEfficiencyFactor.csv`**  
  - TODO: Define usage.  
  - Example: [pEnergyEfficiencyFactor.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/pEnergyEfficiencyFactor.csv)

### **`sRelevants.csv`**  
  - TODO: Define usage.  
  - Example: [sRelevants.csv](https://github.com/ESMAP-World-Bank-Group/EPM/tree/features/epm/input/data_gambia/load/sRelevants.csv)


## Supply Data

