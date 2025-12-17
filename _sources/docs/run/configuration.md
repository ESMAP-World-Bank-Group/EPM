# Configuration Flow

This page explains how EPM loads and uses configuration files, settings, and resources when running a simulation.

## Overview

EPM uses a layered configuration system:

```
Command Line Arguments
        ↓
    config.csv (maps parameter names → CSV files)
        ↓
    pSettings.csv (model behavior switches)
        ↓
    resources/ (shared model constants)
        ↓
    Input CSV files (data)
```

## Entry Point: `epm.py`

When you run EPM from Python:

```bash
python epm.py --folder_input data_test --config config.csv
```

The `main()` function in `epm/epm.py` parses command-line arguments and calls `launch_epm_multi_scenarios()` to orchestrate execution.

### Key Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--folder_input` | `data_test` | Input folder name inside `epm/input/` |
| `--config` | `config.csv` | Configuration file relative to folder_input |
| `--modeltype` | `None` (uses config) | Solver type: `MIP` or `RMIP` |
| `--scenarios` | `None` | Scenarios file for variant runs |
| `--selected_scenarios` | `None` | Subset of scenarios to run |
| `--cpu` | `1` | Parallel processes |
| `--sensitivity` | `False` | Enable sensitivity analysis |
| `--montecarlo` | `False` | Enable Monte Carlo analysis |
| `--debug` | `False` | Enable verbose GAMS output |

## config.csv: The Master Index

The `config.csv` file maps parameter names to their corresponding CSV data files. It serves as the central routing table for all model inputs.

**Location**: `epm/input/<folder_input>/config.csv`

**Structure**:

```csv
metadata,paramNames,file
modeltype type (MIP or RMIP),modeltype,RMIP
Cplex file path,cplexfile,cplex/cplex_baseline.opt
,GENERAL,
Global model settings,pSettings,pSettings.csv
Planning horizon years,y,y.csv
Zone-country mapping,zcmap,zcmap.csv
...
```

**Columns**:
- `metadata`: Human-readable description (ignored by code)
- `paramNames`: Parameter name used internally by the model
- `file`: Relative path to the CSV file within the input folder

### How config.csv is Loaded

1. Python reads `config.csv` and creates a dictionary mapping `paramNames → file`
2. File paths are normalized to POSIX format for cross-platform compatibility
3. Full absolute paths are constructed by prepending `folder_input`
4. The dictionary becomes the "baseline" scenario

```python
# Example of what happens internally
config = {
    'pSettings': '/path/to/epm/input/data_test/pSettings.csv',
    'y': '/path/to/epm/input/data_test/y.csv',
    'pGenDataInput': '/path/to/epm/input/data_test/supply/pGenDataInput.csv',
    ...
}
```

### Parameter Categories in config.csv

| Category | Parameters |
|----------|------------|
| **GENERAL** | pSettings, y, zcmap, pHours |
| **STATIC** | pDays, mapTS |
| **LOAD** | pDemandProfile, pDemandData, pDemandForecast, sRelevant, pEnergyEfficiencyFactor |
| **SUPPLY** | pGenDataInput, pGenDataInputDefault, pAvailability, pAvailabilityDefault, pCapexTrajectories, pCapexTrajectoriesDefault, pFuelPrice, pVREProfile, pVREgenProfile, pStorageDataInput, pCSPData |
| **RESERVE** | pPlanningReserveMargin, pSpinningReserveReqCountry, pSpinningReserveReqSystem |
| **CONSTRAINT** | pEmissionsTotal, pEmissionsCountry, pMaxFuellimit, pCarbonPrice |
| **TRADE** | zext, pExtTransferLimit, pLossFactorInternal, pMaxPriceImportShare, pMaxAnnualExternalTradeShare, pMinImport, pNewTransmission, pTradePrice, pTransferLimit |
| **H2** | pH2DataExcel, pAvailabilityH2, pCapexTrajectoryH2, pExternalH2, pFuelDataH2 |

## pSettings.csv: Model Behavior

The `pSettings.csv` file controls model behavior through feature flags and economic parameters.

**Location**: `epm/input/<folder_input>/pSettings.csv`

**Structure**:

```csv
Abbreviation,Value
fEnableCapacityExpansion,1
WACC,0.08
VoLL,5000
...
```

### Key Settings Categories

#### Capacity & Dispatch
| Setting | Description |
|---------|-------------|
| `fEnableCapacityExpansion` | Enable capacity expansion (1) or dispatch-only (0) |
| `fDispatchMode` | Dispatch mode configuration |

#### Economic Parameters
| Setting | Description | Typical Value |
|---------|-------------|---------------|
| `WACC` | Weighted average cost of capital | 0.08 (8%) |
| `DR` | Discount rate | 0.05 (5%) |
| `VoLL` | Value of Lost Load ($/MWh) | 5000 |
| `CostSurplus` | Cost of surplus energy | 0 |
| `CostCurtail` | Cost of curtailment | 0 |

#### Interconnection & Trade
| Setting | Description |
|---------|-------------|
| `fEnableInternalExchange` | Allow internal zone transfers |
| `fRemoveInternalTransferLimit` | Remove internal transfer limits |
| `fAllowTransferExpansion` | Allow new transmission investment |
| `fEnableExternalExchange` | Allow external trade |
| `sMaxHourlyImportExternalShare` | Max hourly import share |
| `sMaxHourlyExportExternalShare` | Max hourly export share |

#### Reserves
| Setting | Description |
|---------|-------------|
| `fApplyPlanningReserveConstraint` | Enforce planning reserve margin |
| `sReserveMarginPct` | Planning reserve margin (%) |
| `fApplyCountrySpinReserveConstraint` | Country spinning reserve |
| `fApplySystemSpinReserveConstraint` | System spinning reserve |
| `sVREForecastErrorPct` | VRE forecast error (%) |

#### Emissions & Renewables
| Setting | Description |
|---------|-------------|
| `fApplyCountryCo2Constraint` | Country CO2 limits |
| `fApplySystemCo2Constraint` | System CO2 limit |
| `fEnableCarbonPrice` | Enable carbon pricing |
| `sMinRenewableSharePct` | Minimum renewable share (%) |
| `sRenewableTargetYear` | Year for renewable target |

#### Special Features
| Setting | Description |
|---------|-------------|
| `fEnableCSP` | Enable CSP modeling |
| `fEnableStorage` | Enable storage modeling |
| `fEnableH2Production` | Enable hydrogen production |
| `fEnableEconomicRetirement` | Allow economic retirement |
| `fApplyStartupCost` | Include startup costs |
| `fApplyRampConstraint` | Apply ramping constraints |

## Resources: Shared Model Constants

Resource files are located in `epm/resources/` and provide model-wide definitions shared across all input folders and scenarios.

### pTechFuel.csv

Maps technologies to fuels with classification attributes.

```csv
tech,fuel,HourlyVariation,RETechnology,FuelIndex
CCGT,Gas,0,0,2
PV,Solar,1,1,4
OnshoreWind,Wind,1,1,5
ReservoirHydro,Water,0,1,3
...
```

| Column | Description |
|--------|-------------|
| `tech` | Technology name |
| `fuel` | Associated fuel |
| `HourlyVariation` | 1 = variable renewable (needs hourly profile) |
| `RETechnology` | 1 = renewable energy technology |
| `FuelIndex` | Numeric fuel identifier |

### pFuelCarbonContent.csv

Carbon emission factors by fuel type.

```csv
Fuel,value
Coal,0.1031
Gas,0.0592
Diesel,0.0784
Solar,0
Wind,0
...
```

Values are in tCO2/MMBtu.

### pSettingsHeader.csv

Defines all valid setting abbreviations (validates pSettings.csv).

### Other Resource Files

| File | Purpose |
|------|---------|
| `pGenDataInputHeader.csv` | Generator parameter definitions |
| `pStorageDataHeader.csv` | Storage parameter definitions |
| `pTransmissionHeader.csv` | Transmission parameter definitions |
| `pH2Header.csv` | Hydrogen parameter definitions |
| `colors.csv` | Visualization colors |
| `geojson_to_epm.csv` | Geographic mapping |

## Complete Execution Flow

```
1. main() in epm.py
   │
   ├── Parse command-line arguments
   │
   └── launch_epm_multi_scenarios()
       │
       ├── Read config.csv → Create baseline scenario dictionary
       ├── Read scenarios.csv (if provided) → Create variant scenarios
       ├── Apply selected_scenarios filter
       ├── Perform sensitivity analysis (if enabled)
       ├── Perform Monte Carlo sampling (if enabled)
       ├── Create output folder with timestamp
       └── Export input_scenarios.csv for documentation
       │
       └── For each scenario (in parallel via multiprocessing):
           │
           └── launch_epm() for individual scenario
               │
               ├── Create scenario subfolder
               ├── Build GAMS command with arguments
               └── Execute: gams main.gms <options>
                   │
                   └── GAMS Execution:
                       │
                       ├── Include input_readers.gms
                       │   ├── Load CSV files via GAMS Connect
                       │   └── Write all data to input.gdx
                       │
                       ├── Load input.gdx
                       │
                       ├── Run Python input_treatment (embedded):
                       │   ├── Filter to allowed zones
                       │   ├── Interpolate time series
                       │   ├── Auto-fill hydro availability
                       │   ├── Fill defaults
                       │   └── Expand availability by year
                       │
                       ├── Run input_verification
                       ├── Include base.gms (model definitions)
                       ├── Solve optimization problem
                       ├── Include generate_report.gms
                       └── Write epmresults.gdx
       │
       └── postprocess_output()
           ├── Extract results from epmresults.gdx
           ├── Generate CSV outputs
           └── Create visualizations
```

## Input Treatment and Preprocessing

Before the model runs, `input_treatment.py` automatically processes input data:

1. **Column Renaming**: Standardizes column names (e.g., `uni` → `g`, `zone` → `z`)

2. **Zone Filtering**: Removes generators/transmission not in `zcmap.csv`

3. **Status Validation**:
   - Removes generators with invalid Status
   - Sets Capacity=0 for status not in {1, 2, 3}

4. **Time Series Interpolation**: Linearly interpolates yearly parameters

5. **Default Value Handling**:
   - `pGenDataInputDefault` fills missing `pGenDataInput` values
   - `pAvailabilityDefault` fills missing `pAvailability` values
   - `pCapexTrajectoriesDefault` fills missing `pCapexTrajectories`

6. **Hydro Processing** (controlled by pSettings flags):
   - `EPM_FILL_HYDRO_AVAILABILITY`: Auto-fill missing hydro availability
   - `EPM_FILL_HYDRO_CAPEX`: Auto-fill missing hydro capex
   - `EPM_FILL_ROR_FROM_AVAILABILITY`: Fill ROR profiles from seasonal availability

7. **Availability Expansion**:
   - `pAvailabilityInput(g,q)` → `pAvailability(g,y,q)` by year
   - Applies `pEvolutionAvailability(g,y)` multipliers

## Scenarios

Scenarios are created by overlaying changes on the baseline configuration.

### scenarios.csv

```csv
paramNames,ScenarioA,ScenarioB
pSettings,pSettings_alt.csv,
pDemandForecast,,demand/high_demand.csv
pFuelPrice,supply/fuel_high.csv,supply/fuel_low.csv
```

- Empty cells inherit from baseline
- Non-empty cells override the baseline file path

### Supported Scenario Types

| Type | Description |
|------|-------------|
| **Standard** | Direct overrides via scenarios.csv |
| **Sensitivity** | Systematic parameter variation |
| **Monte Carlo** | Sample from uncertainty distributions |
| **Project Assessment** | Remove specific generators |
| **Interconnection Assessment** | Remove transmission corridors |

## Summary

1. **config.csv is the master index** - maps parameter names to CSV files
2. **pSettings.csv controls behavior** - enables/disables features and sets economic parameters
3. **Resources are model constants** - shared across all scenarios
4. **Input treatment is automatic** - fills gaps, validates, expands dimensions
5. **Scenarios inherit from baseline** - only changed parameters need specification
6. **Data flow**: CSV → GDX → Python treatment → GAMS optimization → Results GDX → Python postprocessing
