# Resource Files

Resource files are model-wide constants shared across all input folders and scenarios. They are located in `epm/resources/` and define technologies, fuels, emission factors, and parameter headers.

## Location

```
epm/
└── resources/
    ├── pTechFuel.csv
    ├── pFuelCarbonContent.csv
    ├── pSettingsHeader.csv
    ├── pGenDataInputHeader.csv
    ├── pStorageDataHeader.csv
    ├── pTransmissionHeader.csv
    ├── pH2Header.csv
    ├── pTechFuelProcessing.csv
    ├── colors.csv
    └── geojson_to_epm.csv
```

## pTechFuel.csv

Maps technologies to fuels and classifies them for model processing.

### Structure

| Column | Type | Description |
|--------|------|-------------|
| `tech` | string | Technology name (must match `pGenDataInput.tech`) |
| `fuel` | string | Associated fuel type |
| `HourlyVariation` | binary | 1 = variable renewable requiring hourly profile |
| `RETechnology` | binary | 1 = renewable energy technology |
| `FuelIndex` | integer | Numeric fuel identifier |

### Current Technologies

| tech | fuel | HourlyVariation | RETechnology | Notes |
|------|------|-----------------|--------------|-------|
| Nuclear | Uranium | 0 | 0 | Baseload |
| CCGT | Gas | 0 | 0 | Combined cycle |
| ST | Coal | 0 | 0 | Steam turbine |
| ST | Lignite | 0 | 0 | Steam turbine |
| OCGT | Diesel | 0 | 0 | Open cycle peaker |
| OCGT | Gas | 0 | 0 | Open cycle peaker |
| OCGT | HFO | 0 | 0 | Open cycle peaker |
| ICE | Diesel | 0 | 0 | Internal combustion |
| ICE | HFO | 0 | 0 | Internal combustion |
| BiomassPlant | Biomass | 0 | 1 | Renewable |
| ReservoirHydro | Water | 0 | 1 | Dispatchable hydro |
| ROR | Water | 1 | 1 | Run-of-river (variable) |
| CSPPlant | CSP | 0 | 1 | Concentrated solar |
| OffshoreWind | Wind | 1 | 1 | Variable renewable |
| OnshoreWind | Wind | 1 | 1 | Variable renewable |
| PV | Solar | 1 | 1 | Variable renewable |
| PVwSTO | Solar | 1 | 1 | PV with storage |
| STOPV | Solar | 0 | 1 | Storage paired PV |
| Storage | Battery | 0 | 0 | Battery storage |
| Storage | Water | 0 | 0 | Pumped hydro |
| ImportTransmission | Import | 0 | 0 | Import capacity |

### Adding New Technologies

To add a new technology:

1. Add a row to `pTechFuel.csv`
2. Set `HourlyVariation=1` if the technology requires hourly generation profiles (VRE)
3. Set `RETechnology=1` if it should count towards renewable targets
4. Assign a unique `FuelIndex`

## pFuelCarbonContent.csv

Defines CO2 emission factors by fuel type.

### Structure

| Column | Type | Description |
|--------|------|-------------|
| `Fuel` | string | Fuel name |
| `value` | float | CO2 emissions (tCO2/MMBtu) |

### Current Values

| Fuel | tCO2/MMBtu | Notes |
|------|------------|-------|
| Coal | 0.1031 | Highest emissions |
| Gas | 0.0592 | ~57% of coal |
| Diesel | 0.0784 | ~76% of coal |
| HFO | 0.0819 | Heavy fuel oil |
| LFO | 0.0819 | Light fuel oil |
| LNG | 0.0592 | Same as gas |
| Water | 0 | Zero emissions |
| Solar | 0 | Zero emissions |
| Wind | 0 | Zero emissions |
| Uranium | 0 | Zero direct emissions |
| CSP | 0 | Zero emissions |
| Battery | 0 | Zero direct emissions |
| Biomass | 0 | Carbon neutral |
| Geothermal | 0 | Zero emissions |
| Import | 0 | Emissions at source |

### Usage

These values are used in:
- CO2 constraint calculations (`pEmissionsTotal`, `pEmissionsCountry`)
- Carbon cost calculations when `fEnableCarbonPrice=1`
- Renewable share calculations

## pSettingsHeader.csv

Defines all valid setting abbreviations for `pSettings.csv`.

### Purpose

- Validates that settings in `pSettings.csv` are recognized
- Documents available model switches

### Current Settings (48 total)

```
fEnableCapacityExpansion
fDispatchMode
WACC
DR
VoLL
ReserveVoLL
SpinReserveVoLL
CostSurplus
CostCurtail
CO2backstop
H2UnservedCost
fEnableInternalExchange
fRemoveInternalTransferLimit
fAllowTransferExpansion
fEnableExternalExchange
sMaxHourlyImportExternalShare
sMaxHourlyExportExternalShare
fEnableCarbonPrice
fEnableEnergyEfficiency
fEnableCSP
fEnableStorage
InitialSOCforBattery
fEnableEconomicRetirement
fUseSimplifiedDemand
fCountIntercoForReserves
fApplyPlanningReserveConstraint
sReserveMarginPct
fApplyCountrySpinReserveConstraint
fApplySystemSpinReserveConstraint
sVREForecastErrorPct
sIntercoReserveContributionPct
fEnableCapexTrajectoryH2
fEnableH2Production
fApplyCountryCo2Constraint
fApplySystemCo2Constraint
sMinRenewableSharePct
sRenewableTargetYear
fApplyFuelConstraint
fApplyCapitalConstraint
sMaxCapitalInvestment
fApplyMinGenShareAllHours
fApplyMinGenCommitment
fApplyStartupCost
fApplyRampConstraint
fApplyMUDT
sPeakLoadProximityThreshold
```

## Header Files

These files define column headers for input validation.

### pGenDataInputHeader.csv

Defines valid columns for `pGenDataInput.csv` and `pGenDataInputDefault.csv`:
- Generator identification (uni, zone, tech, fuel, country)
- Capacity parameters (CapacityMW, MaxCapMW, MinCapMW)
- Cost parameters (Capex, fOM, vOM, HeatRate)
- Operational parameters (Efficiency, MinStableLevel, RampRate)
- Timing parameters (COD, RetirementYear, LifeTime)
- Status and flags (Status, Committed, DiscreteCap)

### pStorageDataHeader.csv

Defines valid columns for `pStorageDataInput.csv`:
- Storage identification (uni, zone, tech)
- Energy capacity (EnergyCapacityMWh)
- Power capacity (ChargingCapacityMW, DischargingCapacityMW)
- Efficiency (RoundTripEfficiency)
- Duration and cycling parameters

### pTransmissionHeader.csv

Defines valid columns for transmission files:
- Corridor identification (from_zone, to_zone)
- Capacity (TransferCapacityMW)
- Investment parameters (InvestmentCost, LeadTime)
- Operational parameters (Losses)

### pH2Header.csv

Defines valid columns for hydrogen input files:
- Electrolyzer parameters
- Hydrogen storage parameters
- Transport and demand parameters

## pTechFuelProcessing.csv

Additional technology classification for input processing and postprocessing.

Used by Python preprocessing to:
- Identify technologies needing special treatment
- Map technologies for aggregation in outputs
- Handle unit conversion

## colors.csv

Color definitions for visualization consistency across all outputs.

Maps technologies and fuels to hex color codes for:
- Generation mix charts
- Capacity expansion plots
- Dispatch visualizations

## geojson_to_epm.csv

Geographic mapping for visualization. Maps EPM zone names to GeoJSON identifiers for map generation.

### Format

```csv
EPM,Geojson,region,division
```

| Column | Required | Description |
|--------|----------|-------------|
| `EPM` | Yes | Zone name in your model |
| `Geojson` | Yes | Zone/country name in GeoJSON file (matches ADMIN field) |
| `region` | No | For split zones: north, south, east, west, center |
| `division` | No | Split pattern: NS (North-South), EW (East-West), NSE (3-way), NCS (3 bands) |

### Examples

**Simple zones (no splitting):**
```csv
EPM,Geojson,region,division
Angola,Angola,,
Kenya,Kenya,,
```

**Split zones (dividing a country into sub-regions):**
```csv
EPM,Geojson,region,division
DRC,Democratic Republic of the Congo,north,NSE
DRC_South,Democratic Republic of the Congo,south,NSE
DRC_East,Democratic Republic of the Congo,east,NSE
```

## How Resources Are Loaded

Resources are loaded by GAMS via `input_readers.gms`:

```gams
$if not set FOLDER_RESOURCES $set FOLDER_RESOURCES "%modeldir%resources"
$if not set pTechFuel $set pTechFuel %FOLDER_RESOURCES%/pTechFuel.csv
$if not set pFuelCarbonContent $set pFuelCarbonContent %FOLDER_RESOURCES%/pFuelCarbonContent.csv
```

The default path is `epm/resources/`, but can be overridden via GAMS command-line arguments.

## Customizing Resources

### When to Modify Resources

Modify resource files when:
- Adding new technology types not in `pTechFuel.csv`
- Changing emission factors (different fuel quality)
- Adding new model settings

### When NOT to Modify Resources

Do not modify resource files for:
- Study-specific data (use input folder files)
- Scenario variations (use scenarios.csv)
- Temporary testing (copy to input folder)

### Best Practices

1. **Version control**: Always commit resource changes with clear documentation
2. **Backward compatibility**: Adding rows is safe; removing/renaming breaks existing studies
3. **Validation**: After changes, run with test data to verify model behavior
4. **Documentation**: Update this documentation when adding new resources
