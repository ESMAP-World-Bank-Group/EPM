# Input Treatment

The input treatment process automatically validates, transforms, and fills missing data before the GAMS model runs. This happens transparently through embedded Python code in `input_treatment.py`.

## Overview

Input treatment ensures:
1. Data consistency across all input files
2. Zone filtering based on `zcmap.csv`
3. Status validation for generators and transmission
4. Time series interpolation
5. Default value filling
6. Availability expansion across years

## Processing Pipeline

The diagram below shows the complete input treatment pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INPUT TREATMENT PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐                                                          │
│   │ Raw Input    │                                                          │
│   │ CSV Files    │                                                          │
│   └──────┬───────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  1. Zone Filtering          Filter to zones in zcmap.csv        │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  2. Generator Status        Set Capacity=0 for invalid Status   │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  3. Transmission Status     Validate transmission corridors     │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  4. Time Series Interpolation   Fill all model years            │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  5-6. Hydro Checks          Availability & Capex monitoring     │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  7-8. Default Filling       Fill NaN from defaults + StYr       │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  9-12. Availability         Expand, fill, evolve over years     │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  13. Capex Trajectories     Fill from defaults                  │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                          │
│   │ Treated      │                                                          │
│   │ Input GDX    │                                                          │
│   └──────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Processing Steps

The input treatment runs these steps in order:

```
1. Filter inputs to allowed zones
2. Zero capacity for invalid generator status
3. Zero capacity for invalid transmission status
4. Interpolate time series parameters
5. Monitor hydro availability
6. Monitor hydro capex
7. Overwrite NaN values for pGenDataInput
8. Set missing StYr for existing generators
9. Prepare pAvailability (expand to years)
10. Fill pAvailability with defaults
11. Warn about missing availability
12. Apply availability evolution
13. Fill pCapexTrajectories with defaults
```

## Step 1: Zone Filtering

Removes rows from input parameters whose zones are not defined in `zcmap.csv`.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ZONE FILTERING                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   zcmap.csv                         pGenDataInput.csv                       │
│   ┌──────────────┐                  ┌────────────────────────────────┐      │
│   │ zone,country │                  │ generator, zone, ...           │      │
│   │ ZoneA, USA   │                  │ Gen1, ZoneA, ...    ✓ Keep     │      │
│   │ ZoneB, USA   │      ───►        │ Gen2, ZoneB, ...    ✓ Keep     │      │
│   │ ZoneC, MEX   │                  │ Gen3, ZoneX, ...    ✗ Remove   │      │
│   └──────────────┘                  └────────────────────────────────┘      │
│                                                                             │
│   Result: Only data for zones defined in zcmap.csv is kept                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Affected Parameters

| Parameter | Zone Columns |
|-----------|--------------|
| `pGenDataInput` | z |
| `pGenDataInputDefault` | z |
| `pAvailabilityDefault` | z |
| `pCapexTrajectoriesDefault` | z |
| `pDemandForecast` | z |
| `pNewTransmission` | z, z2 |
| `pLossFactorInternal` | z, z2 |
| `pTransferLimit` | z, z2 |

### Behavior

- If `zcmap.csv` is missing or empty, filtering is skipped
- Rows with zones not in `zcmap` are removed
- Warnings are logged for removed rows

### Example Log Output

```
pGenDataInput: removing 5 row(s) with zones outside zcmap.
```

## Step 2: Invalid Generator Status

Generators with invalid or missing Status values have their Capacity set to 0.

### Valid Status Values

| Status | Meaning |
|--------|---------|
| 1 | Existing |
| 2 | Committed |
| 3 | Candidate |

### Behavior

- Generators with Status=0, NaN, or any value not in {1,2,3} are zeroed
- Only the Capacity field is set to 0; the generator remains in the dataset
- Warnings are logged per zone

### Example Log Output

```
Setting Capacity=0 for 3 generator row(s) with invalid/missing Status (allowed values: 1, 2, 3).
```

## Step 3: Invalid Transmission Status

Transmission corridors with invalid Status have their CapacityPerLine set to 0.

### Behavior

- Corridors with Status=0 or missing Status are zeroed
- Corridors with valid Status (non-zero) are kept
- Warnings are logged for each removed corridor

### Example Log Output

```
All transmission corridor(s) status are valid.
```
or
```
Removing 2 transmission corridor(s) from pNewTransmission due to Status=0 or missing Status.
  - ZoneA -> ZoneB (Status=missing)
  - ZoneC -> ZoneD (Status=0)
```

## Step 4: Time Series Interpolation

Linearly interpolates yearly parameters to match all model years in `y.csv`.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TIME SERIES INTERPOLATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: Sparse yearly data              Model years (y.csv):               │
│   ┌────────────────────────┐             2025, 2030, 2035, 2040, 2050       │
│   │ y=2025: 100            │                                                │
│   │ y=2035: 150            │                                                │
│   └────────────────────────┘                                                │
│              │                                                              │
│              ▼                                                              │
│   Output: Interpolated & Extrapolated                                       │
│   ┌────────────────────────────────────────────────────────┐                │
│   │ y=2025: 100.0   (original)                             │                │
│   │ y=2030: 125.0   (interpolated)                         │                │
│   │ y=2035: 150.0   (original)                             │                │
│   │ y=2040: 150.0   (extrapolated using last known value)  │                │
│   │ y=2050: 150.0   (extrapolated using last known value)  │                │
│   └────────────────────────────────────────────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Interpolated Parameters

- `pDemandForecast`
- `pCapexTrajectories`
- `pTradePrice`
- `pTransferLimit`

### Behavior

- Groups data by non-year columns (e.g., zone, technology)
- Performs linear interpolation between provided years
- Extrapolates using edge values for years outside the range

### Example

If input provides data for years 2025 and 2035:

```csv
z,y,value
ZoneA,2025,100
ZoneA,2035,150
```

And `y.csv` contains 2025, 2030, 2035:

```text
Interpolated result:
ZoneA,2025,100.0
ZoneA,2030,125.0  # interpolated
ZoneA,2035,150.0
```

### Example Log Output

```text
[input_treatment][interpolate] Linear interpolation performed on pDemandForecast to match model years 2025-2050.
```

## Step 5: Hydro Availability Monitoring

Checks that hydro generators (ReservoirHydro, ROR) have availability data.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     HYDRO AVAILABILITY AUTO-FILL                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Missing ReservoirHydro availability for generator G:                      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ 1. Same zone ReservoirHydro generators exist with availability? │       │
│   │    YES → Use mean of their availability                         │       │
│   │    NO  ↓                                                        │       │
│   ├─────────────────────────────────────────────────────────────────┤       │
│   │ 2. Global ReservoirHydro availability exists?                   │       │
│   │    YES → Use global mean                                        │       │
│   │    NO  ↓                                                        │       │
│   ├─────────────────────────────────────────────────────────────────┤       │
│   │ 3. WARNING: No availability data found                          │       │
│   │    Generator will have availability = 0 (won't dispatch)        │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│   Note: Auto-fill only runs when EPM_FILL_HYDRO_AVAILABILITY=1              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ReservoirHydro Check

Verifies each ReservoirHydro generator has entries in `pAvailability`.

### ROR Check

Verifies each ROR (run-of-river) generator has hourly profiles in `pVREgenProfile`.

### Auto-Fill (Optional)

When `EPM_FILL_HYDRO_AVAILABILITY=1` in pSettings:

- Missing ReservoirHydro availability is filled from zone/tech averages
- Missing ROR profiles can be derived from seasonal availability

### Enabling Auto-Fill

1. **Via pSettings.csv**:

   ```csv
   Abbreviation,Value
   EPM_FILL_HYDRO_AVAILABILITY,1
   ```

2. **Via environment variable**:

   ```bash
   export EPM_FILL_HYDRO_AVAILABILITY=1
   ```

### Example Log Output

```text
Reservoir hydro availability check: all 15 generator(s) defined in pGenDataInput have entries in pAvailability.
```

or

```text
[input_treatment][hydro_avail] Reservoir hydro warning: 3 generator(s) lack entries in pAvailability.
  Missing reservoir capacity-factor rows by zone:
    zone ZoneA: ['Hydro1', 'Hydro2']
    zone ZoneB: ['Hydro3']
```

## Step 6: Hydro Capex Monitoring

Checks that committed/candidate hydro generators have Capex defined.

### Target Generators

- Technologies: `ReservoirHydro`, `ROR`
- Status: 2 (Committed) or 3 (Candidate)

### Auto-Fill (Optional)

When `EPM_FILL_HYDRO_CAPEX=1` in pSettings:
- Missing Capex is filled using the mean from existing generators in the same zone and technology

### Enabling Auto-Fill

1. **Via pSettings.csv**:
   ```csv
   Abbreviation,Value
   EPM_FILL_HYDRO_CAPEX,1
   ```

2. **Via environment variable**:
   ```bash
   export EPM_FILL_HYDRO_CAPEX=1
   ```

### Example Log Output

```
Hydro capex warning: 2 generator(s) in {'ROR', 'ReservoirHydro'} with status 2 or 3 have no Capex defined.
  Missing hydro capex entries by zone:
    zone ZoneA: ['NewHydro1']
    zone ZoneB: ['NewHydro2']
  -> Auto-filled Capex for NewHydro1 (ReservoirHydro, zone: ZoneA) using the mean value 2500.000 from existing generators in the same zone and technology.
```

## Step 7: Default Value Filling (pGenDataInput)

Fills missing values in `pGenDataInput` using `pGenDataInputDefault`.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEFAULT VALUE FILLING                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Priority: Custom Input  >  Zone Defaults  >  Generic Defaults             │
│                                                                             │
│   ┌────────────────────────┐                                                │
│   │ pGenDataInput          │ ◄─── Has value? Use it (highest priority)     │
│   │ (Custom)               │                                                │
│   └───────────┬────────────┘                                                │
│               │ NaN?                                                        │
│               ▼                                                             │
│   ┌────────────────────────┐                                                │
│   │ pGenDataInputDefault   │ ◄─── Match by (zone, tech, fuel)              │
│   │ (Zone-specific)        │                                                │
│   └───────────┬────────────┘                                                │
│               │ Still NaN?                                                  │
│               ▼                                                             │
│   ┌────────────────────────┐                                                │
│   │ pGenDataInputGeneric   │ ◄─── Match by (tech, fuel) only               │
│   │ (from resources/)      │                                                │
│   └────────────────────────┘                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How It Works

1. Unstack both parameters by header (Capacity, Capex, vOM, etc.)
2. Fill NaN values in `pGenDataInput` with corresponding values from `pGenDataInputDefault`
3. Add any columns present in defaults but missing from custom input

### Matching Logic

- Matches on zone (z), technology (tech), and fuel (f)
- Custom values always take priority over defaults

### Example

`pGenDataInputDefault.csv`:

```csv
z,tech,f,pGenDataInputHeader,value
ZoneA,CCGT,Gas,Capex,800
ZoneA,CCGT,Gas,vOM,3
```

`pGenDataInput.csv`:

```csv
g,z,tech,f,pGenDataInputHeader,value
CCGT_ZoneA_1,ZoneA,CCGT,Gas,Capacity,500
```

Result: CCGT_ZoneA_1 gets Capex=800 and vOM=3 from defaults.

## Step 8: StYr for Existing Generators

Sets the start year (StYr) for existing generators (Status=1) if missing.

### Default Value

StYr is set to `(first model year) - 1`

### Behavior

- Only affects generators with Status=1
- If StYr is already set, no change is made
- Adds StYr row if completely missing

### Example Log Output

```
[input_treatment][defaults] Added StYr=2024 for 5 existing generator(s) (Status=1): Gen1, Gen2, Gen3, Gen4, Gen5.
```

## Step 9: Availability Expansion

Expands `pAvailabilityInput(g,q)` to `pAvailability(g,y,q)` by copying across all years.

### Input Format

`pAvailabilityCustom.csv` contains seasonal availability without year dimension:
```csv
g,q,value
Hydro1,q1,0.8
Hydro1,q2,0.9
```

### Output Format

Expanded to all model years:
```csv
g,y,q,value
Hydro1,2025,q1,0.8
Hydro1,2025,q2,0.9
Hydro1,2030,q1,0.8
Hydro1,2030,q2,0.9
...
```

## Step 10: Availability Default Filling

Fills missing `pAvailability` values using `pAvailabilityDefault`.

### Matching Logic

- Matches generators in `pGenDataInput` with defaults by (zone, tech, fuel)
- Default availability is then assigned to generators missing availability data

### Error Handling

If a generator has no availability after filling:
```
Warning: the following generator(s) have no entries in pAvailability and will have implicit availability of 0: ['Gen1', 'Gen2']
```

## Step 11: Availability Evolution

Applies year-dependent availability evolution factors from `pEvolutionAvailability`.

### Formula

```
pAvailability(g,y,q) = pAvailability(g,y,q) * (1 + pEvolutionAvailability(g,y))
```

### Behavior

- Only generators with entries in `pEvolutionAvailability` are affected
- Linear interpolation is performed for missing years
- Default evolution factor is 0 (no change)

### Example

`pEvolutionAvailability.csv`:
```csv
g,y,value
Hydro1,2025,0.0
Hydro1,2050,-0.1
```

This reduces Hydro1's availability by 10% by 2050, with linear interpolation between years.

## Step 12: Capex Trajectories

Fills missing `pCapexTrajectories` values using `pCapexTrajectoriesDefault`.

### Behavior

Same as availability defaults - matches by (zone, tech, fuel) and fills missing values.

## Column Renaming

The input treatment automatically renames columns to match GAMS expectations:

| Parameter | Renames |
|-----------|---------|
| `pGenDataInput` | uni → pGenDataInputHeader, gen → g, zone → z, fuel → f |
| `pAvailabilityInput` | uni → q, gen → g |
| `pNewTransmission` | From → z, To → z2, uni → pTransmissionHeader |
| `zcmap` | country → c, zone → z |
| `pDemandForecast` | type → pe, uni → y, zone → z |
| `pTransferLimit` | From → z, To → z2, uni → y |

## Debugging Input Treatment

### Run Standalone

You can test input treatment outside of GAMS:

```bash
cd epm
python input_treatment.py
```

This reads `epm/test/input.gdx`, applies all treatments, and writes to `epm/test/input_treated.gdx`.

### Enable Verbose Logging

Check the GAMS log file (e.g., `baseline_main.log`) for detailed input treatment messages:

```
============================================================
[input_treatment] starting
============================================================
------------------------------------------------------------
Filter inputs to allowed zones
------------------------------------------------------------
pGenDataInput: removing 0 row(s) with zones outside zcmap.
...
```

## Auto-Fill Settings Summary

| Setting | pSettings Key | Environment Variable | Default |
|---------|---------------|---------------------|---------|
| Hydro Availability | `EPM_FILL_HYDRO_AVAILABILITY` | `EPM_FILL_HYDRO_AVAILABILITY` | 0 (disabled) |
| Hydro Capex | `EPM_FILL_HYDRO_CAPEX` | `EPM_FILL_HYDRO_CAPEX` | 0 (disabled) |
| ROR from Availability | `EPM_FILL_ROR_FROM_AVAILABILITY` | - | 0 (disabled) |

## Common Issues

### Missing Zones

**Symptom**: Generators removed unexpectedly

**Cause**: Zone names in input files don't match `zcmap.csv`

**Solution**: Ensure zone names are consistent across all files

### Missing Availability

**Symptom**: Warning about generators with availability=0

**Cause**: Generator in `pGenDataInput` has no matching entry in `pAvailability`

**Solution**: Add availability data or check default file coverage

### NaN Values in Defaults

**Symptom**: Error about missing values in defaults

**Cause**: `pAvailabilityDefault` or `pCapexTrajectoriesDefault` missing required combinations

**Solution**: Ensure all (zone, tech, fuel) combinations have entries in default files

### Invalid Status

**Symptom**: Generator capacity unexpectedly zero

**Cause**: Status value is not 1, 2, or 3

**Solution**: Set valid Status in `pGenDataInput.csv`
