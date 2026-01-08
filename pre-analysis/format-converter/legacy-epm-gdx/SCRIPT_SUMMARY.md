# Legacy EPM GDX to CSV Converter - Summary

Converts legacy EPM GDX files into the newer CSV layout used by the EPM model.

## Core Conversion Process

1. **Reads legacy GDX file** via GAMS Transfer API and loads CSV→GDX symbol mapping from `symbol_mapping.csv`.
2. **Transforms multi-dimensional GDX parameters/sets**: Pivots specified dimensions to CSV column headers (e.g., `pFuelPrice` 3D: zone,fuel,year → zone/fuel index columns, years as headers).
3. **Writes organized CSV files** to `output/data/` following folder structure (constraint/, h2/, load/, supply/, trade/, etc.).
4. **Handles optional symbols**: Writes empty CSV stubs if symbol is missing in GDX (defined in `OPTIONAL_SYMBOLS`).
5. **Exports unmapped GDX symbols** to `extras/` folder for inspection.
6. **Skips `pSettings.csv`** export (not needed in new format).

## Configuration

- **`symbol_mapping.csv`**: Central configuration file with columns:
  - `csv_symbol`: Output CSV file name (without .csv)
  - `gdx_symbol`: GDX symbol name to read
  - `folder`: Subfolder for output (e.g., "supply", "constraint")
  - `header_cols`: Column index to unstack (0-indexed, empty if no unstacking)

## Post-Processing Pipeline

After all CSV files are exported, the script applies optional post-processing rules:

### 1. pGenDataInput Processing
- Renames first column to `gen`
- Finds and renames `fuel`, `tech`, `zone` columns (case-insensitive, handles variations like `fuel1`, `Fuel`, `type`)
- Reorders columns: `gen`, `zone`, `tech`, `fuel`, then others from `pGenDataInputHeader.csv`
- Removes columns not in `pGenDataInputHeader.csv`
- Applies lookup mappings:
  - `zone`: from `extras/pZoneIndex.csv`
  - `tech`: from `extras/pTechDataExcel.csv`
  - `fuel`: from `extras/ftfindex.csv`

### 2. Column Renaming Rules (by file)
- **Global**: Removes `element_text` column from all CSV files
- **pCarbonPrice**: First column → `year`
- **pEmissionsTotal**: First column → `year`
- **pDemandForecast**: First → `zone`, second → `type`
- **pDemandProfile**: First three → `zone`, `season`, `daytype`
- **pHours**: First three → `season`, `daytype`, `year`
- **pSpinningReserveReqSystem**: First → `year`
- **pMaxAnnualExternalTradeShare**: First → `year`
- **pAvailabilityCustom**: First → `gen`
- **pCapexTrajectoriesCustom**: First → `gen`
- **pStorageDataInput**: First → `gen`, second → `Linked plants`; renames `Capacity`→`CapacityMWh`, `Capex`→`CapexMWh`, `FixedOM`→`FixedOMMWh`, `VOMMWh`→`VOM`
- **pVREProfile**: First four → `zone`, `tech`, `season`, `daytype`
- **pExtTransferLimit**: First three → `zone`, `zext`, `season`
- **pLossFactorInternal**: First two → `From`, `To`
- **pTradePrice**: First four → `zone`, `season`, `daytype`, `year`
- **y**: First → `year`
- **zcmap**: First two → `zone`, `country`

### 3. Storage Merge
- Identifies rows in `pGenDataInput` where `tech` contains "Storage" or "Sto" (case-insensitive), excluding "STO HY" (reservoir)
- Merges these rows into `pStorageDataInput`:
  - For existing `gen` values: adds new columns from `pGenDataInput`, updates existing columns only if `pStorageDataInput` value is empty/null
  - For new `gen` values: adds as new rows
- Removes merged rows from `pGenDataInput`
- Reorders columns in `pStorageDataInput`: `gen`, `zone`, `tech`, `fuel`, `Linked plants`, then columns from `pStorageDataHeader.csv`

### 4. Default File Expansion
- Copies default files from `epm/input/data_test/supply/`:
  - `pCapexTrajectoriesDefault.csv`
  - `pAvailabilityDefault.csv`
  - `pGenDataInputDefault.csv`
- Expands each file: takes first zone as reference, duplicates rows for all zones found in `zcmap.csv`

### 5. Tech-Fuel Validation
- Validates unique tech-fuel combinations in `pGenDataInput` and `pStorageDataInput` against `epm/resources/pTechFuel.csv`
- Reports invalid combinations with suggestions based on string similarity matching
- Format: `Found in {file} tech: '{tech}' - fuel: '{fuel}'. Suggest replacement of tech: '{suggested_tech}' and fuel is valid (or should be replaced by '{suggested_fuel}').`

## Execution Order

1. Export all GDX symbols to CSV files
2. Print conversion summary report
3. Apply post-processing pipeline:
   - Process `pGenDataInput.csv`
   - Process all other CSV files (column renaming)
   - Merge Storage rows from `pGenDataInput` to `pStorageDataInput`
   - Copy and expand default files
   - Validate tech-fuel combinations

## Example Output

```
Symbols missing in GDX: pNewTransmission (expected 'pNewTransmission')
Optional symbols absent in GDX; wrote empty CSV: pAvailabilityH2 (stubbed as 'pAvailabilityH2')
Extras written:
  - MapGG -> extras/MapGG.csv
  - peak -> extras/peak.csv

Exports written under: /path/to/output/data

[Post-processing] Applying rules to pGenDataInput.csv
  [pGenDataInput] Found zone column: zone
  [pGenDataInput] Found tech column: tech
  [pGenDataInput] Applied zone mapping from pZoneIndex.csv (45/50 values mapped)

[Post-processing] Copying and expanding default files...
  [Default files] Expanded pCapexTrajectoriesDefault.csv: 10 rows × 5 zones = 50 rows

[Post-processing] Validating tech-fuel combinations...
  Found in pGenDataInput tech: 'BIOMAS' - fuel: 'Biomass'. Suggest replacement of tech: 'BiomassPlant' and fuel is valid.
```

## Usage

```bash
python legacy_to_new_format.py --gdx input/input_epm_Turkiye_v8.gdx --mapping symbol_mapping.csv
```

Optional arguments:
- `--output-base`: Base output directory (default: `./output`)
- `--target-folder`: Subfolder under output-base (default: `data`)
- `--overwrite`: Overwrite existing files (default: True)
- `--no-overwrite`: Skip existing files
