# Legacy GDX to CSV Converter - Summary

1. Reads legacy GDX file via GAMS Transfer API and loads CSV→GDX symbol mapping from `symbol_mapping.csv`.
2. Transforms multi-dimensional GDX parameters/sets: pivots specified dimensions to CSV column headers (e.g., `pFuelPrice` 3D: zone,fuel,year → zone/fuel index columns, years as headers).
3. Writes organized CSV files to `output/data/` following `CSV_LAYOUT` specification (constraint/, h2/, load/, supply/, trade/, etc.).
4. Handles optional symbols (writes empty CSV if missing) and exports unmapped GDX symbols to `extras/` folder.
5. Final log output: conversion summary DataFrame (csv_symbol, gdx_symbol, rows, path), warnings for missing/optional/empty symbols, list of extras written, and output directory path.

## Example Final Log Output

```
        csv_symbol      gdx_symbol  rows                              path
0    pCarbonPrice    pCarbonPrice    10            constraint/pCarbonPrice.csv
1  pEmissionsCountry pEmissionsCountry    15  constraint/pEmissionsCountry.csv
...
Optional symbols absent in GDX; wrote empty CSV: pAvailabilityH2 (stubbed as 'pAvailabilityH2')
Extras written:
  - MapGG -> extras/MapGG.csv
  - peak -> extras/peak.csv

Exports written under: /path/to/output/data
```

