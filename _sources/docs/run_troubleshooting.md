# Troublehooting

## Common issues

### Time definition

- The parameters `pHours`, `pVREProfile`, `pVREgenProfile`, and `pDemandProfile` should have the same (q, d, t) combinations. An error is raised if they do not.

### Default dataframes

Files such as `pAvailabilityDefault.csv` and `pCapexTrajectoriesDefault.csv` should include all combinations of zone, tech, and fuel defined in pGenDataExcel. Typical error:

``` 
  File "<string>", line 134, in prepare_generatorbased_parameter
Exception from Python (line 264): <class 'ValueError'>: Missing values in default is not permitted. To fix this bug ensure that all combination in pAvailabilityDefault are included.
```

### Fuels and technologies definition

The new version of EPM uses updated naming conventions for fuels and technologies. When transitioning from an older version of the model (e.g., using Excel-based inputs), special attention must be given to aligning fuel and technology names with the new standard, as some of these are referenced directly in the GAMS code.
These appear across several input files and must be harmonized accordingly. Refer to the `Data Structure Documentation` section for the full list of authorized names. Key files involved include:

- ftfindex.csv: Contains the list of allowed fuel names and their associated indices. Primarily used to define secondary fuels where applicable. This file is standardized and should not be modified.
- pTechData.csv: Contains the list of allowed tech names and associated characteristics. This file is standardized and should not be modified.
- pFuelCarbonContent.csv: Defines fuel-specific carbon content. This file is standardized and should not be modified.
- pGenDataExcelCustom.csv and pGenDataExcelDefault: Core files where fuel and technology types are defined for each generation asset, and where default values are defined per set of fuel and technology.

### pGenDataExcelCustom

- Candidate plants are missing from the output file: **Please verify that `BuildLimitperYear` is properly defined.**
- If power plant names are too long, the following error may appear:
``` 
Exception from Connect: <class 'UnboundLocalError'>: cannot access local variable 'was_relaxed' where it is not associated with a value
*** Error executing embedded code section:
```

### pStorageDataExcel 

Storage technologies listed in `pGenDataExcelCustom` should also appear in `pStorageDataExcel`. Otherwise, an error such as the following will be raised:

``` 
Exception from Python (line 455): <class 'ValueError'>: Error: The following fuels are in gendata but not defined in pStorData: 
{'New_BESS_AGO'}
```

### Transmission data
- All (z, z2) pairs in 'pTransferLimit' should have their corresponding (z2, z) pairs in the dataframe. An error is raised otherwise.
- Each candidate transmission line should only specified once in pNewTransmission. An error is raised otherwise.