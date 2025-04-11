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