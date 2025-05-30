# Common Issues & Troubleshooting

The EPM code performs several automatic checks on input data. If the model fails or produces unexpected outputs, inspect the **log file** for error messages and failed tests. This page lists common issues and how to resolve them.

If you encounter new issues, please contact the EPM team so we can expand this list.

---

## Debugging Input File Errors

Modifying input files or reading functions may introduce errors. To help identify which input is causing issues, we recommend enabling **trace mode** in GAMS Studio.

To do this, **replace** all instances of `trace: 0` with `trace: 1` in the `emp_readers.gms` file. This will enable detailed logs for each input being read.

You can use the **Find and Replace All** feature in GAMS Studio to do this efficiently.

Example:
```gams
$onEmbeddedCode Connect:

- CSVReader:
    trace: 1
    file: input/data/pAvailability.csv
    name: pAvailability
    indexColumns: [1]
    header: [1,2]
    type: par
```

Once debugging is complete and the issue is resolved, set `trace` back to `0` to reduce log output

---

## Time Definitions

The following parameters must share the same time structure (q, d, t):
- `pHours`
- `pVREProfile`
- `pVREgenProfile`
- `pDemandProfile`

An error is raised if they are inconsistent.

---

## Default DataFrames

Files like `pAvailabilityDefault.csv` and `pCapexTrajectoriesDefault.csv` must include all combinations of **zone**, **tech**, and **fuel** defined in `pGenDataExcel`.

**Typical error**:
```
Exception from Python (line 264): <class 'ValueError'>:
Missing values in default is not permitted.
To fix this bug ensure that all combinations in pAvailabilityDefault are included.
```

---

## Fuel and Technology Naming

EPM uses standardized names for fuels and technologies. If you're upgrading from an older version (e.g., Excel-based inputs), make sure to **align your names with the updated standard**.

Do not modify the following reference files:

- `ftfindex.csv`: Authorized fuel names and indices (used for secondary fuels)
- `pTechData.csv`: Authorized tech names and properties
- `pFuelCarbonContent.csv`: Fuel-specific carbon content

Also review:

- `pGenDataExcelCustom.csv` and `pGenDataExcelDefault.csv`: Core generator data inputs. All fuel and technology names must be consistent with the files above.

Refer to the **Data Structure Documentation** for accepted naming conventions.

---

## Issues with `pGenDataExcelCustom`

- **Missing candidate plants in output**: Check that `BuildLimitperYear` is properly filled.
- **Error due to long plant names**. When some plant names are too long, this will raise an error:
  ```
  Exception from Connect: <class 'UnboundLocalError'>:
  cannot access local variable 'was_relaxed' where it is not associated with a value
  ```

---

## Issues with `pStorageDataExcel`

All **storage technologies** listed in `pGenDataExcelCustom` must also be included in `pStorageDataExcel`.

**Typical error**:
```
Exception from Python (line 455): <class 'ValueError'>:
Error: The following fuels are in gendata but not defined in pStorData:
{'New_BESS_AGO'}
```

### Transmission data
- All (z, z2) pairs in `pTransferLimit` should have their corresponding (z2, z) pairs in the dataframe. An error is raised otherwise.
- Each candidate transmission line should only specified once in `pNewTransmission`. An error is raised otherwise.

### Zone definition

Zones are defined in the file `zcmap.csv`. All other files containing zones will only consider zones which are defined in zcmap. Therefore, you should pay attention in how you spell those zones, to make sure they are being considered. In particular:
pGenDataExcel, pGenDataExcelDefault, pCapexTrajectoriesDefault, pAvailabilityDefault, pNewTransmission, pDemandProfile, pDemandForecast, pTransferLimit, pLossFactor, pVREProfile

The following input files must use zone names that match those defined in `zcmap.csv`:

The list of zones used in your model is defined in the file `zcmap.csv`. This file acts as the master reference for zone names across the entire model workflow. 
> ⚠️ All other input files that refer to zones will **only** recognize zones that are listed in `zcmap.csv`.
 
- `pGenDataExcel`
- `pGenDataExcelDefault`
- `pCapexTrajectoriesDefault`
- `pAvailabilityDefault`
- `pNewTransmission`
- `pDemandProfile`
- `pDemandForecast`
- `pTransferLimit`
- `pLossFactor`
- `pVREProfile`

We recommend validating zone names across files before launching the model to avoid silent errors or ignored data.
