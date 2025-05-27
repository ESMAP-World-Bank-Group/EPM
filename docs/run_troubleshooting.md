# Common Issues & Troubleshooting

The EPM code performs several automatic checks on input data. If the model fails or produces unexpected outputs, inspect the **log file** for error messages and failed tests. This page lists common issues and how to resolve them.

If you encounter new issues, please contact the EPM team so we can expand this list.

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
- **Error due to long plant names**:
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

---

## Transmission Data

- All zone pairs `(z, z2)` defined in `pTransferLimit` must have their reverse `(z2, z)` also present. Missing pairs trigger an error.
- Each candidate transmission line should be listed **only once** in `pNewTransmission`. Duplicates will trigger an error.

---

Let us know if you encounter other recurring problems—we’ll keep this list updated.