# Common Issues & Troubleshooting

The EPM code performs several automatic checks on input data. If the model fails or produces unexpected outputs, inspect the **log file** for error messages and failed tests. This page lists common issues and how to resolve them.

> If you encounter new issues, please contact the EPM team so we can expand this list. Specifically, we suggest to raise an issue on the [EPM GitHub repository](https://github.com/ESMAP-World-Bank-Group/EPM/issues). You can assign your issue to Célia Escribe, Maëlle Baronnet and Lucas Vivier. **In any case, please do not send your personnal issue by email.** Other user may have the same issue, and it is better to have a single place where we can track all issues.

---

## Debugging Input File Errors

`90% of errors come from incorrectly formatted input files, and GAMS logs are not always helpful for pinpointing the faulty one.

To identify which input is causing the issue, enable trace mode by setting `--TRACE` to 1 in GAMS Studio. Trace mode prints the full log when reading `.csv` files, making it easier to see which file fails.

Example in `input_readers.gms`:
```gams
$onEmbeddedCode Connect:

- CSVReader:
    trace: %TRACE%
    file: input/data/pAvailability.csv
    name: pAvailability
    indexColumns: [1]
    header: [1,2]
    type: par
```

In GAMS Studio, pass --TRACE=1 as a command-line argument when running the model.
Alternatively, change $if not set TRACE $set TRACE 0 to $if not set TRACE $set TRACE 1 directly in input_readers.gms.
You may add an `$exit` statement in the code after input loading statement to simplify debugging.

Then run the model again. 
Each input file being read will appear in sequence; the faulty one is either the next one that does not appear or the last printed (based on the order in `input_readers.gms`).

Trace parameter can also be defined in the `config.csv` when using Python API with `trace=0 or 1`, but you should debug with GAMS Studio.
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

## Transmission data
- All (z, z2) pairs in `pTransferLimit` should have their corresponding (z2, z) pairs in the dataframe. An error is raised otherwise.
- Each candidate transmission line should only specified once in `pNewTransmission`. An error is raised otherwise.

## Zone definition

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

## Transmission lines

Two different files are used to define transmission capacity in the model:

- `pTransferLimit.csv`: Defines fixed capacity for existing transmission lines. Capacity must be specified for each year.
- `pNewTransmission.csv`: Defines committed and candidate lines, with status committed (`2`) or candidate (`3`). Only active when `allowTrnsferExpansion` is set to 1 in `pSettings.csv`.

If lines from `pNewTransmission.csv` do not appear in the results and no error is raised, check the following: make sure allowTransferExpansion is set to 1 in pSettings.csv.
If it is set to 0, the model will ignore `pNewTransmission.csv` completely.

## Sensitivity analysis

A number of issues may cause bugs when running a sensitivity analysis. The main ones are:

- The `sensitivity.csv` file is not correctly defined: it may be missing the sensitivity header (see section `Run EPM from Python - Advanced features` for an example), or the activation of sensitivity options may not be specified using `TRUE/FALSE`.
- Some input files may be missing required headers, which will not cause errors in regular runs but will break the sensitivity analysis. For instance, `pFuelPrice.csv` must include country and fuel as headers.
