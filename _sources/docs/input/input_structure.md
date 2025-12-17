# Input Folder Structure

All input files must be placed inside the `epm/input` folder.

Each dataset lives in its own subfolder, which is passed to the model with `--folder_input`.

A `config.csv` file maps model parameters to the corresponding `.csv` files; the path of this file is passed with `--config` from `--folder_input`. See [Configuration](../run/configuration.md) for details on how config.csv works.

**Related Documentation:**

- [Configuration Flow](../run/configuration.md) - How config.csv, pSettings, and resources work together
- [Input Overview](input_overview.md) - Summary of all input files
- [Input Description](input_detailed.md) - Detailed parameter descriptions
- [Resources](resources.md) - Shared model constants (technologies, fuels, carbon content)
- [Input Treatment](input_treatment.md) - Automatic data validation and filling

Below is the current baseline structure for the `data_test` dataset.

To run the model with Python:

```bash
python epm.py --folder_input data_test --config config.csv
```

```plaintext
data_test/
│
├── config.csv                          # Main configuration file for baseline run
├── pHours.csv                          # Time-slice definitions
├── pSettings.csv                       # Default simulation settings
├── pSettings_NoTransmissionExpansion.csv
├── scenarios.csv                       # Scenario definitions (e.g., NoTransmissionExpansion, OptimalExpansion)
├── sensitivity.csv                     # Optional sensitivity inputs referenced by `sensitivity/`
├── y.csv                               # Year list (full horizon)
├── zcmap.csv                           # Zone-country mapping
│
├── constraint/                         # Policy and emissions constraints
│   ├── pCarbonPrice.csv
│   ├── pEmissionsCountry.csv
│   ├── pEmissionsTotal.csv
│   └── pMaxFuellimit.csv
│
├── h2/                                 # Hydrogen-specific parameters
│   ├── pAvailabilityH2.csv
│   ├── pCapexTrajectoryH2.csv
│   ├── pExternalH2.csv
│   ├── pFuelDataH2.csv
│   └── pH2DataExcel.csv
│
├── load/                               # Load and demand data
│   ├── pDemandData.csv
│   ├── pDemandForecast.csv
│   ├── pDemandProfile.csv
│   ├── pEnergyEfficiencyFactor.csv
│   └── sRelevant.csv
│
├── reserve/                            # Reserve requirements
│   ├── pPlanningReserveMargin.csv
│   ├── pSpinningReserveReqCountry.csv
│   └── pSpinningReserveReqSystem.csv
│
├── sensitivity/                        # Reduced input files listed in `sensitivity.csv`
│   └── y_reduced.csv
│
├── supply/                             # Generation and availability data
│   ├── pAvailabilityCustom.csv
│   ├── pAvailabilityDefault.csv
│   ├── pCSPData.csv
│   ├── pCapexTrajectoriesCustom.csv
│   ├── pCapexTrajectoriesDefault.csv
│   ├── pFuelPrice.csv
│   ├── pGenDataInput.csv
│   ├── pGenDataInputDefault.csv
│   ├── pStorageDataInput.csv
│   ├── pVREProfile.csv
│   ├── pVREgenProfile.csv
│   └── sensitivity/
│       └── pGenDataInput_linear.csv
│
└── trade/                              # Cross-border trade and transmission
    ├── pExtTransferLimit.csv
    ├── pLossFactorInternal.csv
    ├── pMaxAnnualExternalTradeShare.csv
    ├── pMaxPriceImportShare.csv
    ├── pMinImport.csv
    ├── pNewTransmission.csv
    ├── pNewTransmission_optimal.csv
    ├── pTradePrice.csv
    ├── pTransferLimit.csv
    └── zext.csv
```

### Scenario and Sensitivity Files

- `scenarios.csv`: Lists the core scenario names and the specific files that each scenario should pull from the dataset.
- `sensitivity.csv`: Points to alternative data sources (e.g., reduced year sets) contained in the `sensitivity/` directories at the dataset root or within specific subfolders such as `supply/sensitivity/`.
